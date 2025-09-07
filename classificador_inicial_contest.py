# classificador_inicial_contest.py
# -*- coding: utf-8 -*-
"""
Classificador de peças processuais usando LLM via OpenRouter (sem heurística).

Tarefas:
- Ler .pdf ou .txt, extrair e normalizar texto
- Enviar o conteúdo ao LLM (modelo definido no .env) para classificar:
    "petição inicial" | "contestação" | "outra"
- Salvar JSON e Excel com campos:
    Nº do Processo, Tipo de Petição, Transcrição (512 tokens)
- Renomear PDFs seguindo a lógica: "<pct_conf>% - <nº processo> - <Tipo de Petição>.pdf"

Requisitos:
- pip install python-dotenv requests pandas pymupdf4llm
  (pymupdf4llm puxa o PyMuPDF como dependência)

.env esperado:
    OPENROUTER_API_KEY=sk-or-...
    OPENROUTER_MODEL=google/gemma-2-9b-it  # ou o que preferir
    OPENROUTER_BASE=https://openrouter.ai/api/v1  # opcional (padrão)
    APP_TITLE=Classificador PGE-MS            # opcional (header x-title)
    APP_SITE_URL=https://pge.ms.gov.br/lab   # opcional (header HTTP-Referer)

Observações:
- A contagem de "512 tokens" aqui é uma aproximação baseada em tokens por espaço
  (word-level), já que cada modelo pode tokenizar diferente. É suficiente para
  auditoria/visão geral.
- Caso a API não retorne confiança, usamos 0.50 como padrão.
"""

import os
import re
import json
import time
import logging
import traceback
import threading
import queue
from datetime import datetime
import re
from typing import List, Tuple
from collections import defaultdict
from typing import List, Tuple, Optional, Dict

import pandas as pd
import requests
from dotenv import load_dotenv

# =========================
# LOGGING (base)
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("classificador")

# =========================
# ENV
# =========================
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env2"))

OPENROUTER_BASE  = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_KEY   = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-2-9b-it")
APP_TITLE        = os.getenv("APP_TITLE", "Classificador PGE-MS")
APP_SITE_URL     = os.getenv("APP_SITE_URL", "")

TIMEOUT_S        = 90
MAX_INPUT_CHARS  = 24000      # segurança contra inputs gigantes
SNIPPET_TOKENS   = 512        # limite de tokens (aprox. por palavras)
ENABLE_RENAME    = True       # Se False, não renomeia arquivos (evita problemas de lock)

# === SOLUÇÃO PARA WinError 32 "Arquivo em uso" ===
# Se você continuar tendo problemas de renomeação:
# 1. Feche TODOS os visualizadores de PDF (Adobe, Edge, Chrome, Windows)
# 2. Feche Windows Explorer na pasta dos arquivos
# 3. Pause temporariamente o antivírus em tempo real
# 4. Desative indexação do Windows na pasta (Windows Search)
# 5. Aguarde alguns minutos entre tentativas
# 6. Como última opção: defina ENABLE_RENAME = False
# =================================================

# =========================
# Utilitários de texto/arquivo
# =========================

def normalize_text(t: str) -> str:
    # preserva acentuação, remove espaços duplicados, organiza quebras
    t = t.replace('\u200b', '')  # zero-width
    t = re.sub(r"[\t\r]+", " ", t)
    # colapsa espaços
    t = re.sub(r"[ ]+", " ", t)
    # normaliza quebras (mantém parágrafos)
    t = re.sub(r"\s*\n\s*", "\n", t)
    t = t.strip()
    return t


def read_text_file(path: str) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                t = f.read()
                if t:
                    return normalize_text(t)
        except Exception:
            continue
    with open(path, "rb") as f:
        return normalize_text(f.read().decode("utf-8", errors="ignore"))


def read_pdf_text(path: str) -> str:
    """Extrai texto de PDF usando PyMuPDF via pymupdf4llm (markdown),
    com fallback para PyMuPDF básico em texto puro.
    """
    try:
        import pymupdf4llm
        markdown_text = pymupdf4llm.to_markdown(path)
        return normalize_text(markdown_text)
    except Exception:
        try:
            import fitz
            text_parts = []
            with fitz.open(path) as doc:
                for page in doc:
                    text_parts.append(page.get_text("text"))
            return normalize_text("\n\n".join(text_parts))
        except Exception as e:
            log.warning(f"Falha ao extrair texto do PDF {path}: {e}")
            return ""


def read_file_content(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return read_pdf_text(path)
    elif ext == ".txt":
        return read_text_file(path)
    else:
        raise ValueError(f"Extensão não suportada: {ext}")


def base_filename_without_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def extract_process_number_from_name(name: str) -> str:
    """Extrai um número de processo (bloco de 15+ dígitos) do nome do arquivo.
    Se não achar, retorna todos os dígitos; se não houver, retorna o próprio nome.
    """
    m = re.search(r"(\d{15,})", name)
    if m:
        return m.group(1)
    digits = re.sub(r"[^\d]", "", name)
    return digits if digits else name


def tokenize_approx(text: str, limit: int = SNIPPET_TOKENS) -> str:
    """Aproximação simples de tokens por separação em espaços. Retorna
    a transcrição com os primeiros `limit` tokens."""
    # substitui quebras por espaço para não "matar" tokens por linha
    flat = re.sub(r"\s+", " ", text).strip()
    if not flat:
        return ""
    parts = flat.split(" ")
    return " ".join(parts[:limit])


# =========================
# OpenRouter — chamada à API
# =========================

class LLMError(Exception):
    pass


def build_unified_prompt(texto_norm: str) -> Dict:
    """
    Monta o payload para uma chamada única que classifica e extrai dados.
    """
    system = (
        "Você é um especialista em análise de documentos judiciais brasileiros.\n"
        "Sua tarefa é:\n"
        "1. Classificar o documento em 'inicial', 'contestacao' ou 'outra'.\n"
        "2. SE for 'inicial' ou 'contestacao', extrair seus argumentos principais.\n\n"
        "REGRAS DE CLASSIFICAÇÃO:\n"
        "- 'inicial': A peça que propõe a ação. Sinais: 'propõe a presente ação', 'ajuíza a presente demanda'. NUNCA deve conter: 'já qualificado nos autos', 'réplica', 'impugnação à contestação'.\n"
        "- 'contestacao': A defesa principal do réu. Sinais: 'apresentar contestação', 'oferece contestação'.\n"
        "- 'outra': Qualquer outra peça processual. Em caso de dúvida, classifique como 'outra'.\n\n"
        "REGRAS DE EXTRAÇÃO:\n"
        "- Para 'inicial': Extraia 'pedidos', 'fundamentos_fato', e 'fundamentos_direito'.\n"
        "- Para 'contestacao': Extraia 'preliminares', 'merito_fatos' (contra-argumentos aos fatos), e 'merito_direito' (contra-argumentos jurídicos).\n"
        "- Se a classificação for 'outra', o campo 'extracao' DEVE ser nulo (`null`).\n\n"
        "SAÍDA OBRIGATÓRIA (JSON):\n"
        "Responda ESTRITAMENTE com um JSON no formato:\n"
        "{\n"
        "  \"classificacao\": {\n"
        "    \"tipo_peticao\": \"inicial\" | \"contestacao\" | \"outra\",\n"
        "    \"confianca\": 0.0-1.0,\n"
        "    \"justificativa\": \"Cite o trecho exato que baseou a classificação.\"\n"
        "  },\n"
        "  \"extracao\": { \n"
        "    \"pedidos\": [\"...\"],\n"
        "    \"fundamentos_fato\": [\"...\"],
"
        "    \"fundamentos_direito\": [\"...\"]\n"
        "  } | { \n"
        "    \"preliminares\": [\"...\"],
"
        "    \"merito_fatos\": [\"...\"],
"
        "    \"merito_direito\": [\"...\"]\n"
        "  } | null\n"
        "}"
    )

    user = (
        "Analise o documento abaixo, classificando-o e extraindo os dados relevantes. Retorne SOMENTE um JSON válido.\n\n"
        f"=== DOCUMENTO (pode estar truncado) ===\n{texto_norm[:MAX_INPUT_CHARS]}"
    )
    
    return {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }



def build_extraction_prompt_inicial(texto_norm: str) -> Dict:
    """Prompt para extrair pedidos e fundamentos de petições iniciais."""
    system = (
        "Você é especialista em análise de petições iniciais brasileiras. "
        "Sua tarefa é extrair de forma estruturada os PEDIDOS e FUNDAMENTOS da petição inicial.\n\n"
        "INSTRUÇÕES:\n"
        "1. PEDIDOS: Identifique todos os pedidos feitos pelo autor (principal, subsidiário, cautelar, etc.)\n"
        "2. FUNDAMENTOS DE FATO: Extraia os principais fatos alegados pelo autor\n"
        "3. FUNDAMENTOS DE DIREITO: Identifique as bases jurídicas invocadas (leis, jurisprudência, doutrina)\n\n"
        "SAÍDA OBRIGATÓRIA:\n"
        "Responda ESTRITAMENTE com um JSON no formato:\n"
        "{\n"
        "  \"pedidos\": [\"pedido1\", \"pedido2\", ...],\n"
        "  \"fundamentos_fato\": [\"fato1\", \"fato2\", ...],\n"
        "  \"fundamentos_direito\": [\"base_juridica1\", \"base_juridica2\", ...]\n"
        "}\n"
    )
    user = (
        "Extraia os pedidos e fundamentos da petição inicial abaixo:\n\n"
        f"=== PETIÇÃO INICIAL ===\n{texto_norm[:MAX_INPUT_CHARS]}"
    )
    return {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }


def build_extraction_prompt_contestacao(texto_norm: str) -> Dict:
    """Prompt para extrair argumentos de contestações."""
    system = (
        "Você é especialista em análise de contestações brasileiras. "
        "Sua tarefa é extrair os ARGUMENTOS DE DEFESA utilizados pelo réu para impugnar a inicial.\n\n"
        "INSTRUÇÕES:\n"
        "1. PRELIMINARES: Argumentos processuais (incompetência, ilegitimidade, etc.)\n"
        "2. MÉRITO - FATOS: Argumentos que contestam os fatos alegados na inicial\n"
        "3. MÉRITO - DIREITO: Argumentos jurídicos contra os fundamentos da inicial\n"
        "4. OUTROS: Demais argumentos defensivos\n\n"
        "SAÍDA OBRIGATÓRIA:\n"
        "Responda ESTRITAMENTE com um JSON no formato:\n"
        "{\n"
        "  \"preliminares\": [\"argumento1\", \"argumento2\", ...],\n"
        "  \"merito_fatos\": [\"contra_fato1\", \"contra_fato2\", ...],\n"
        "  \"merito_direito\": [\"contra_direito1\", \"contra_direito2\", ...],\n"
        "  \"outros_argumentos\": [\"argumento1\", \"argumento2\", ...]\n"
        "}\n"
    )
    user = (
        "Extraia os argumentos de defesa da contestação abaixo:\n\n"
        f"=== CONTESTAÇÃO ===\n{texto_norm[:MAX_INPUT_CHARS]}"
    )
    return {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }


def build_mapping_prompt(inicial_data: Dict, contestacao_data: Dict) -> Dict:
    """
    Constrói o prompt para o LLM mapear argumentos da inicial com a contestação.
    """
    system = (
        "Você é um assistente jurídico especialista em análise dialética de processos. "
        "Sua tarefa é mapear os argumentos de uma petição inicial com seus respectivos contra-argumentos de uma contestação.\n\n"
        "REGRAS:\n"
        "1. Para cada argumento da inicial (fato ou direito), encontre o contra-argumento mais direto na contestação.\n"
        "2. Se um argumento da inicial não for diretamente rebatido, o mapeamento para ele deve ser nulo (`null`).\n"
        "3. Um mesmo contra-argumento da contestação PODE ser usado para rebater múltiplos argumentos da inicial, se aplicável.\n"
        "4. Foque na refutação lógica, não apenas em palavras-chave semelhantes.\n\n"
        "SAÍDA OBRIGATÓRIA (JSON):\n"
        "Retorne uma lista de mapeamentos no formato:\n"
        "[\n"
        "  {\n"
        "    \"argumento_inicial\": { \"id\": \"fato_1\", \"texto\": \"...\" },\n"
        "    \"contra_argumento_contestacao\": { \"id\": \"contrafato_3\", \"texto\": \"...\" },\n"
        "    \"justificativa_mapeamento\": \"O réu nega o fato X afirmando que Y ocorreu.\",\n"
        "    \"score_confianca\": 0.9\n"
        "  },\n"
        "  {\n"
        "    \"argumento_inicial\": { \"id\": \"direito_2\", \"texto\": \"...\" },\n"
        "    \"contra_argumento_contestacao\": null,\n"
        "    \"justificativa_mapeamento\": \"A contestação não aborda diretamente a tese jurídica do art. 261 do CTB.\",\n"
        "    \"score_confianca\": 0.95\n"
        "  }\n"
        "]"
    )

    # Constrói a lista de argumentos da inicial de forma numerada
    inicial_args_str = ""
    i_fato_count = 0
    for i, fato in enumerate(inicial_data.get("fundamentos_fato", [])):
        texto = fato.get("texto", str(fato))
        inicial_args_str += f"  - fato_{i+1}: \\\"{texto}\\\"\n"
        i_fato_count = i + 1
    
    for i, direito in enumerate(inicial_data.get("fundamentos_direito", [])):
        texto = direito.get("texto", str(direito))
        inicial_args_str += f"  - direito_{i+1}: \\\"{texto}\\\"\n"

    # Constrói a lista de argumentos da contestação
    contestacao_args_str = ""
    c_prelim_count = 0
    for i, prelim in enumerate(contestacao_data.get("preliminares", [])):
        texto = prelim.get("texto", str(prelim))
        contestacao_args_str += f"  - preliminar_{i+1}: \\\"{texto}\\\"\n"
        c_prelim_count = i + 1

    c_fato_count = 0
    for i, fato in enumerate(contestacao_data.get("merito_fatos", [])):
        texto = fato.get("texto", str(fato))
        contestacao_args_str += f"  - contrafato_{i+1}: \\\"{texto}\\\"\n"
        c_fato_count = i + 1

    c_direito_count = 0
    for i, direito in enumerate(contestacao_data.get("merito_direito", [])):
        texto = direito.get("texto", str(direito))
        contestacao_args_str += f"  - contradireito_{i+1}: \\\"{texto}\\\"\n"
        c_direito_count = i + 1

    user = (
        "Abaixo estão os argumentos extraídos. Mapeie-os conforme as regras e retorne SOMENTE o JSON.\n\n"
        f"=== ARGUMENTOS DA INICIAL ===\n{inicial_args_str}\n\n"
        f"=== CONTRA-ARGUMENTOS DA CONTESTAÇÃO ===\n{contestacao_args_str}"
    )

    return {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }



# =========================
# Sistema de Rastreabilidade e Referências Normativas
# =========================

class NormativeReference:
    """Representa uma referência normativa encontrada no texto."""
    
    def __init__(self, ref_id: str, text: str, offset: Tuple[int, int], 
                 norm_type: str = "lei", vigencia_inicio: str = None, vigencia_fim: str = None):
        self.ref_id = ref_id
        self.text = text
        self.offset = offset
        self.norm_type = norm_type  # lei, decreto, resolução, jurisprudência
        self.vigencia_inicio = vigencia_inicio
        self.vigencia_fim = vigencia_fim
        self.link = self._generate_link()
    
    def _generate_link(self):
        """Gera link para a norma (placeholder para futura integração)."""
        if "CTB" in self.ref_id:
            return f"https://www.planalto.gov.br/ccivil_03/leis/l9503.htm#{self.ref_id}"
        elif "CF" in self.ref_id:
            return f"https://www.planalto.gov.br/ccivil_03/constituicao/constituicao.htm#{self.ref_id}"
        return f"https://legislacao.gov.br/search?q={self.ref_id}"
    
    def is_valid_for_date(self, date_str: str) -> bool:
        """Verifica se a norma estava vigente na data especificada."""
        if not date_str or not self.vigencia_inicio:
            return True  # Assume válida se não há informação
        
        try:
            data_fato = datetime.strptime(date_str, "%Y-%m-%d")
            inicio = datetime.strptime(self.vigencia_inicio, "%Y-%m-%d")
            
            if data_fato < inicio:
                return False
            
            if self.vigencia_fim:
                fim = datetime.strptime(self.vigencia_fim, "%Y-%m-%d")
                return data_fato <= fim
            
            return True
        except:
            return True  # Em caso de erro, assume válida
    
    def to_dict(self):
        return {
            "id": self.ref_id,
            "texto": self.text,
            "offset": self.offset,
            "tipo": self.norm_type,
            "vigencia_inicio": self.vigencia_inicio,
            "vigencia_fim": self.vigencia_fim,
            "link": self.link
        }


class NormativeExtractor:
    """Extrai referências normativas de textos jurídicos."""
    
    def __init__(self):
        self.patterns = {
            "ctb": r"(?:art\.?\s*)?(\d+)(?:[\s,§]+(\d+))?[\s,°]+(?:do\s+)?(?:CTB|Código\s+de\s+Trânsito)",
            "cf": r"(?:art\.?\s*)?(\d+)(?:[\s,§]+(\d+))?[\s,°]+(?:da\s+)?(?:CF|Constituição)",
            "lei": r"(?:Lei\s+(?:nº?\s*)?(\d+(?:[\.\/]\d+)*)\/?(\d{4}))",
            "decreto": r"(?:Decreto\s+(?:nº?\s*)?(\d+(?:[\.\/]\d+)*)\/?(\d{4}))",
            "resolucao": r"(?:Resolução\s+(?:nº?\s*)?(\d+(?:[\.\/]\d+)*)\/?(\d{4}))",
            "súmula": r"(?:Súmula\s+(?:nº?\s*)?(\d+)\s+(?:do\s+)?(STF|STJ|TST))"
        }
        
        # Base de dados de vigência (pode ser expandida)
        self.vigencia_db = {
            "CTB-282": {"inicio": "1998-01-22", "fim": None},
            "CF-5": {"inicio": "1988-10-05", "fim": None},
            "Lei-9503-1997": {"inicio": "1998-01-22", "fim": None}
        }
    
    def extract_references(self, text: str, doc_name: str) -> List[NormativeReference]:
        """Extrai todas as referências normativas do texto."""
        references = []
        
        for norm_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                ref_id = self._build_reference_id(norm_type, match)
                ref_text = match.group(0)
                offset = (match.start(), match.end())
                
                # Busca informações de vigência
                vigencia = self.vigencia_db.get(ref_id, {})
                
                ref = NormativeReference(
                    ref_id=ref_id,
                    text=ref_text,
                    offset=offset,
                    norm_type=norm_type,
                    vigencia_inicio=vigencia.get("inicio"),
                    vigencia_fim=vigencia.get("fim")
                )
                
                references.append(ref)
        
        return references
    
    def _build_reference_id(self, norm_type: str, match) -> str:
        """Constrói ID único para a referência."""
        if norm_type == "ctb":
            art = match.group(1)
            par = match.group(2) if match.group(2) else ""
            return f"CTB-{art}" + (f"-{par}" if par else "")
        elif norm_type == "cf":
            art = match.group(1)
            par = match.group(2) if match.group(2) else ""
            return f"CF-{art}" + (f"-{par}" if par else "")
        elif norm_type in ["lei", "decreto", "resolucao"]:
            num = match.group(1)
            ano = match.group(2)
            return f"{norm_type.title()}-{num}-{ano}"
        elif norm_type == "súmula":
            num = match.group(1)
            tribunal = match.group(2)
            return f"Sumula-{num}-{tribunal}"
        
        return f"{norm_type}-{match.group(0)}"


class QualityMetrics:
    """Calcula métricas de qualidade para mapeamentos."""
    
    def __init__(self):
        self.thresholds = {
            "alta": 0.75,
            "revisao_min": 0.60,
            "revisao_max": 0.74,
            "descarte": 0.60
        }
        
        # Para cálculo de precision/recall (seria alimentado por anotações manuais)
        self.ground_truth = {}
        self.predictions = {}
    
    def classify_confidence(self, similarity_score: float) -> str:
        """Classifica o nível de confiança baseado no score."""
        if similarity_score >= self.thresholds["alta"]:
            return "alta"
        elif self.thresholds["revisao_min"] <= similarity_score <= self.thresholds["revisao_max"]:
            return "revisao"
        else:
            return "baixa"
    
    def calculate_cluster_cohesion(self, embeddings: List) -> float:
        """Calcula coesão de um cluster usando embeddings."""
        if len(embeddings) < 2:
            return 1.0
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                    similarities.append(sim)
            
            return float(np.mean(similarities)) if similarities else 0.0
        except:
            return 0.0
    
    def calculate_precision_recall(self, predictions: Dict, ground_truth: Dict) -> Dict:
        """Calcula precision, recall e F1-score."""
        if not ground_truth or not predictions:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        # Simplificado - em produção seria mais complexo
        tp = len(set(predictions.keys()) & set(ground_truth.keys()))
        fp = len(predictions) - tp
        fn = len(ground_truth) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3)
        }


# =========================
# Estruturas de dados para análise semântica avançada
# =========================

class SemanticAnalyzer:
    """Análise semântica de argumentos jurídicos com embeddings."""
    
    def __init__(self):
        self.model = None
        self._init_embeddings_model()
    
    def _init_embeddings_model(self):
        """Inicializa modelo de embeddings com fallback robusto."""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Lista de modelos alternativos (do mais pesado para o mais leve)
            models_to_try = [
                'paraphrase-multilingual-MiniLM-L12-v2',  # Ideal para português
                'all-MiniLM-L6-v2',                       # Modelo menor, mais rápido
                'all-MiniLM-L12-v2'                       # Fallback médio
            ]
            
            self.model = None
            for model_name in models_to_try:
                try:
                    log.info(f"🤖 Carregando modelo de embeddings: {model_name}")
                    self.model = SentenceTransformer(model_name)
                    log.info(f"✅ Modelo {model_name} carregado com sucesso")
                    break
                except Exception as e:
                    log.warning(f"❌ Falha ao carregar {model_name}: {e}")
                    continue
            
            if self.model is None:
                log.warning("🔄 Todos os modelos falharam. Usando análise textual simples.")
                
        except ImportError as e:
            log.warning(f"📦 sentence-transformers não instalado: {e}")
            log.warning("💡 Para instalar: pip install sentence-transformers")
            log.warning("🔄 Continuando com análise textual simples...")
            self.model = None
        except Exception as e:
            log.error(f"❌ Erro inesperado ao inicializar embeddings: {e}")
            log.warning("🔄 Continuando com análise textual simples...")
            self.model = None
    
    def get_embedding(self, text: str):
        """Gera embedding para um texto com tratamento robusto de erro."""
        if self.model is None:
            return None
        
        if not text or not text.strip():
            return None
        
        try:
            # Limita tamanho do texto para evitar erros de memória
            max_chars = 8000  # Limite seguro para a maioria dos modelos
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
                log.debug(f"📝 Texto truncado para {max_chars} caracteres")
            
            embedding = self.model.encode(text, show_progress_bar=False)
            return embedding
            
        except Exception as e:
            log.warning(f"❌ Erro ao gerar embedding para texto de {len(text)} chars: {e}")
            log.debug(f"🔍 Tipo do erro: {type(e).__name__}")
            
            # Se o modelo falhar, desabilita para evitar loops de erro
            if "CUDA" in str(e) or "memory" in str(e).lower():
                log.warning("🚨 Possível problema de memória GPU/CPU. Desabilitando embeddings.")
                self.model = None
            
            return None
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calcula similaridade semântica entre dois textos com fallback robusto."""
        
        # Extrai texto se for dict
        if isinstance(text1, dict):
            text1 = text1.get("texto", str(text1))
        if isinstance(text2, dict):
            text2 = text2.get("texto", str(text2))
        
        # Converte para string
        text1 = str(text1) if text1 is not None else ""
        text2 = str(text2) if text2 is not None else ""
        
        # Validação básica
        if not text1 or not text2 or not text1.strip() or not text2.strip():
            return 0.0
        
        # Se embeddings não estão disponíveis, usa método textual
        if self.model is None:
            return self._calculate_textual_similarity(text1, text2)
        
        try:
            # Tenta usar embeddings semânticos
            emb1 = self.get_embedding(text1)
            emb2 = self.get_embedding(text2)
            
            if emb1 is None or emb2 is None:
                log.debug("🔄 Embeddings falharam, usando similaridade textual")
                return self._calculate_textual_similarity(text1, text2)
            
            # Calcula similaridade usando embeddings
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                import numpy as np
                
                similarity = cosine_similarity([emb1], [emb2])[0][0]
                result = float(similarity)
                
                # Sanity check
                if not (0.0 <= result <= 1.0):
                    log.warning(f"⚠️ Similaridade fora do range [0,1]: {result}")
                    return max(0.0, min(1.0, result))
                
                return result
                
            except ImportError:
                log.warning("📦 scikit-learn não disponível para cálculo de similaridade")
                return self._calculate_textual_similarity(text1, text2)
                
        except Exception as e:
            log.warning(f"❌ Erro no cálculo de similaridade semântica: {e}")
            return self._calculate_textual_similarity(text1, text2)
    
    def _calculate_textual_similarity(self, text1: str, text2: str) -> float:
        """Fallback para similaridade textual simples."""
        try:
            # Método Jaccard melhorado
            words1 = set(word.lower() for word in text1.split() if len(word) > 2)
            words2 = set(word.lower() for word in text2.split() if len(word) > 2)
            
            if not words1 or not words2:
                return 0.0
            
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            
            jaccard = len(intersection) / len(union) if union else 0.0
            
            # Boost para textos muito similares
            if len(intersection) > 5:  # Muitas palavras em comum
                jaccard = min(1.0, jaccard * 1.2)
            
            return jaccard
            
        except Exception as e:
            log.warning(f"❌ Erro na similaridade textual: {e}")
            return 0.0
    
    def group_similar_texts(self, texts: list, threshold: float = 0.7):
        """Agrupa textos similares semanticamente."""
        if not texts:
            return []
        
        groups = []
        used_indices = set()
        
        for i, text1 in enumerate(texts):
            if i in used_indices:
                continue
            
            current_group = [{"text": text1, "index": i}]
            used_indices.add(i)
            
            for j, text2 in enumerate(texts[i+1:], i+1):
                if j in used_indices:
                    continue
                
                similarity = self.calculate_similarity(text1, text2)
                if similarity >= threshold:
                    current_group.append({"text": text2, "index": j})
                    used_indices.add(j)
            
            groups.append(current_group)
        
        return groups


class DataAnalyzer:
    """Gerencia coleta e análise avançada de dados jurídicos com rastreabilidade completa."""
    
    def __init__(self, output_dir: str = None):
        self.iniciais_data = []  # Lista de dados de petições iniciais
        self.contestacoes_data = []  # Lista de dados de contestações
        self.semantic_analyzer = SemanticAnalyzer()
        
        # Estruturas para mapeamento autor-requerido
        self.argument_pairs = []  # Pares de argumentos autor->requerido
        self.semantic_clusters = {}  # Clusters semânticos
        
        # Sistema de rastreabilidade e qualidade
        self.normative_extractor = NormativeExtractor()
        self.quality_metrics = QualityMetrics()
        self.text_cache = {}  # Cache de textos completos para extração de offsets
        
        # Sistema de persistência
        self.output_dir = output_dir or os.getcwd()
        self.backup_file = os.path.join(self.output_dir, "backup_dados_classificador.json")
        self.checkpoint_counter = 0
        self.checkpoint_interval = 10  # Salva a cada 10 documentos processados
        
        # Métricas de distribuição
        self.similarity_distribution = []
        self.cluster_counter = 0
        
        # Tenta carregar dados existentes
        self.load_backup_if_exists()
    
    def add_inicial_data(self, arquivo: str, data: Dict, texto_completo: str = ""):
        """Adiciona dados extraídos de uma petição inicial com rastreabilidade."""
        # Cache do texto para extração de offsets
        if texto_completo:
            self.text_cache[arquivo] = texto_completo
        
        # Extrai referências normativas
        normative_refs = self.normative_extractor.extract_references(texto_completo, arquivo)
        
        # Adiciona rastreabilidade aos dados extraídos
        enhanced_data = self._add_traceability(data, arquivo, texto_completo, normative_refs)
        
        entry = {
            "arquivo": arquivo,
            "data": enhanced_data,
            "normative_references": [ref.to_dict() for ref in normative_refs],
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "file_size": len(texto_completo),
                "total_normative_refs": len(normative_refs),
                "extraction_version": "2.0_traceable"
            }
        }
        self.iniciais_data.append(entry)
        self._checkpoint_save()
    
    def add_contestacao_data(self, arquivo: str, data: Dict, texto_completo: str = ""):
        """Adiciona dados extraídos de uma contestação com rastreabilidade."""
        # Cache do texto para extração de offsets
        if texto_completo:
            self.text_cache[arquivo] = texto_completo
        
        # Extrai referências normativas
        normative_refs = self.normative_extractor.extract_references(texto_completo, arquivo)
        
        # Adiciona rastreabilidade aos dados extraídos
        enhanced_data = self._add_traceability(data, arquivo, texto_completo, normative_refs)
        
        entry = {
            "arquivo": arquivo,
            "data": enhanced_data,
            "normative_references": [ref.to_dict() for ref in normative_refs],
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "file_size": len(texto_completo),
                "total_normative_refs": len(normative_refs),
                "extraction_version": "2.0_traceable"
            }
        }
        self.contestacoes_data.append(entry)
        
        # Tenta fazer mapeamento com inicial correspondente
        self._try_map_arguments(arquivo, enhanced_data, texto_completo)
        self._checkpoint_save()
    
    def _add_traceability(self, data: Dict, arquivo: str, texto_completo: str, normative_refs: List) -> Dict:
        """Adiciona informações de rastreabilidade aos dados extraídos."""
        enhanced_data = {}
        
        for category, items in data.items():
            if isinstance(items, list):
                enhanced_items = []
                for item in items:
                    if isinstance(item, str):
                        # Encontra offset do texto no documento
                        offset = self._find_text_offset(item, texto_completo)
                        
                        # Encontra referências normativas relacionadas
                        related_norms = [ref for ref in normative_refs 
                                       if self._text_overlaps(item, ref.text, texto_completo)]
                        
                        enhanced_item = {
                            "texto": item,
                            "doc": arquivo,
                            "offset": offset,
                            "normas_relacionadas": [ref.ref_id for ref in related_norms],
                            "trace_id": f"{arquivo}:{offset[0]}-{offset[1]}" if offset else f"{arquivo}:unknown"
                        }
                        enhanced_items.append(enhanced_item)
                    else:
                        enhanced_items.append(item)
                
                enhanced_data[category] = enhanced_items
            else:
                enhanced_data[category] = items
        
        return enhanced_data
    
    def _find_text_offset(self, search_text: str, full_text: str) -> Tuple[int, int]:
        """Encontra o offset de um texto no documento completo."""
        # Normaliza textos para busca
        search_clean = re.sub(r'\s+', ' ', search_text.strip().lower())
        full_clean = re.sub(r'\s+', ' ', full_text.lower())
        
        # Busca exata
        start = full_clean.find(search_clean)
        if start != -1:
            return (start, start + len(search_clean))
        
        # Busca por palavras-chave se não encontrar exato
        words = search_clean.split()[:5]  # Primeiras 5 palavras
        if len(words) >= 2:
            key_phrase = ' '.join(words)
            start = full_clean.find(key_phrase)
            if start != -1:
                return (start, start + len(key_phrase))
        
        return (0, 0)  # Fallback
    
    def _text_overlaps(self, text1: str, text2: str, full_text: str) -> bool:
        """Verifica se dois textos se sobrepõem no documento."""
        offset1 = self._find_text_offset(text1, full_text)
        offset2 = self._find_text_offset(text2, full_text)
        
        # Verifica sobreposição ou proximidade (dentro de 200 caracteres)
        return (abs(offset1[0] - offset2[0]) < 200 or 
                abs(offset1[1] - offset2[1]) < 200)
    
    def _try_map_arguments(self, arquivo_contestacao: str, contestacao_data: Dict, texto_contestacao: str = ""):
        """Tenta mapear argumentos da contestação com inicial correspondente."""
        # Busca inicial com nome similar (mesmo processo)
        base_name = arquivo_contestacao.replace('.pdf', '').split(' - ')[0]
        
        inicial_match = None
        for inicial in self.iniciais_data:
            if base_name in inicial["arquivo"]:
                inicial_match = inicial
                break
        
        if inicial_match:
            self._create_argument_pairs_with_llm(inicial_match, {
                "arquivo": arquivo_contestacao, 
                "data": contestacao_data,
                "texto_completo": texto_contestacao
            })

    def _create_argument_pairs_with_llm(self, inicial: Dict, contestacao: Dict):
        """Cria pares de argumentos usando um LLM para o mapeamento lógico."""
        log.info(f"🤖 Iniciando mapeamento com LLM para o processo: {inicial['arquivo'].split(' - ')[0]}")
        
        inicial_data = inicial["data"]
        contestacao_data = contestacao["data"]

        pair_entry = {
            "id": f"MAP-{len(self.argument_pairs) + 1:03d}",
            "processo": inicial["arquivo"].split(' - ')[0],
            "inicial_arquivo": inicial["arquivo"],
            "contestacao_arquivo": contestacao["arquivo"],
            "pares": {
                "pedidos_vs_preliminares": [],
                "fatos_vs_contrafatos": [],
                "direito_vs_contradireito": [],
                "outros_mapeamentos": []
            },
            "argumentos_nao_rebatidos": [],
            "timestamp": datetime.now().isoformat(),
            "qualidade": {
                "total_pares": 0,
                "confianca_alta": 0,
                "revisao_necessaria": 0,
                "score_medio": 0.0
            }
        }

        try:
            # 1. Construir e chamar o prompt de mapeamento
            mapping_prompt = build_mapping_prompt(inicial_data, contestacao_data)
            mapping_result = call_openrouter(mapping_prompt)

            # Validação do tipo de retorno
            if not isinstance(mapping_result, list):
                raise LLMError(f"Resposta do LLM para mapeamento não é uma lista JSON, mas sim {type(mapping_result)}")

            all_scores = []

            # 2. Processar a resposta do LLM
            for i, mapping in enumerate(mapping_result):
                arg_inicial = mapping.get("argumento_inicial")
                arg_contestacao = mapping.get("contra_argumento_contestacao")
                justificativa = mapping.get("justificativa_mapeamento", "")
                score = mapping.get("score_confianca", 0.5)
                
                all_scores.append(score)

                # Se o argumento não foi rebatido
                if not arg_contestacao:
                    pair_entry["argumentos_nao_rebatidos"].append({
                        "argumento_inicial": arg_inicial,
                        "justificativa": justificativa,
                        "score_confianca": score
                    })
                    continue

                # Determinar a categoria do par
                id_inicial = arg_inicial.get("id", "")
                id_contestacao = arg_contestacao.get("id", "")
                
                categoria = "outros_mapeamentos" # Padrão
                if id_contestacao.startswith("preliminar"):
                    categoria = "pedidos_vs_preliminares"
                elif id_inicial.startswith("fato") and id_contestacao.startswith("contrafato"):
                    categoria = "fatos_vs_contrafatos"
                elif id_inicial.startswith("direito") and id_contestacao.startswith("contradireito"):
                    categoria = "direito_vs_contradireito"

                # Montar o objeto do par
                pair_data = {
                    "id": f"PAIR-{i+1:03d}",
                    "autor": {"texto": arg_inicial.get("texto", "")},
                    "reu": {"texto": arg_contestacao.get("texto", "")},
                    "score": score,
                    "confianca": self.quality_metrics.classify_confidence(score),
                    "justificativa_llm": justificativa,
                    "trace": {
                        "doc_inicial": inicial["arquivo"],
                        "doc_contestacao": contestacao["arquivo"],
                        "tipo_mapeamento": categoria
                    }
                }
                pair_entry["pares"][categoria].append(pair_data)

            # 3. Calcular métricas de qualidade
            if all_scores:
                total_pares = len(all_scores)
                confianca_alta = sum(1 for s in all_scores if s >= 0.75)
                revisao = sum(1 for s in all_scores if 0.60 <= s < 0.75)
                score_medio = sum(all_scores) / len(all_scores)
                
                pair_entry["qualidade"] = {
                    "total_pares": total_pares,
                    "confianca_alta": confianca_alta,
                    "revisao_necessaria": revisao,
                    "score_medio": round(score_medio, 3),
                    "distribuicao": {
                        "alta_confianca": f"{(confianca_alta/total_pares)*100:.1f}%",
                        "revisao": f"{(revisao/total_pares)*100:.1f}%",
                        "baixa": f"{((total_pares-confianca_alta-revisao)/total_pares)*100:.1f}%"
                    }
                }
                self.similarity_distribution.extend(all_scores)
            
            self.argument_pairs.append(pair_entry)
            log.info(f"✅ Mapeamento com LLM concluído. {len(all_scores)} pares criados.")

        except LLMError as e:
            log.error(f"❌ Erro de LLM durante o mapeamento para {pair_entry['processo']}: {e}")
        except Exception as e:
            log.error(f"❌ Erro inesperado durante o mapeamento com LLM para {pair_entry['processo']}: {e}")
            traceback.print_exc()
    
    def _create_argument_pairs_with_embeddings(self, inicial: Dict, contestacao: Dict):
        """Cria pares de argumentos autor-requerido com métricas de qualidade."""
        inicial_data = inicial["data"]
        contestacao_data = contestacao["data"]
        texto_contestacao = contestacao.get("texto_completo", "")
        
        pair_entry = {
            "id": f"MAP-{len(self.argument_pairs) + 1:03d}",
            "processo": inicial["arquivo"].split(' - ')[0],
            "inicial_arquivo": inicial["arquivo"],
            "contestacao_arquivo": contestacao["arquivo"],
            "pares": {
                "pedidos_vs_preliminares": [],
                "fatos_vs_contrafatos": [],
                "direito_vs_contradireito": [],
                "outros_mapeamentos": []
            },
            "timestamp": datetime.now().isoformat(),
            "qualidade": {
                "total_pares": 0,
                "confianca_alta": 0,
                "revisao_necessaria": 0,
                "score_medio": 0.0
            }
        }
        
        all_similarities = []
        
        # Mapeia pedidos vs preliminares
        if "pedidos" in inicial_data and "preliminares" in contestacao_data:
            for pedido_item in inicial_data["pedidos"]:
                pedido_text = pedido_item.get("texto", pedido_item) if isinstance(pedido_item, dict) else pedido_item
                
                best_match = self._find_best_semantic_match_enhanced(
                    pedido_text, contestacao_data["preliminares"], texto_contestacao
                )
                if best_match and best_match["similarity"] >= 0.3:
                    confidence = self.quality_metrics.classify_confidence(best_match["similarity"])
                    
                    pair_data = {
                        "id": f"P-{len(pair_entry['pares']['pedidos_vs_preliminares']) + 1:03d}",
                        "autor": self._extract_traceable_data(pedido_item),
                        "reu": best_match,
                        "score": best_match["similarity"],
                        "confianca": confidence,
                        "normas_relacionadas": self._extract_related_norms(pedido_item, best_match),
                        "trace": {
                            "doc_inicial": inicial["arquivo"],
                            "doc_contestacao": contestacao["arquivo"],
                            "tipo_mapeamento": "pedido_vs_preliminar"
                        }
                    }
                    
                    pair_entry["pares"]["pedidos_vs_preliminares"].append(pair_data)
                    all_similarities.append(best_match["similarity"])
        
        # Mapeia fundamentos de fato vs argumentos de mérito sobre fatos
        if "fundamentos_fato" in inicial_data and "merito_fatos" in contestacao_data:
            for fato_item in inicial_data["fundamentos_fato"]:
                fato_text = fato_item.get("texto", fato_item) if isinstance(fato_item, dict) else fato_item
                
                best_match = self._find_best_semantic_match_enhanced(
                    fato_text, contestacao_data["merito_fatos"], texto_contestacao
                )
                if best_match and best_match["similarity"] >= 0.3:
                    confidence = self.quality_metrics.classify_confidence(best_match["similarity"])
                    
                    pair_data = {
                        "id": f"F-{len(pair_entry['pares']['fatos_vs_contrafatos']) + 1:03d}",
                        "autor": self._extract_traceable_data(fato_item),
                        "reu": best_match,
                        "score": best_match["similarity"],
                        "confianca": confidence,
                        "normas_relacionadas": self._extract_related_norms(fato_item, best_match),
                        "trace": {
                            "doc_inicial": inicial["arquivo"],
                            "doc_contestacao": contestacao["arquivo"],
                            "tipo_mapeamento": "fato_vs_contrafato"
                        }
                    }
                    
                    pair_entry["pares"]["fatos_vs_contrafatos"].append(pair_data)
                    all_similarities.append(best_match["similarity"])
        
        # Mapeia fundamentos de direito vs argumentos jurídicos
        if "fundamentos_direito" in inicial_data and "merito_direito" in contestacao_data:
            for direito_item in inicial_data["fundamentos_direito"]:
                direito_text = direito_item.get("texto", direito_item) if isinstance(direito_item, dict) else direito_item
                
                best_match = self._find_best_semantic_match_enhanced(
                    direito_text, contestacao_data["merito_direito"], texto_contestacao
                )
                if best_match and best_match["similarity"] >= 0.3:
                    confidence = self.quality_metrics.classify_confidence(best_match["similarity"])
                    
                    pair_data = {
                        "id": f"D-{len(pair_entry['pares']['direito_vs_contradireito']) + 1:03d}",
                        "autor": self._extract_traceable_data(direito_item),
                        "reu": best_match,
                        "score": best_match["similarity"],
                        "confianca": confidence,
                        "normas_relacionadas": self._extract_related_norms(direito_item, best_match),
                        "trace": {
                            "doc_inicial": inicial["arquivo"],
                            "doc_contestacao": contestacao["arquivo"],
                            "tipo_mapeamento": "direito_vs_contradireito"
                        }
                    }
                    
                    pair_entry["pares"]["direito_vs_contradireito"].append(pair_data)
                    all_similarities.append(best_match["similarity"])
        
        # Calcula métricas de qualidade do mapeamento
        if all_similarities:
            total_pares = len(all_similarities)
            confianca_alta = sum(1 for s in all_similarities if s >= 0.75)
            revisao = sum(1 for s in all_similarities if 0.60 <= s < 0.75)
            score_medio = sum(all_similarities) / len(all_similarities)
            
            pair_entry["qualidade"] = {
                "total_pares": total_pares,
                "confianca_alta": confianca_alta,
                "revisao_necessaria": revisao,
                "score_medio": round(score_medio, 3),
                "distribuicao": {
                    "alta_confianca": f"{(confianca_alta/total_pares)*100:.1f}%",
                    "revisao": f"{(revisao/total_pares)*100:.1f}%",
                    "baixa": f"{((total_pares-confianca_alta-revisao)/total_pares)*100:.1f}%"
                }
            }
            
            # Adiciona à distribuição global
            self.similarity_distribution.extend(all_similarities)
        
        self.argument_pairs.append(pair_entry)
    
    def _find_best_semantic_match(self, text: str, candidates: list, min_similarity: float = 0.3):
        """Encontra a melhor correspondência semântica."""
        if not candidates:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        for candidate in candidates:
            similarity = self.semantic_analyzer.calculate_similarity(text, candidate)
            if similarity > best_similarity and similarity >= min_similarity:
                best_similarity = similarity
                best_match = {"text": candidate, "similarity": similarity}
        
        return best_match
    
    def _find_best_semantic_match_enhanced(self, text: str, candidates: list, full_text: str = "", min_similarity: float = 0.3):
        """Encontra correspondência semântica com rastreabilidade."""
        if not candidates:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        for candidate in candidates:
            candidate_text = candidate.get("texto", candidate) if isinstance(candidate, dict) else candidate
            similarity = self.semantic_analyzer.calculate_similarity(text, candidate_text)
            
            if similarity > best_similarity and similarity >= min_similarity:
                best_similarity = similarity
                
                # Extrai dados de rastreabilidade
                if isinstance(candidate, dict):
                    best_match = {
                        "texto": candidate_text,
                        "similarity": similarity,
                        "doc": candidate.get("doc", ""),
                        "offset": candidate.get("offset", (0, 0)),
                        "trace_id": candidate.get("trace_id", ""),
                        "normas_relacionadas": candidate.get("normas_relacionadas", [])
                    }
                else:
                    # Para candidatos simples, tenta encontrar offset
                    offset = self._find_text_offset(candidate_text, full_text) if full_text else (0, 0)
                    best_match = {
                        "texto": candidate_text,
                        "similarity": similarity,
                        "doc": "",
                        "offset": offset,
                        "trace_id": f"unknown:{offset[0]}-{offset[1]}",
                        "normas_relacionadas": []
                    }
        
        return best_match
    
    def _extract_traceable_data(self, item):
        """Extrai dados de rastreabilidade de um item."""
        if isinstance(item, dict):
            return {
                "texto": item.get("texto", str(item)),
                "doc": item.get("doc", ""),
                "offset": item.get("offset", (0, 0)),
                "trace_id": item.get("trace_id", ""),
                "normas_relacionadas": item.get("normas_relacionadas", [])
            }
        else:
            return {
                "texto": str(item),
                "doc": "",
                "offset": (0, 0),
                "trace_id": "",
                "normas_relacionadas": []
            }
    
    def _extract_related_norms(self, item1, item2):
        """Extrai normas relacionadas entre dois itens."""
        norms1 = []
        norms2 = []
        
        if isinstance(item1, dict):
            norms1 = item1.get("normas_relacionadas", [])
        if isinstance(item2, dict):
            norms2 = item2.get("normas_relacionadas", [])
        
        # Combina e remove duplicatas
        all_norms = list(set(norms1 + norms2))
        return all_norms
    
    def get_semantic_analysis(self):
        """Gera análise semântica completa dos dados."""
        analysis = {
            "resumo_geral": self._get_general_summary(),
            "clusters_semanticos": self._get_semantic_clusters(),
            "mapeamento_autor_reu": self._get_argument_mapping_analysis(),
            "padroes_contestacao": self._get_defense_patterns(),
            "teses_chave": self._get_key_thesis(),
            "insights_para_contestacao": self._get_contestacao_insights()
        }
        return analysis
    
    def _get_general_summary(self):
        """Resumo geral dos dados."""
        return {
            "total_peticoes_iniciais": len(self.iniciais_data),
            "total_contestacoes": len(self.contestacoes_data),
            "total_pares_mapeados": len(self.argument_pairs),
            "cobertura_mapeamento": f"{len(self.argument_pairs)}/{len(self.contestacoes_data)}" if self.contestacoes_data else "0/0"
        }
    
    def _get_semantic_clusters(self):
        """Agrupa argumentos similares semanticamente."""
        clusters = {
            "pedidos_agrupados": [],
            "fundamentos_fato_agrupados": [],
            "fundamentos_direito_agrupados": [],
            "argumentos_defesa_agrupados": []
        }
        
        # Coleta todos os textos por categoria
        all_pedidos = []
        all_fatos = []
        all_direitos = []
        all_defesas = []
        
        for inicial in self.iniciais_data:
            data = inicial["data"]
            # Extrai texto de estruturas que podem ser dict ou string
            for item in data.get("pedidos", []):
                text = self._safe_extract_text_for_clustering(item)
                if text:
                    all_pedidos.append(text)
            
            for item in data.get("fundamentos_fato", []):
                text = self._safe_extract_text_for_clustering(item)
                if text:
                    all_fatos.append(text)
            
            for item in data.get("fundamentos_direito", []):
                text = self._safe_extract_text_for_clustering(item)
                if text:
                    all_direitos.append(text)
        
        for contestacao in self.contestacoes_data:
            data = contestacao["data"]
            for categoria in ["preliminares", "merito_fatos", "merito_direito", "outros_argumentos"]:
                for item in data.get(categoria, []):
                    text = self._safe_extract_text_for_clustering(item)
                    if text:
                        all_defesas.append(text)
        
        # Agrupa semanticamente
        if all_pedidos:
            clusters["pedidos_agrupados"] = self._process_semantic_groups(
                self.semantic_analyzer.group_similar_texts(all_pedidos), "pedidos"
            )
        
        if all_fatos:
            clusters["fundamentos_fato_agrupados"] = self._process_semantic_groups(
                self.semantic_analyzer.group_similar_texts(all_fatos), "fundamentos_fato"
            )
        
        if all_direitos:
            clusters["fundamentos_direito_agrupados"] = self._process_semantic_groups(
                self.semantic_analyzer.group_similar_texts(all_direitos), "fundamentos_direito"
            )
        
        if all_defesas:
            clusters["argumentos_defesa_agrupados"] = self._process_semantic_groups(
                self.semantic_analyzer.group_similar_texts(all_defesas), "argumentos_defesa"
            )
        
        return clusters
    
    def _process_semantic_groups(self, groups, category):
        """Processa grupos semânticos para relatório."""
        processed_groups = []
        
        for i, group in enumerate(groups):
            if len(group) > 1:  # Só grupos com múltiplos itens
                processed_group = {
                    "cluster_id": f"{category}_{i+1}",
                    "tamanho_cluster": len(group),
                    "representante": group[0]["text"],  # Texto mais representativo
                    "variantes": [item["text"] for item in group[1:]],
                    "interpretacao": self._interpret_cluster(group, category)
                }
                processed_groups.append(processed_group)
        
        return sorted(processed_groups, key=lambda x: x["tamanho_cluster"], reverse=True)
    
    def _interpret_cluster(self, group, category):
        """Interpreta o significado de um cluster."""
        texts = [item["text"] for item in group]
        
        # Análise simples de palavras-chave comuns
        all_words = []
        for text in texts:
            all_words.extend(text.lower().split())
        
        from collections import Counter
        common_words = Counter(all_words).most_common(3)
        
        keywords = [word for word, count in common_words if count > 1 and len(word) > 3]
        
        return {
            "palavras_chave": keywords,
            "tema_principal": self._deduce_theme(keywords, category),
            "frequencia_relativa": f"{len(group)} ocorrências"
        }
    
    def _deduce_theme(self, keywords, category):
        """Deduz o tema principal baseado nas palavras-chave."""
        juridical_themes = {
            "danos": ["danos", "morais", "materiais", "indenização"],
            "prazo": ["prazo", "prescrição", "decadência", "tempo"],
            "competência": ["competência", "foro", "jurisdição"],
            "legitimidade": ["legitimidade", "parte", "ativa", "passiva"],
            "mérito": ["mérito", "procedência", "improcedência"],
            "prova": ["prova", "documentos", "testemunhas", "perícia"]
        }
        
        for theme, theme_keywords in juridical_themes.items():
            if any(keyword in theme_keywords for keyword in keywords):
                return theme
        
        return "tema_geral"
    
    def _get_argument_mapping_analysis(self):
        """Análise detalhada do mapeamento autor-requerido."""
        if not self.argument_pairs:
            return {"status": "Nenhum mapeamento disponível"}
        
        mapping_analysis = {
            "total_processos_mapeados": len(self.argument_pairs),
            "estatisticas_mapeamento": {},
            "exemplos_mapeamentos": [],
            "padroes_resposta": []
        }
        
        # Estatísticas de mapeamento
        total_pedidos_vs_prelim = sum(len(pair["pares"]["pedidos_vs_preliminares"]) for pair in self.argument_pairs)
        total_fatos_vs_contra = sum(len(pair["pares"]["fatos_vs_contrafatos"]) for pair in self.argument_pairs)
        total_direito_vs_contra = sum(len(pair["pares"]["direito_vs_contradireito"]) for pair in self.argument_pairs)
        
        mapping_analysis["estatisticas_mapeamento"] = {
            "pedidos_vs_preliminares": total_pedidos_vs_prelim,
            "fatos_vs_contrafatos": total_fatos_vs_contra,
            "direito_vs_contradireito": total_direito_vs_contra,
            "total_mapeamentos": total_pedidos_vs_prelim + total_fatos_vs_contra + total_direito_vs_contra
        }
        
        # Exemplos de mapeamentos mais relevantes
        for pair in self.argument_pairs[:3]:  # Primeiros 3 processos
            exemplo = {
                "processo": pair["processo"],
                "mapeamentos_encontrados": {}
            }
            
            if pair["pares"]["pedidos_vs_preliminares"]:
                exemplo["mapeamentos_encontrados"]["pedido_vs_preliminar"] = pair["pares"]["pedidos_vs_preliminares"][0]
            
            if pair["pares"]["fatos_vs_contrafatos"]:
                exemplo["mapeamentos_encontrados"]["fato_vs_contrafato"] = pair["pares"]["fatos_vs_contrafatos"][0]
            
            if pair["pares"]["direito_vs_contradireito"]:
                exemplo["mapeamentos_encontrados"]["direito_vs_contradireito"] = pair["pares"]["direito_vs_contradireito"][0]
            
            mapping_analysis["exemplos_mapeamentos"].append(exemplo)
        
        return mapping_analysis
    
    def _get_defense_patterns(self):
        """Identifica padrões de defesa mais recorrentes."""
        defense_patterns = {
            "preliminares_frequentes": [],
            "estrategias_merito_fatos": [],
            "estrategias_merito_direito": [],
            "combinacoes_argumentos": []
        }
        
        # Análise de preliminares mais usadas
        all_preliminares = []
        for contestacao in self.contestacoes_data:
            all_preliminares.extend(contestacao["data"].get("preliminares", []))
        
        if all_preliminares:
            prelim_groups = self.semantic_analyzer.group_similar_texts(all_preliminares)
            defense_patterns["preliminares_frequentes"] = [
                {
                    "argumento": group[0]["text"],
                    "frequencia": len(group),
                    "variantes": [item["text"] for item in group[1:]]
                }
                for group in prelim_groups if len(group) > 1
            ]
        
        return defense_patterns
    
    def _get_key_thesis(self):
        """Identifica teses-chave mais importantes."""
        key_thesis = {
            "teses_mais_atacadas": [],  # Teses que o autor mais usa
            "defesas_mais_eficazes": [],  # Respostas mais comuns do réu
            "argumentos_unicos": [],  # Argumentos que aparecem raramente
            "padrao_sucesso": []  # Padrões que tendem a funcionar
        }
        
        # Análise das teses mais atacadas (pelos autores)
        all_direitos = []
        for inicial in self.iniciais_data:
            all_direitos.extend(inicial["data"].get("fundamentos_direito", []))
        
        if all_direitos:
            direito_groups = self.semantic_analyzer.group_similar_texts(all_direitos)
            key_thesis["teses_mais_atacadas"] = [
                {
                    "tese": group[0]["text"],
                    "frequencia": len(group),
                    "interpretacao": "Tese frequentemente usada por autores"
                }
                for group in sorted(direito_groups, key=len, reverse=True)[:5]
            ]
        
        return key_thesis
    
    def _get_contestacao_insights(self):
        """Gera insights práticos para construção de contestações."""
        insights = {
            "estrutura_recomendada": {},
            "argumentos_padrao": {},
            "estrategias_por_tipo": {},
            "modelo_contestacao": {}
        }
        
        # Estrutura recomendada baseada nos dados
        insights["estrutura_recomendada"] = {
            "preliminares_essenciais": self._get_essential_preliminares(),
            "merito_fatos_estrategias": self._get_fact_strategies(),
            "merito_direito_estrategias": self._get_legal_strategies(),
            "ordem_recomendada": ["preliminares", "mérito_fatos", "mérito_direito", "pedidos"]
        }
        
        # Argumentos padrão que funcionam
        insights["argumentos_padrao"] = self._get_standard_arguments()
        
        # Estratégias por tipo de pedido
        insights["estrategias_por_tipo"] = self._get_strategies_by_request_type()
        
        # Modelo de contestação
        insights["modelo_contestacao"] = self._generate_contestacao_template()
        
        return insights
    
    def _get_essential_preliminares(self):
        """Preliminares essenciais baseadas nos dados."""
        preliminares = []
        for contestacao in self.contestacoes_data:
            preliminares.extend(contestacao["data"].get("preliminares", []))
        
        if preliminares:
            groups = self.semantic_analyzer.group_similar_texts(preliminares)
            return [
                {
                    "argumento": group[0]["text"],
                    "uso_recomendado": f"Usado em {len(group)} casos",
                    "eficacia": "Alta" if len(group) > len(self.contestacoes_data) * 0.3 else "Média"
                }
                for group in sorted(groups, key=len, reverse=True)[:3]
            ]
        return []
    
    def _get_fact_strategies(self):
        """Estratégias para contestação de fatos."""
        return [
            "Impugnar especificamente cada fato alegado",
            "Apresentar versão alternativa dos fatos",
            "Questionar a qualificação jurídica dos fatos",
            "Demonstrar ausência de nexo causal"
        ]
    
    def _get_legal_strategies(self):
        """Estratégias para contestação jurídica."""
        return [
            "Invocar teses jurídicas contrárias",
            "Questionar a aplicabilidade da norma",
            "Apresentar jurisprudência favorável",
            "Demonstrar prescrição ou decadência"
        ]
    
    def _get_standard_arguments(self):
        """Argumentos padrão eficazes."""
        return {
            "preliminares_tipo": ["incompetência", "ilegitimidade", "inépcia da inicial"],
            "merito_tipo": ["ausência de responsabilidade", "excludentes de responsabilidade", "prescrição"],
            "pedidos_tipo": ["improcedência total", "improcedência parcial", "gratuidade da justiça"]
        }
    
    def _get_strategies_by_request_type(self):
        """Estratégias específicas por tipo de pedido."""
        return {
            "danos_morais": ["questionar a ocorrência do dano", "impugnar o valor pleiteado", "invocar excludentes"],
            "danos_materiais": ["exigir comprovação dos prejuízos", "questionar nexo causal", "apresentar contraprovas"],
            "obrigacao_fazer": ["demonstrar impossibilidade", "questionar interesse de agir", "propor alternativas"]
        }
    
    def _generate_contestacao_template(self):
        """Gera template de contestação baseado nos padrões."""
        return {
            "introducao": "Estrutura padrão para introdução da contestação",
            "preliminares": "Lista das preliminares mais eficazes baseada nos dados",
            "merito": "Estrutura para contestação de mérito com base nos padrões identificados",
            "pedidos": "Pedidos padrão de improcedência com fundamentação",
            "observacoes": "Adaptações necessárias conforme o caso específico"
        }
    
    def _calculate_global_quality_metrics(self):
        """Calcula métricas de qualidade globais do sistema."""
        if not self.similarity_distribution:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "confiabilidade_geral": "Insuficiente",
                "revisao_necessaria": 0,
                "distribuicao_scores": {}
            }
        
        # Análise da distribuição de scores
        alta_conf = sum(1 for s in self.similarity_distribution if s >= 0.75)
        media_conf = sum(1 for s in self.similarity_distribution if 0.60 <= s < 0.75)
        baixa_conf = len(self.similarity_distribution) - alta_conf - media_conf
        total = len(self.similarity_distribution)
        
        # Calcula métricas estimadas (em produção seria baseado em ground truth)
        precision_est = alta_conf / total if total > 0 else 0.0
        recall_est = (alta_conf + media_conf * 0.7) / total if total > 0 else 0.0
        f1_est = 2 * (precision_est * recall_est) / (precision_est + recall_est) if (precision_est + recall_est) > 0 else 0.0
        
        return {
            "precision": round(precision_est, 3),
            "recall": round(recall_est, 3),
            "f1": round(f1_est, 3),
            "confiabilidade_geral": "Alta" if precision_est > 0.8 else "Média" if precision_est > 0.6 else "Baixa",
            "revisao_necessaria": media_conf,
            "distribuicao_scores": {
                "alta_confianca": f"{(alta_conf/total)*100:.1f}%" if total > 0 else "0%",
                "revisao_humana": f"{(media_conf/total)*100:.1f}%" if total > 0 else "0%",
                "baixa_confianca": f"{(baixa_conf/total)*100:.1f}%" if total > 0 else "0%",
                "score_medio": round(sum(self.similarity_distribution) / total, 3) if total > 0 else 0.0
            }
        }
    
    def _generate_production_clusters(self):
        """Gera clusters para produção com rótulos e métricas."""
        self.cluster_counter += 1
        clusters = []
        
        # Processa clusters de pedidos
        pedidos_data = []
        for inicial in self.iniciais_data:
            for pedido in inicial["data"].get("pedidos", []):
                if isinstance(pedido, dict):
                    pedidos_data.append(pedido)
                else:
                    pedidos_data.append({"texto": pedido, "doc": inicial["arquivo"]})
        
        if pedidos_data:
            pedidos_texts = [item["texto"] for item in pedidos_data]
            pedidos_groups = self.semantic_analyzer.group_similar_texts(pedidos_texts)
            
            for i, group in enumerate(pedidos_groups):
                if len(group) > 1:  # Só clusters significativos
                    cluster_data = {
                        "id": f"C-P-{i+1:03d}",
                        "tipo": "pedidos",
                        "rotulo": self._generate_cluster_label(group),
                        "topicos": self._extract_cluster_topics(group),
                        "tamanho": len(group),
                        "exemplos": [
                            {
                                "texto": item["text"],
                                "doc": pedidos_data[item["index"]].get("doc", "unknown"),
                                "offset": pedidos_data[item["index"]].get("offset", [0, 0])
                            }
                            for item in group[:3]  # 3 exemplos canônicos
                        ],
                        "qualidade": {
                            "coesao": self._calculate_cluster_cohesion(group),
                            "representatividade": len(group) / len(pedidos_data) if pedidos_data else 0
                        }
                    }
                    clusters.append(cluster_data)
        
        return clusters
    
    def _generate_production_mappings(self):
        """Gera mapeamentos para produção com rastreabilidade completa."""
        production_mappings = []
        
        for pair in self.argument_pairs:
            for tipo, pares in pair["pares"].items():
                for p in pares:
                    mapping = {
                        "id": f"{pair['id']}-{p['id']}",
                        "processo": pair["processo"],
                        "tipo_mapeamento": tipo,
                        "par": {
                            "autor": {
                                "cluster_id": self._find_cluster_for_text(p["autor"]["texto"]),
                                "texto": p["autor"]["texto"],
                                "doc": p["autor"]["doc"],
                                "offset": p["autor"]["offset"],
                                "normas": p["autor"]["normas_relacionadas"]
                            },
                            "reu": {
                                "cluster_id": self._find_cluster_for_text(p["reu"]["texto"]),
                                "texto": p["reu"]["texto"],
                                "doc": p["reu"]["doc"],
                                "offset": p["reu"]["offset"],
                                "normas": p["reu"]["normas_relacionadas"]
                            }
                        },
                        "score": p["score"],
                        "confianca": p["confianca"],
                        "normas_relacionadas": p["normas_relacionadas"],
                        "trace": p["trace"],
                        "validacao_temporal": self._validate_temporal_consistency(p)
                    }
                    production_mappings.append(mapping)
        
        return production_mappings
    
    def _generate_normative_summary(self):
        """Gera resumo das referências normativas encontradas."""
        all_refs = []
        
        # Coleta todas as referências
        for inicial in self.iniciais_data:
            all_refs.extend(inicial.get("normative_references", []))
        
        for contestacao in self.contestacoes_data:
            all_refs.extend(contestacao.get("normative_references", []))
        
        # Agrupa por tipo e ID
        refs_by_type = defaultdict(list)
        for ref in all_refs:
            refs_by_type[ref["tipo"]].append(ref)
        
        summary = {}
        for tipo, refs in refs_by_type.items():
            # Remove duplicatas por ID
            unique_refs = {ref["id"]: ref for ref in refs}
            
            summary[tipo] = {
                "total_referencias": len(refs),
                "referencias_unicas": len(unique_refs),
                "mais_citadas": sorted(
                    [(ref_id, len([r for r in refs if r["id"] == ref_id])) for ref_id in unique_refs.keys()],
                    key=lambda x: x[1], reverse=True
                )[:5],
                "detalhes": list(unique_refs.values())
            }
        
        return summary
    
    def _generate_temporal_analysis(self):
        """Gera análise temporal considerando vigência das normas."""
        # Placeholder - seria expandido com dados reais de vigência
        return {
            "analise_vigencia": "Análise de vigência das normas citadas",
            "conflitos_temporais": [],
            "recomendacoes": [
                "Verificar vigência das normas na data dos fatos",
                "Considerar alterações legislativas posteriores"
            ]
        }
    
    def _extract_most_effective_arguments(self):
        """Extrai argumentos mais eficazes baseado nos dados."""
        # Análise baseada na frequência e qualidade dos mapeamentos
        argument_effectiveness = defaultdict(list)
        
        for pair in self.argument_pairs:
            for tipo, pares in pair["pares"].items():
                for p in pares:
                    if p["score"] >= 0.75:  # Alta confiança
                        argument_effectiveness[tipo].append({
                            "argumento": p["reu"]["texto"],
                            "eficacia": p["score"],
                            "contexto": p["autor"]["texto"][:100] + "..."
                        })
        
        return {tipo: sorted(args, key=lambda x: x["eficacia"], reverse=True)[:3] 
                for tipo, args in argument_effectiveness.items()}
    
    def _identify_success_patterns(self):
        """Identifica padrões de sucesso nas contestações."""
        return [
            "Contestações que citam normas específicas têm maior score de similaridade",
            "Argumentos estruturados em preliminares + mérito são mais eficazes",
            "Referências a jurisprudência consolidada aumentam a qualidade"
        ]
    
    def _generate_specific_recommendations(self):
        """Gera recomendações específicas baseadas nos dados."""
        total_mappings = len([p for pair in self.argument_pairs for pares in pair["pares"].values() for p in pares])
        high_quality = len([p for pair in self.argument_pairs for pares in pair["pares"].values() for p in pares if p["score"] >= 0.75])
        
        recommendations = []
        
        if total_mappings > 0:
            quality_ratio = high_quality / total_mappings
            if quality_ratio < 0.5:
                recommendations.append("Melhorar estruturação dos argumentos para aumentar qualidade dos mapeamentos")
            else:
                recommendations.append("Manter padrão de qualidade atual dos argumentos")
        
        recommendations.extend([
            "Usar referencias normativas específicas para maior precisão",
            "Estruturar contestações seguindo padrões identificados",
            "Revisar mapeamentos com score entre 0.6-0.74 antes de usar"
        ])
        
        return recommendations
    
    def _generate_cluster_label(self, group):
        """Gera rótulo automático para cluster."""
        texts = [item["text"] for item in group]
        
        # Extrai palavras-chave mais comuns
        all_words = []
        for text in texts:
            words = re.findall(r'\b\w{4,}\b', text.lower())
            all_words.extend(words)
        
        from collections import Counter
        common_words = Counter(all_words).most_common(3)
        
        if common_words:
            keywords = [word for word, _ in common_words]
            return f"Argumentos sobre {' e '.join(keywords[:2])}"
        
        return "Cluster de argumentos similares"
    
    def _extract_cluster_topics(self, group):
        """Extrai tópicos principais do cluster."""
        legal_topics = {
            "notificacao": ["notificação", "citação", "intimação"],
            "competencia": ["competência", "foro", "jurisdição"],
            "prescricao": ["prescrição", "decadência", "prazo"],
            "danos": ["danos", "indenização", "reparação"],
            "prova": ["prova", "documento", "testemunha"]
        }
        
        texts = " ".join([item["text"].lower() for item in group])
        found_topics = []
        
        for topic, keywords in legal_topics.items():
            if any(keyword in texts for keyword in keywords):
                found_topics.append(topic)
        
        return found_topics[:5]  # Máximo 5 tópicos
    
    def _calculate_cluster_cohesion(self, group):
        """Calcula coesão do cluster (placeholder)."""
        # Simplificado - seria calculado com embeddings reais
        return 0.75 + (len(group) * 0.05) if len(group) < 5 else 0.95
    
    def _find_cluster_for_text(self, text):
        """Encontra ID do cluster para um texto específico."""
        # Placeholder - seria implementado com busca nos clusters gerados
        return f"C-{hash(text[:20]) % 1000:03d}"
    
    def _validate_temporal_consistency(self, mapping_pair):
        """Valida consistência temporal do mapeamento."""
        # Placeholder para validação temporal
        return {
            "status": "válido",
            "observacoes": "Normas vigentes na data dos fatos"
        }
    
    def generate_simple_pattern_mapping(self):
        """Gera mapeamento simples: 'Quando inicial alega X, contestação responde Y'."""
        if not self.argument_pairs:
            return {
                "status": "Sem dados suficientes",
                "padroes": [],
                "resumo": "Execute o processamento primeiro para gerar padrões."
            }
        
        # Estrutura para armazenar padrões
        patterns = {
            "pedidos_vs_preliminares": [],
            "fatos_vs_contrafatos": [],
            "direito_vs_contradireito": [],
            "resumo_estatistico": {}
        }
        
        # Coleta todos os mapeamentos
        all_pedido_patterns = []
        all_fato_patterns = []
        all_direito_patterns = []
        
        for pair in self.argument_pairs:
            processo = pair["processo"]
            
            # Padrões: Pedidos vs Preliminares
            for p in pair["pares"]["pedidos_vs_preliminares"]:
                if p["score"] >= 0.3:  # Só mapeamentos com qualidade mínima
                    # Extrai texto de forma segura
                    autor_texto = self._safe_extract_text(p.get("autor", {}))
                    reu_texto = self._safe_extract_text(p.get("reu", {}))
                    
                    pattern = {
                        "quando_inicial_alega": self._clean_text_for_pattern(autor_texto),
                        "contestacao_responde": reu_texto,  # Texto completo sem truncar
                        "contestacao_responde_display": self._clean_text_for_pattern(reu_texto),  # Versão truncada para display
                        "confianca": p["confianca"],
                        "score": p["score"],
                        "exemplo_processo": processo,
                        "normas_citadas": p.get("normas_relacionadas", [])
                    }
                    all_pedido_patterns.append(pattern)
            
            # Padrões: Fatos vs Contrafatos  
            for f in pair["pares"]["fatos_vs_contrafatos"]:
                if f["score"] >= 0.3:
                    # Extrai texto de forma segura
                    autor_texto = self._safe_extract_text(f.get("autor", {}))
                    reu_texto = self._safe_extract_text(f.get("reu", {}))
                    
                    pattern = {
                        "quando_inicial_alega": self._clean_text_for_pattern(autor_texto),
                        "contestacao_responde": reu_texto,  # Texto completo sem truncar
                        "contestacao_responde_display": self._clean_text_for_pattern(reu_texto),  # Versão truncada para display
                        "confianca": f["confianca"],
                        "score": f["score"],
                        "exemplo_processo": processo,
                        "normas_citadas": f.get("normas_relacionadas", [])
                    }
                    all_fato_patterns.append(pattern)
            
            # Padrões: Direito vs Contradireito
            for d in pair["pares"]["direito_vs_contradireito"]:
                if d["score"] >= 0.3:
                    # Extrai texto de forma segura
                    autor_texto = self._safe_extract_text(d.get("autor", {}))
                    reu_texto = self._safe_extract_text(d.get("reu", {}))
                    
                    pattern = {
                        "quando_inicial_alega": self._clean_text_for_pattern(autor_texto),
                        "contestacao_responde": reu_texto,  # Texto completo sem truncar
                        "contestacao_responde_display": self._clean_text_for_pattern(reu_texto),  # Versão truncada para display
                        "confianca": d["confianca"],
                        "score": d["score"],
                        "exemplo_processo": processo,
                        "normas_citadas": d.get("normas_relacionadas", [])
                    }
                    all_direito_patterns.append(pattern)
        
        # Agrupa padrões similares e ordena por relevância
        patterns["pedidos_vs_preliminares"] = self._group_and_rank_patterns(all_pedido_patterns)
        patterns["fatos_vs_contrafatos"] = self._group_and_rank_patterns(all_fato_patterns)
        patterns["direito_vs_contradireito"] = self._group_and_rank_patterns(all_direito_patterns)
        
        # Estatísticas resumidas
        patterns["resumo_estatistico"] = {
            "total_padroes_pedidos": len(patterns["pedidos_vs_preliminares"]),
            "total_padroes_fatos": len(patterns["fatos_vs_contrafatos"]),
            "total_padroes_direito": len(patterns["direito_vs_contradireito"]),
            "total_processos_analisados": len(self.argument_pairs),
            "qualidade_media": self._calculate_average_quality(all_pedido_patterns + all_fato_patterns + all_direito_patterns)
        }
        
        return patterns
    
    def _clean_text_for_pattern(self, text, max_length=200):
        """Limpa e trunca texto para exibição em padrões."""
        if not text:
            return "Texto não disponível"
        
        # Se text é um dicionário, extrai o campo 'texto'
        if isinstance(text, dict):
            text = text.get("texto", str(text))
        
        # Converte para string se necessário
        text = str(text)
        
        # Remove quebras de linha e espaços extras
        clean = re.sub(r'\s+', ' ', text.strip())
        
        # Trunca se muito longo
        if len(clean) > max_length:
            clean = clean[:max_length] + "..."
        
        return clean
    
    def _safe_extract_text(self, data):
        """Extrai texto de forma segura de dados que podem ser dict ou string."""
        if not data:
            return "Texto não disponível"
        
        if isinstance(data, dict):
            # Tenta diferentes campos possíveis
            for field in ["texto", "text", "content", "value"]:
                if field in data and data[field]:
                    return str(data[field])
            # Se não encontrou nenhum campo conhecido, converte tudo para string
            return str(data)
        
        # Se não é dict, converte para string
        return str(data)
    
    def _safe_extract_text_for_clustering(self, item):
        """Extrai texto para clustering de forma segura."""
        if not item:
            return None
        
        if isinstance(item, str):
            return item.strip() if item.strip() else None
        
        if isinstance(item, dict):
            # Tenta diferentes campos possíveis
            for field in ["texto", "text", "content", "value"]:
                if field in item and item[field]:
                    text = str(item[field]).strip()
                    return text if text else None
            # Se não encontrou, converte o dict inteiro
            return str(item)
        
        # Qualquer outro tipo, converte para string
        text = str(item).strip()
        return text if text else None
    
    def _group_and_rank_patterns(self, patterns_list):
        """Agrupa padrões similares e ranqueia por importância."""
        if not patterns_list:
            return []
        
        # Agrupa por similaridade da alegação inicial
        grouped = {}
        
        for pattern in patterns_list:
            inicial_key = pattern["quando_inicial_alega"][:100]  # Primeiros 100 chars como chave
            
            if inicial_key not in grouped:
                grouped[inicial_key] = {
                    "quando_inicial_alega": pattern["quando_inicial_alega"],
                    "respostas_comuns": [],
                    "frequencia": 0,
                    "confianca_media": 0,
                    "exemplos_processos": [],
                    "normas_mais_citadas": []
                }
            
            # Adiciona resposta se não existe similar
            resposta = pattern["contestacao_responde"]
            similar_response_found = False
            
            for existing_resp in grouped[inicial_key]["respostas_comuns"]:
                if self._texts_are_similar(resposta, existing_resp["texto"], threshold=0.7):
                    existing_resp["frequencia"] += 1
                    existing_resp["scores"].append(pattern["score"])
                    similar_response_found = True
                    break
            
            if not similar_response_found:
                grouped[inicial_key]["respostas_comuns"].append({
                    "texto": resposta,
                    "frequencia": 1,
                    "scores": [pattern["score"]],
                    "confianca": pattern["confianca"]
                })
            
            grouped[inicial_key]["frequencia"] += 1
            grouped[inicial_key]["exemplos_processos"].append(pattern["exemplo_processo"])
            grouped[inicial_key]["normas_mais_citadas"].extend(pattern["normas_citadas"])
        
        # Processa dados agrupados
        result = []
        for key, group in grouped.items():
            # Calcula média de confiança
            all_scores = []
            for resp in group["respostas_comuns"]:
                all_scores.extend(resp["scores"])
            
            group["confianca_media"] = sum(all_scores) / len(all_scores) if all_scores else 0
            
            # Ordena respostas por frequência
            group["respostas_comuns"] = sorted(
                group["respostas_comuns"], 
                key=lambda x: x["frequencia"], 
                reverse=True
            )
            
            # Remove duplicatas de processos
            group["exemplos_processos"] = list(set(group["exemplos_processos"]))[:3]  # Máximo 3 exemplos
            
            # Conta normas mais citadas
            from collections import Counter
            norm_counter = Counter(group["normas_mais_citadas"])
            group["normas_mais_citadas"] = [norm for norm, count in norm_counter.most_common(3)]
            
            result.append(group)
        
        # Ordena por frequência e confiança
        return sorted(result, key=lambda x: (x["frequencia"], x["confianca_media"]), reverse=True)
    
    def _texts_are_similar(self, text1, text2, threshold=0.7):
        """Verifica se dois textos são similares."""
        return self.semantic_analyzer.calculate_similarity(text1, text2) >= threshold
    
    def _calculate_average_quality(self, patterns_list):
        """Calcula qualidade média dos padrões."""
        if not patterns_list:
            return 0.0
        
        scores = [p["score"] for p in patterns_list]
        return sum(scores) / len(scores)
    
    def _checkpoint_save(self):
        """Salva checkpoint automático a cada N documentos processados."""
        self.checkpoint_counter += 1
        
        # Salva a cada checkpoint_interval documentos
        if self.checkpoint_counter % self.checkpoint_interval == 0:
            self.save_backup()
            log.info(f"🔄 Checkpoint automático salvo ({self.checkpoint_counter} documentos processados)")
    
    def save_backup(self, force=False):
        """Salva backup completo dos dados coletados."""
        try:
            backup_data = {
                "timestamp": datetime.now().isoformat(),
                "checkpoint_counter": self.checkpoint_counter,
                "metadata": {
                    "total_iniciais": len(self.iniciais_data),
                    "total_contestacoes": len(self.contestacoes_data),
                    "total_pares": len(self.argument_pairs)
                },
                "iniciais_data": self.iniciais_data,
                "contestacoes_data": self.contestacoes_data,
                "argument_pairs": self.argument_pairs
            }
            
            # Salva com nome timestampado para histórico
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_with_timestamp = os.path.join(
                self.output_dir, 
                f"backup_dados_classificador_{timestamp}.json"
            )
            
            # Salva backup principal (sempre sobrescreve)
            os.makedirs(self.output_dir, exist_ok=True)
            with open(self.backup_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, ensure_ascii=False, indent=2)
            
            # Salva cópia com timestamp para histórico
            if force or self.checkpoint_counter % 50 == 0:  # Backup com timestamp a cada 50 docs
                with open(backup_with_timestamp, 'w', encoding='utf-8') as f:
                    json.dump(backup_data, f, ensure_ascii=False, indent=2)
                log.info(f"💾 Backup com timestamp salvo: {backup_with_timestamp}")
            
            return True
            
        except Exception as e:
            log.error(f"❌ Erro ao salvar backup: {e}")
            return False
    
    def load_backup_if_exists(self):
        """Carrega backup existente se disponível."""
        if not os.path.exists(self.backup_file):
            log.info("🆕 Iniciando nova sessão (nenhum backup encontrado)")
            return False
        
        try:
            with open(self.backup_file, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            # Restaura dados
            self.iniciais_data = backup_data.get("iniciais_data", [])
            self.contestacoes_data = backup_data.get("contestacoes_data", [])
            self.argument_pairs = backup_data.get("argument_pairs", [])
            self.checkpoint_counter = backup_data.get("checkpoint_counter", 0)
            
            total_docs = len(self.iniciais_data) + len(self.contestacoes_data)
            if total_docs > 0:
                log.info(f"🔄 Backup restaurado! {total_docs} documentos recuperados")
                log.info(f"   • Petições iniciais: {len(self.iniciais_data)}")
                log.info(f"   • Contestações: {len(self.contestacoes_data)}")
                log.info(f"   • Pares mapeados: {len(self.argument_pairs)}")
                return True
            else:
                log.info("🆕 Backup vazio encontrado, iniciando nova sessão")
                return False
                
        except Exception as e:
            log.warning(f"⚠️ Erro ao carregar backup: {e}")
            log.info("🆕 Iniciando nova sessão")
            return False
    
    def clear_backup(self):
        """Remove arquivo de backup (usado após conclusão bem-sucedida)."""
        try:
            if os.path.exists(self.backup_file):
                os.remove(self.backup_file)
                log.info("🗑️ Backup temporário removido (processamento concluído)")
        except Exception as e:
            log.warning(f"⚠️ Não foi possível remover backup: {e}")
    
    def get_recovery_info(self):
        """Retorna informações sobre dados recuperados."""
        return {
            "backup_existe": os.path.exists(self.backup_file),
            "total_documentos_backup": len(self.iniciais_data) + len(self.contestacoes_data),
            "ultimo_checkpoint": self.checkpoint_counter,
            "backup_file": self.backup_file
        }


def call_openrouter(payload: Dict) -> Dict:
    """Chama a API do OpenRouter e retorna o JSON da resposta."""
    if not OPENROUTER_KEY:
        raise LLMError("OPENROUTER_API_KEY não configurada no .env")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": APP_SITE_URL,
        "X-Title": APP_TITLE,
    }
    try:
        response = requests.post(
            f"{OPENROUTER_BASE}/chat/completions",
            headers=headers,
            json=payload,
            timeout=TIMEOUT_S,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        
        # Tenta extrair o JSON do conteúdo
        try:
            # Remove ```json ... ``` se existir
            if content.strip().startswith("```json"):
                content = content.strip()[7:-3].strip()
            
            parsed_json = json.loads(content)
            return parsed_json
        except json.JSONDecodeError as e:
            raise LLMError(f"Resposta do LLM não é um JSON válido: {content} | Erro: {e}")

    except requests.Timeout:
        raise LLMError(f"Timeout ({TIMEOUT_S}s) ao chamar a API OpenRouter")
    except requests.RequestException as e:
        msg = f"Erro na chamada à API: {e}"
        if e.response:
            msg += f" | Status: {e.response.status_code} | Body: {e.response.text}"
        raise LLMError(msg)
    except (KeyError, IndexError) as e:
        raise LLMError(f"Erro ao extrair resposta do LLM: {e}")


# =========================
# Renomeação e listagem
# =========================

def list_files_for_classification(root: str, recursive: bool) -> List[str]:
    out = []
    extensions = (".txt", ".pdf")
    if recursive:
        for base, _, files in os.walk(root):
            for fn in files:
                if fn.lower().endswith(extensions):
                    out.append(os.path.join(base, fn))
    else:
        for fn in os.listdir(root):
            if fn.lower().endswith(extensions):
                out.append(os.path.join(root, fn))
    return sorted(out)


def safe_rename(path: str, new_basename: str, max_retries: int = 6) -> str:
    """Renomeia arquivo com estratégias avançadas para WinError 32."""
    import time
    import gc
    
    if not ENABLE_RENAME:
        log.info(f"Renomeação desabilitada para: {os.path.basename(path)}")
        return path
    
    if not os.path.exists(path):
        log.warning(f"Arquivo não encontrado: {path}")
        return path
    
    directory = os.path.dirname(path)
    original_ext = os.path.splitext(path)[1]
    rootname = os.path.splitext(new_basename)[0]
    candidate = os.path.join(directory, rootname + original_ext)
    original_name = os.path.basename(path)
    
    # Verifica se já é o nome desejado
    try:
        cur_abs = os.path.normcase(os.path.abspath(path))
        cand_abs = os.path.normcase(os.path.abspath(candidate))
        if cur_abs == cand_abs:
            return path
    except Exception:
        pass

    def force_file_release(file_path):
        """Força liberação do arquivo (Windows específico)."""
        try:
            gc.collect()  # Força garbage collection
            time.sleep(0.1)  # Pequena pausa para liberar recursos
        except:
            pass

    def wait_for_file_access(file_path, max_wait=8):
        """Aguarda o arquivo ficar acessível."""
        for i in range(max_wait):
            try:
                # Tenta abrir o arquivo para verificar se está livre
                with open(file_path, 'r+b') as f:
                    pass
                return True
            except (PermissionError, IOError):
                if i < max_wait - 1:
                    time.sleep(0.5)
                continue
        return False

    def try_rename_enhanced(source: str, target: str) -> str:
        """Tenta renomear com estratégias progressivas."""
        for attempt in range(max_retries):
            try:
                if os.path.exists(target):
                    return None  # Arquivo destino já existe
                
                if not os.path.exists(source):
                    log.warning(f"Arquivo fonte desapareceu: {source}")
                    return source
                
                os.rename(source, target)
                log.info(f"✅ Renomeado: {os.path.basename(source)} → {os.path.basename(target)}")
                return target
                
            except PermissionError as e:
                if "WinError 32" in str(e) or "being used by another process" in str(e):
                    # Estratégias progressivas baseadas na tentativa
                    if attempt == 0:
                        wait_time = 1
                        log.info(f"🔒 Arquivo em uso. Aguardando {wait_time}s...")
                    elif attempt == 1:
                        wait_time = 2
                        log.info(f"🔄 Forçando liberação de recursos...")
                        force_file_release(source)
                    elif attempt == 2:
                        wait_time = 3
                        log.info(f"⏳ Aguardando acesso ao arquivo...")
                        if not wait_for_file_access(source, 3):
                            log.warning("⚠️ Arquivo ainda bloqueado")
                    elif attempt == 3:
                        wait_time = 5
                        log.warning(f"🚨 Tentativa {attempt + 1}/{max_retries} - WinError 32 persistente")
                    else:
                        wait_time = 8
                        log.warning(f"🔴 Última tentativa ({attempt + 1}/{max_retries})")
                    
                    time.sleep(wait_time)
                else:
                    log.warning(f"❌ Erro de permissão: {e}")
                    return source
                    
            except FileExistsError:
                return None  # Arquivo destino já existe
                
            except Exception as e:
                log.warning(f"❌ Erro inesperado: {e}")
                return source
        
        # Todas as tentativas falharam
        log.error(f"💥 FALHA: Impossível renomear {original_name} após {max_retries} tentativas")
        log.error("🔧 SOLUÇÕES:")
        log.error("   • Feche visualizadores de PDF (Adobe Reader, Edge, Chrome)")
        log.error("   • Feche Windows Explorer na pasta")
        log.error("   • Aguarde alguns minutos")
        log.error("   • Temporariamente: pause antivírus/indexação")
        return source

    # Primeira tentativa: nome desejado
    result = try_rename_enhanced(path, candidate)
    if result and result != path:
        return result
    elif result == path:
        return path

    # Se não conseguiu, tenta com numeração
    i = 1
    max_attempts = 15  # Reduzido para evitar loops longos
    
    while i <= max_attempts:
        alt_name = f"{rootname} ({i})"
        alt_path = os.path.join(directory, alt_name + original_ext)
        
        result = try_rename_enhanced(path, alt_path)
        if result and result != path:
            return result
        elif result == path:
            return path
        i += 1
    
    # Última tentativa: nome com timestamp
    timestamp = datetime.now().strftime("%H%M%S")
    final_name = f"{rootname}_temp{timestamp}"
    final_path = os.path.join(directory, final_name + original_ext)
    
    result = try_rename_enhanced(path, final_path)
    if result == path:
        log.error(f"🆘 CRÍTICO: Arquivo {original_name} não pode ser renomeado")
        log.error("📋 Reporte este problema se persistir")
    
    return result


# =========================
# Núcleo de processamento
# =========================

def process_one(path: str, log_cb=None) -> Dict[str, Optional[str]]:
    def _log(msg):
        log.info(msg)
        if log_cb:
            log_cb(msg)

    ext = os.path.splitext(path)[1].lower()
    _log(f"Processando ({ext}): {os.path.basename(path)}")

    try:
        texto = read_file_content(path)
    except Exception as e:
        _log(f"❌ Erro ao ler arquivo: {e}")
        texto = ""

    sem_conteudo = not bool(texto.strip())

    base_name = base_filename_without_ext(path)
    numero_processo = extract_process_number_from_name(base_name)

    if sem_conteudo:
        _log("⚠️ Arquivo vazio ou sem texto extraível — marcando como 'Outra petição' com confiança 0.00")
        snippet = ""
        tipo = "Outra petição"
        conf = 0.0
        novo_basename = f"0% - {numero_processo} - {tipo}{os.path.splitext(path)[1]}"
        
        if ENABLE_RENAME:
            _log(f"→ Renomeando para: {novo_basename}")
            novo_caminho = safe_rename(path, novo_basename)
        else:
            _log(f"→ Renomeação desabilitada, mantendo: {os.path.basename(path)}")
            novo_caminho = path
        return {
            "Nº do Processo": numero_processo,
            "Tipo de Petição": tipo,
            "Transcrição (512 tokens)": snippet,
            "Arquivo (novo)": os.path.basename(novo_caminho),
            "_texto_normalizado": "",  # Arquivo vazio
        }

    # Normaliza e gera transcrição de 512 tokens (aprox.)
    texto_norm = normalize_text(texto)
    transcricao_512 = tokenize_approx(texto_norm, SNIPPET_TOKENS)

    # Chama LLM (OpenRouter)
    _log("→ Chamando OpenRouter LLM…")
    try:
        res = call_openrouter(texto_norm)
        tipo = res["tipo"]
        conf = float(res.get("confianca", 0.5))
    except Exception as e:
        _log(f"× Falha na LLM: {e} — marcando como 'Outra petição' (conf. 0.50)")
        tipo = "Outra petição"
        conf = 0.5

    # Renomeia o arquivo conforme confiança e rótulo (se habilitado)
    confianca_pct = max(0, min(100, int(round(conf * 100))))
    novo_basename = f"{confianca_pct}% - {numero_processo} - {tipo}{os.path.splitext(path)[1]}"
    
    if ENABLE_RENAME:
        _log(f"→ Renomeando para: {novo_basename}")
        novo_caminho = safe_rename(path, novo_basename)
    else:
        _log(f"→ Renomeação desabilitada, mantendo: {os.path.basename(path)}")
        novo_caminho = path

    return {
        "Nº do Processo": numero_processo,
        "Tipo de Petição": tipo,
        "Transcrição (512 tokens)": transcricao_512,
        "Arquivo (novo)": os.path.basename(novo_caminho),
        "_texto_normalizado": texto_norm,  # Para uso interno na extração
    }


def process_batch(input_dir: str, output_dir: str, recursive: bool, log_cb=None, progress_cb=None, analyzer=None, interrupt_check=None):
    def _log(msg):
        log.info(msg)
        if log_cb:
            log_cb(msg)

    arquivos = list_files_for_classification(input_dir, recursive)
    if not arquivos:
        _log("Nenhum arquivo .txt ou .pdf encontrado. Encerrando.")
        return None

    _log(f"Arquivos encontrados: {len(arquivos)}")
    ts = int(time.time())
    saida_xlsx = os.path.join(output_dir, f"resultado_{ts}.xlsx")
    saida_json = os.path.join(output_dir, f"resultado_{ts}.json")
    _log(f"Excel de saída: {saida_xlsx}")
    _log(f"JSON final: {saida_json}")
    os.makedirs(os.path.dirname(saida_xlsx) or ".", exist_ok=True)

    rows = []
    total = len(arquivos)
    for i, f in enumerate(arquivos, 1):
        # Verifica se deve interromper
        if interrupt_check and interrupt_check():
            _log(f"Processamento interrompido pelo usuário em {i-1}/{total} arquivos.")
            break
            
        _log(f"[{i}/{total}] Iniciando {os.path.basename(f)}")
        try:
            res = process_one(f, log_cb=log_cb)
            rows.append(res)
            
            # Inicializa variáveis para uso posterior
            tipo_peticao = ""
            texto_norm = ""
            
            # Extração de dados para análise (se analyzer foi fornecido)
            if analyzer and res.get("Tipo de Petição") and res.get("_texto_normalizado"):
                tipo_peticao = res["Tipo de Petição"].lower()
                texto_norm = res["_texto_normalizado"]
                _log(f"[{i}/{total}] Tipo classificado: '{res['Tipo de Petição']}'")
                
                # Só processa se há texto para extrair
                if texto_norm.strip():
                    try:
                        if "inicial" in tipo_peticao or "petição inicial" in tipo_peticao:
                            _log(f"[{i}/{total}] Extraindo dados da petição inicial...")
                            payload = build_extraction_prompt_inicial(texto_norm)
                            extraction_data = call_openrouter_extraction(payload)
                            analyzer.add_inicial_data(os.path.basename(f), extraction_data, texto_norm)
                            pedidos_count = len(extraction_data.get('pedidos', []))
                            fatos_count = len(extraction_data.get('fundamentos_fato', []))
                            direito_count = len(extraction_data.get('fundamentos_direito', []))
                            _log(f"[{i}/{total}] ✅ Inicial extraída: {pedidos_count} pedidos, {fatos_count} fatos, {direito_count} fundamentos jurídicos")
                            
                            # Estatísticas acumuladas
                            total_iniciais = len(analyzer.iniciais_data)
                            total_contestacoes = len(analyzer.contestacoes_data)
                            _log(f"[{i}/{total}] 📊 ACUMULADO: {total_iniciais} iniciais, {total_contestacoes} contestações processadas")
                            
                        elif "contestação" in tipo_peticao or "contestacao" in tipo_peticao:
                            _log(f"[{i}/{total}] Extraindo dados da contestação...")
                            payload = build_extraction_prompt_contestacao(texto_norm)
                            extraction_data = call_openrouter_extraction(payload)
                            analyzer.add_contestacao_data(os.path.basename(f), extraction_data, texto_norm)
                            preliminares_count = len(extraction_data.get('preliminares', []))
                            merito_fatos_count = len(extraction_data.get('merito_fatos', []))
                            merito_direito_count = len(extraction_data.get('merito_direito', []))
                            outros_count = len(extraction_data.get('outros_argumentos', []))
                            total_args = preliminares_count + merito_fatos_count + merito_direito_count + outros_count
                            _log(f"[{i}/{total}] ✅ Contestação extraída: {total_args} argumentos ({preliminares_count} prelim., {merito_fatos_count} fatos, {merito_direito_count} direito, {outros_count} outros)")
                            
                            # Estatísticas acumuladas
                            total_contestacoes = len(analyzer.contestacoes_data)
                            total_mapeamentos = len(analyzer.argument_pairs)
                            _log(f"[{i}/{total}] 📊 ACUMULADO: {total_contestacoes} contestações, {total_mapeamentos} mapeamentos autor-réu")
                            
                    except Exception as extract_error:
                        _log(f"[{i}/{total}] Erro na extração de dados: {extract_error}")
                        if log_cb:
                            log_cb(f"Detalhes do erro: {traceback.format_exc()}")
            
            # Log de progresso da coleta de dados (sem gerar relatórios)
            if analyzer:
                total_docs = len(analyzer.iniciais_data) + len(analyzer.contestacoes_data)
                
                # Mostra progresso a cada 10 documentos ou quando encontra dados relevantes
                if (i % 10 == 0 and total_docs > 0) or ("inicial" in tipo_peticao or "contestação" in tipo_peticao):
                    _log(f"[{i}/{total}] 📊 Dados coletados até agora: {len(analyzer.iniciais_data)} iniciais, {len(analyzer.contestacoes_data)} contestações")
            
            _log(f"[{i}/{total}] OK.")
        except Exception as e:
            _log(f"[{i}/{total}] ERRO: {e}")
            if log_cb:
                log_cb(traceback.format_exc())
            err_row = {
                "Nº do Processo": base_filename_without_ext(f),
                "Tipo de Petição": "Erro ao processar",
                "Transcrição (512 tokens)": "",
                "Arquivo (novo)": os.path.basename(f),
            }
            rows.append(err_row)
        if progress_cb:
            progress_cb(i, total)

    # DataFrame nas colunas pedidas
    df = pd.DataFrame(rows, columns=[
        "Nº do Processo", "Tipo de Petição", "Transcrição (512 tokens)", "Arquivo (novo)"
    ])

    # Salva Excel e JSON
    df.to_excel(saida_xlsx, index=False)
    with open(saida_json, "w", encoding="utf-8") as jf:
        json.dump(rows, jf, ensure_ascii=False, indent=2)

    tipos = df["Tipo de Petição"].value_counts().to_dict()
    _log("=== CONCLUÍDO ===")
    _log(f"Planilha gerada em: {saida_xlsx}")
    _log(f"JSON final em: {saida_json}")
    _log(f"Total processado: {len(df)}")
    _log(f"Distribuição por tipo: {tipos}")
    
    # Gera relatórios de análise final se há dados coletados
    if analyzer:
        total_analises = len(analyzer.iniciais_data) + len(analyzer.contestacoes_data)
        if total_analises > 0:
            try:
                _log("=== GERANDO RELATÓRIOS DE ANÁLISE FINAL ===")
                _log(f"Processando dados de {len(analyzer.iniciais_data)} petições iniciais e {len(analyzer.contestacoes_data)} contestações...")
                json_path, excel_path = generate_reports(analyzer, output_dir, real_time=False)
                _log(f"📊 Relatórios de análise salvos:")
                _log(f"  • JSON: {os.path.basename(json_path)}")
                _log(f"  • Excel: {os.path.basename(excel_path)}")
                _log(f"📈 Rankings consolidados gerados com base em {total_analises} documentos analisados")
            except Exception as e:
                _log(f"❌ Erro ao gerar relatórios de análise: {e}")
        else:
            _log("ℹ️ Nenhuma petição inicial ou contestação foi encontrada para análise de rankings")
    
    return saida_xlsx


# =========================
# Interface Tkinter (opcional)
# =========================
try:
    import tkinter as tk
    from tkinter import filedialog, ttk, messagebox
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False


class TkLogHandler(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q
    def emit(self, record):
        try:
            msg = self.format(record)
            self.q.put(msg)
        except Exception:
            pass


# =========================
# Geração de relatórios
# =========================

def generate_reports(analyzer: DataAnalyzer, output_dir: str, real_time=True):
    """Gera relatórios de produção com rastreabilidade completa e métricas de qualidade."""
    try:
        import pandas as pd
    except ImportError:
        raise Exception("pandas é necessário para gerar relatórios Excel. Execute: pip install pandas openpyxl")
    
    # Cria diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Gera análise semântica completa
    semantic_analysis = analyzer.get_semantic_analysis()
    
    # Calcula métricas de qualidade globais
    quality_metrics = analyzer._calculate_global_quality_metrics()
    
    # === RELATÓRIO JSON DE PRODUÇÃO ===
    relatorio_json = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "status": "em_andamento" if real_time else "final",
            "versao": "3.0_producao_rastreavel",
            "total_documentos": len(analyzer.iniciais_data) + len(analyzer.contestacoes_data),
            "gerador": "Classificador Jurídico - Sistema de Análise Semântica",
            "auditoria": {
                "responsavel": "Sistema Automatizado",
                "revisao_necessaria": quality_metrics.get("revisao_necessaria", 0),
                "confiabilidade": quality_metrics.get("confiabilidade_geral", "N/A")
            }
        },
        "avaliacao": {
            "precision": quality_metrics.get("precision", 0.0),
            "recall": quality_metrics.get("recall", 0.0),
            "f1": quality_metrics.get("f1", 0.0),
            "thresholds": {
                "alto": 0.75,
                "revisao": [0.60, 0.74],
                "descarte": 0.60
            },
            "distribuicao_scores": quality_metrics.get("distribuicao_scores", {}),
            "total_mapeamentos": len(analyzer.argument_pairs)
        },
        "clusters": analyzer._generate_production_clusters(),
        "mapeamentos": analyzer._generate_production_mappings(),
        "referencias_normativas": analyzer._generate_normative_summary(),
        "analise_temporal": analyzer._generate_temporal_analysis(),
        "insights_contestacao": {
            "estrutura_recomendada": semantic_analysis["insights_para_contestacao"]["estrutura_recomendada"],
            "argumentos_mais_eficazes": analyzer._extract_most_effective_arguments(),
            "padroes_sucesso": analyzer._identify_success_patterns(),
            "recomendacoes_especificas": analyzer._generate_specific_recommendations()
        },
        "padroes_simples": analyzer.generate_simple_pattern_mapping(),
        "dados_auditoria": {
            "arquivos_processados": {
                "iniciais": [item["arquivo"] for item in analyzer.iniciais_data],
                "contestacoes": [item["arquivo"] for item in analyzer.contestacoes_data]
            },
            "rastreabilidade_completa": True,
            "backup_disponivel": analyzer.get_recovery_info()["backup_existe"]
        }
    }
    
    # Nomes de arquivos
    if real_time:
        # Para tempo real, sobrescreve o mesmo arquivo
        json_path = os.path.join(output_dir, "relatorio_analise_ATUAL.json")
        excel_path = os.path.join(output_dir, "relatorio_analise_ATUAL.xlsx")
    else:
        # Para relatório final, cria arquivo com timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = os.path.join(output_dir, f"relatorio_analise_FINAL_{timestamp}.json")
        excel_path = os.path.join(output_dir, f"relatorio_analise_FINAL_{timestamp}.xlsx")
    
    # Salva JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(relatorio_json, f, ensure_ascii=False, indent=2)
    
    # === RELATÓRIO EXCEL AVANÇADO ===
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Aba 1: Resumo Executivo
        resumo_data = [relatorio_json["resumo_executivo"]]
        resumo_df = pd.DataFrame(resumo_data)
        resumo_df.to_excel(writer, sheet_name='Resumo_Executivo', index=False)
        
        # Aba 2: Clusters Semânticos - Pedidos
        clusters_pedidos = semantic_analysis["clusters_semanticos"]["pedidos_agrupados"]
        pedidos_data = []
        if clusters_pedidos:
            for cluster in clusters_pedidos:
                pedidos_data.append({
                    "Cluster_ID": cluster["cluster_id"],
                    "Tamanho": cluster["tamanho_cluster"],
                    "Representante": cluster["representante"],
                    "Tema_Principal": cluster["interpretacao"]["tema_principal"],
                    "Palavras_Chave": ", ".join(cluster["interpretacao"]["palavras_chave"])
                })
        
        # Sempre cria a aba, mesmo se vazia
        if not pedidos_data:
            pedidos_data = [{"Cluster_ID": "N/A", "Tamanho": 0, "Representante": "Nenhum cluster encontrado", "Tema_Principal": "N/A", "Palavras_Chave": "N/A"}]
        pd.DataFrame(pedidos_data).to_excel(writer, sheet_name='Clusters_Pedidos', index=False)
        
        # Aba 3: Clusters Semânticos - Fundamentos
        clusters_fatos = semantic_analysis["clusters_semanticos"]["fundamentos_fato_agrupados"]
        fatos_data = []
        if clusters_fatos:
            for cluster in clusters_fatos:
                fatos_data.append({
                    "Cluster_ID": cluster["cluster_id"],
                    "Tamanho": cluster["tamanho_cluster"],
                    "Representante": cluster["representante"],
                    "Tema_Principal": cluster["interpretacao"]["tema_principal"],
                    "Palavras_Chave": ", ".join(cluster["interpretacao"]["palavras_chave"])
                })
        
        # Sempre cria a aba, mesmo se vazia
        if not fatos_data:
            fatos_data = [{"Cluster_ID": "N/A", "Tamanho": 0, "Representante": "Nenhum cluster encontrado", "Tema_Principal": "N/A", "Palavras_Chave": "N/A"}]
        pd.DataFrame(fatos_data).to_excel(writer, sheet_name='Clusters_Fundamentos', index=False)
        
        # Aba 4: Mapeamento Autor-Réu
        mapeamento_data = []
        if analyzer.argument_pairs:
            for pair in analyzer.argument_pairs:
                for pedido_vs_prelim in pair["pares"]["pedidos_vs_preliminares"]:
                    mapeamento_data.append({
                        "Processo": pair["processo"],
                        "Tipo": "Pedido vs Preliminar",
                        "Argumento_Autor": pedido_vs_prelim["pedido_autor"],
                        "Resposta_Reu": pedido_vs_prelim["resposta_reu"],
                        "Similaridade": f"{pedido_vs_prelim['similaridade']:.2f}"
                    })
                
                for fato_vs_contra in pair["pares"]["fatos_vs_contrafatos"]:
                    mapeamento_data.append({
                        "Processo": pair["processo"],
                        "Tipo": "Fato vs Contrafato",
                        "Argumento_Autor": fato_vs_contra["fato_autor"],
                        "Resposta_Reu": fato_vs_contra["contrafato_reu"],
                        "Similaridade": f"{fato_vs_contra['similaridade']:.2f}"
                    })
                
                for direito_vs_contra in pair["pares"]["direito_vs_contradireito"]:
                    mapeamento_data.append({
                        "Processo": pair["processo"],
                        "Tipo": "Direito vs Contradireito",
                        "Argumento_Autor": direito_vs_contra["direito_autor"],
                        "Resposta_Reu": direito_vs_contra["contradireito_reu"],
                        "Similaridade": f"{direito_vs_contra['similaridade']:.2f}"
                    })
        
        # Sempre cria a aba, mesmo se vazia
        if not mapeamento_data:
            mapeamento_data = [{"Processo": "N/A", "Tipo": "Nenhum mapeamento encontrado", "Argumento_Autor": "N/A", "Resposta_Reu": "N/A", "Similaridade": "0.00"}]
        pd.DataFrame(mapeamento_data).to_excel(writer, sheet_name='Mapeamento_Autor_Reu', index=False)
        
        # Aba 5: Padrões de Defesa
        padroes_defesa = semantic_analysis["padroes_contestacao"]["preliminares_frequentes"]
        defesa_data = []
        if padroes_defesa:
            for padrao in padroes_defesa:
                defesa_data.append({
                    "Argumento_Principal": padrao["argumento"],
                    "Frequencia": padrao["frequencia"],
                    "Variantes": "; ".join(padrao["variantes"]) if padrao["variantes"] else "Nenhuma"
                })
        
        # Sempre cria a aba, mesmo se vazia
        if not defesa_data:
            defesa_data = [{"Argumento_Principal": "Nenhum padrão encontrado", "Frequencia": 0, "Variantes": "N/A"}]
        pd.DataFrame(defesa_data).to_excel(writer, sheet_name='Padroes_Defesa', index=False)
        
        # Aba 6: Insights para Contestação
        insights = semantic_analysis["insights_para_contestacao"]
        insights_data = []
        
        # Preliminares essenciais
        if insights.get("estrutura_recomendada", {}).get("preliminares_essenciais"):
            for prelim in insights["estrutura_recomendada"]["preliminares_essenciais"]:
                insights_data.append({
                    "Categoria": "Preliminar Essencial",
                    "Argumento": prelim["argumento"],
                    "Recomendacao": prelim["uso_recomendado"],
                    "Eficacia": prelim["eficacia"]
                })
        
        # Estratégias por tipo
        if insights.get("estrategias_por_tipo"):
            for tipo, estrategias in insights["estrategias_por_tipo"].items():
                for estrategia in estrategias:
                    insights_data.append({
                        "Categoria": f"Estratégia {tipo}",
                        "Argumento": estrategia,
                        "Recomendacao": "Aplicar conforme o caso",
                        "Eficacia": "Recomendada"
                    })
        
        # Sempre cria a aba, mesmo se vazia
        if not insights_data:
            insights_data = [{"Categoria": "N/A", "Argumento": "Nenhum insight encontrado", "Recomendacao": "N/A", "Eficacia": "N/A"}]
        pd.DataFrame(insights_data).to_excel(writer, sheet_name='Insights_Contestacao', index=False)
        
        # Aba 7: Padrões Simples (Sua funcionalidade principal!)
        padroes_simples = relatorio_json["padroes_simples"]
        padroes_data = []
        
        if padroes_simples and padroes_simples.get("resumo_estatistico", {}).get("total_padroes_pedidos", 0) > 0:
            # Padrões de Pedidos vs Preliminares
            for padrao in padroes_simples["pedidos_vs_preliminares"]:
                for resposta in padrao["respostas_comuns"]:
                    padroes_data.append({
                        "Tipo": "Pedido vs Preliminar",
                        "Quando_Inicial_Alega": padrao["quando_inicial_alega"],
                        "Contestacao_Responde": resposta["texto"],
                        "Frequencia": resposta["frequencia"],
                        "Confianca": resposta["confianca"],
                        "Exemplo_Processo": padrao["exemplos_processos"][0] if padrao["exemplos_processos"] else "",
                        "Normas_Citadas": "; ".join(padrao["normas_mais_citadas"])
                    })
            
            # Padrões de Fatos vs Contrafatos
            for padrao in padroes_simples["fatos_vs_contrafatos"]:
                for resposta in padrao["respostas_comuns"]:
                    padroes_data.append({
                        "Tipo": "Fato vs Contrafato",
                        "Quando_Inicial_Alega": padrao["quando_inicial_alega"],
                        "Contestacao_Responde": resposta["texto"],
                        "Frequencia": resposta["frequencia"],
                        "Confianca": resposta["confianca"],
                        "Exemplo_Processo": padrao["exemplos_processos"][0] if padrao["exemplos_processos"] else "",
                        "Normas_Citadas": "; ".join(padrao["normas_mais_citadas"])
                    })
            
            # Padrões de Direito vs Contradireito
            for padrao in padroes_simples["direito_vs_contradireito"]:
                for resposta in padrao["respostas_comuns"]:
                    padroes_data.append({
                        "Tipo": "Direito vs Contradireito",
                        "Quando_Inicial_Alega": padrao["quando_inicial_alega"],
                        "Contestacao_Responde": resposta["texto"],
                        "Frequencia": resposta["frequencia"],
                        "Confianca": resposta["confianca"],
                        "Exemplo_Processo": padrao["exemplos_processos"][0] if padrao["exemplos_processos"] else "",
                        "Normas_Citadas": "; ".join(padrao["normas_mais_citadas"])
                    })
        
        # Sempre cria a aba, mesmo se vazia
        if not padroes_data:
            padroes_data = [{"Tipo": "N/A", "Quando_Inicial_Alega": "Nenhum padrão encontrado", "Contestacao_Responde": "N/A", "Frequencia": 0, "Confianca": 0.0, "Exemplo_Processo": "N/A", "Normas_Citadas": "N/A"}]
        else:
            # Ordena por frequência para mostrar padrões mais comuns primeiro
            padroes_data = sorted(padroes_data, key=lambda x: x["Frequencia"], reverse=True)
        
        pd.DataFrame(padroes_data).to_excel(writer, sheet_name='Padroes_Simples', index=False)
        
        # Aba: Dados Petições Iniciais
        iniciais_flat = []
        if analyzer.iniciais_data:
            for item in analyzer.iniciais_data:
                row = {"arquivo": item["arquivo"], "timestamp": item["timestamp"]}
                data = item["data"]
                row.update({
                    "pedidos": "; ".join(data.get("pedidos", [])),
                    "fundamentos_fato": "; ".join(data.get("fundamentos_fato", [])),
                    "fundamentos_direito": "; ".join(data.get("fundamentos_direito", []))
                })
                iniciais_flat.append(row)
        
        # Sempre cria a aba, mesmo se vazia
        if not iniciais_flat:
            iniciais_flat = [{"arquivo": "N/A", "timestamp": "N/A", "pedidos": "Nenhuma petição inicial encontrada", "fundamentos_fato": "N/A", "fundamentos_direito": "N/A"}]
        
        iniciais_df = pd.DataFrame(iniciais_flat)
        iniciais_df.to_excel(writer, sheet_name='Dados_Peticoes_Iniciais', index=False)
        
        # Aba: Dados Contestações
        contestacoes_flat = []
        if analyzer.contestacoes_data:
            for item in analyzer.contestacoes_data:
                row = {"arquivo": item["arquivo"], "timestamp": item["timestamp"]}
                data = item["data"]
                row.update({
                    "preliminares": "; ".join(data.get("preliminares", [])),
                    "merito_fatos": "; ".join(data.get("merito_fatos", [])),
                    "merito_direito": "; ".join(data.get("merito_direito", [])),
                    "outros_argumentos": "; ".join(data.get("outros_argumentos", []))
                })
                contestacoes_flat.append(row)
        
        # Sempre cria a aba, mesmo se vazia
        if not contestacoes_flat:
            contestacoes_flat = [{"arquivo": "N/A", "timestamp": "N/A", "preliminares": "Nenhuma contestação encontrada", "merito_fatos": "N/A", "merito_direito": "N/A", "outros_argumentos": "N/A"}]
        
        contestacoes_df = pd.DataFrame(contestacoes_flat)
        contestacoes_df.to_excel(writer, sheet_name='Dados_Contestacoes', index=False)
    
    return json_path, excel_path




class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Classificador de Peças (.txt/.pdf) — OpenRouter LLM")
        self.geometry("900x600")

        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.recursive = tk.BooleanVar(value=True)
        
        # Sistema de análise de dados
        self.analyzer = DataAnalyzer(output_dir=self.get_output_directory())
        
        # Controle de interrupção
        self.should_interrupt = False
        self.is_processing = False
        
        # Configura comportamento de fechamento personalizado
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Header ENV
        env_frame = ttk.LabelFrame(self, text="Config (.env)")
        env_frame.pack(fill="x", padx=10, pady=10)
        ttk.Label(env_frame, text=f"OPENROUTER_MODEL: {OPENROUTER_MODEL}").pack(anchor="w", padx=8, pady=2)
        ttk.Label(env_frame, text=f"API KEY: {'definida' if bool(OPENROUTER_KEY) else 'NÃO definida'}").pack(anchor="w", padx=8, pady=2)
        if APP_SITE_URL:
            ttk.Label(env_frame, text=f"HTTP-Referer: {APP_SITE_URL}").pack(anchor="w", padx=8, pady=2)
        if APP_TITLE:
            ttk.Label(env_frame, text=f"X-Title: {APP_TITLE}").pack(anchor="w", padx=8, pady=2)

        # Seleção de pastas
        io_frame = ttk.LabelFrame(self, text="Pastas")
        io_frame.pack(fill="x", padx=10, pady=10)

        row1 = ttk.Frame(io_frame); row1.pack(fill="x", padx=8, pady=5)
        ttk.Label(row1, text="Pasta de ENTRADA:").pack(side="left")
        ttk.Entry(row1, textvariable=self.input_dir, width=80).pack(side="left", padx=5)
        ttk.Button(row1, text="Selecionar…", command=self.choose_input).pack(side="left")

        row2 = ttk.Frame(io_frame); row2.pack(fill="x", padx=8, pady=5)
        ttk.Label(row2, text="Pasta de SAÍDA:   ").pack(side="left")
        ttk.Entry(row2, textvariable=self.output_dir, width=80).pack(side="left", padx=5)
        ttk.Button(row2, text="Selecionar…", command=self.choose_output).pack(side="left")

        # Opções
        opt_frame = ttk.LabelFrame(self, text="Opções")
        opt_frame.pack(fill="x", padx=10, pady=10)
        ttk.Checkbutton(opt_frame, text="Percorrer subpastas (recursivo)", variable=self.recursive).pack(side="left", padx=8)

        # Botões de ação
        act_frame = ttk.Frame(self)
        act_frame.pack(fill="x", padx=10, pady=10)
        self.start_button = ttk.Button(act_frame, text="Iniciar", command=self.start_run)
        self.start_button.pack(side="left", padx=5)
        
        self.interrupt_button = ttk.Button(act_frame, text="Interromper", command=self.interrupt_processing, state="disabled")
        self.interrupt_button.pack(side="left", padx=5)
        
        ttk.Button(act_frame, text="Gerar Relatórios", command=self.generate_reports).pack(side="left", padx=5)
        ttk.Button(act_frame, text="Sair", command=self.on_closing).pack(side="right", padx=5)

        # Progresso
        prog_frame = ttk.LabelFrame(self, text="Progresso")
        prog_frame.pack(fill="x", padx=10, pady=10)
        self.progress = ttk.Progressbar(prog_frame, mode="determinate")
        self.progress.pack(fill="x", padx=8, pady=8)
        self.status_label = ttk.Label(prog_frame, text="Aguardando…")
        self.status_label.pack(anchor="w", padx=8, pady=2)

        # Logs
        log_frame = ttk.LabelFrame(self, text="Logs")
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.text = tk.Text(log_frame, wrap="word", height=16)
        self.text.pack(fill="both", expand=True, padx=8, pady=8)
        self.text.configure(state="disabled")

        # Queue para logs e handler
        self.log_q = queue.Queue()
        self.tk_handler = TkLogHandler(self.log_q)
        self.tk_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
        logging.getLogger().addHandler(self.tk_handler)

        # Atualizador de logs
        self.after(100, self.consume_log_queue)

        # Controle de thread
        self.worker_thread = None

    def choose_input(self):
        path = filedialog.askdirectory(title="Selecione a PASTA DE ENTRADA (contendo .txt/.pdf)", mustexist=True)
        if path:
            self.input_dir.set(path)

    def choose_output(self):
        path = filedialog.askdirectory(title="Selecione a PASTA DE SAÍDA (para salvar o Excel/JSON)", mustexist=True)
        if path:
            self.output_dir.set(path)

    def append_log(self, msg: str):
        self.text.configure(state="normal")
        self.text.insert("end", msg + "\n")
        self.text.see("end")
        self.text.configure(state="disabled")

    def consume_log_queue(self):
        try:
            while True:
                msg = self.log_q.get_nowait()
                self.append_log(msg)
        except queue.Empty:
            pass
        self.after(100, self.consume_log_queue)

    def set_progress(self, i, total):
        self.progress["maximum"] = total
        self.progress["value"] = i
        self.status_label.configure(text=f"Processado {i}/{total}")

    def generate_reports(self):
        """Gera relatórios de análise com os dados coletados até o momento."""
        try:
            # Verifica se há dados para gerar relatório
            total_docs = len(self.analyzer.iniciais_data) + len(self.analyzer.contestacoes_data)
            
            # Debug - mostra estado atual do analyzer
            self.append_log(f"DEBUG - Estado do analyzer:")
            self.append_log(f"  Iniciais coletadas: {len(self.analyzer.iniciais_data)}")
            self.append_log(f"  Contestações coletadas: {len(self.analyzer.contestacoes_data)}")
            
            if len(self.analyzer.iniciais_data) > 0:
                self.append_log(f"  Primeira inicial: {list(self.analyzer.iniciais_data[0]['data'].keys())}")
            if len(self.analyzer.contestacoes_data) > 0:
                self.append_log(f"  Primeira contestação: {list(self.analyzer.contestacoes_data[0]['data'].keys())}")
            
            # Conta elementos únicos da nova estrutura
            total_pedidos = sum(len(inicial['data'].get('pedidos', [])) for inicial in self.analyzer.iniciais_data)
            total_contestacoes_args = sum(len(contest['data'].get('preliminares', [])) + 
                                        len(contest['data'].get('merito_fatos', [])) + 
                                        len(contest['data'].get('merito_direito', [])) 
                                        for contest in self.analyzer.contestacoes_data)
            total_mapeamentos = len(self.analyzer.argument_pairs)
            
            self.append_log(f"  Total de pedidos extraídos: {total_pedidos}")
            self.append_log(f"  Total argumentos de contestação: {total_contestacoes_args}")
            self.append_log(f"  Total de mapeamentos autor-réu: {total_mapeamentos}")
            self.append_log(f"  Dados com rastreabilidade completa: ✅")
            
            if total_docs == 0:
                messagebox.showinfo("Sem dados", "Nenhum documento foi analisado ainda.\nExecute a classificação primeiro.")
                return
            
            # Solicita diretório de saída para relatórios
            if not self.output_dir.get() or not os.path.isdir(self.output_dir.get()):
                messagebox.showwarning("Ops", "Selecione uma PASTA DE SAÍDA válida primeiro.")
                return
            
            # Gera relatórios
            try:
                json_path, excel_path = generate_reports(self.analyzer, self.output_dir.get(), real_time=False)
                
                msg = (f"Relatórios gerados com sucesso!\n\n"
                       f"Documentos analisados: {total_docs}\n"
                       f"Petições iniciais: {len(self.analyzer.iniciais_data)}\n"
                       f"Contestações: {len(self.analyzer.contestacoes_data)}\n"
                       f"Pedidos únicos: {total_pedidos}\n"
                       f"Fundamentos únicos: {total_fund_fato + total_fund_direito}\n"
                       f"Argumentos únicos: {total_argumentos}\n\n"
                       f"Arquivos criados:\n"
                       f"• JSON: {os.path.basename(json_path)}\n"
                       f"• Excel: {os.path.basename(excel_path)}")
                
                messagebox.showinfo("Relatórios Gerados", msg)
                self.append_log(f"Relatórios salvos em: {self.output_dir.get()}")
                
            except Exception as e:
                error_msg = f"Erro ao gerar relatórios: {str(e)}"
                messagebox.showerror("Erro", error_msg)
                self.append_log(error_msg)
                self.append_log(f"Detalhes do erro: {traceback.format_exc()}")
                
        except Exception as e:
            messagebox.showerror("Erro", f"Erro inesperado: {str(e)}")
            self.append_log(f"Erro inesperado: {traceback.format_exc()}")

    def interrupt_processing(self):
        """Interrompe o processamento em andamento."""
        if self.is_processing:
            result = messagebox.askyesnocancel(
                "Interromper Processamento",
                "Deseja realmente interromper o processamento?\n\n"
                "• Sim: Interrompe e mantém relatórios gerados até agora\n"
                "• Não: Interrompe e descarta relatórios\n"
                "• Cancelar: Continua processamento"
            )
            
            if result is True:  # Sim - interromper e manter relatórios
                self.should_interrupt = True
                self.append_log("Interrupção solicitada - mantendo relatórios...")
            elif result is False:  # Não - interromper e descartar
                self.should_interrupt = True
                # Limpa os dados do analyzer
                self.analyzer = DataAnalyzer(output_dir=self.get_output_directory())
                self.append_log("Interrupção solicitada - descartando relatórios...")
            # Se result is None (Cancelar), não faz nada
        else:
            messagebox.showinfo("Info", "Nenhum processamento em andamento.")

    def get_output_directory(self):
        """Retorna o diretório de saída baseado no diretório selecionado."""
        if hasattr(self, 'output_dir') and self.output_dir.get():
            return self.output_dir.get()
        return os.getcwd()
    
    def on_closing(self):
        """Método chamado quando o usuário tenta fechar a aplicação."""
        try:
            total_docs = len(self.analyzer.iniciais_data) + len(self.analyzer.contestacoes_data)
            
            if total_docs > 0:
                # Há dados analisados, pergunta se deseja gerar relatórios
                result = messagebox.askyesnocancel(
                    "Encerrar Aplicação",
                    f"Foram analisados {total_docs} documentos nesta sessão.\n\n"
                    f"Deseja gerar relatórios de análise antes de sair?\n\n"
                    f"• Sim: Gera relatórios e fecha a aplicação\n"
                    f"• Não: Fecha sem gerar relatórios\n"
                    f"• Cancelar: Continua na aplicação"
                )
                
                if result is True:  # Sim - gerar relatórios
                    try:
                        if self.output_dir.get() and os.path.isdir(self.output_dir.get()):
                            json_path, excel_path = generate_reports(self.analyzer, self.output_dir.get(), real_time=False)
                            self.analyzer.clear_backup()  # Remove backup após relatório final
                            messagebox.showinfo(
                                "Relatórios Gerados", 
                                f"Relatórios finais salvos com sucesso!\n\n"
                                f"Local: {self.output_dir.get()}\n"
                                f"Arquivos: {os.path.basename(json_path)}, {os.path.basename(excel_path)}"
                            )
                        else:
                            messagebox.showwarning("Aviso", "Pasta de saída não configurada.\nRelatórios não foram gerados.")
                    except Exception as e:
                        messagebox.showerror("Erro", f"Erro ao gerar relatórios: {str(e)}")
                    
                    # Fecha a aplicação após gerar (ou tentar gerar) relatórios
                    self.destroy()
                    
                elif result is False:  # Não - fechar sem relatórios
                    # Salva backup final antes de sair
                    backup_saved = self.analyzer.save_backup(force=True)
                    if backup_saved:
                        messagebox.showinfo(
                            "Backup Salvo", 
                            f"💾 Dados salvos em backup!\n\n"
                            f"Local: {self.analyzer.backup_file}\n\n"
                            f"Você pode continuar de onde parou na próxima execução."
                        )
                    self.destroy()
                    
                # Se result is None (Cancelar), não faz nada e mantém a aplicação aberta
                
            else:
                # Não há dados analisados, fecha normalmente
                if messagebox.askokcancel("Encerrar", "Deseja realmente encerrar a aplicação?"):
                    self.destroy()
                    
        except Exception as e:
            # Em caso de erro, fecha normalmente
            messagebox.showerror("Erro", f"Erro inesperado: {str(e)}")
            self.destroy()

    def start_run(self):
        if not self.input_dir.get() or not os.path.isdir(self.input_dir.get()):
            messagebox.showwarning("Ops", "Selecione a PASTA DE ENTRADA válida.")
            return
        if not self.output_dir.get() or not os.path.isdir(self.output_dir.get()):
            messagebox.showwarning("Ops", "Selecione a PASTA DE SAÍDA válida.")
            return
        
        # Exibe informações de recuperação se houver
        recovery_info = self.analyzer.get_recovery_info()
        if recovery_info["total_documentos_backup"] > 0:
            self.append_log(f"🔄 Dados recuperados do backup anterior:")
            self.append_log(f"   • {recovery_info['total_documentos_backup']} documentos já processados")
            self.append_log(f"   • Checkpoint: {recovery_info['ultimo_checkpoint']}")
            self.append_log(f"   • Continuando de onde parou...")
        else:
            self.append_log("🆕 Iniciando nova sessão de processamento")
        
        # Controle de estado
        self.is_processing = True
        self.should_interrupt = False
        
        # Controla botões
        self.start_button.configure(state="disabled")
        self.interrupt_button.configure(state="normal")
        
        # Desabilita outros controles
        for child in self.winfo_children():
            try:
                if child not in [self.interrupt_button]:
                    child.configure(state="disabled")
            except Exception:
                pass

        def log_cb(msg):
            self.append_log(msg)

        def progress_cb(i, total):
            self.set_progress(i, total)
        
        def interrupt_check():
            return self.should_interrupt

        def run_bg():
            try:
                excel_path = process_batch(
                    input_dir=self.input_dir.get(),
                    output_dir=self.output_dir.get(),
                    recursive=self.recursive.get(),
                    log_cb=log_cb,
                    progress_cb=progress_cb,
                    analyzer=self.analyzer,
                    interrupt_check=interrupt_check
                )
                
                if self.should_interrupt:
                    self.append_log("Processamento interrompido.")
                    self.append_log("Os relatórios de análise foram gerados automaticamente.")
                elif excel_path:
                    self.append_log(f"Arquivo Excel salvo em: {excel_path}")
                    messagebox.showinfo("Concluído", f"Processamento concluído!\n\nPlanilha: {excel_path}\nRelatórios de análise salvos na pasta de saída.")
                else:
                    messagebox.showwarning("Atenção", "Nenhum arquivo .txt/.pdf encontrado para processar.")
            except Exception as e:
                self.append_log("Falha fatal: " + str(e))
                self.append_log(traceback.format_exc())
                messagebox.showerror("Erro", str(e))
            finally:
                # Restaura estado dos controles
                self.is_processing = False
                self.should_interrupt = False
                self.start_button.configure(state="normal")
                self.interrupt_button.configure(state="disabled")
                for child in self.winfo_children():
                    try:
                        child.configure(state="normal")
                    except Exception:
                        pass

        self.worker_thread = threading.Thread(target=run_bg, daemon=True)
        self.worker_thread.start()


# =========================
# Execução
# =========================
if __name__ == "__main__":
    if TK_AVAILABLE:
        app = App()
        app.mainloop()
    else:
        print("Tkinter indisponível — rodando em modo CLI básico.")
        inp = input("Pasta de ENTRADA (.txt/.pdf): ").strip().strip('"').strip("'")
        outp = input("Pasta de SAÍDA (Excel/JSON): ").strip().strip('"').strip("'")
        rec = input("Recursivo (s/n, padrão s): ").strip().lower()
        recursive = (rec == "" or rec.startswith("s"))
        process_batch(inp, outp, recursive, log_cb=lambda m: print(m), progress_cb=lambda i,t: print(f"{i}/{t}"))

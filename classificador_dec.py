# classifica_decisoes_tk.py
# -*- coding: utf-8 -*-
import os
import re
import json
import time
import logging
import traceback
import threading
import queue
from typing import List, Tuple, Optional, Dict

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
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
load_dotenv()  # carrega .env na mesma pasta

LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
LLM_API_KEY  = os.getenv("LLM_API_KEY", "")
LLM_MODEL    = os.getenv("LLM_MODEL", "gpt-5-mini")

TIMEOUT_S        = 60
CONTEXT_WINDOW   = 180        # chars para snippet
MAX_INPUT_CHARS  = 18000      # limite enviado à LLM

# =========================
# Heurística (padrões)
# =========================
LIMINAR_PATTERNS = [
    r'\btutela\s+de\s+urg[eê]ncia\b',
    r'\btutela\s+provis[óo]ria\b',
    r'\btutela\s+antecipada\b',
    r'\bmedida\s+liminar\b',
    r'\bliminarmente\b',
    r'\bantecip[aâ]?[cç][aã]o\s+dos?\s+efeitos\s+da\s+tutela\b',
    r'\bpedido\s+liminar\b',
    r'\bprovimento\s+de\s+urg[eê]ncia\b',
]

DEFERIDO_PATTERNS = [
    r'\bdefiro\b',
    r'\bdefere-se\b',
    r'\bfica\s+deferid[ao]?\b',
    r'\bconcedo\b',
    r'\bconcede-se\b',
    r'\bacolho\b',
    r'\bdeferimento\b',
]

INDEFERIDO_PATTERNS = [
    r'\bindefiro\b',
    r'\bindefere-se\b',
    r'\bfica\s+indefirid[ao]?\b',
    r'\bn[aã]o\s+concedo\b',
    r'\bnega-se\b(?:\s+provimento)?',
    r'\bn[eê]ga-se\b(?:\s+provimento)?',
    r'\bindeferimento\b',
]

PARCIAL_PATTERNS = [
    r'\bdefiro\s+em\s+parte\b',
    r'\bparcialmente\s+deferid[ao]?\b',
    r'\bconcedo\s+em\s+parte\b',
    r'\bacolho\s+em\s+parte\b',
    r'\bparcial\s+acolhimento\b',
]

INTERLOCUTORIA_PATTERNS = [
    r'\bdecis[aã]o\s+interlocut[óo]ria\b',
    r'\bdespacho\b',
    r'\bintime?-se\b',
    r'\bmanifeste-se\b',
    r'\bdetermino\b',
    # Execução de tutela já deferida
    r'\btransfer[ari]?-?se\s+(o\s+)?valor',
    r'\bdetermino:\s*transfira',
    r'\bconsiderando\s+que\s+a\s+tutela\s+foi\s+concedida.*determino',
    r'\bexpeça-?se\s+mandado',
    r'\bcumpra-?se\s+(a\s+)?liminar',
    r'\bpara\s+cumprimento\s+da\s+(tutela|liminar)',
    # Mera ratificação/continuidade (sem nova apreciação)
    r'\bmantenho\s+(o\s+)?\w+\s+conforme\s+decisão\s+anterior',
    r'\bratific[ao]\s+(a\s+)?decisão',
    r'\bconfirmo\s+(a\s+)?decisão\s+anterior',
    r'\bpermanece\s+(em\s+)?vigor',
    r'\bcontinua\s+(o\s+)?\w+\s+(institucional|anterior)',
]

# =========================
# Utilitários de texto/arquivo
# =========================
def read_text_file(path: str) -> str:
    """Lê texto de arquivo .txt usando várias codificações"""
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
    """Extrai texto de arquivo PDF usando PyMuPDF"""
    try:
        # Tenta primeiro com pymupdf4llm (melhor qualidade)
        import pymupdf4llm
        markdown_text = pymupdf4llm.to_markdown(path)
        return normalize_text(markdown_text)
    except Exception:
        try:
            # Fallback para PyMuPDF básico
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
    """Lê conteúdo de arquivo .txt ou .pdf automaticamente"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return read_pdf_text(path)
    elif ext == ".txt":
        return read_text_file(path)
    else:
        raise ValueError(f"Extensão não suportada: {ext}")

def normalize_text(t: str) -> str:
    t = re.sub(r'[ \t]+', ' ', t)
    t = re.sub(r'\s+\n', '\n', t)
    t = re.sub(r'\n\s+', '\n', t)
    return t

def base_filename_without_ext(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def extract_process_number_from_name(name: str) -> str:
    """Extrai um número de processo (apenas dígitos) do nome do arquivo.
    Preferimos um bloco de 15+ dígitos (formato CNJ), senão retornamos todos os dígitos encontrados.
    """
    m = re.search(r"(\d{15,})", name)
    if m:
        return m.group(1)
    digits = re.sub(r"[^\d]", "", name)
    return digits if digits else name

def _collect_matches(patterns: List[str], text_lower: str) -> List[Tuple[str, re.Match]]:
    out = []
    for pat in patterns:
        for m in re.finditer(pat, text_lower, flags=re.IGNORECASE):
            out.append((pat, m))
    return out

def _context_snippet(text: str, idx: int, window: int = CONTEXT_WINDOW) -> str:
    start = max(0, idx - window)
    end = min(len(text), idx + window)
    return text[start:end].replace('\n', ' ').strip()

def score_liminar(text_lower: str) -> Tuple[int, List[str], Optional[str]]:
    reasons = []
    hits = _collect_matches(LIMINAR_PATTERNS, text_lower)
    score = 0
    snippet = None
    if hits:
        score += 3 * len(hits)
        reasons.append(f"{len(hits)} termo(s) de tutela/liminar")
        snippet = _context_snippet(text_lower, hits[0][1].start())

    def_hits = _collect_matches(DEFERIDO_PATTERNS, text_lower)
    inde_hits = _collect_matches(INDEFERIDO_PATTERNS, text_lower)
    par_hits  = _collect_matches(PARCIAL_PATTERNS, text_lower)

    if def_hits:
        score += 1; reasons.append("sinal de deferimento")
        snippet = snippet or _context_snippet(text_lower, def_hits[0][1].start())
    if inde_hits:
        score += 1; reasons.append("sinal de indeferimento")
        snippet = snippet or _context_snippet(text_lower, inde_hits[0][1].start())
    if par_hits:
        score += 2; reasons.append("sinal de deferimento parcial")
        snippet = snippet or _context_snippet(text_lower, par_hits[0][1].start())
    return score, reasons, snippet

def score_interlocutoria(text_lower: str) -> Tuple[int, List[str], Optional[str]]:
    reasons = []
    hits = _collect_matches(INTERLOCUTORIA_PATTERNS, text_lower)
    score = 0
    snippet = None
    if hits:
        score += 1 * len(hits)
        reasons.append(f"{len(hits)} marcador(es) interlocutórios")
        snippet = _context_snippet(text_lower, hits[0][1].start())
    return score, reasons, snippet

def classify_resultado_heuristica(text_lower: str) -> Tuple[str, Optional[str]]:
    par = _collect_matches(PARCIAL_PATTERNS, text_lower)
    if par:
        return "Parcialmente deferido", _context_snippet(text_lower, par[0][1].start())
    de = _collect_matches(DEFERIDO_PATTERNS, text_lower)
    inx = _collect_matches(INDEFERIDO_PATTERNS, text_lower)
    if de and not inx:  return "Deferido", _context_snippet(text_lower, de[0][1].start())
    if inx and not de:  return "Indeferido", _context_snippet(text_lower, inx[0][1].start())
    if de and inx:      return "Não identificado (conflito de sinais)", None
    return "Não se aplica", None

def decide_tipo_heuristica(text: str) -> Dict[str, Optional[str]]:
    low = text.lower()
    lim_s, lim_r, lim_sn = score_liminar(low)
    int_s, int_r, int_sn = score_interlocutoria(low)
    
    # Lógica melhorada: se há qualquer menção a tutela, prioriza liminar
    tem_tutela = any(palavra in low for palavra in [
        'tutela de urgência', 'tutela provisória', 'tutela antecipada', 
        'medida liminar', 'liminarmente', 'antecipação dos efeitos',
        'liminar anteriormente', 'mantenho a liminar', 'revogo a liminar'
    ])
    
    if tem_tutela or lim_s >= 3:
        tipo = "Decisão Liminar"
        res, res_sn = classify_resultado_heuristica(low)
        snip = res_sn or lim_sn
    elif int_s > 0 and not tem_tutela:
        tipo = "Decisão Interlocutória"
        res, snip = "Não se aplica", int_sn
    else:
        # Em caso de dúvida, assume interlocutória (mais conservador)
        tipo = "Decisão Interlocutória"
        res, snip = "Não se aplica", int_sn or lim_sn
    
    # Calcula confiança baseada na força dos indicadores
    if tem_tutela and lim_s >= 3:
        confianca_heur = 0.8  # alta confiança quando há tutela explícita
    elif tem_tutela:
        confianca_heur = 0.7  # boa confiança quando detecta tutela
    elif int_s > 0:
        confianca_heur = 0.75  # boa confiança para interlocutórias claras
    else:
        confianca_heur = 0.6   # confiança média quando incerto
    
    score_conf = f"liminar={lim_s}; interlocutoria={int_s}; tem_tutela={tem_tutela}; razões={' | '.join(lim_r + int_r) if (lim_r or int_r) else '—'}"
    return {"tipo": tipo, "resultado": res, "snippet": snip, "score": score_conf, "fonte": "Heurística", "confianca": confianca_heur}

# =========================
# LLM (GPT-5 mini) — PROMPT REFORÇADO + FEW-SHOTS
# =========================
class LLMError(Exception):
    pass

def build_llm_messages(texto: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Prompt reforçado com foco no OBJETO da decisão (se é tutela) e não nos verbos genéricos.
    Mantém as mesmas chaves de saída:
      - tipo_decisao: 'Decisão Liminar' | 'Decisão Interlocutória'
      - resultado_tutela: 'Deferido' | 'Indeferido' | 'Parcialmente deferido' | 'Não se aplica' | 'Não identificado'
      - trecho_justificativa: string curta (1–3 frases)
      - racional_curto: ~25 palavras
    """
    system = (
        "Você é especialista em classificar decisões judiciais brasileiras em texto puro (.txt). "
        "Classifique com base NO TEOR DO QUE É DECIDIDO, e NÃO apenas nos verbos ('defiro', 'decido', etc.).\n\n"
        "DEFINIÇÕES PRECISAS:\n"
        "• 'Decisão Liminar' — quando a decisão APRECIA tutela de urgência/provisória/antecipada (art. 300 CPC), "
        "  ou CONFIRMA/MANTÉM/REVOGA/MODULA liminar/tutela já existente, ou ANTECIPA efeitos da tutela (mesmo sem a palavra 'liminar').\n"
        "  Sinais textuais fortes (exemplos, não exaustivo): 'tutela de urgência', 'tutela provisória', 'tutela antecipada', "
        "  'antecipação dos efeitos da tutela', 'medida liminar', 'liminarmente', menção a art. 300 do CPC, "
        "  expressões de requisitos ('probabilidade do direito', 'perigo de dano', 'periculum in mora', 'fumus boni iuris').\n"
        "• 'Decisão Interlocutória' — demais atos processuais que NÃO apreciam tutela (intimações, prazos, provas, gratuidade, emenda/saneamento, ofícios, juntadas, etc.), "
        "  ainda que usem verbos decisórios ('defiro a produção de prova', 'defiro gratuidade', 'defiro prazo').\n\n"
        "REGRAS CRÍTICAS:\n"
        "1) Verifique se o VERBO está ligado a TUTELA. 'Defiro a tutela provisória' => liminar; "
        "'Defiro a produção de prova'/'defiro dilação de prazo'/'defiro gratuidade' => interlocutória.\n"
        "2) Se há NOVA APRECIAÇÃO dos requisitos da tutela (concede, revoga, cassa, reconsidera), classifique como 'Decisão Liminar'. ATENÇÃO: 'mantém/confirma' só é liminar se houver RE-ANÁLISE dos requisitos; se for mera ratificação/continuidade, é interlocutória.\n"
        "3) EXECUÇÃO/CUMPRIMENTO de tutela já deferida (transferir valores, expedir ofícios, intimar para cumprimento) SEM nova apreciação = 'Decisão Interlocutória'.\n"
        "4) Referência meramente HISTÓRICA à existência de liminar, SEM ato decisório atual sobre tutela, continua sendo 'Decisão Interlocutória'.\n"
        "5) Em dúvida razoável sobre o resultado da tutela (conflito textual/ambiguidade), use 'Não identificado' no campo de resultado; o tipo permanece 'Decisão Liminar' se a tutela foi apreciada.\n"
        "6) Foque no dispositivo e trechos decisórios; cabeçalho/relatório não bastam isoladamente.\n\n"
        "RESULTADO DA TUTELA (apenas se 'Decisão Liminar'):\n"
        "• 'Deferido' — ex.: 'defiro/concedo/acolho a tutela', 'mantenho/confirmo a liminar', 'torno definitiva a tutela'.\n"
        "• 'Indeferido' — ex.: 'indefiro/nego a tutela', 'revogo/casso a liminar', 'reconsidero para indeferir'.\n"
        "• 'Parcialmente deferido' — ex.: 'defiro/concedo/acolho em parte'.\n"
        "• 'Não identificado' — quando não há clareza suficiente no trecho fornecido.\n"
        "Se 'Decisão Interlocutória', o resultado_tutela é SEMPRE 'Não se aplica'.\n\n"
        "SAÍDA — FORMATO ESTRITO (UM ÚNICO JSON, sem texto fora do JSON):\n"
        "{\n"
        "  \"tipo_decisao\": \"Decisão Liminar\" | \"Decisão Interlocutória\",\n"
        "  \"resultado_tutela\": \"Deferido\" | \"Indeferido\" | \"Parcialmente deferido\" | \"Não se aplica\" | \"Não identificado\",\n"
        "  \"trecho_justificativa\": \"frase(s) curta(s) citando o trecho e o porquê (p. ex., ligação do verbo à tutela)\",\n"
        "  \"racional_curto\": \"por que decidiu assim, até ~25 palavras\",\n"
        "  \"confianca\": número entre 0.0 e 1.0 indicando certeza da classificação\n"
        "}\n"
        "CONFIANÇA: 0.9-1.0=muito alta, 0.7-0.9=alta, 0.5-0.7=média, 0.3-0.5=baixa, 0.0-0.3=muito baixa.\n"
        "Se o texto estiver truncado e não permitir conclusão segura do RESULTADO, use 'Não identificado' e explique.\n"
    )

    # Few-shots para evitar falsos positivos por verbo genérico
    fewshots = [
        # LIMINAR — deferido
        {
            "role": "user",
            "content": (
                "Classifique. Responda apenas o JSON.\n\n"
                "=== TEXTO ===\n"
                "Presentes os requisitos do art. 300 do CPC (probabilidade do direito e perigo de dano), "
                "defiro a tutela de urgência para determinar o fornecimento do medicamento X."
            )
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "tipo_decisao": "Decisão Liminar",
                "resultado_tutela": "Deferido",
                "trecho_justificativa": "Menção ao art. 300 e 'defiro a tutela de urgência' ligada ao objeto tutela.",
                "racional_curto": "Há apreciação expressa de tutela com concessão.",
                "confianca": 0.95
            }, ensure_ascii=False)
        },
        # INTERLOCUTÓRIA — verbos decisórios, mas não tutela
        {
            "role": "user",
            "content": (
                "Classifique. Responda apenas o JSON.\n\n"
                "=== TEXTO ===\n"
                "Defiro a produção de prova pericial e intime-se a parte autora para apresentação de quesitos em 15 dias."
            )
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "tipo_decisao": "Decisão Interlocutória",
                "resultado_tutela": "Não se aplica",
                "trecho_justificativa": "Verbo 'defiro' aplicado à prova pericial, não à tutela.",
                "racional_curto": "Ato ordinário de instrução, sem exame de tutela.",
                "confianca": 0.9
            }, ensure_ascii=False)
        },
        # INTERLOCUTÓRIA — execução de tutela já deferida (não é nova apreciação)
        {
            "role": "user",
            "content": (
                "Classifique. Responda apenas o JSON.\n\n"
                "=== TEXTO ===\n"
                "Considerando que a tutela foi concedida em decisão liminar, determino: "
                "Transfira-se imediatamente o valor de R$ 6.490,80 para conta da parte autora."
            )
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "tipo_decisao": "Decisão Interlocutória",
                "resultado_tutela": "Não se aplica",
                "trecho_justificativa": "Apenas executa tutela já concedida ('determino: Transfira-se'), sem nova apreciação.",
                "racional_curto": "Ato de cumprimento/execução, não apreciação de tutela.",
                "confianca": 0.85
            }, ensure_ascii=False)
        },
        # INTERLOCUTÓRIA — mera ratificação/continuidade (sem nova apreciação)
        {
            "role": "user",
            "content": (
                "Classifique. Responda apenas o JSON.\n\n"
                "=== TEXTO ===\n"
                "Mantenho o acolhimento institucional da menor conforme decisão anterior. "
                "Determino que seja intimado o MP para manifestação."
            )
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "tipo_decisao": "Decisão Interlocutória",
                "resultado_tutela": "Não se aplica",
                "trecho_justificativa": "Mera continuidade/ratificação sem nova apreciação dos requisitos da tutela.",
                "racional_curto": "Ratificação administrativa, não re-análise de tutela.",
                "confianca": 0.8
            }, ensure_ascii=False)
        },
        # INTERLOCUTÓRIA — gratuidade de justiça (não é tutela)
        {
            "role": "user",
            "content": (
                "Classifique. Responda apenas o JSON.\n\n"
                "=== TEXTO ===\n"
                "Defiro os benefícios da justiça gratuita. Intime-se."
            )
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "tipo_decisao": "Decisão Interlocutória",
                "resultado_tutela": "Não se aplica",
                "trecho_justificativa": "Concessão de gratuidade não diz respeito à tutela de urgência.",
                "racional_curto": "Matéria acessória; sem apreciação de tutela.",
                "confianca": 0.9
            }, ensure_ascii=False)
        },
        # LIMINAR — manutenção/confirmacao
        {
            "role": "user",
            "content": (
                "Classifique. Responda apenas o JSON.\n\n"
                "=== TEXTO ===\n"
                "Analisando novamente os requisitos do art. 300, mantenho a liminar anteriormente concedida, "
                "pois permanecem presentes o fumus boni iuris e o periculum in mora."
            )
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "tipo_decisao": "Decisão Liminar",
                "resultado_tutela": "Deferido",
                "trecho_justificativa": "Nova análise dos requisitos (art. 300) com manutenção da liminar.",
                "racional_curto": "Há RE-APRECIAÇÃO atual dos requisitos da tutela.",
                "confianca": 0.85
            }, ensure_ascii=False)
        },
        # LIMINAR — revogação (resultado negativo)
        {
            "role": "user",
            "content": (
                "Classifique. Responda apenas o JSON.\n\n"
                "=== TEXTO ===\n"
                "Diante da perda superveniente do objeto, revogo a liminar anteriormente deferida."
            )
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "tipo_decisao": "Decisão Liminar",
                "resultado_tutela": "Indeferido",
                "trecho_justificativa": "revogo a liminar anteriormente deferida.",
                "racional_curto": "Ato atual suprime a tutela, configurando resultado negativo.",
                "confianca": 0.9
            }, ensure_ascii=False)
        },
        # LIMINAR — ambíguo/resultado não claro
        {
            "role": "user",
            "content": (
                "Classifique. Responda apenas o JSON.\n\n"
                "=== TEXTO ===\n"
                "Anoto o pedido de tutela de urgência. Voltem conclusos após manifestação do MP."
            )
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "tipo_decisao": "Decisão Interlocutória",
                "resultado_tutela": "Não se aplica",
                "trecho_justificativa": "Há pedido de tutela, mas sem apreciação/resultado; apenas andamento.",
                "racional_curto": "Sem decisão sobre tutela neste ato; é mero impulso processual.",
                "confianca": 0.8
            }, ensure_ascii=False)
        },
        # INTERLOCUTÓRIA — prazos/andamento
        {
            "role": "user",
            "content": (
                "Classifique. Responda apenas o JSON.\n\n"
                "=== TEXTO ===\n"
                "Decido: intime-se a parte ré para contestar no prazo legal. Após, venham conclusos."
            )
        },
        {
            "role": "assistant",
            "content": json.dumps({
                "tipo_decisao": "Decisão Interlocutória",
                "resultado_tutela": "Não se aplica",
                "trecho_justificativa": "Decisão organiza o procedimento (intimação/contestação), sem tratar de tutela.",
                "racional_curto": "Ato ordinário, sem exame de tutela.",
                "confianca": 0.95
            }, ensure_ascii=False)
        }
    ]

    user = (
        "Analise CUIDADOSAMENTE o texto a seguir. Classifique com PRECISÃO, focando no objeto decidido "
        "(se é tutela ou não). Responda SOMENTE o JSON no formato exigido.\n\n"
        f"=== TEXTO (pode estar truncado) ===\n{texto[:MAX_INPUT_CHARS]}"
    )

    messages = [{"role": "system", "content": system}] + fewshots + [{"role": "user", "content": user}]
    return {"messages": messages}

@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(LLMError)
)
def call_llm(texto: str) -> Optional[Dict[str, Optional[str]]]:
    if not LLM_API_KEY:
        log.warning("LLM_API_KEY não definida - usando fallback heurístico")
        return None

    payload = {
        "model": LLM_MODEL,
        "messages": build_llm_messages(texto)["messages"],
        # Aumentado para evitar finish_reason: 'length'
        "max_completion_tokens": 100000,
    }
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    url = f"{LLM_API_BASE}/chat/completions"

    resp = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT_S)
    
    if not resp.ok:
        # Fallback entre max_completion_tokens e max_tokens conforme erro da API
        if resp.status_code == 400:
            try:
                err_txt = resp.text or ""
            except Exception:
                err_txt = ""
            # Se reclamar do max_completion_tokens, tenta max_tokens
            if ("max_completion_tokens" in err_txt.lower()) and ("unsupported" in err_txt.lower() or "invalid" in err_txt.lower()):
                alt_payload = {
                    "model": LLM_MODEL,
                    "messages": payload["messages"],
                    "max_tokens": 1000,
                }
                resp2 = requests.post(url, headers=headers, json=alt_payload, timeout=TIMEOUT_S)
                if not resp2.ok:
                    raise LLMError(f"HTTP {resp2.status_code}: {resp2.text}")
                data = resp2.json()
            # Se reclamar do max_tokens, volta para max_completion_tokens
            elif ("max_tokens" in err_txt.lower()) and ("unsupported" in err_txt.lower() or "invalid" in err_txt.lower()):
                alt_payload = {
                    "model": LLM_MODEL,
                    "messages": payload["messages"],
                    "max_completion_tokens": 1000,
                }
                resp2 = requests.post(url, headers=headers, json=alt_payload, timeout=TIMEOUT_S)
                if not resp2.ok:
                    raise LLMError(f"HTTP {resp2.status_code}: {resp2.text}")
                data = resp2.json()
            else:
                raise LLMError(f"HTTP {resp.status_code}: {resp.text}")
        else:
            raise LLMError(f"HTTP {resp.status_code}: {resp.text}")
    else:
        data = resp.json()

    content = ""
    # 1) Formato Chat Completions
    if isinstance(data, dict) and isinstance(data.get("choices"), list) and data.get("choices"):
        try:
            message_obj = data["choices"][0].get("message", {})
            content = message_obj.get("content", "") or ""
            # Se vier tool_calls com function.arguments JSON, usar como conteúdo
            if (not content) and isinstance(message_obj.get("tool_calls"), list):
                for tc in message_obj.get("tool_calls", []):
                    if isinstance(tc, dict) and tc.get("type") == "function":
                        args = tc.get("function", {}).get("arguments")
                        if isinstance(args, str) and args.strip():
                            content = args.strip()
                            break
        except Exception as e:
            log.warning(f"Erro ao extrair content da resposta: {e}")
            content = ""
    # 2) Formato Responses API (output_text)
    if not content and isinstance(data, dict) and isinstance(data.get("output_text"), str):
        content = data.get("output_text", "") or ""
    # 3) Formato Responses API (output: [...])
    if not content and isinstance(data, dict) and isinstance(data.get("output"), list):
        try:
            parts = []
            for item in data.get("output", []):
                # message com content: [{type: output_text, text: ...}]
                cont_list = item.get("content") if isinstance(item, dict) else None
                if isinstance(cont_list, list):
                    for c in cont_list:
                        if isinstance(c, dict) and c.get("type") in ("output_text", "text") and isinstance(c.get("text"), str):
                            parts.append(c.get("text"))
            content = "\n".join([p for p in parts if p])
        except Exception:
            content = content or ""
    content = (content or "").strip()

    # 4) Fallback para Responses API quando sem conteúdo
    if not content:
        responses_url = f"{LLM_API_BASE}/responses"
        try:
            # Converte messages -> input (Responses API)
            messages = build_llm_messages(texto)["messages"]
            alt_body = {
                "model": LLM_MODEL,
                "input": messages,
                "max_output_tokens": 1000,
                "response_format": {"type": "json_object"}
            }
            r2 = requests.post(responses_url, headers=headers, json=alt_body, timeout=TIMEOUT_S)
            if r2.ok:
                d2 = r2.json()
                content = (d2.get("output_text") or "").strip()
                if not content and isinstance(d2.get("output"), list):
                    try:
                        parts = []
                        for item in d2.get("output", []):
                            cont_list = item.get("content") if isinstance(item, dict) else None
                            if isinstance(cont_list, list):
                                for c in cont_list:
                                    if isinstance(c, dict) and c.get("type") in ("output_text", "text") and isinstance(c.get("text"), str):
                                        parts.append(c.get("text"))
                        content = "\n".join([p for p in parts if p]).strip()
                    except Exception:
                        content = content or ""
        except Exception:
            content = content or ""

    if not content:
        # Fallback 5: Forçar function call com schema (tools)
        schema = {
            "type": "object",
            "properties": {
                "tipo_decisao": {
                    "type": "string",
                    "enum": [
                        "Decisão Liminar",
                        "Decisão Interlocutória"
                    ]
                },
                "resultado_tutela": {
                    "type": "string",
                    "enum": [
                        "Deferido",
                        "Indeferido",
                        "Parcialmente deferido",
                        "Não se aplica",
                        "Não identificado"
                    ]
                },
                "trecho_justificativa": {"type": "string"},
                "racional_curto": {"type": "string"}
            },
            "required": ["tipo_decisao", "resultado_tutela"],
            "additionalProperties": False
        }
        tool_payload = {
            "model": LLM_MODEL,
            "messages": build_llm_messages(texto)["messages"],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "emitir_classificacao",
                        "description": "Emite a classificação em JSON conforme o schema.",
                        "parameters": schema
                    }
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "emitir_classificacao"}},
            "max_tokens": 1000
        }
        r3 = requests.post(url, headers=headers, json=tool_payload, timeout=TIMEOUT_S)
        if r3.ok:
            d3 = r3.json()
            try:
                tcalls = d3.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
                for tc in tcalls:
                    if isinstance(tc, dict) and tc.get("type") == "function":
                        args = tc.get("function", {}).get("arguments")
                        if isinstance(args, str) and args.strip():
                            content = args.strip()
                            break
            except Exception:
                content = content or ""
        if not content:
            # Diagnóstico da resposta vazia
            reason = "desconhecido"
            if isinstance(data, dict) and "choices" in data and data["choices"]:
                choice = data["choices"][0]
                finish_reason = choice.get("finish_reason", "")
                if finish_reason == "length":
                    reason = "limite de tokens atingido"
                    log.warning("LLM atingiu limite de tokens - resposta truncada")
                elif finish_reason == "content_filter":
                    reason = "filtro de conteúdo"
                    log.warning("LLM bloqueada por filtro de conteúdo")
                else:
                    reason = f"finish_reason={finish_reason}"
                    log.warning(f"LLM parou por: {finish_reason}")
            else:
                log.warning("LLM retornou estrutura inesperada")
            
            # Último fallback: simular resposta baseada em heurística
            heur_fallback = decide_tipo_heuristica(texto)
            heur_fallback["fonte"] = "Heurística (LLM fallback)"
            heur_fallback["score"] = heur_fallback.get("score", "") + f" | LLM_fallback_{reason}"
            log.info(f"× LLM falhou ({reason}); usando heurística como fallback.")
            return heur_fallback

    # Tenta limpar cercas de código e extrair JSON
    cleaned = content.strip().strip("`")
    first_brace = cleaned.find("{")
    last_brace  = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1:
        cleaned = cleaned[first_brace:last_brace+1]
    try:
        parsed = json.loads(cleaned)
    except Exception as e:
        log.warning(f"Falha ao parsear JSON da LLM: {e}")
        log.warning(f"Conteúdo recebido: {content[:200]}...")
        # Retorna None para usar fallback heurístico
        return None

    tipo = parsed.get("tipo_decisao") or ""
    resultado = parsed.get("resultado_tutela") or "Não identificado"
    trecho = parsed.get("trecho_justificativa") or ""
    racional = parsed.get("racional_curto") or ""
    confianca = parsed.get("confianca")
    
    # Normaliza confiança
    if confianca is not None:
        try:
            confianca = float(confianca)
            confianca = max(0.0, min(1.0, confianca))  # clamp entre 0 e 1
        except (ValueError, TypeError):
            confianca = 0.5  # default em caso de erro
    else:
        confianca = 0.5

    if "liminar" in tipo.lower() or "tutela" in tipo.lower():
        tipo_final = "Decisão Liminar"
    else:
        tipo_final = "Decisão Interlocutória"

    if tipo_final != "Decisão Liminar":
        resultado = "Não se aplica"

    return {
        "tipo": tipo_final,
        "resultado": resultado,
        "snippet": trecho if trecho else None,
        "score": f"LLM:{LLM_MODEL}; rationale={racional[:120]}",
        "fonte": "LLM",
        "confianca": confianca
    }

def ensemble(heur: Dict[str, Optional[str]], llm: Optional[Dict[str, Optional[str]]]) -> Dict[str, Optional[str]]:
    if not llm:
        return heur
    final = llm.copy()
    if heur and llm["tipo"] != heur["tipo"]:
        final["score"] = (llm.get("score","") + " | conflito_com_heuristica=" + (heur.get("score","")))
    if not final.get("snippet") and heur.get("snippet"):
        final["snippet"] = heur["snippet"]
    return final

# =========================
# Renomeação e listagem
# =========================
def list_files_for_classification(root: str, recursive: bool) -> List[str]:
    """Lista arquivos .txt e .pdf para classificação"""
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

def safe_rename(path: str, new_basename: str) -> str:
    """Renomeia arquivo mantendo a extensão original"""
    directory = os.path.dirname(path)
    original_ext = os.path.splitext(path)[1]  # .txt ou .pdf
    rootname = os.path.splitext(new_basename)[0]
    candidate = os.path.join(directory, rootname + original_ext)
    
    # Evita renomear para o mesmo caminho
    try:
        cur_abs = os.path.normcase(os.path.abspath(path))
        cand_abs = os.path.normcase(os.path.abspath(candidate))
        if cur_abs == cand_abs:
            return path
    except Exception:
        pass
        
    if not os.path.exists(candidate):
        os.rename(path, candidate)
        return candidate
        
    # Se já existe, adiciona numeração
    i = 1
    while True:
        alt = os.path.join(directory, f"{rootname} ({i}){original_ext}")
        if not os.path.exists(alt):
            os.rename(path, alt)
            return alt
        i += 1

# =========================
# Núcleo de processamento (para usar no Tk)
# =========================
def process_one(path: str, use_llm: bool, log_cb=None) -> Dict[str, Optional[str]]:
    def _log(msg):
        log.info(msg)
        if log_cb:
            log_cb(msg)

    file_ext = os.path.splitext(path)[1].lower()
    _log(f"Processando ({file_ext}): {os.path.basename(path)}")
    
    try:
        texto = read_file_content(path)
        sem_conteudo = not texto.strip()
        if sem_conteudo:
            _log("⚠️ Arquivo vazio ou sem texto extraível")
    except Exception as e:
        _log(f"❌ Erro ao ler arquivo: {e}")
        sem_conteudo = True
        texto = ""

    # Extrai número do processo primeiro
    base_name = base_filename_without_ext(path)
    numero_processo = extract_process_number_from_name(base_name)
    
    # Se não há conteúdo, não classifica
    if sem_conteudo:
        _log("→ Sem conteúdo extraível - não classificando")
        novo_basename = f"{numero_processo} - sem classificação{os.path.splitext(path)[1]}"
        _log(f"→ Renomeando para: {novo_basename}")
        novo_caminho = safe_rename(path, novo_basename)
        
        return {
            "Processo": numero_processo,
            "Tipo de Decisão": "Sem classificação",
            "Resultado Tutela": "Não se aplica",
            "Justificativa/ Trecho": "Arquivo sem conteúdo extraível",
            "Arquivo (novo)": os.path.basename(novo_caminho),
            "Fonte da Classificação": "Sistema",
            "Score/Confiança": "Sem conteúdo",
            "Nível de Confiança": 0.0,
        }

    _log("→ Rodando heurística...")
    heur = decide_tipo_heuristica(texto)

    llm_res = None
    if use_llm:
        _log("→ Chamando LLM (GPT-5 mini)...")
        try:
            llm_res = call_llm(texto)
            if llm_res:
                _log("→ LLM OK.")
            else:
                _log("× LLM não disponível ou falhou; usando heurística.")
                llm_res = None
        except Exception as e:
            _log(f"× LLM falhou com exceção: {e}")
            heur["score"] = (heur.get("score","") + f" | LLM_falhou={str(e)[:50]}")
            llm_res = None

    final = ensemble(heur, llm_res)
    _log(f"→ Classificação final: tipo={final['tipo']} | resultado={final['resultado']} | fonte={final.get('fonte')}")

    # Prepara nome com confiança
    confianca = final.get("confianca", 0.5)
    confianca_pct = int(confianca * 100)
    tipo_para_nome = final["tipo"] if final["tipo"] else "Decisão Indefinida"
    novo_basename = f"{confianca_pct}% - {numero_processo} - {tipo_para_nome}{os.path.splitext(path)[1]}"

    _log(f"→ Renomeando para: {novo_basename}")
    novo_caminho = safe_rename(path, novo_basename)

    return {
        "Processo": numero_processo,
        "Tipo de Decisão": final["tipo"] or "",
        "Resultado Tutela": final["resultado"] or "",
        "Justificativa/ Trecho": final["snippet"] or "",
        "Arquivo (novo)": os.path.basename(novo_caminho),
        "Fonte da Classificação": final.get("fonte","Heurística"),
        "Score/Confiança": final.get("score",""),
        "Nível de Confiança": final.get("confianca", 0.5),
    }

def process_batch(input_dir: str, output_dir: str, recursive: bool, use_llm: bool, log_cb=None, progress_cb=None):
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
    saida_jsonl = os.path.join(output_dir, f"resultado_{ts}.jsonl")
    saida_json = os.path.join(output_dir, f"resultado_{ts}.json")
    saida_csv = os.path.join(output_dir, f"resultado_{ts}.csv")
    saida_csv_estruturado = os.path.join(output_dir, f"estruturado_{ts}.csv")
    _log(f"Excel de saída: {saida_xlsx}")
    _log(f"JSONL incremental: {saida_jsonl}")
    _log(f"JSON final: {saida_json}")
    _log(f"CSV geral: {saida_csv}")
    _log(f"CSV estruturado: {saida_csv_estruturado}")
    os.makedirs(os.path.dirname(saida_xlsx) or ".", exist_ok=True)

    rows = []
    total = len(arquivos)
    for i, f in enumerate(arquivos, 1):
        _log(f"[{i}/{total}] Iniciando {os.path.basename(f)}")
        try:
            res = process_one(f, use_llm, log_cb=log_cb)
            rows.append(res)
            try:
                with open(saida_jsonl, "a", encoding="utf-8") as jf:
                    jf.write(json.dumps(res, ensure_ascii=False) + "\n")
            except Exception as e:
                _log(f"[WARN] Falha ao gravar JSONL: {e}")
            _log(f"[{i}/{total}] OK.")
        except Exception as e:
            _log(f"[{i}/{total}] ERRO: {e}")
            if log_cb:
                log_cb(traceback.format_exc())
            err_row = {
                "Processo": base_filename_without_ext(f),
                "Tipo de Decisão": "Erro ao processar",
                "Resultado Tutela": "",
                "Justificativa/ Trecho": f"Erro: {e}",
                "Arquivo (novo)": os.path.basename(f),
                "Fonte da Classificação": "—",
                "Score/Confiança": "",
                "Nível de Confiança": 0.0,
            }
            rows.append(err_row)
            try:
                with open(saida_jsonl, "a", encoding="utf-8") as jf:
                    jf.write(json.dumps(err_row, ensure_ascii=False) + "\n")
            except Exception as e2:
                _log(f"[WARN] Falha ao gravar JSONL (erro): {e2}")
        if progress_cb:
            progress_cb(i, total)

    df = pd.DataFrame(rows, columns=[
        "Processo","Tipo de Decisão","Resultado Tutela",
        "Justificativa/ Trecho","Arquivo (novo)","Fonte da Classificação","Score/Confiança","Nível de Confiança"
    ])
    
    # Salva Excel completo
    df.to_excel(saida_xlsx, index=False)
    
    # Salva CSV geral
    df.to_csv(saida_csv, index=False, encoding="utf-8")
    
    # Salva CSV estruturado conforme solicitado
    df_estruturado = df[["Processo", "Tipo de Decisão", "Resultado Tutela", "Justificativa/ Trecho", "Nível de Confiança"]].copy()
    # Renomeia colunas para o formato solicitado
    df_estruturado.columns = ["Nº do Processo", "Tipo de Decisão", "Resultado", "Trecho Utilizado", "Nível de Confiança"]
    df_estruturado.to_csv(saida_csv_estruturado, index=False, encoding="utf-8")
    
    try:
        with open(saida_json, "w", encoding="utf-8") as jf:
            json.dump(rows, jf, ensure_ascii=False, indent=2)
    except Exception as e:
        _log(f"[WARN] Falha ao gravar JSON final: {e}")

    tipos = df["Tipo de Decisão"].value_counts().to_dict()
    confianca_media = df["Nível de Confiança"].mean()
    _log("=== CONCLUÍDO ===")
    _log(f"Planilha gerada em: {saida_xlsx}")
    _log(f"JSONL consolidado em: {saida_jsonl}")
    _log(f"JSON final em: {saida_json}")
    _log(f"CSV geral em: {saida_csv}")
    _log(f"CSV estruturado em: {saida_csv_estruturado}")
    _log(f"Total processado: {len(df)}")
    _log(f"Distribuição por tipo: {tipos}")
    _log(f"Confiança média: {confianca_media:.2f}")
    return saida_xlsx

# =========================
# Interface Tkinter
# =========================
try:
    import tkinter as tk
    from tkinter import filedialog, ttk, messagebox
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

class TkLogHandler(logging.Handler):
    """ encaminha logs para uma queue consumida pela UI """
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q
    def emit(self, record):
        try:
            msg = self.format(record)
            self.q.put(msg)
        except Exception:
            pass

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Classificador de Decisões (.txt/.pdf) — Heurística + GPT-5 mini")
        self.geometry("900x600")

        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.use_llm = tk.BooleanVar(value=True)
        self.recursive = tk.BooleanVar(value=True)

        # Header ENV
        env_frame = ttk.LabelFrame(self, text="API (.env)")
        env_frame.pack(fill="x", padx=10, pady=10)
        ttk.Label(env_frame, text=f"LLM_API_BASE: {LLM_API_BASE}").pack(anchor="w", padx=8, pady=2)
        ttk.Label(env_frame, text=f"LLM_MODEL: {LLM_MODEL}").pack(anchor="w", padx=8, pady=2)
        ttk.Label(env_frame, text=f"LLM_KEY: {'definida' if bool(LLM_API_KEY) else 'NÃO definida'}").pack(anchor="w", padx=8, pady=2)

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
        ttk.Checkbutton(opt_frame, text="Usar GPT-5 mini (LLM)", variable=self.use_llm).pack(side="left", padx=8)
        ttk.Checkbutton(opt_frame, text="Percorrer subpastas (recursivo)", variable=self.recursive).pack(side="left", padx=8)

        # Botões de ação
        act_frame = ttk.Frame(self)
        act_frame.pack(fill="x", padx=10, pady=10)
        ttk.Button(act_frame, text="Iniciar", command=self.start_run).pack(side="left", padx=5)
        ttk.Button(act_frame, text="Sair", command=self.destroy).pack(side="right", padx=5)

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
        path = filedialog.askdirectory(title="Selecione a PASTA DE SAÍDA (para salvar o Excel)", mustexist=True)
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

    def start_run(self):
        if not self.input_dir.get() or not os.path.isdir(self.input_dir.get()):
            messagebox.showwarning("Ops", "Selecione a PASTA DE ENTRADA válida.")
            return
        if not self.output_dir.get() or not os.path.isdir(self.output_dir.get()):
            messagebox.showwarning("Ops", "Selecione a PASTA DE SAÍDA válida.")
            return
        # trava UI
        for child in self.winfo_children():
            try:
                child.configure(state="disabled")
            except Exception:
                pass

        def log_cb(msg):  # logs em tempo real
            self.append_log(msg)

        def progress_cb(i, total):
            self.set_progress(i, total)

        def run_bg():
            try:
                excel_path = process_batch(
                    input_dir=self.input_dir.get(),
                    output_dir=self.output_dir.get(),
                    recursive=self.recursive.get(),
                    use_llm=self.use_llm.get(),
                    log_cb=log_cb,
                    progress_cb=progress_cb
                )
                if excel_path:
                    self.append_log(f"Arquivo Excel salvo em: {excel_path}")
                    messagebox.showinfo("Concluído", f"Planilha gerada:\n{excel_path}")
                else:
                    messagebox.showwarning("Atenção", "Nenhum arquivo .txt/.pdf encontrado para processar.")
            except Exception as e:
                self.append_log("Falha fatal: " + str(e))
                self.append_log(traceback.format_exc())
                messagebox.showerror("Erro", str(e))
            finally:
                # destrava UI
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
        # Fallback CLI simples se Tk não estiver disponível
        print("Tkinter indisponível — rodando em modo CLI básico.")
        inp = input("Pasta de ENTRADA (.txt/.pdf): ").strip().strip('"').strip("'")
        outp = input("Pasta de SAÍDA (Excel): ").strip().strip('"').strip("'")
        use = input("Usar LLM (s/n, padrão s): ").strip().lower()
        rec = input("Recursivo (s/n, padrão s): ").strip().lower()
        use_llm = (use == "" or use.startswith("s"))
        recursive = (rec == "" or rec.startswith("s"))
        process_batch(inp, outp, recursive, use_llm, log_cb=lambda m: print(m), progress_cb=lambda i,t: print(f"{i}/{t}"))

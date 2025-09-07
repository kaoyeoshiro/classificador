# 📋 Classificador de Decisões Judiciais

Sistema inteligente para classificação automática de decisões judiciais usando **Heurística** + **LLM (GPT-5 mini)** com análise de confiança e renomeação inteligente de arquivos.

## 🎯 **Objetivo**

Automatizar a classificação de documentos jurídicos (.txt/.pdf) distinguindo entre:
- **Decisão Liminar** - Aprecia tutela de urgência/antecipada
- **Decisão Interlocutória** - Atos ordinários processuais
- **Sem classificação** - Arquivos sem conteúdo extraível

## ✨ **Principais Funcionalidades**

### 🧠 **Classificação Híbrida**
- **Heurística**: Padrões regex para termos jurídicos específicos
- **LLM**: GPT-5 mini com prompt especializado e few-shot examples
- **Ensemble**: Combina ambas as abordagens para máxima precisão

### 📊 **Sistema de Confiança (0.0 - 1.0)**
- **0.9-1.0**: Muito alta - classificação extremamente confiável
- **0.7-0.9**: Alta - classificação confiável
- **0.5-0.7**: Média - revisar se necessário
- **0.3-0.5**: Baixa - revisar obrigatoriamente
- **0.0-0.3**: Muito baixa - classificação duvidosa

### 📁 **Renomeação Inteligente**
- **Com conteúdo**: `85% - 00123456789 - Decisão Liminar.pdf`
- **Sem conteúdo**: `00987654321 - sem classificação.txt`
- **Duplicatas**: Numeração automática `(1), (2), (3)...`
- **Preservação**: Mantém extensão original (.txt/.pdf)

### 📄 **Suporte Multi-formato**
- **TXT**: Leitura direta
- **PDF**: Extração com `pymupdf4llm` (alta qualidade) + fallback `PyMuPDF`

## 🏛️ **Lógica de Classificação Jurídica**

### **Decisão Liminar** ⚖️
Quando há **NOVA APRECIAÇÃO** dos requisitos da tutela:
- ✅ `"Defiro a tutela de urgência"`
- ✅ `"Analisando novamente os requisitos do art. 300, mantenho a liminar"`
- ✅ `"Revogo a tutela anteriormente concedida"`
- ✅ `"Reconsidero para indeferir a liminar"`

### **Decisão Interlocutória** 📋
Atos ordinários, execução ou mera ratificação:
- ✅ `"Intime-se a parte para manifestação"`
- ✅ `"Transfira-se o valor de R$ X conforme tutela deferida"`
- ✅ `"Mantenho o acolhimento conforme decisão anterior"`
- ✅ `"Defiro a produção de prova pericial"`

### **Regras Críticas Implementadas**
1. **Verbo + Objeto**: `"Defiro a tutela"` = Liminar | `"Defiro prazo"` = Interlocutória
2. **Nova Apreciação vs. Ratificação**: Re-análise dos requisitos = Liminar | Mera continuidade = Interlocutória
3. **Execução vs. Concessão**: Executar tutela já deferida = Interlocutória
4. **Referência Histórica**: Mencionar liminar sem decidir sobre ela = Interlocutória

## 📊 **Saídas Geradas**

O sistema gera **5 tipos de arquivo** com timestamp:

### 1. **Excel Completo** (`resultado_TIMESTAMP.xlsx`)
```
Processo | Tipo | Resultado | Trecho | Arquivo | Fonte | Score | Confiança
```

### 2. **CSV Geral** (`resultado_TIMESTAMP.csv`)
Mesma estrutura do Excel em formato CSV

### 3. **CSV Estruturado** (`estruturado_TIMESTAMP.csv`)
```
Nº do Processo | Tipo de Decisão | Resultado | Trecho Utilizado | Nível de Confiança
```

### 4. **JSON Consolidado** (`resultado_TIMESTAMP.json`)
Todos os registros em formato JSON estruturado

### 5. **JSONL Incremental** (`resultado_TIMESTAMP.jsonl`)
Stream de processamento (um JSON por linha)

## 🚀 **Como Usar**

### **Interface Gráfica (Tkinter)**
```bash
python classificador_dec.py
```

1. **Selecione pasta** com arquivos .txt/.pdf
2. **Escolha pasta de saída** para relatórios
3. **Configure opções**:
   - ☑️ Usar LLM (recomendado)
   - ☑️ Busca recursiva
4. **Execute** e acompanhe o progresso

### **Programaticamente**
```python
from classificador_dec import process_batch

result = process_batch(
    input_dir="caminho/para/documentos",
    output_dir="caminho/para/saidas", 
    recursive=True,
    use_llm=True
)
```

## ⚙️ **Configuração**

### **Variáveis de Ambiente (.env)**
```env
LLM_API_BASE=https://api.openai.com/v1
LLM_API_KEY=sua_chave_openai_aqui
LLM_MODEL=gpt-5-mini
```

### **Dependências (requirements.txt)**
```
flask
requests
python-dotenv
tenacity
pandas
openpyxl
pymupdf
pymupdf4llm
pillow
tkinter
```

## 🎓 **Exemplos de Classificação**

### **Caso 1: Liminar Clara**
```
Texto: "Presentes os requisitos do art. 300, defiro a tutela de urgência."
→ Tipo: Decisão Liminar
→ Resultado: Deferido  
→ Confiança: 95%
→ Arquivo: "95% - 00123456789 - Decisão Liminar.pdf"
```

### **Caso 2: Interlocutória**
```
Texto: "Intime-se a parte autora para emendar a inicial em 15 dias."
→ Tipo: Decisão Interlocutória
→ Resultado: Não se aplica
→ Confiança: 85%
→ Arquivo: "85% - 00987654321 - Decisão Interlocutória.txt"
```

### **Caso 3: Execução de Tutela**
```
Texto: "Considerando a tutela deferida, determino: transfira-se R$ 5.000,00."
→ Tipo: Decisão Interlocutória (não é nova apreciação)
→ Resultado: Não se aplica
→ Confiança: 80%
→ Arquivo: "80% - 00111222333 - Decisão Interlocutória.pdf"
```

### **Caso 4: Sem Conteúdo**
```
Texto: [PDF corrompido/vazio]
→ Tipo: Sem classificação
→ Resultado: Não se aplica
→ Confiança: 0%
→ Arquivo: "00444555666 - sem classificação.pdf"
```

## 🔧 **Arquitetura Técnica**

### **Fluxo de Processamento**
1. **Listagem**: Encontra arquivos .txt/.pdf
2. **Extração**: Lê conteúdo textual (pymupdf4llm para PDFs)
3. **Validação**: Verifica se há conteúdo extraível
4. **Heurística**: Aplica padrões regex especializados
5. **LLM**: Consulta GPT-5 mini com prompt otimizado
6. **Ensemble**: Combina resultados (prioriza LLM se disponível)
7. **Renomeação**: Aplica nova nomenclatura com confiança
8. **Exportação**: Gera relatórios em múltiplos formatos

### **Componentes Principais**

#### **Padrões Heurísticos**
```python
LIMINAR_PATTERNS = [
    r'\btutela\s+de\s+urgência\b',
    r'\btutela\s+antecipada\b', 
    r'\bmedida\s+liminar\b',
    # ... mais padrões
]
```

#### **Prompt LLM Especializado**
- **Sistema**: Definições jurídicas precisas e regras críticas
- **Few-shots**: 8 exemplos cobrindo casos complexos
- **Saída**: JSON estruturado com confiança obrigatória

#### **Sistema de Confiança**
- **LLM**: Campo obrigatório no JSON (0.0-1.0)
- **Heurística**: Calculada pela força dos indicadores
- **Validação**: Clamp automático entre 0.0 e 1.0

## 📈 **Estatísticas e Qualidade**

### **Métricas Fornecidas**
- **Total processado**: Quantidade de arquivos
- **Distribuição por tipo**: Liminar vs. Interlocutória vs. Sem classificação
- **Confiança média**: Qualidade geral das classificações
- **Taxa de sucesso**: Arquivos classificados vs. sem conteúdo

### **Indicadores de Qualidade**
- **Confiança > 80%**: Classificação confiável
- **Conflito Heur vs. LLM**: Indica casos complexos
- **Arquivos sem conteúdo**: Possíveis problemas de digitalização

## 🛡️ **Robustez e Fallbacks**

### **Tratamento de Erros**
- **LLM indisponível**: Fallback para heurística
- **PDF corrompido**: Marca como "sem classificação"
- **API timeout**: Retry automático com backoff exponencial
- **JSON inválido**: Parsing defensivo com múltiplos formatos

### **Validações**
- **Confiança**: Normalizada entre 0.0-1.0
- **Tipos**: Apenas valores válidos aceitos
- **Processos**: Extração inteligente de números
- **Arquivos**: Verificação de existência antes de renomear

## 🎯 **Precisão Jurídica**

### **Diferenciações Implementadas**
- **Apreciação vs. Execução**: Decidir tutela vs. cumprir decisão
- **Nova análise vs. Ratificação**: Re-examinar vs. confirmar
- **Tutela vs. Atos ordinários**: Urgência vs. procedimento normal
- **Concessão vs. Manutenção**: Primeira vez vs. continuidade

### **Casos Especiais Tratados**
- Manutenção **COM** nova apreciação → Liminar
- Manutenção **SEM** nova apreciação → Interlocutória  
- Execução de tutela já deferida → Interlocutória
- Referência histórica a liminar → Interlocutória

## 🚀 **Resultados Esperados**

### **Organização Melhorada**
- Arquivos ordenados por confiança visual
- Identificação rápida de casos duvidosos
- Separação clara entre tipos de decisão

### **Eficiência Operacional**
- Processamento em lote de centenas de documentos
- Múltiplos formatos de saída para diferentes usos
- Redução drástica do trabalho manual

### **Qualidade Jurídica**
- Distinções precisas baseadas em critérios jurídicos sólidos
- Consistent application of legal principles
- Confidence scoring para priorização de revisão

---

**Desenvolvido para automatizar a classificação jurídica com precisão, confiabilidade e eficiência.** ⚖️✨

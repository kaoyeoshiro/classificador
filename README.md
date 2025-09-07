# Classificador de Petições Jurídicas

Sistema inteligente para classificação e análise de petições iniciais e contestações jurídicas, com extração automática de dados e geração de relatórios avançados.

## 🚀 Funcionalidades

### Classificação Automática
- **Petição Inicial**: Identifica e classifica petições iniciais
- **Contestação**: Detecta e classifica contestações
- **Outras Petições**: Categoriza demais tipos de peças processuais

### Extração de Dados
- **Pedidos**: Extrai pedidos das petições iniciais
- **Fundamentos de Fato**: Identifica fundamentos fáticos
- **Fundamentos de Direito**: Extrai fundamentos jurídicos
- **Argumentos de Defesa**: Captura argumentos das contestações

### Análise Semântica Avançada
- **Clustering Inteligente**: Agrupa argumentos similares semanticamente
- **Mapeamento Autor-Réu**: Correlaciona alegações com respostas
- **Padrões de Defesa**: Identifica estratégias recorrentes
- **Insights Jurídicos**: Gera recomendações práticas

### Relatórios Completos
- **JSON Estruturado**: Dados completos com rastreabilidade
- **Excel Avançado**: Planilhas organizadas por categoria
- **Padrões Simples**: "Quando inicial alega X, contestação responde Y"
- **Métricas de Qualidade**: Precisão, recall e F1-score

## 📋 Pré-requisitos

- Python 3.8+
- Bibliotecas listadas em `requirements.txt`

## 🛠️ Instalação

1. **Clone o repositório:**
```bash
git clone https://github.com/seu-usuario/classificador.git
cd classificador
```

2. **Crie um ambiente virtual:**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

4. **Configure as variáveis de ambiente:**
```bash
# Copie o arquivo de exemplo
cp .env.example .env
# Edite o arquivo .env com suas configurações
```

## ⚙️ Configuração

Crie um arquivo `.env` com suas configurações:

```env
OPENROUTER_API_KEY=sua_chave_api_aqui
APP_SITE_URL=https://openrouter.ai
```

## 🚀 Uso

### Interface Gráfica
```bash
python classificador_inicial_contest.py
```

### Funcionalidades Principais
1. **Selecione a pasta** com os arquivos PDF
2. **Configure o diretório de saída** para os relatórios
3. **Inicie o processamento** em lote
4. **Interrompa quando necessário** e gere relatórios parciais
5. **Visualize os resultados** em Excel e JSON

## 📊 Estrutura dos Relatórios

### JSON (Dados Completos)
- Metadados e auditoria
- Clusters semânticos
- Mapeamentos autor-réu
- Referências normativas
- Padrões simples
- Métricas de qualidade

### Excel (Visualização)
- **Resumo Executivo**: Métricas gerais
- **Clusters**: Agrupamentos semânticos
- **Mapeamentos**: Correlações detalhadas
- **Padrões de Defesa**: Estratégias recorrentes
- **Insights**: Recomendações práticas
- **Padrões Simples**: "Quando X, então Y"
- **Dados Brutos**: Petições e contestações

## 🔧 Tecnologias Utilizadas

- **Python 3.8+**: Linguagem principal
- **Tkinter**: Interface gráfica
- **OpenRouter API**: LLM para classificação
- **sentence-transformers**: Análise semântica
- **pandas**: Manipulação de dados
- **openpyxl**: Geração de Excel
- **scikit-learn**: Algoritmos de ML

## 📈 Recursos Avançados

### Rastreabilidade Completa
- Documento de origem
- Offset no texto original
- Referências normativas
- IDs de rastreamento

### Métricas de Qualidade
- Scores de confiança
- Distribuição de similaridade
- Precisão estimada
- Thresholds automáticos

### Persistência de Dados
- Backup automático
- Recuperação de falhas
- Checkpoints incrementais
- Histórico de processamento

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 📞 Suporte

Para dúvidas ou sugestões, abra uma issue no repositório.

---

**Desenvolvido para análise jurídica inteligente e geração de insights práticos.**

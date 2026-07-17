# MedQA-USMLE com LightRAG

Este é um kit completo para reproduzir um experimento que avalia o LightRAG em questões de medicina (USMLE). Se você nunca rodou isso antes, siga os passos abaixo.




## Pipeline 

1. **Indexação** – Processa 18 livros de medicina e constrói um Grafo de Conhecimento Médico utilizando o LightRAG.
2. **Filtragem** – Seleciona as questões do MedQA com cobertura suficiente no grafo para a avaliação.
3. **Inferência** – Recupera o contexto relevante e gera respostas para as questões selecionadas.
4. **Avaliação** – Compara as respostas com o gabarito utilizando métricas do RAGAS e acurácia.


## Customizações Importantes

Este kit usa versões modificadas de dois arquivos do LightRAG:

- **`lightrag/prompt.py`** – Prompt especializado para domínio médico (extração de entidades clínicas)
- **`lightrag/constants.py`** – Constantes ajustadas para MedQA (chunk size, overlap, etc.)

Se você clonar o repositório original, precisará aplicar essas mudanças manualmente ou usar os arquivos deste kit.

## Antes de começar

Você precisa de:
- Python 3.10+
- OpenAI API key com créditos 
- 50 GB de espaço em disco
- 16 GB de RAM

## Setup rápido

```bash
# 1. Clonar e entrar no diretório
cd /home/iris/LightRAG/reproduce/steps_replicacao

# 2. Criar ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# 3. Instalar dependências
pip install -e ../..
pip install -r requirements.txt

# 4. Configurar API key
export OPENAI_API_KEY="sk-..."
```

## Rodar o pipeline

Se você já tem o grafo indexado (Passo 1), pode começar daqui:

```bash
# Passo 2: Filtrar questões (rápido)
python3 Step_2_build_benchmark.py --max-questions 300

# Passo 3: Gerar respostas (demora)
MEDQA_NUM_QUESTIONS=125 python3 Step_3_query_medqa.py

# Passo 4: Avaliar qualidade
python3 Step_4_ragas_eval.py
```

Ou rodar tudo de uma vez:

```bash
cd /home/iris/LightRAG/reproduce/steps_replicacao && \
python3 Step_2_build_benchmark.py --max-questions 300 && \
MEDQA_NUM_QUESTIONS=125 python3 Step_3_query_medqa.py && \
python3 Step_4_ragas_eval.py
```

## Arquivos gerados

Após rodar, você vai ter:
- `experiments/results/filtered_125_results.json` - Respostas do sistema
- `experiments/results/eval_*_ragas_results.json` - Métricas de qualidade
- `experiments/results/eval_*_ragas_report.md` - Relatório final

## Monitorar progresso

```bash
# Ver logs em tempo real
tail -f /home/iris/LightRAG/experiments/logs/query_*.log

# Ver quantas questões já foram processadas
jq 'length' /home/iris/LightRAG/experiments/results/filtered_125_results.json
```

## Resultados esperados

- **Acurácia**: 50-70% (baseline é 73.92%)
- **Answer Correctness**: 0.4-0.6
- **Semantic Similarity**: 0.6-0.8

## Customizar

```bash
# Rodar com 50 questões em vez de 125
MEDQA_NUM_QUESTIONS=50 python3 Step_3_query_medqa.py

# Rodar com 1.200 questões (produção)
MEDQA_NUM_QUESTIONS=1200 python3 Step_3_query_medqa.py
```

## Problemas comuns

| Erro | O que fazer |
|------|-----------|
| `OPENAI_API_KEY not found` | Rode `export OPENAI_API_KEY="sk-..."` |
| Query muito lenta | Normal na primeira rodada. Cache vai acelerar depois. |
| Acurácia baixa | Verificar se indexação completou >50% dos livros |
| Processo travou | Seguro. Próxima rodada continua de onde parou. |

## Estrutura de pastas

```
steps_replicacao/
├─ Step_1_index_medqa.py      (indexação)
├─ Step_2_build_benchmark.py  (filtragem)
├─ Step_3_query_medqa.py      (query)
├─ Step_4_ragas_eval.py       (avaliação)
├─ requirements.txt
└─ README.md (você está aqui)
```

## Referências

- Framework: [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)
- Benchmark: [MedQA-USMLE](https://github.com/jind11/MedQA)
- Avaliação: [RAGAS](https://github.com/explodinggradients/ragas)

---



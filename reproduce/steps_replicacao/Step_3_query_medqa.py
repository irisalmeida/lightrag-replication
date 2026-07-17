import os
import sys
import re
import json
import time
import asyncio
import logging
import subprocess
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import wrap_embedding_func_with_attrs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils_medqa import (
    carregar_questoes_medqa,
    extrair_alternativa,
    calcular_metricas,
    salvar_resultados_csv,
)

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKING_DIR = PROJECT_ROOT / "working_dir"
BENCHMARK_FILE = PROJECT_ROOT / "experiments" / "results" / "benchmark_filtered.json"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
LOGS_DIR = PROJECT_ROOT / "experiments" / "logs"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

NUM_QUESTIONS = int(os.getenv("MEDQA_NUM_QUESTIONS", "125"))
PHASE_NAME = "filtered_125" if NUM_QUESTIONS == 125 else f"filtered_{NUM_QUESTIONS}"
RESULTS_FILE = RESULTS_DIR / f"{PHASE_NAME}_results.json"
CSV_FILE = RESULTS_DIR / f"{PHASE_NAME}_results.csv"
DOC_FILE = RESULTS_DIR / f"{PHASE_NAME}_experiment_documentation.txt"

from datetime import datetime

_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS_DIR / f"query_filtered_{NUM_QUESTIONS}_{_timestamp}.log"

logger = logging.getLogger("step3_query_medqa")
logger.setLevel(logging.INFO)
_fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_fh)
logger.addHandler(_sh)

async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.openai.com/v1",
        **kwargs,
    )


_embedding_model = None


@wrap_embedding_func_with_attrs(embedding_dim=768, model_name="all-mpnet-base-v2")
async def sentence_transformer_embed(texts: list[str]) -> np.ndarray:
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer("all-mpnet-base-v2")
    if isinstance(texts, str):
        texts = [texts]
    return _embedding_model.encode(texts, convert_to_numpy=True)

ANSWER_USER_PROMPT = (
    "You are answering a USMLE-style multiple-choice question. You MUST respond "
    "with a single valid JSON object, and nothing else, in exactly this format: "
    '{"analysis": "<brief clinical reasoning grounded strictly in the Context above>", '
    '"answer": "<single letter A, B, C, D, or E>"}. '
    "Do not include any text before or after the JSON object. Do not use markdown code fences."
)


def build_question_prompt(question: dict) -> str:
    options_text = "\n".join(f"{k}: {v}" for k, v in question["options"].items())
    return (
        f"Question: {question['question']}\n\n"
        f"Options:\n{options_text}\n\n"
        f"Which option is correct? Respond only with the required JSON object."
    )


def parse_llm_json_answer(raw_response) -> tuple[str, str]:
    if raw_response is None:
        return "", ""
    text = raw_response if isinstance(raw_response, str) else str(raw_response)
    cleaned = re.sub(r"```json|```", "", text).strip()
    try:
        parsed = json.loads(cleaned)
        answer = str(parsed.get("answer", "")).strip().upper()
        analysis = str(parsed.get("analysis", "")).strip()
        match = re.search(r"\b([A-E])\b", answer)
        if match:
            return match.group(1), analysis
    except (json.JSONDecodeError, AttributeError):
        pass
    return extrair_alternativa(text), text


async def initialize_rag() -> LightRAG:
    rag = LightRAG(
        working_dir=str(WORKING_DIR),
        llm_model_func=llm_model_func,
        embedding_func=sentence_transformer_embed,
    )
    await rag.initialize_storages()
    return rag


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def load_benchmark_filtered(benchmark_file: Path, limite: int | None = None) -> list[dict]:
    if not benchmark_file.exists():
        logger.error(f"Benchmark não encontrado: {benchmark_file}")
        sys.exit(1)
    with open(benchmark_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
    if limite:
        questions = questions[:limite]
    logger.info(f"✓ Carregadas {len(questions)} questões filtradas")
    return questions


async def run_pilot() -> None:
    if not WORKING_DIR.exists() or not any(WORKING_DIR.iterdir()):
        logger.error(f"working_dir vazio: {WORKING_DIR}")
        sys.exit(1)

    logger.info(f"FASE: Query ({NUM_QUESTIONS} questões)")
    logger.info(f"Iniciado: {datetime.now().isoformat()}")

    questions = load_benchmark_filtered(BENCHMARK_FILE, limite=NUM_QUESTIONS)
    if not questions:
        logger.error(f"Nenhuma questão carregada")
        sys.exit(1)

    rag = await initialize_rag()
    logger.info(f"✓ LightRAG inicializado")

    try:
        await rag.chunk_entity_relation_graph.get_graph()
        logger.info(f"✓ Grafo carregado")
    except Exception as e:
        logger.warning(f"Aviso: {e}")

    query_param = QueryParam(mode="hybrid", user_prompt=ANSWER_USER_PROMPT)

    resultados = []
    execution_start = time.time()

    for idx, q in enumerate(questions, start=1):
        pergunta_texto = build_question_prompt(q)

        start = time.time()
        api_calls = 1
        try:
            raw_response = await rag.aquery(
                pergunta_texto,
                param=query_param,
            )
        except Exception as e:
            logger.error(f"[{idx}/{len(questions)}] Erro na query: {e}")
            raw_response = None
        elapsed_ms = (time.time() - start) * 1000

        resposta_predita, analysis = parse_llm_json_answer(raw_response)
        acerto = resposta_predita == q["answer_idx"]

        resultado = {
            "indice": idx,
            "pergunta": q["question"],
            "opcoes": q["options"],
            "resposta_verdadeira": q["answer_idx"],
            "resposta_predita": resposta_predita,
            "analise_llm": analysis if analysis else "",
            "resposta_bruta": raw_response if raw_response else "",
            "acerto": acerto,
            "chamadas_api": api_calls,
            "tempo_ms": round(elapsed_ms, 2),
        }
        resultados.append(resultado)

        acertos_ate_aqui = sum(1 for r in resultados if r["acerto"])
        acuracia_ate_aqui = (acertos_ate_aqui / idx * 100) if idx > 0 else 0
        logger.info(
            f"[{idx:4d}/{len(questions)}] Pred={resposta_predita} Real={q['answer_idx']} "
            f"{'✓' if acerto else '✗'} Acur={acuracia_ate_aqui:5.1f}% ({elapsed_ms:6.0f}ms)"
        )

        if idx % 10 == 0 or idx == len(questions):
            with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                json.dump(resultados, f, indent=2, ensure_ascii=False)

    total_execution_time = time.time() - execution_start

    salvar_resultados_csv(resultados, str(CSV_FILE))

    y_true = [r["resposta_verdadeira"] for r in resultados]
    y_pred = [r["resposta_predita"] for r in resultados]
    metricas = calcular_metricas(y_true, y_pred)

    total_api_calls = sum(r["chamadas_api"] for r in resultados)
    tempo_medio_ms = sum(r["tempo_ms"] for r in resultados) / len(resultados)

    logger.info(f"Questões: {len(resultados)} | Acurácia: {metricas['acuracia']:.1f}% | Tempo: {total_execution_time/60:.1f}min")
    logger.info(f"JSON: {RESULTS_FILE} | CSV: {CSV_FILE} | DOC: {DOC_FILE}")

    # Gera documentacao
    await write_documentation(metricas, total_api_calls, tempo_medio_ms, len(resultados), NUM_QUESTIONS)


async def write_documentation(metricas: dict, total_api_calls: int, tempo_medio_ms: float, n: int, num_questions_param: int) -> None:
    phase_status = "FILTRADO POR COBERTURA"
    completion_pct = (n / 287 * 100) if num_questions_param <= 287 else 100

    content = f"""EXPERIMENT DOCUMENTATION - MedQA-USMLE via LightRAG
{'=' * 70}

FASE: {phase_status}
Status de Conclusao: {n}/{min(num_questions_param, 287)} questoes ({completion_pct:.1f}%)
Data/Hora: {datetime.now().isoformat()}

ARQUITETURA DE INDEXACAO
{'='*70}
Estrategia: Insercao incremental por livro (checkpoint por segmento)
  - Cada livro dividido em segmentos (~100 chunks cada)
  - Falha em 1 segmento nao descarta o livro inteiro
  - working_dir/ persiste entre execucoes para recuperacao
  - Cache LLM reutilizado para evitar reprocessamento

Parametros de Indexacao (paridade AMG-RAG)
-------------------------------------------
chunk_token_size: 512
chunk_overlap_token_size: 100
entity_extract_max_gleaning: 1
embedding_model: sentence-transformers/all-mpnet-base-v2 (768 dim)
llm_backbone: gpt-4o-mini
max_concurrent_llm: 2
timeout_por_chunk: 600s

Parametros de Query
-------------------
mode: hybrid
num_questions: {n}
top_k_entities: 60
top_k_chunks: 20
formato_saida_llm: {{"analysis": "...", "answer": "A-E"}}

METRICAS ({n} QUESTOES)
{'='*70}
Acuracia: {metricas['acuracia']:.1f}%
F1-score (weighted): {metricas['f1_score']:.1f}%
Questoes validas: {metricas['questoes_validas']}
Questoes invalidas (resposta nao A-E): {metricas['questoes_invalidas']}
Chamadas de API por query (media): {total_api_calls / n:.2f}
Tempo medio por query: {tempo_medio_ms:.0f}ms

BASELINE DE COMPARACAO (AMG-RAG)
{'='*70}
Acuracia: 73.92%
F1-score: 74.1%
Tamanho do dataset: 1.200 questoes

REPRODUCIBILIDADE
{'='*70}
Commit: {get_git_commit()}
Python: {sys.version.split()[0]}
Dataset: reproduce/MedQA-USMLE/
Knowledge Graph: working_dir/
Cache LLM: working_dir/kv_store_llm_response_cache.json

INSTRUCOS PARA REPLICA
{'='*70}
1. Indexacao (uma vez):
   python reproduce/steps_replicacao/Step_1_run_all_books.py

2. Filtragem por cobertura (uma vez):
   python reproduce/steps_replicacao/Step_2_build_benchmark.py

3. Inferência com 125 questões filtradas (validação):
   python reproduce/steps_replicacao/Step_3_query_medqa.py

4. Inferência com todas as questões filtradas (até 287):
   MEDQA_NUM_QUESTIONS=287 python reproduce/steps_replicacao/Step_3_query_medqa.py

Todos os resultados sao salvos em experiments/results/

Gerado em: {datetime.now().isoformat()}
"""
    with open(DOC_FILE, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"✓ Documentacao gerada: {DOC_FILE}")


if __name__ == "__main__":
    asyncio.run(run_pilot())

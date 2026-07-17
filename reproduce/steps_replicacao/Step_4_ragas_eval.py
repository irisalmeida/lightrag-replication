import os
import re
import sys
import json
import time
import asyncio
import logging
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from dotenv import load_dotenv

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import wrap_embedding_func_with_attrs

from ragas.metrics.collections.answer_correctness import AnswerCorrectness
from ragas.metrics.collections import SemanticSimilarity
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from openai import AsyncOpenAI

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKING_DIR = PROJECT_ROOT / "working_dir"
QUESTIONS_FILE = PROJECT_ROOT / "data" / "medqa" / "questions" / "US" / "test.jsonl"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
LOGS_DIR = PROJECT_ROOT / "experiments" / "logs"

for _d in (RESULTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS_DIR / f"ragas_eval_{_ts}.log"

logger = logging.getLogger("step4_ragas")
logger.setLevel(logging.INFO)
_fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_fh)
logger.addHandler(_sh)

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
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
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        _embedding_model = SentenceTransformer("all-mpnet-base-v2")
    if isinstance(texts, str):
        texts = [texts]
    return _embedding_model.encode(texts, convert_to_numpy=True)

ANSWER_SYSTEM_PROMPT = (
    "You are a medical expert answering USMLE-style questions. "
    "Use ONLY the provided context. Be concise and factually accurate."
)

ANSWER_USER_PROMPT = (
    "Based on the context above, answer the following multiple-choice question. "
    "Respond with a JSON object in this exact format: "
    '{"reasoning": "<brief clinical reasoning>", "answer_letter": "<A|B|C|D|E>", '
    '"answer_text": "<full text of the chosen option>"}. '
    "Output only the JSON object, no markdown."
)


def build_question_text(q: dict) -> str:
    options = "\n".join(f"{k}: {v}" for k, v in q["options"].items())
    return f"Question: {q['question']}\n\nOptions:\n{options}"


def parse_answer(raw: str | None, options: dict) -> tuple[str, str, str]:
    if not raw:
        return "", "", ""
    cleaned = re.sub(r"```json|```", "", raw).strip()
    try:
        parsed = json.loads(cleaned)
        letter = str(parsed.get("answer_letter", "")).strip().upper()
        match = re.search(r"\b([A-E])\b", letter)
        letter = match.group(1) if match else ""
        text = parsed.get("answer_text", options.get(letter, ""))
        reasoning = parsed.get("reasoning", "")
        return letter, str(text), str(reasoning)
    except (json.JSONDecodeError, AttributeError):
        match = re.search(r"\b([A-E])\b", raw.upper())
        letter = match.group(1) if match else ""
        return letter, options.get(letter, raw[:200]), ""

def load_benchmark_json(path: Path, limit: int | None) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if limit:
        data = data[:limit]
    return data


def load_full_medqa(path: Path, limit: int | None) -> list[dict]:
    questions = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                q = json.loads(line.strip())
                if "question" in q and "options" in q and "answer_idx" in q:
                    questions.append({
                        "question_id": i + 1,
                        "question": q["question"],
                        "options": q["options"],
                        "answer": q.get("answer", ""),
                        "answer_idx": q["answer_idx"],
                        "coverage": 1.0,
                    })
            except json.JSONDecodeError:
                continue
    return questions

async def run_evaluation(questions: list[dict], rag: LightRAG) -> list[dict]:
    records = []

    for idx, q in enumerate(questions, 1):
        question_text = build_question_text(q)
        start = time.time()

        context = ""
        try:
            context = await rag.aquery(
                question_text,
                param=QueryParam(mode="hybrid", only_need_context=True),
            )
        except Exception as e:
            logger.error(f"[{idx}/{len(questions)}] Erro contexto: {e}")
            context = ""

        response_raw = None
        try:
            response_raw = await rag.aquery(
                question_text,
                param=QueryParam(mode="hybrid", user_prompt=ANSWER_USER_PROMPT),
            )
        except Exception as e:
            logger.error(f"[{idx}/{len(questions)}] Erro resposta: {e}")
            response_raw = None

        elapsed = time.time() - start
        answer_letter, answer_text, reasoning = parse_answer(
            response_raw, q["options"]
        )

        # ground_truth: resposta descritiva completa do MedQA
        correct_letter = q["answer_idx"]
        ground_truth_text = q.get("answer") or q["options"].get(correct_letter, correct_letter)

        correct = answer_letter == correct_letter

        record = {
            "question_id": q.get("question_id", idx),
            "coverage": q.get("coverage", 1.0),
            "user_input": question_text,
            "retrieved_contexts": [context] if context else [""],
            "response": f"{answer_letter}: {answer_text}" if answer_text else answer_letter,
            "reference": ground_truth_text,
            "answer_letter_pred": answer_letter,
            "answer_letter_true": correct_letter,
            "correct": correct,
            "reasoning": reasoning[:300],
            "elapsed_s": round(elapsed, 2),
        }
        records.append(record)

        logger.info(
            f"[{idx:4d}/{len(questions)}] Pred={answer_letter} True={correct_letter} "
            f"{'✓' if correct else '✗'} cov={q.get('coverage', 1.0):.2f} ({elapsed:.1f}s)"
        )

        if idx % 10 == 0 or idx == len(questions):
            ckpt = RESULTS_DIR / "ragas_checkpoint.json"
            with open(ckpt, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)

    return records

async def run_ragas_async(records: list[dict], skip_answer_correctness: bool = False) -> pd.DataFrame:
    n_samples = len(records)
    MIN_RAGAS_SAMPLES = 5

    if n_samples < MIN_RAGAS_SAMPLES:
        logger.warning(f"Dataset pequeno (n={n_samples} < {MIN_RAGAS_SAMPLES})")

    for i, r in enumerate(records[:5]):
        for col in ("user_input", "response", "reference"):
            if not str(r.get(col, "")).strip():
                logger.error(f"Row {i}: '{col}' vazio")

    logger.info("Configurando RAGAS...")
    embeddings = embedding_factory("huggingface", model="sentence-transformers/all-mpnet-base-v2")

    ac_scorer = None
    if not skip_answer_correctness:
        logger.info("Carregando LLM para AnswerCorrectness...")
        client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=60,
            max_retries=1,
        )
        llm = llm_factory("gpt-4o-mini", client=client)
        ac_scorer = AnswerCorrectness(llm=llm, embeddings=embeddings)

    ss_scorer = SemanticSimilarity(embeddings=embeddings)

    logger.info("Executando RAGAS...")

    rows = []
    for idx, r in enumerate(records, 1):
        ac_value = None
        ss_value = None

        # AnswerCorrectness (opcional)
        if ac_scorer:
            try:
                ac_result = await asyncio.wait_for(
                    ac_scorer.ascore(
                        user_input=r["user_input"],
                        response=r["response"],
                        reference=r["reference"],
                    ),
                    timeout=90,  # segundos por amostra
                )
                ac_value = ac_result.value if hasattr(ac_result, 'value') else ac_result
            except asyncio.TimeoutError:
                logger.warning(f"[{idx}/{n_samples}] AnswerCorrectness timeout (90s) — pulando amostra")
            except Exception as e:
                logger.warning(f"[{idx}/{n_samples}] Erro AnswerCorrectness: {type(e).__name__}: {e}")

        # SemanticSimilarity
        try:
            ss_result = await ss_scorer.ascore(
                reference=r["reference"],
                response=r["response"],
            )
            ss_value = ss_result.value if hasattr(ss_result, 'value') else ss_result
        except Exception as e:
            logger.warning(f"[{idx}/{n_samples}] Erro SemanticSimilarity: {e}")

        row = {
            "semantic_similarity": ss_value,
        }
        if ac_scorer:
            row["answer_correctness"] = ac_value
        rows.append(row)

        if idx % 10 == 0 or idx == n_samples:
            logger.info(f"RAGAS: {idx}/{n_samples} amostras avaliadas")

        # Pequeno delay entre amostras para evitar rate limit
        if ac_scorer and idx < n_samples:
            await asyncio.sleep(1)

    df = pd.DataFrame(rows)

    # Log dos resultados
    logger.info(f"Avaliação RAGAS concluída: {len(df)} amostras processadas")

    metrics_to_check = ["answer_correctness", "semantic_similarity"] if ac_scorer else ["semantic_similarity"]
    for col in metrics_to_check:
        if col in df.columns:
            non_null = df[col].notna().sum()
            logger.info(f"✓ {col}: {non_null}/{len(df)} valores válidos")
            if non_null > 0:
                valid = df[col].dropna()
                logger.info(
                    f"    Média: {valid.mean():.3f} | Std: {valid.std():.3f} | "
                    f"Min: {valid.min():.3f} | Max: {valid.max():.3f}"
                )
            else:
                logger.error(f"    ❌ CRÍTICO: Todos os valores de {col} são NULL!")

    return df


# ---------------------------------------------------------------------------
# Exportação
# ---------------------------------------------------------------------------
def export_results(
    ragas_df: pd.DataFrame,
    records: list[dict],
    prefix: str,
) -> tuple[Path, Path, pd.DataFrame]:
    # Mescla metadados originais com scores RAGAS
    meta_df = pd.DataFrame([
        {
            "question_id": r["question_id"],
            "coverage": r["coverage"],
            "answer_letter_pred": r["answer_letter_pred"],
            "answer_letter_true": r["answer_letter_true"],
            "correct": r["correct"],
            "elapsed_s": r["elapsed_s"],
        }
        for r in records
    ])
    full_df = pd.concat([ragas_df.reset_index(drop=True), meta_df], axis=1)

    csv_path = RESULTS_DIR / f"{prefix}_ragas_results.csv"
    json_path = RESULTS_DIR / f"{prefix}_ragas_results.json"

    full_df.to_csv(csv_path, index=False, encoding="utf-8")
    full_df.to_json(json_path, orient="records", indent=2, force_ascii=False)

    logger.info(f"CSV exportado:  {csv_path}")
    logger.info(f"JSON exportado: {json_path}")
    return csv_path, json_path, full_df


# Visualizações removidas (apenas dados numéricos)


# ---------------------------------------------------------------------------
# Relatório final em Markdown
# ---------------------------------------------------------------------------
def generate_report(df: pd.DataFrame, records: list[dict], prefix: str) -> Path:
    n = len(df)
    acc = df["correct"].mean() * 100 if "correct" in df.columns else float("nan")

    ac_col = "answer_correctness"
    as_col = "semantic_similarity"

    ac_mean = df[ac_col].mean() if ac_col in df.columns else float("nan")
    ac_std = df[ac_col].std() if ac_col in df.columns else float("nan")
    as_mean = df[as_col].mean() if as_col in df.columns else float("nan")
    as_std = df[as_col].std() if as_col in df.columns else float("nan")

    # Alta semantic_similarity + resposta errada (problema no LLM, não na recuperação)
    high_sim_wrong = 0
    if as_col in df.columns and "correct" in df.columns:
        high_sim_wrong = int(((df[as_col] >= 0.7) & (df["correct"] == False)).sum())

    avg_elapsed = sum(r["elapsed_s"] for r in records) / max(n, 1)

    report_path = RESULTS_DIR / f"{prefix}_ragas_report.md"
    content = f"""# Relatório de Avaliação RAGAS — LightRAG × MedQA-USMLE

**Data**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Questões avaliadas**: {n}
**Prefixo de arquivos**: `{prefix}`

---

## 1. Resultados

### Acurácia (seleção de alternativa)

| Métrica | Valor |
|---------|-------|
| Acurácia (letter match) | **{acc:.1f}%** |
| Baseline AMG-RAG | 73,92% |
| Δ vs. baseline | {acc - 73.92:+.1f} pp |

### Métricas RAGAS

| Métrica | Média | Desvio Padrão | Intervalo [0, 1] | Descrição |
|---------|-------|--------------|-----------------|-----------|
| Answer Correctness | **{ac_mean:.3f}** | {ac_std:.3f} | [0, 1] | Factual + Semantic |
| Answer Similarity | **{as_mean:.3f}** | {as_std:.3f} | [0, 1] | Semantic Only |

---

## 2. Discussão

### Answer Correctness ({ac_mean:.3f})

A métrica **Answer Correctness** combina:
- **Factual Correctness**: Análise de fatos verdadeiros (TP), falsos (FP) e ausentes (FN)
- **Semantic Similarity**: Similaridade semântica entre respostas

Um valor de {ac_mean:.3f} indica que o LightRAG {"consegue recuperar conhecimento suficiente para gerar respostas factualmente corretas e semanticamente similares" if ac_mean >= 0.5 else "ainda apresenta dificuldades na geração de respostas factualmente corretas, possivelmente devido à cobertura parcial do grafo"}.

### Answer Similarity ({as_mean:.3f})

A **Answer Similarity** mede apenas a proximidade semântica entre a resposta gerada e
o ground truth, usando cross-encoder ou embeddings para calcular similaridade.

**Padrão observado — alta similarity + resposta errada**:
{high_sim_wrong} questões ({high_sim_wrong/max(n,1)*100:.1f}%) apresentaram answer_similarity ≥ 0,70
mas a alternativa escolhida estava incorreta. Isso indica que, nesses casos, o
conhecimento foi corretamente recuperado pelo grafo, mas o LLM falhou na etapa
final de decisão entre alternativas próximas — o problema está na seleção da
alternativa, não na recuperação.

---

## 3. Ameaças à Validade

| Ameaça | Descrição | Impacto |
|--------|-----------|---------|
| Cobertura parcial do grafo | {n} questões selecionadas por cobertura de entidades; questões sem cobertura excluídas | Viés de seleção — resultados podem superestimar desempenho |
| Modelo LLM | gpt-4o-mini (menor que GPT-4) | Subestima o teto de qualidade do pipeline |
| Embedding | all-mpnet-base-v2 (768 dim) vs. modelos maiores | Pode limitar recall de entidades relevantes |
| Prompt de extração | Prompt especializado em MedQA pode favorecer questões clínicas típicas | Generalização limitada para sub-especialidades raras |
| Ground truth RAGAS | Usa resposta textual completa do MedQA, não apenas a letra | Penaliza respostas corretas com paráfrase diferente |

---

## 4. Conclusões

{"O LightRAG, mesmo com grafo parcialmente indexado, demonstra capacidade de recuperação de conhecimento médico suficiente para gerar respostas textualmente próximas ao ground truth" if as_mean >= 0.5 else "O grafo parcialmente indexado limita a qualidade da recuperação, sugerindo que a indexação completa é necessária para atingir o baseline AMG-RAG"}.

A Semantic Similarity de {as_mean:.3f} {"confirma que o pipeline de recuperação funciona corretamente; a diferença na acurácia de alternativas reflete limitação do LLM na discriminação final" if as_mean >= 0.65 else "indica que o contexto recuperado ainda não é suficientemente próximo ao conhecimento necessário para as questões"}.

Para a pergunta de pesquisa — *"Um Knowledge Graph parcialmente indexado consegue recuperar conhecimento suficiente para gerar respostas textualmente corretas?"* — a evidência {"suporta uma resposta afirmativa" if as_mean >= 0.6 and ac_mean >= 0.4 else "é ainda inconclusiva, recomendando-se completar a indexação antes de uma avaliação definitiva"}.

---

## 5. Arquivos Gerados

| Arquivo | Conteúdo |
|---------|---------|
| `{prefix}_ragas_results.csv` | Resultados completos + métricas |
| `{prefix}_ragas_results.json` | Idem em JSON |
| `{prefix}_answer_correctness.png` | Visualização Answer Correctness |
| `{prefix}_semantic_similarity.png` | Visualização Semantic Similarity |
| `{prefix}_ragas_report.md` | Este relatório |

---

## 6. Parâmetros de Execução

| Parâmetro | Valor |
|-----------|-------|
| Modo de query LightRAG | hybrid |
| LLM de resposta | gpt-4o-mini |
| LLM RAGAS | gpt-4o-mini |
| Embedding RAGAS | text-embedding-3-small |
| Embedding LightRAG | all-mpnet-base-v2 (768 dim) |
| Tempo médio por questão | {avg_elapsed:.1f}s |
| Total de questões | {n} |

*Gerado automaticamente por Step_4_ragas_eval.py em {datetime.now().isoformat()}*
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Relatório gerado: {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=str,
        default=str(RESULTS_DIR / "benchmark_filtered.json"),
        help="JSON gerado pelo Step_2_build_benchmark.py (default: benchmark_filtered.json)",
    )
    parser.add_argument(
        "--use-full-medqa",
        action="store_true",
        help="Ignora --input e usa o test.jsonl completo (grafo totalmente indexado)",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limita o número de questões a avaliar (útil para testes rápidos)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=f"eval_{_ts}",
        help="Prefixo para os arquivos de saída (default: eval_<timestamp>)",
    )
    parser.add_argument(
        "--skip-answer-correctness",
        action="store_true",
        help="Pula AnswerCorrectness (métrica cara com LLM), mantém apenas SemanticSimilarity (mais rápido)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY não configurada")
        sys.exit(1)

    args = parse_args()

    # Carrega questões
    if args.use_full_medqa:
        logger.info(f"Modo: MedQA completo ({QUESTIONS_FILE})")
        questions = load_full_medqa(QUESTIONS_FILE, args.max_questions)
        prefix = args.prefix or f"fullmedqa_{_ts}"
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(
                f"Arquivo de benchmark não encontrado: {input_path}\n"
                "Execute Step_2_build_benchmark.py primeiro, ou use --use-full-medqa"
            )
            sys.exit(1)
        logger.info(f"Modo: benchmark filtrado ({input_path})")
        questions = load_benchmark_json(input_path, args.max_questions)
        prefix = args.prefix

    if not questions:
        logger.error("Nenhuma questão carregada.")
        sys.exit(1)

    logger.info(f"Questões a avaliar: {len(questions)}")

    # Inicializa LightRAG
    if not WORKING_DIR.exists():
        logger.error(f"working_dir não encontrado: {WORKING_DIR}")
        sys.exit(1)

    rag = LightRAG(
        working_dir=str(WORKING_DIR),
        llm_model_func=llm_model_func,
        embedding_func=sentence_transformer_embed,
    )
    await rag.initialize_storages()
    logger.info("LightRAG inicializado")

    # — Etapa 1: recuperação + geração
    logger.info("\n" + "=" * 60)
    logger.info("ETAPA 1 — Recuperação e geração de respostas")
    logger.info("=" * 60)
    records = await run_evaluation(questions, rag)
    await rag.finalize_storages()

    # — Etapa 2: avaliação RAGAS
    logger.info("\n" + "=" * 60)
    logger.info("ETAPA 2 — Avaliação RAGAS")
    logger.info("=" * 60)
    logger.info(f"Records para RAGAS: {len(records)} amostras")
    ragas_df = await run_ragas_async(records, skip_answer_correctness=args.skip_answer_correctness)

    # — Exportação
    csv_path, json_path, full_df = export_results(ragas_df, records, prefix)

    # — Visualizações (desabilitadas - apenas números)
    # logger.info("\n" + "=" * 60)
    # logger.info("ETAPA 3 — Visualizações")
    # logger.info("=" * 60)

    # — Relatório
    report_path = generate_report(full_df, records, prefix)

    # Sumário final
    logger.info("\n" + "=" * 60)
    logger.info("SUMÁRIO FINAL")
    logger.info("=" * 60)

    # Métricas RAGAS
    for col_name, description in [("answer_correctness", "Factual + Semantic"),
                                   ("answer_similarity", "Semantic Only")]:
        if col_name in full_df.columns:
            valid_values = full_df[col_name].dropna()
            if len(valid_values) > 0:
                logger.info(
                    f"{col_name} ({description}): "
                    f"média={valid_values.mean():.3f}  "
                    f"std={valid_values.std():.3f}  "
                    f"min={valid_values.min():.3f}  "
                    f"max={valid_values.max():.3f}"
                )
            else:
                logger.error(f"{col_name}: TODOS OS VALORES SÃO NULL")

    # Acurácia
    if "correct" in full_df.columns:
        acc = full_df['correct'].mean() * 100
        logger.info(f"Acurácia (letter): {acc:.1f}%")

    logger.info(f"\nArquivos gerados em: {RESULTS_DIR}")
    logger.info(f"  CSV:        {csv_path.name}")
    logger.info(f"  JSON:       {json_path.name}")
    logger.info(f"  Relatório:  {report_path.name}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

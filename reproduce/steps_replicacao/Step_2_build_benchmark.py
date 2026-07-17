import re
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKING_DIR = PROJECT_ROOT / "working_dir"
QUESTIONS_FILE = PROJECT_ROOT / "data" / "medqa" / "questions" / "US" / "test.jsonl"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
LOGS_DIR = PROJECT_ROOT / "experiments" / "logs"
GRAPH_FILE = WORKING_DIR / "graph_chunk_entity_relation.graphml"

for _d in (RESULTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logger = logging.getLogger("step2_benchmark")
logger.setLevel(logging.INFO)
_fh = logging.FileHandler(LOGS_DIR / f"benchmark_{_ts}.log", encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_fh)
logger.addHandler(_sh)

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "on", "at", "by", "for", "with", "about",
    "against", "between", "through", "during", "before", "after", "above",
    "below", "from", "up", "down", "out", "off", "over", "under", "again",
    "further", "then", "once", "and", "but", "or", "nor", "so", "yet",
    "both", "either", "neither", "not", "only", "own", "same", "than",
    "too", "very", "just", "because", "as", "until", "while", "if",
    "this", "that", "these", "those", "which", "who", "whom", "what",
    "when", "where", "why", "how", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "her", "his", "he", "she", "it", "they",
    "we", "you", "i", "me", "him", "us", "them", "my", "your", "its",
    "our", "their", "also", "any", "into", "there", "here", "now", "then",
    # termos muito genéricos em contexto médico
    "patient", "history", "following", "likely", "most", "shows", "present",
    "comes", "physician", "due", "associated", "cause", "caused", "which",
    "correct", "diagnosis", "treatment", "management", "finding", "findings",
    "physical", "examination", "result", "results", "old", "year", "man",
    "woman", "male", "female", "weeks", "days", "hours", "months", "years",
}


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [w for w in words if w not in STOPWORDS]

def load_graph_entities(graph_file: Path) -> set[str]:
    logger.info(f"Carregando grafo: {graph_file}")
    G = nx.read_graphml(str(graph_file))
    entities: set[str] = set()
    for node_id, data in G.nodes(data=True):
        name = data.get("entity_name") or node_id
        entities.add(name.lower())
        for tok in tokenize(name):
            entities.add(tok)
    logger.info(f"Entidades: {G.number_of_nodes()} nós → {len(entities)} tokens")
    return entities

def compute_coverage(question: dict, entity_tokens: set[str]) -> float:
    text = question["question"]
    for opt_text in question.get("options", {}).values():
        text += " " + opt_text
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    matched = sum(1 for t in tokens if t in entity_tokens)
    return matched / len(tokens)

def load_medqa(path: Path, limit: int | None = None) -> list[dict]:
    questions = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                q = json.loads(line.strip())
                if "question" in q and "options" in q and "answer_idx" in q:
                    questions.append(q)
            except json.JSONDecodeError:
                continue
    return questions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-coverage", type=float, default=0.10)
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--output", type=str, default=str(RESULTS_DIR / "benchmark_filtered.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    if not GRAPH_FILE.exists():
        logger.error(f"Grafo não encontrado: {GRAPH_FILE}")
        logger.error("Execute Step_1_run_all_books_parallel.py antes.")
        sys.exit(1)

    if not QUESTIONS_FILE.exists():
        logger.error(f"Arquivo de questões não encontrado: {QUESTIONS_FILE}")
        sys.exit(1)

    entity_tokens = load_graph_entities(GRAPH_FILE)
    questions = load_medqa(QUESTIONS_FILE)
    logger.info(f"Questões carregadas: {len(questions)}")

    scored = []
    for i, q in enumerate(questions):
        coverage = compute_coverage(q, entity_tokens)
        scored.append((coverage, i, q))

    scored.sort(key=lambda x: x[0], reverse=True)

    filtered = [
        {
            "question_id": i + 1,
            "original_index": orig_idx,
            "question": q["question"],
            "options": q["options"],
            "answer": q.get("answer", ""),
            "answer_idx": q["answer_idx"],
            "meta_info": q.get("meta_info", ""),
            "coverage": round(cov, 4),
        }
        for cov, orig_idx, q in scored
        if cov >= args.min_coverage
    ]

    if args.max_questions:
        filtered = filtered[: args.max_questions]

    all_scores = [s[0] for s in scored]
    logger.info(f"Total: {len(questions)} | Filtradas: {len(filtered)} | Cobertura média: {sum(all_scores)/len(all_scores):.3f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    logger.info(f"✓ Benchmark: {output_path} ({len(filtered)} questões)")


if __name__ == "__main__":
    main()


import os
import sys
import json
import time
import argparse
import asyncio
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import RateLimitError

from lightrag import LightRAG
from lightrag.base import DocStatus
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.utils import EmbeddingFunc, wrap_embedding_func_with_attrs

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEXTBOOKS_DIR = PROJECT_ROOT / "data" / "medqa" / "textbooks" / "en"
WORKING_DIR = PROJECT_ROOT / "working_dir"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
LOGS_DIR = PROJECT_ROOT / "experiments" / "logs"
PROGRESS_FILE = RESULTS_DIR / "indexing_progress.json"

WORKING_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS_DIR / f"indexing_{_timestamp}.log"

logger = logging.getLogger("step1_index_medqa")
logger.setLevel(logging.INFO)
_fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_fh)
logger.addHandler(_sh)

CHUNK_TOKEN_SIZE = 512
CHUNK_OVERLAP_TOKEN_SIZE = 100
ENTITY_EXTRACT_MAX_GLEANING = 1
CHUNKS_PER_SEGMENT = int(os.getenv("MEDQA_CHUNKS_PER_SEGMENT", "100"))

LLM_MODEL_MAX_ASYNC = int(os.getenv("MEDQA_LLM_MAX_ASYNC", "2"))
LLM_TIMEOUT = int(os.getenv("MEDQA_LLM_TIMEOUT", "600"))

class AdaptiveDelay:
    def __init__(self, initial=1.5, minimum=0.5, maximum=10.0, decay_step=0.1, decay_after=5):
        self.delay = initial
        self.minimum = minimum
        self.maximum = maximum
        self.decay_step = decay_step
        self.decay_after = decay_after
        self._consecutive_successes = 0

    def on_success(self):
        self._consecutive_successes += 1
        if self._consecutive_successes >= self.decay_after:
            self.delay = max(self.minimum, self.delay - self.decay_step)
            self._consecutive_successes = 0

    def on_rate_limit(self, suggested_wait: float | None = None):
        self._consecutive_successes = 0
        if suggested_wait is not None:
            self.delay = min(self.maximum, max(self.delay, suggested_wait))
        else:
            self.delay = min(self.maximum, self.delay * 2)


_adaptive_delay = AdaptiveDelay()


def _extract_suggested_wait(error_message: str) -> float | None:
    import re
    match = re.search(r"try again in ([\d.]+)s", error_message, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    await asyncio.sleep(_adaptive_delay.delay)
    try:
        result = await openai_complete_if_cache(
            "gpt-4o-mini",
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.openai.com/v1",
            **kwargs,
        )
        _adaptive_delay.on_success()
        return result
    except RateLimitError as e:
        suggested = _extract_suggested_wait(str(e))
        _adaptive_delay.on_rate_limit(suggested)
        logger.warning(
            f"RateLimitError - novo delay adaptativo: {_adaptive_delay.delay:.2f}s "
            f"(suggested_wait={suggested})"
        )
        raise

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

def split_book_into_segments(text: str, chunks_per_segment: int = CHUNKS_PER_SEGMENT) -> list[str]:
    approx_chars_per_token = 4  # heuristica grosseira, so para dimensionar segmentos
    target_chars = chunks_per_segment * CHUNK_TOKEN_SIZE * approx_chars_per_token

    paragraphs = text.split("\n\n")
    segments: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2  # +2 pelo separador "\n\n" reconstituido
        if current and current_len + para_len > target_chars:
            segments.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += para_len

    if current:
        segments.append("\n\n".join(current))

    return segments if segments else [text]

def discover_books() -> list[tuple[str, Path]]:
    books = sorted(TEXTBOOKS_DIR.glob("*.txt"))
    if not books:
        raise FileNotFoundError(f"Nenhum livro .txt encontrado em {TEXTBOOKS_DIR}")
    return [(p.stem, p) for p in books]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--book", type=str, default=None)
    return parser.parse_args()

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            data.setdefault("segments", {})
            return data
    return {"completed": [], "failed": [], "current_book": None, "segments": {}}


def save_progress(progress: dict) -> None:
    progress["last_update"] = datetime.now(timezone.utc).isoformat()
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)

async def initialize_rag() -> LightRAG:
    rag = LightRAG(
        working_dir=str(WORKING_DIR),
        llm_model_func=llm_model_func,
        embedding_func=sentence_transformer_embed,
        chunk_token_size=CHUNK_TOKEN_SIZE,
        chunk_overlap_token_size=CHUNK_OVERLAP_TOKEN_SIZE,
        entity_extract_max_gleaning=ENTITY_EXTRACT_MAX_GLEANING,
        llm_model_max_async=LLM_MODEL_MAX_ASYNC,
        default_llm_timeout=LLM_TIMEOUT,
    )
    await rag.initialize_storages()
    logger.info(
        f"LightRAG configurado com llm_model_max_async={LLM_MODEL_MAX_ASYNC}, "
        f"default_llm_timeout={LLM_TIMEOUT}s"
    )
    return rag


async def insert_segment(rag: LightRAG, segment_id: str, segment_text: str) -> bool:
    start = time.time()
    try:
        await rag.ainsert(segment_text, ids=segment_id)
    except Exception as e:
        logger.error(f"[{segment_id}] Excecao nao tratada em ainsert(): {e}")
        return False
    elapsed = time.time() - start

    status_doc = await rag.doc_status.get_by_id(segment_id)
    real_status = status_doc.get("status") if status_doc else None

    if real_status == DocStatus.PROCESSED.value:
        logger.info(
            f"[{segment_id}] Concluido em {elapsed:.1f}s "
            f"(delay atual: {_adaptive_delay.delay:.2f}s) - status real: processed"
        )
        return True

    error_msg = status_doc.get("error_msg") if status_doc else "doc_status ausente"
    logger.error(
        f"[{segment_id}] Falhou apos {elapsed:.1f}s - status real: {real_status} - "
        f"erro: {error_msg}"
    )
    return False


async def insert_book(rag: LightRAG, book_name: str, book_path: Path, progress: dict) -> bool:
    size_mb = book_path.stat().st_size / 1024 / 1024
    with open(book_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    segments = split_book_into_segments(text)
    total_segments = len(segments)
    logger.info(
        f"[{book_name}] {size_mb:.1f} MB dividido em {total_segments} segmento(s) "
        f"(~{CHUNKS_PER_SEGMENT} chunks cada)"
    )

    book_progress = progress["segments"].setdefault(
        book_name, {"total": total_segments, "completed": [], "failed": []}
    )
    book_progress["total"] = total_segments

    all_ok = True
    for idx, segment_text in enumerate(segments):
        segment_id = f"{book_name}__part{idx:04d}"

        if segment_id in book_progress["completed"]:
            logger.info(f"[{segment_id}] Ja concluido em execucao anterior, pulando")
            continue

        logger.info(f"[{book_name}] Inserindo segmento {idx + 1}/{total_segments} ({segment_id})")
        success = await insert_segment(rag, segment_id, segment_text)

        if success:
            if segment_id not in book_progress["completed"]:
                book_progress["completed"].append(segment_id)
            if segment_id in book_progress["failed"]:
                book_progress["failed"].remove(segment_id)
        else:
            all_ok = False
            if segment_id not in book_progress["failed"]:
                book_progress["failed"].append(segment_id)

        save_progress(progress)

    remaining = total_segments - len(book_progress["completed"])
    if remaining:
        logger.warning(
            f"[{book_name}] {remaining}/{total_segments} segmento(s) ainda pendente(s) "
            f"ou falho(s): {book_progress['failed']}"
        )

    return all_ok


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY nao configurada")
        sys.exit(1)

    args = parse_args()
    books = discover_books()
    logger.info(f"Livros encontrados: {len(books)}")

    if args.book:
        matching = [(name, path) for name, path in books if name == args.book]
        if not matching:
            logger.error(f"Livro '{args.book}' nao encontrado em {TEXTBOOKS_DIR}")
            sys.exit(2)
        books_to_run = matching
    else:
        books_to_run = books

    progress = load_progress()
    logger.info(f"Estado anterior: {len(progress['completed'])} completos, {len(progress['failed'])} falhados")

    rag = await initialize_rag()
    logger.info(f"LightRAG inicializado em {WORKING_DIR}")

    pending = [
        (name, path)
        for name, path in books_to_run
        if name not in progress["completed"]
    ]
    logger.info(f"Pendentes nesta execucao: {len(pending)} livros")

    any_failure = False
    execution_start = time.time()

    for book_idx, (book_name, book_path) in enumerate(pending, 1):
        progress["current_book"] = book_name
        save_progress(progress)

        logger.info(f"[{book_idx}/{len(pending)}] Iniciando: {book_name}")

        try:
            success = await insert_book(rag, book_name, book_path, progress)
        except Exception as e:
            logger.error(f"[{book_name}] Excecao nao tratada: {e}")
            logger.debug(traceback.format_exc())
            success = False

        if success:
            if book_name not in progress["completed"]:
                progress["completed"].append(book_name)
            if book_name in progress["failed"]:
                progress["failed"].remove(book_name)
            logger.info(f"✓ {book_name} concluido com SUCESSO")
        else:
            any_failure = True
            if book_name not in progress["failed"]:
                progress["failed"].append(book_name)
            logger.warning(f"✗ {book_name} falhou ou foi parcialmente processado")

        progress["current_book"] = None
        save_progress(progress)

    execution_elapsed = time.time() - execution_start

    logger.info(f"Total: {len(books)} | Completos: {len(progress['completed'])} | Falhados: {len(progress['failed'])}")
    logger.info(f"Tempo: {execution_elapsed/3600:.1f}h | Grafo: {WORKING_DIR}")

    if args.book and any_failure:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

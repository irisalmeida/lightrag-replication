import os
import re
import json
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc, always_get_an_event_loop
import numpy as np


## For Upstage API
# please check if embedding_dim=4096 in lightrag.py and llm.py in lightrag direcotry
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # return await openai_complete_if_cache(
    #     "solar-mini",
    #     prompt,
    #     system_prompt=system_prompt,
    #     history_messages=history_messages,
    #     api_key=os.getenv("UPSTAGE_API_KEY"),
    #     base_url="https://api.upstage.ai/v1/solar",
    #     **kwargs,
    # )
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.openai.com/v1",
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    # return await openai_embed(
    #     texts,
    #     model="solar-embedding-1-large-query",
    #     api_key=os.getenv("UPSTAGE_API_KEY"),
    #     base_url="https://api.upstage.ai/v1/solar",
    # )
    return await openai_embed.func(
        texts,
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.openai.com/v1",
        embedding_dim=1536,
    )


## /For Upstage API


def extract_queries(file_path):
    with open(file_path, "r") as f:
        data = f.read()

    data = data.replace("**", "")

    queries = re.findall(r"- Question \d+: (.+)", data)

    return queries


async def process_query(query_text, rag_instance, query_param):
    try:
        result = await rag_instance.aquery(query_text, param=query_param)
        return {"query": query_text, "result": result}, None
    except Exception as e:
        return None, {"query": query_text, "error": str(e)}


def run_queries_and_save_to_json(
    queries, rag_instance, query_param, output_file, error_file
):
    loop = always_get_an_event_loop()

    with (
        open(output_file, "a", encoding="utf-8") as result_file,
        open(error_file, "a", encoding="utf-8") as err_file,
    ):
        result_file.write("[\n")
        first_entry = True

        for query_text in queries:
            result, error = loop.run_until_complete(
                process_query(query_text, rag_instance, query_param)
            )

            if result:
                if not first_entry:
                    result_file.write(",\n")
                json.dump(result, result_file, ensure_ascii=False, indent=4)
                first_entry = False
            elif error:
                json.dump(error, err_file, ensure_ascii=False, indent=4)
                err_file.write("\n")

        result_file.write("\n]")


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(embedding_dim=1536, func=embedding_func),
    )
    await rag.initialize_storages()
    return rag


if __name__ == "__main__":
    # cls = "mix"
    # cls = "agriculture"  # original
    cls = os.environ.get("LIGHTRAG_CLS", "agriculture")  # controlado por run_all_domains.py
    mode = "hybrid"
    # WORKING_DIR = f"../{cls}"
    WORKING_DIR = f"./outputs/UltraDomain_2samples/{cls}"

    # rag = LightRAG(working_dir=WORKING_DIR)
    # rag = LightRAG(
    #     working_dir=WORKING_DIR,
    #     llm_model_func=llm_model_func,
    #     embedding_func=EmbeddingFunc(embedding_dim=1536, func=embedding_func),
    # )
    query_param = QueryParam(mode=mode)

    # base_dir = "../datasets/questions"
    base_dir = "./datasets/questions"
    queries = extract_queries(f"{base_dir}/{cls}_questions.txt")

    rag = asyncio.run(initialize_rag())
    # Antes: f"{base_dir}/result.json", f"{base_dir}/errors.json"  (hardcoded, sobrescrevia entre domínios)
    run_queries_and_save_to_json(
        queries, rag, query_param, f"{base_dir}/{cls}_result.json", f"{base_dir}/{cls}_errors.json"
    )

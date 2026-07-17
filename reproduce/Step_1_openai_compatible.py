import os
import json
import time
import asyncio
import numpy as np

from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import openai_complete_if_cache, openai_embed


## For Upstage API
# please check if embedding_dim=4096 in lightrag.py and llm.py in lightrag direcotry
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # Upstage solar-mini (original)
    # return await openai_complete_if_cache(
    #     "solar-mini",
    #     prompt,
    #     system_prompt=system_prompt,
    #     history_messages=history_messages,
    #     api_key=os.getenv("UPSTAGE_API_KEY"),
    #     base_url="https://api.upstage.ai/v1/solar",
    #     **kwargs,
    # )
    # gpt-4o-mini via OpenAI (padronizado para todos os domínios)
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.openai.com/v1",
        **kwargs,
    )


#VERSÃO ANTIGA
# async def embedding_func(texts: list[str]) -> np.ndarray:
#     return await openai_embed(
#         texts,
#         model="solar-embedding-1-large-query",
#         api_key=os.getenv("UPSTAGE_API_KEY"),
#         base_url="https://api.upstage.ai/v1/solar",
#         embedding_dim=4096 #mudança devido ao erro do tamanho dos embeddings
#     )




async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed.func(
        texts,
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://api.openai.com/v1",
        embedding_dim=1536,
    )






## /For Upstage API


def insert_text(rag, file_path):
    with open(file_path, mode="r") as f:
        unique_contexts = json.load(f)

    retries = 0
    max_retries = 3
    while retries < max_retries:
        try:
            rag.insert(unique_contexts)
            break
        except Exception as e:
            retries += 1
            print(f"Insertion failed, retrying ({retries}/{max_retries}), error: {e}")
            time.sleep(10)
    if retries == max_retries:
        print("Insertion failed after exceeding the maximum number of retries")


# cls = "mix"
# WORKING_DIR = f"../{cls}"

# if not os.path.exists(WORKING_DIR):
#     os.mkdir(WORKING_DIR)


# async def initialize_rag():
#     rag = LightRAG(
#         working_dir=WORKING_DIR,
#         llm_model_func=llm_model_func,
#         embedding_func=EmbeddingFunc(embedding_dim=4096, func=embedding_func),
#     )

#     await rag.initialize_storages()  # Auto-initializes pipeline_status
#     return rag


# def main():
#     # Initialize RAG instance
#     rag = asyncio.run(initialize_rag())
#     insert_text(rag, f"../datasets/unique_contexts/{cls}_unique_contexts.json")


# if __name__ == "__main__":
#     main()



#----------------------------------------------------------------------------------------------------------------------------------------------------------

#REPRODUÇÃO MINIMA INICIALMENTE:

cls = os.environ.get("LIGHTRAG_CLS", "agriculture")  # controlado por run_all_domains.py

WORKING_DIR = f"./outputs/UltraDomain_2samples/{cls}"

os.makedirs(WORKING_DIR, exist_ok=True)


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        #embedding_func=EmbeddingFunc(embedding_dim=4096, func=embedding_func),
        embedding_func=EmbeddingFunc(embedding_dim=1536, func=embedding_func),
    )

    await rag.initialize_storages()
    return rag


def main():
    rag = asyncio.run(initialize_rag())
    insert_text(
        rag,
        f"./datasets/UltraDomain_2samples_unique_contexts/{cls}_unique_contexts.json",
    )


if __name__ == "__main__":
    main()
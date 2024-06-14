"""
Hyde implementation based on OCI Cohere
"""

from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever

from oci_command_r_oo import OCICommandR
from factory_rfx import get_embed_model
from factory_vector_store import get_vector_store

from preamble_libraries import preamble_dict

from config import (
    EMBED_MODEL_TYPE,
    VECTOR_STORE_TYPE,
    COHERE_RERANKER_MODEL,
    TOP_K,
    TOP_N,
)
from config_private import COMPARTMENT_ID, COHERE_API_KEY


def get_task_step1(query):
    """
    to complete
    """

    task = f"""
    Given a question, write a documentation passage to answer the question
    Question: {query}
    Passage:
    """

    return task


def get_llm():
    """
    to complete
    """
    chat = OCICommandR(
        model="cohere.command-r-plus",
        service_endpoint="https://ppe.inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id=COMPARTMENT_ID,
        max_tokens=1024,
        is_streaming=False,
    )
    return chat


def get_retriever(add_reranker=False):
    # this doesn't change
    embed_model = get_embed_model(EMBED_MODEL_TYPE)

    v_store = get_vector_store(
        vector_store_type=VECTOR_STORE_TYPE,
        embed_model=embed_model,
        local_index_dir=None,
        books_dir=None,
    )

    base_retriever = v_store.as_retriever(k=TOP_K)

    if add_reranker:
        compressor = CohereRerank(
            cohere_api_key=COHERE_API_KEY, top_n=TOP_N, model=COHERE_RERANKER_MODEL
        )

        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )
    else:
        retriever = base_retriever

    return retriever


def hyde_step1_2(query, add_reranker=False, lang="en"):
    """
    to complete
    """

    # this doesn't change
    retriever = get_retriever(add_reranker)

    chat = get_llm()

    # step1
    task = get_task_step1(query)

    # print("Step 1...")

    # resetting preamble
    chat.preamble_override = None

    response1 = chat.invoke(query=task, chat_history=[], documents=[])

    # this is the hypotethical doc produced by step1
    hyde_doc = response1.data.chat_response.text

    # step 2
    # do the semantic search
    docs = retriever.invoke(hyde_doc)

    documents_txt = [
        {
            "id": str(i + 1),
            "snippet": doc.page_content,
            "source": doc.metadata["source"],
            "page": str(doc.metadata["page"]),
        }
        for i, doc in enumerate(docs)
    ]

    # print("Step 2...")
    # choose the preamble based also on language
    chat.preamble_override = preamble_dict[f"preamble_{lang}"]

    response2 = chat.invoke(query=query, chat_history=[], documents=documents_txt)

    return response2.data.chat_response.text

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
    MAX_TOKENS,
    TOP_K,
    TOP_N,
)
from config_private import COMPARTMENT_ID, COHERE_API_KEY


def format_docs_for_cohere(l_docs):
    """ "
    format documents in the format expected by Cohere command-r/plus
    l_docs: list of Document
    """

    # Cohere wants a map
    documents_txt = [
        {
            "id": str(i + 1),
            "snippet": doc.page_content,
            "source": doc.metadata["source"],
            "page": str(doc.metadata["page"]),
        }
        for i, doc in enumerate(l_docs)
    ]

    return documents_txt


def get_task_step1(query):
    """
    Create the query for an Hyde doc
    """

    task = f"""
    Given a question, write a documentation passage to answer the question
    Question: {query}
    Passage:
    """

    return task


def get_text_from_response(response):
    """
    extract text from OCI response
    """
    return response.data.chat_response.text


def get_llm():
    """
    to complete
    """
    chat = OCICommandR(
        model="cohere.command-r-plus",
        service_endpoint="https://ppe.inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        compartment_id=COMPARTMENT_ID,
        max_tokens=MAX_TOKENS,
        is_streaming=False,
    )
    return chat


#
# This has been modified to support selection over
# multiple collections
# for now only O23AI
#
def get_retriever(add_reranker=False, selected_collection="ORACLE_KNOWLEDGE"):
    """
    selected_collection: the name of the Oracle table in OracleVS
    """
    embed_model = get_embed_model(EMBED_MODEL_TYPE)

    v_store = get_vector_store(
        vector_store_type=VECTOR_STORE_TYPE,
        embed_model=embed_model,
        selected_collection=selected_collection,
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


def hyde_step1_2(
    query, add_reranker=False, lang="en", selected_collection="ORACLE_KNOWLEDGE"
):
    """
    This method supports the implementation of hyde
    see: https://arxiv.org/abs/2212.10496
    """

    # this doesn't change
    retriever = get_retriever(add_reranker, selected_collection)

    chat = get_llm()

    # step1: ask to the llm to answer to the query
    # creating an hypothetical document
    task = get_task_step1(query)

    # resetting preamble
    chat.preamble_override = None

    # get the hyde doc
    response1 = chat.invoke(query=task, chat_history=[], documents=[])

    # this is the hypotethical doc produced by step1
    hyde_doc = response1.data.chat_response.text

    # step 2
    # do the semantic search searching for docs similar to hyde_doc
    docs = retriever.invoke(hyde_doc)

    documents_txt = format_docs_for_cohere(docs)

    # print("Step 2...")
    # choose the preamble based also on language
    chat.preamble_override = preamble_dict[f"preamble_{lang}"]

    response2 = chat.invoke(query=query, chat_history=[], documents=documents_txt)

    return get_text_from_response(response2)


def classic_rag(
    query, add_reranker=False, lang="en", selected_collection="ORACLE_KNOWLEDGE"
):
    """
    Do the classic rag
    """
    # this doesn't change
    retriever = get_retriever(add_reranker, selected_collection)

    chat = get_llm()

    docs = retriever.invoke(query)

    documents_txt = format_docs_for_cohere(docs)

    chat.preamble_override = preamble_dict[f"preamble_{lang}"]

    response = chat.invoke(query=query, chat_history=[], documents=documents_txt)

    return get_text_from_response(response)

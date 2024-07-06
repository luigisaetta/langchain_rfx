"""
Factory methods implementation based on OCI Cohere
* supports classi RAG, HyDE
"""

from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI

from oci_cohere_embeddings_utils import OCIGenAIEmbeddingsWithBatch
from oci_command_r_oo import OCICommandR
from factory_vector_store import get_vector_store
from oci_citations_utils import extract_complete_citations, extract_document_list
from utils import get_console_logger, check_value_in_list

from preamble_libraries import preamble_dict

from config import (
    EMBED_MODEL_TYPE,
    OCI_EMBED_MODEL,
    VECTOR_STORE_TYPE,
    ENDPOINT,
    TEMPERATURE,
    COHERE_RERANKER_MODEL,
    MAX_TOKENS,
    TOP_K,
    TOP_N,
)
from config_private import COMPARTMENT_ID, COHERE_API_KEY

logger = get_console_logger()


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


def get_embed_model(model_type="OCI"):
    """
    get the Embeddings Model
    """
    check_value_in_list(model_type, ["OCI"])

    embed_model = None

    if model_type == "OCI":
        embed_model = OCIGenAIEmbeddingsWithBatch(
            auth_type="API_KEY",
            model_id=OCI_EMBED_MODEL,
            service_endpoint=ENDPOINT,
            compartment_id=COMPARTMENT_ID,
        )

    return embed_model


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


def get_text_from_response(response, llm_model):
    """
    extract text from OCI response
    """
    if llm_model.startswith("cohere"):
        return response.data.chat_response.text

    # here it is Llama3
    return response.content


def get_citations_from_response(response):
    """
    Extract from the response the citations
    only to be used with Cohere
    """
    citations = extract_complete_citations(response)

    return citations


def get_documents_from_response(response):
    """
    get the documents for citations
    """
    return extract_document_list(response)


def get_llm(llm_model, temperature=TEMPERATURE):
    """
    return the llm model

    for now supports command-r and command-r-plus
    """
    chat = None

    if llm_model.startswith("cohere"):
        # Command R/R-plus
        chat = OCICommandR(
            model=llm_model,
            service_endpoint=ENDPOINT,
            compartment_id=COMPARTMENT_ID,
            max_tokens=MAX_TOKENS,
            temperature=temperature,
            is_streaming=False,
        )
    elif llm_model.startswith("meta"):
        # Llama3
        chat = ChatOCIGenAI(
            model_id=llm_model,
            service_endpoint=ENDPOINT,
            compartment_id=COMPARTMENT_ID,
            is_stream=False,
            model_kwargs={"temperature": temperature, "max_tokens": MAX_TOKENS},
        )
    return chat


#
# This has been modified to support selection over
# multiple collections
#
def get_retriever(add_reranker=False, selected_collection="ORACLE_KNOWLEDGE"):
    """
    selected_collection: the name of the Oracle table in OracleVS
    or index in OpenSearch
    """
    embed_model = get_embed_model(EMBED_MODEL_TYPE)

    v_store = get_vector_store(
        vector_store_type=VECTOR_STORE_TYPE,
        embed_model=embed_model,
        selected_collection=selected_collection,
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


def hyde_rag(
    query,
    llm_model,
    add_reranker=False,
    lang="en",
    selected_collection="ORACLE_KNOWLEDGE",
    temperature=TEMPERATURE,
):
    """
    This method supports the implementation of HyDE
    see: https://arxiv.org/abs/2212.10496
    """

    # this doesn't change
    retriever = get_retriever(add_reranker, selected_collection)

    chat = get_llm(llm_model, temperature)

    # Hyde step1: ask to the llm to answer to the query
    # creating an hypothetical document

    # formulate the task
    task = get_task_step1(query)

    if llm_model.startswith("cohere"):
        # resetting preamble
        chat.preamble_override = None

        # get the hyde doc
        response1 = chat.invoke(query=task, chat_history=[], documents=[])

        # this is the hypotethical doc produced by step1
        hyde_doc = get_text_from_response(response1, llm_model)

        # step 2
        # do the semantic search searching for docs similar to hyde_doc
        docs = retriever.invoke(hyde_doc)

        documents_txt = format_docs_for_cohere(docs)

        # print("Step 2...")
        # choose the preamble based on target language
        chat.preamble_override = preamble_dict[f"preamble_{lang}"]

        response2 = chat.invoke(query=query, chat_history=[], documents=documents_txt)
    else:
        # meta
        response1 = chat.invoke([HumanMessage(task)])

        hyde_doc = get_text_from_response(response1, llm_model)

        docs = retriever.invoke(hyde_doc)

        context = (
            "Use the following context: \n"
            + "\n".join(doc.page_content for doc in docs)
            + "\n"
        )

        messages = [
            SystemMessage(content=preamble_dict[f"preamble_{lang}"]),
            SystemMessage(content=context),
            HumanMessage(content=query),
        ]

        response2 = chat.invoke(messages)

    return response2


def classic_rag(
    query,
    llm_model,
    add_reranker=False,
    lang="en",
    selected_collection="ORACLE_KNOWLEDGE",
    temperature=TEMPERATURE,
):
    """
    Do the classic rag
    """

    # this doesn't change
    retriever = get_retriever(add_reranker, selected_collection)

    docs = retriever.invoke(query)

    chat = get_llm(llm_model, temperature)

    if llm_model.startswith("cohere"):
        # using Cohere native interface for citations

        documents_txt = format_docs_for_cohere(docs)

        chat.preamble_override = preamble_dict[f"preamble_{lang}"]

        response = chat.invoke(query=query, chat_history=[], documents=documents_txt)

    else:
        # meta
        context = (
            "Use the following context: \n"
            + "\n".join(doc.page_content for doc in docs)
            + "\n"
        )

        messages = [
            SystemMessage(content=preamble_dict[f"preamble_{lang}"]),
            SystemMessage(content=context),
            HumanMessage(content=query),
        ]

        response = chat.invoke(messages)

    return response

"""
Hyde implementation based on OCI Cohere
"""

from oci_command_r_oo import OCICommandR
from factory_rfx import get_embed_model
from factory_vector_store import get_vector_store

from config import EMBED_MODEL_TYPE, VECTOR_STORE_TYPE, TOP_K
from config_private import COMPARTMENT_ID


def get_task_step1(query):
    """
    to complete
    """

    task = f"""
    Please write a documentation passage to answer the question
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


def get_retriever():
    # this doesn't change
    embed_model = get_embed_model(EMBED_MODEL_TYPE)

    v_store = get_vector_store(
        vector_store_type=VECTOR_STORE_TYPE,
        embed_model=embed_model,
        local_index_dir=None,
        books_dir=None,
    )

    base_retriever = v_store.as_retriever(k=TOP_K)

    return base_retriever


def hyde_step1_2(query):
    """
    to complete
    """

    # this doesn't change
    retriever = get_retriever()

    preamble01 = """
    
    ## Task & Context
    You are an assistant responsible for answering questions 
    using the provided documents. 
    Respond with detailed information and compose 
    a comprehensive and thorough one-page document.
    
    """

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
    chat.preamble_override = preamble01

    response2 = chat.invoke(query=query, chat_history=[], documents=documents_txt)

    return response2.data.chat_response.text

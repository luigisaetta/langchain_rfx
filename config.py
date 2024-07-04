"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-04-27
Python Version: 3.11
"""

# title for the UI
TITLE = "AI Assistant with LangChain 🦜"
HELLO_MSG = "Ciao, come posso aiutarti?"

ADD_REFERENCES = True
VERBOSE = True

LANG_SUPPORTED = ["en", "it", "es", "fr", "de", "el", "nl", "ro"]

# enable tracing with LangSmith
ENABLE_TRACING = False

# for chunking
# in chars
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

# per ora usiamo il tokenizer di Cohere...
TOKENIZER = "Cohere/Cohere-embed-multilingual-v3.0"

# OCI GenAI model used for Embeddings
# to batch embedding with OCI
# with Cohere embeddings max is 96
# value: COHERE, OCI
EMBED_MODEL_TYPE = "OCI"
EMBED_BATCH_SIZE = 90
OCI_EMBED_MODEL = "cohere.embed-multilingual-v3.0"
COHERE_EMBED_MODEL = "embed-multilingual-v3.0"

# current endpoint for OCI GenAI (embed and llm) models
# switched to FRA (19/06)
ENDPOINT = "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com"

# reranker, True only to experiment
ADD_RERANKER = True
COHERE_RERANKER_MODEL = "rerank-multilingual-v3.0"

# only for rfx, instead of reranker
ADD_LLMLINGUA = False

# Alternative to above
ADD_LLM_CHAIN_EXTRACTOR = False

# retriever
TOP_K = 8
TOP_N = 4

# to limit chat_history
# probably in rfp can be kept low
MAX_MSGS_IN_CHAT = 2

# Oracle VS
EMBEDDINGS_BITS = 32

# Vector Store
# VECTOR_STORE_TYPE = "OPENSEARCH"
VECTOR_STORE_TYPE = "23AI"
# VECTOR_STORE_TYPE = "QDRANT"

# OPENSEARCH
# using local as docker
OPENSEARCH_URL = "https://localhost:9200"
OPENSEARCH_INDEX_NAME = "med01"

# QDRANT local
QDRANT_URL = "http://localhost:6333"

# 23AI
# the name of the table with text and embeddings
COLLECTION_NAME = "MY_BOOKS"

# COHERE, OCI
LLM_MODEL_TYPE = "OCI"

# OCI
# OCI_GENAI_MODEL = "cohere.command"
# OCI_GENAI_MODEL = "meta.llama-3-70b-instruct"
# OCI_GENAI_MODEL = "cohere.command-r-16k"
OCI_GENAI_MODEL = "cohere.command-r-plus"

# params for LLM
TEMPERATURE = 0.1
MAX_TOKENS = 2048

# to enable streaming
DO_STREAMING = False

# for TRACING
LANGCHAIN_PROJECT = "rfx-05"

# Opensearch shared params
OPENSEARCH_SHARED_PARAMS = {
    "opensearch_url": OPENSEARCH_URL,
    "use_ssl": True,
    "verify_certs": False,
    "ssl_assert_hostname": False,
    "ssl_show_warn": False,
    "bulk_size": 5000,
    "index_name": OPENSEARCH_INDEX_NAME,
    "engine": "faiss",
}

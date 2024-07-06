"""
Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-04-30
Python Version: 3.11

Usage: contains the functions to split in chunks and create the index
"""

from glob import glob
from tqdm.auto import tqdm
import oracledb

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import get_console_logger, remove_path_from_ref
from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    OPENSEARCH_URL,
    OPENSEARCH_INDEX_NAME,
)

from config_private import (
    OPENSEARCH_USER,
    OPENSEARCH_PWD,
    DB_USER,
    DB_PWD,
    DB_HOST_IP,
    DB_SERVICE,
)


def get_recursive_text_splitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    return a recursive text splitter
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter


def load_book_and_split(book_path, chunk_size, chunk_overlap):
    """
    load a single book
    """
    logger = get_console_logger()

    text_splitter = get_recursive_text_splitter(chunk_size, chunk_overlap)

    loader = PyPDFLoader(file_path=book_path)

    docs = loader.load_and_split(text_splitter=text_splitter)

    # remove path from source
    for doc in docs:
        doc.metadata["source"] = remove_path_from_ref(doc.metadata["source"])

    logger.info("Loaded %s chunks...", len(docs))

    return docs


def create_collection_and_add_docs_to_23ai(docs, embed_model, collection_name):
    """
    create the collection and load docs in that collection
    To be used only for a NEW collection
    """
    logger = get_console_logger()

    try:
        dsn = f"{DB_HOST_IP}:1521/{DB_SERVICE}"

        connection = oracledb.connect(user=DB_USER, password=DB_PWD, dsn=dsn)

        OracleVS.from_documents(
            docs,
            embed_model,
            client=connection,
            table_name=collection_name,
            distance_strategy=DistanceStrategy.COSINE,
        )

        logger.info("Created collection and documents saved...")

    except oracledb.Error as e:
        err_msg = "An error occurred in create_collection_and_add_docs: " + str(e)
        logger.error(err_msg)


def add_docs_to_23ai(docs, embed_model, collection_name):
    """
    add docs from a book to Oracle vector store
    This is used for an existing collection
    """
    logger = get_console_logger()

    try:
        dsn = f"{DB_HOST_IP}:1521/{DB_SERVICE}"

        connection = oracledb.connect(user=DB_USER, password=DB_PWD, dsn=dsn)

        v_store = OracleVS(
            client=connection,
            table_name=collection_name,
            distance_strategy=DistanceStrategy.COSINE,
            embedding_function=embed_model,
        )

        logger.info("Saving new documents to Vector Store...")

        v_store.add_documents(docs)

        logger.info("Saved new documents to Vector Store !")

    except oracledb.Error as e:
        err_msg = "An error occurred in add_docs_to_23ai: " + str(e)
        logger.error(err_msg)


def add_docs_to_opensearch(docs, embed_model):
    """
    add docs from a book to opensearch vector store
    """
    logger = get_console_logger()

    v_store = OpenSearchVectorSearch(
        embedding_function=embed_model,
        opensearch_url=OPENSEARCH_URL,
        http_auth=(OPENSEARCH_USER, OPENSEARCH_PWD),
        use_ssl=True,
        verify_certs=False,
        ssl_assert_hostname=False,
        ssl_show_warn=False,
        bulk_size=5000,
        index_name=OPENSEARCH_INDEX_NAME,
        engine="faiss",
    )

    logger.info("Saving new documents to Vector Store...")

    v_store.add_documents(docs)

    logger.info("Saved new documents to Vector Store !")


def load_books_and_split(books_dir) -> list:
    """
    load a set of books from books_dir and split in chunks
    """
    logger = get_console_logger()

    logger.info("Loading documents from %s...", books_dir)

    text_splitter = get_recursive_text_splitter()

    books_list = sorted(glob(books_dir + "/*.pdf"))

    logger.info("Loading books: ")
    for book in books_list:
        logger.info("* %s", book)

    docs = []

    for book in tqdm(books_list):
        loader = PyPDFLoader(file_path=book)

        docs += loader.load_and_split(text_splitter=text_splitter)

    logger.info("Loaded %s chunks of text...", len(docs))

    return docs

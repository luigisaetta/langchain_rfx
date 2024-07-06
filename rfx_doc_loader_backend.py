"""
RFX docs loader backend function

to separate UI logic from backend logic
"""

import os
import tempfile
import oracledb

from oraclevs_4_rfx import OracleVS4RFX
from translations import translations
from factory_rfx import get_embed_model
from utils import get_console_logger

from chunk_index_utils import (
    load_book_and_split,
    add_docs_to_23ai,
    create_collection_and_add_docs_to_23ai,
)

from config_private import DB_USER, DB_PWD, DB_HOST_IP, DB_SERVICE

logger = get_console_logger()


def get_db_connection():
    """
    get a connection to db
    """
    dsn = f"{DB_HOST_IP}/{DB_SERVICE}"

    conn = oracledb.connect(user=DB_USER, password=DB_PWD, dsn=dsn, retry_count=3)

    return conn


# to handle multilingual use the dictionary in translations.py
def translate(text, v_lang):
    """
    to handle labels in different lang
    """
    return translations.get(v_lang, {}).get(text, text)


def get_list_collections():
    """
    return the list of available collections in the DB
    """
    conn = get_db_connection()

    list_collections = OracleVS4RFX.list_collections(conn)

    return list_collections


def get_books(collection_name):
    """
    return the list of books in collection
    """
    conn = get_db_connection()

    list_books_in_collection = OracleVS4RFX.list_books_in_collection(
        connection=conn, collection_name=collection_name
    )

    return list_books_in_collection


def write_temporary_file(v_tmp_dir_name, v_uploaded_file):
    """
    Write the uploaded file as a temporary file
    """
    temp_file_path = os.path.join(v_tmp_dir_name, v_uploaded_file.name)

    with open(temp_file_path, "wb") as f:
        f.write(v_uploaded_file.getbuffer())

    return temp_file_path


def load_uploaded_file_in_vector_store(
    v_uploaded_file, collection_name, chunk_size, chunk_overlap
):
    """
    load the uploaded file in the Vector Store and index

    this handles also the check to see if the file alredy exists
    """
    embed_model = get_embed_model()

    result_status = ""

    # write a temporary file with the content
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        temp_file_path = write_temporary_file(tmp_dir_name, v_uploaded_file)

        # split in docs and prepare for loading
        docs = load_book_and_split(temp_file_path, chunk_size, chunk_overlap)

    # check if collection exists
    if collection_name in get_list_collections():
        # existing collection

        # check that the book has not already been loaded
        if v_uploaded_file.name not in get_books(collection_name):
            # add books to existing
            logger.info(
                "Add book %s to an existing collection...", v_uploaded_file.name
            )

            add_docs_to_23ai(docs, embed_model, collection_name)

            result_status = "OK"
        else:
            logger.info("Book %s already in collection...", v_uploaded_file.name)

            result_status = "KO"
    else:
        # new collection
        # this way it is safe that the collection doesn't exists
        logger.info("Creating the collection and adding documents...")
        logger.info("Add book %s to new collection...", v_uploaded_file.name)

        create_collection_and_add_docs_to_23ai(docs, embed_model, collection_name)

        result_status = "OK"

    return result_status


def delete_documents_in_collection(collection_name, doc_names):
    """
    drop documents in the given collection
    """
    if len(doc_names) > 0:
        conn = get_db_connection()

        logger.info("Delete docs: %s in collection %s", doc_names, collection_name)
        OracleVS4RFX.delete_documents(conn, collection_name, doc_names)

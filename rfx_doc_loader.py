""""
Document Loader for RAG

for now it manages only 23AI as Vector Store
"""

import os
import tempfile
import pandas as pd
import oracledb
import streamlit as st

from utils import get_console_logger
from oraclevs_4_rfx import OracleVS4RFX
from translations import translations
from factory_rfx import get_embed_model

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

    conn = oracledb.connect(user=DB_USER, password=DB_PWD, dsn=dsn)

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

    list_books = OracleVS4RFX.list_books_in_collection(
        connection=conn, collection_name=collection_name
    )

    return list_books


def write_temporary_file(v_tmp_dir_name, v_uploaded_file):
    """
    Write the uploaded file as a temporary file
    """
    temp_file_path = os.path.join(v_tmp_dir_name, v_uploaded_file.name)

    with open(temp_file_path, "wb") as f:
        f.write(v_uploaded_file.getbuffer())

    return temp_file_path


def load_uploaded_file_in_vector_store(v_uploaded_file, collection_name):
    """
    load the uploaded file in the Vector Store and index
    """
    # write a temporary file with the content
    with tempfile.TemporaryDirectory() as tmp_dir_name:
        temp_file_path = write_temporary_file(tmp_dir_name, v_uploaded_file)

        # prepare for loading
        docs = load_book_and_split(temp_file_path)

    embed_model = get_embed_model()

    result_status = ""

    # check if collection exists
    if collection_name in get_list_collections():
        # existing collection
        logger.info("Add a book to an existing collection...")

        # check that the book has not already been loaded
        if v_uploaded_file.name not in get_books(collection_name):
            # add books to existing
            add_docs_to_23ai(docs, embed_model, collection_name)

            result_status = "OK"
        else:
            logger.info("Book already in collection...")
            st.info("Book already in collection!")

            result_status = "KO"
    else:
        # new collection
        # this way it is safe that the collection doesn't exists
        logger.info("Creating the collection and adding documents...")

        create_collection_and_add_docs_to_23ai(docs, embed_model, collection_name)

        result_status = "OK"

    return result_status


#
# Main
#
# st.set_page_config(layout="wide")
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


st.title("Oracle 23AI Document Loader")

col1, col2 = st.columns(2)

is_debug = st.sidebar.checkbox("Debug")

lang = st.sidebar.selectbox("Select Language", ["en", "es", "fr", "it"])

# Init list of collections
oraclecs_collections_list = get_list_collections()

# add NEW to enable to create a new one
shown_collections_list = oraclecs_collections_list + ["NEW"]

selected_collection = st.sidebar.selectbox(
    "Select documents collections", shown_collections_list
)

if selected_collection == "NEW":
    selected_collection = st.sidebar.text_input("Insert the name of the new collection")

# Caricamento del file
uploaded_file = st.sidebar.file_uploader(
    translate("Choose a pdf file", lang), type=["pdf"]
)

if uploaded_file is not None:
    logger.info("Loading file: %s", uploaded_file.name)

    with st.spinner("Loading in progress.."):
        status = load_uploaded_file_in_vector_store(
            uploaded_file, selected_collection
        )

    if status == "OK":
        # add to the list
        st.session_state.uploaded_files.append(uploaded_file.name)

    df_dict = {"Document name": st.session_state.uploaded_files}
    df = pd.DataFrame(df_dict)

    with col1:
        st.dataframe(df, hide_index=True)

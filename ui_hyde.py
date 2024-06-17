"""
UI for HYDE test
"""

import oracledb
import streamlit as st
import pandas as pd

from factory_hyde import hyde_step1_2, classic_rag
from translations import translations
from utils import get_console_logger
from oraclevs_4_rfx import OracleVS4RFX

from config_private import DB_USER, DB_PWD, DB_HOST_IP, DB_SERVICE

# the name of the column with all the questions
QUESTION_COL_NAME = "Question"
LIMITS = 6


# to handle multilingual use the dictionary in translations.py
def translate(text, v_lang):
    """
    to handle labels in different lang
    """
    return translations.get(v_lang, {}).get(text, text)


def get_list_collections():
    """
    return the list of available collections
    """
    DSN = f"{DB_HOST_IP}/{DB_SERVICE}"

    conn = oracledb.connect(user=DB_USER, password=DB_PWD, dsn=DSN)

    list_collections = OracleVS4RFX.list_vs_collections(conn)

    return list_collections


def get_books(collection_name):
    """
    return the list of books in collection
    """
    DSN = f"{DB_HOST_IP}/{DB_SERVICE}"

    conn = oracledb.connect(user=DB_USER, password=DB_PWD, dsn=DSN)

    list_books = OracleVS4RFX.list_books_in_collection(
        connection=conn, collection_name=collection_name
    )

    return list_books


def show_books(selected_collection):
    """
    show in the log the list of books in the collection
    """
    list_books = get_books(selected_collection)
    logger.info("List of books:")
    for book in list_books:
        logger.info("%s", book)
    logger.info("")


# Funzione per processare il file XLS
def process_file(file):
    """
    read a file and return a df
    """

    # Legge il file XLS in un DataFrame di Pandas
    input_df = pd.read_excel(file)
    # Esempio di elaborazione: mostra le prime 5 righe del DataFrame

    input_df = input_df[:LIMITS]

    return input_df


logger = get_console_logger()


# Titolo dell'applicazione
st.set_page_config(layout="wide")

is_debug = st.sidebar.checkbox("Debug")

lang = st.sidebar.selectbox("Select Language", ["en", "es", "fr", "it"])

add_reranker = st.sidebar.checkbox("Add reranker")
enable_hyde = st.sidebar.checkbox("Enable Hyde")

# Init list of collections
oraclecs_collections_list = get_list_collections()
selected_collection = st.sidebar.selectbox(
    "Select documents collections", oraclecs_collections_list
)


st.title("RFx AI Assistant")

col1, col2 = st.columns(2)

# to handle progress status and green tick
if "processed_questions" not in st.session_state:
    st.session_state.processed_questions = set()

# Caricamento del file
uploaded_file = st.sidebar.file_uploader(
    translate("Choose an xls file", lang), type=["xls", "xlsx"]
)

if uploaded_file is not None:

    if add_reranker:
        logger.info("Added reranker...")
        logger.info("")

    # Carica il file e mostra il contenuto
    df = process_file(uploaded_file)

    # init the dataframe with no green tick
    df["Processed"] = df.index.map(
        lambda i: "✅" if i in st.session_state.processed_questions else ""
    )

    questions = list(df[QUESTION_COL_NAME].values)

    # Show results
    # to update the dataframe in place

    with col1:
        st.header("Questions:")

        dataframe_placeholder = st.empty()

        dataframe_placeholder.dataframe(df)

        progress_bar = st.progress(0)

        info_placeholder = st.empty()
        info_placeholder.info(translate("Processing started!", lang))

    with col2:
        st.header("Results:")

    answers = []

    # show the collection chosen
    logger.info("Collection chosen: %s", selected_collection)
    logger.info("")

    if is_debug:
        show_books(selected_collection)

    for i, question in enumerate(questions):

        logger.info("Processing: %s ...", question)

        # call the GenAI
        if enable_hyde:
            if is_debug:
                logger.info("Enabled hyde...")

            answer = hyde_step1_2(
                question,
                add_reranker=add_reranker,
                lang=lang,
                selected_collection=selected_collection,
            )
        else:
            answer = classic_rag(
                question,
                add_reranker=add_reranker,
                lang=lang,
                selected_collection=selected_collection,
            )

        answers.append(answer)

        if is_debug:
            logger.info("Answer: %s", answer)
            logger.info("")

        # register it has been processed
        st.session_state.processed_questions.add(i)
        df["Processed"] = df.index.map(
            lambda i: "✅" if i in st.session_state.processed_questions else ""
        )

        with col1:
            dataframe_placeholder.dataframe(df)

            # update the progress bar
            progress_bar.progress(int(i + 1) / len(questions))

    with col1:
        info_placeholder.success(translate("Processing completed!", lang))

    # handle output
    dict_out = {"Answers": answers}

    df_out = pd.DataFrame(dict_out)

    with col2:
        st.dataframe(df_out)

else:
    st.write(translate("Load the xls file with the questions.", lang))

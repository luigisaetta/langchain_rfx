"""
UI for HYDE test
"""

import oracledb
import streamlit as st
import pandas as pd

from factory_hyde import (
    hyde_rag,
    classic_rag,
    get_text_from_response,
    get_citations_from_response,
    get_documents_from_response,
)
from translations import translations
from utils import get_console_logger, remove_path_from_ref
from oraclevs_4_rfx import OracleVS4RFX
from opensearch_4_rfx import OpenSearchRFX

from config import VECTOR_STORE_TYPE, LANG_SUPPORTED
from config_private import DB_USER, DB_PWD, DB_HOST_IP, DB_SERVICE

# the name of the column with all the questions
QUESTION_COL_NAME = "Question"

# TODO: remove in longer tests
LIMITS = 6


def reset_ui():
    """
    reset the UI
    """
    st.session_state.processed_questions = set()


# to handle multilingual use the dictionary in translations.py
def translate(text, v_lang):
    """
    to handle labels in different lang
    """
    return translations.get(v_lang, {}).get(text, text)


def get_model_list():
    """
    return list of available llm
    """

    # aligned with official names
    return [
        "cohere.command-r-plus",
        "cohere.command-r-16k",
        "meta.llama-3-70b-instruct",
    ]


def get_db_connection():
    """
    get a connection to db
    """
    dsn = f"{DB_HOST_IP}/{DB_SERVICE}"

    conn = oracledb.connect(user=DB_USER, password=DB_PWD, dsn=dsn)

    return conn


def get_list_collections(vector_store_type="23AI"):
    """
    return the list of available collections
    """
    if vector_store_type == "23AI":
        conn = get_db_connection()

        list_collections = OracleVS4RFX.list_collections(conn)
    else:
        # OpenSearch
        list_collections = OpenSearchRFX.list_collections()

    return list_collections


def get_books(collection_name, vector_store_type="23AI"):
    """
    return the list of books in collection
    """
    if vector_store_type == "23AI":
        conn = get_db_connection()

        list_books = OracleVS4RFX.list_books_in_collection(
            connection=conn, collection_name=collection_name
        )
    else:
        # OpenSearch
        list_books = OpenSearchRFX.list_books(collection_name)

    return list_books


def show_books(the_collection):
    """
    show in the log the list of books in the collection
    """
    list_books = get_books(the_collection, VECTOR_STORE_TYPE)
    logger.info("List of books:")
    for book in list_books:
        logger.info("%s", remove_path_from_ref(book))

    logger.info("")


def highlight_substrings(text, delimiters, doc_ids_list):
    """
    For citations
    Evidenzia le sottostringhe in un testo e aggiunge le liste di doc_id tra parentesi quadre.

    Args:
    - text (str): il testo originale.
    - delimiters (list of tuples): ogni tupla contiene (start, end).
    - doc_ids_list (list of lists): lista delle liste di doc_id associati ai delimitatori.

    Returns:
    - str: il testo con le sottostringhe evidenziate e le liste di doc_id aggiunte.
    """
    combined = list(zip(delimiters, doc_ids_list))
    combined.sort(key=lambda x: x[0][0], reverse=True)

    for (start, end), doc_ids in combined:
        original_substring = text[start:end]
        highlighted_substring = f"{original_substring}"
        doc_id_str = f' [{", ".join(doc_ids)}]'
        text = text[:start] + highlighted_substring + doc_id_str + text[end:]

    return text


def add_citations_to_answer(orig_answer, v_response):
    """
    for Cohere models add citations to the llm answer

    v_response: the complete response from the llm (including citations)
    """
    citations = get_citations_from_response(v_response)

    span_demarks = []
    doc_ids = []

    for citation in citations:
        span_demarks.append(citation["interval"])

        docs_for_this_citation = []
        for doc in citation["documents"]:
            docs_for_this_citation.append(doc["id"])
        doc_ids.append(docs_for_this_citation)

        assert len(doc_ids) == len(span_demarks)

    # this add the doc_id enclosed in []
    new_answer = highlight_substrings(orig_answer, span_demarks, doc_ids)

    # adding docs list
    cited_docs = get_documents_from_response(v_response)

    new_answer += "\n\n"
    for cited_doc in cited_docs:
        new_answer += str(cited_doc) + "\n"

    return new_answer


# Funzione per processare il file XLS
def read_input_file(file):
    """
    read a file and return a df
    """

    # Legge il file XLS in un DataFrame di Pandas
    input_df = pd.read_excel(file)
    # Esempio di elaborazione: mostra le prime 5 righe del DataFrame

    input_df = input_df[:LIMITS]

    return input_df


def create_output_file(all_questions, all_answers, input_file_name):
    """
    Save all the results in an xls file
    """
    out_dict = {"Questions": all_questions, "Answers": all_answers}
    out_df = pd.DataFrame(out_dict)

    # take what preceed .xls
    only_name = input_file_name.split(".")[0]
    new_name = only_name + "_out.xlsx"

    out_df.to_excel(new_name, index=None)

    return new_name


#
# Main
#
logger = get_console_logger()


# Titolo dell'applicazione
st.set_page_config(layout="wide")
st.title("RFx AI Assistant")

# to reset
if st.sidebar.button("Reset"):
    # reset the UI
    reset_ui()

is_debug = st.sidebar.checkbox("Debug")

lang = st.sidebar.selectbox("Select Language", LANG_SUPPORTED)

st.sidebar.header("RAG/LLM")

model_list = get_model_list()
llm_model = st.sidebar.selectbox(translate("Select LLM", lang), model_list)

add_reranker = st.sidebar.checkbox(translate("Add reranker", lang))
enable_hyde = st.sidebar.checkbox(translate("Enable HyDE", lang))
enable_citations = st.sidebar.checkbox(translate("Enable citations", lang))

# Init list of collections
oraclecs_collections_list = get_list_collections(VECTOR_STORE_TYPE)
selected_collection = st.sidebar.selectbox(
    translate("Select documents collection", lang), oraclecs_collections_list
)
st.sidebar.markdown("----------")

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
    df = read_input_file(uploaded_file)

    # init the dataframe with no green tick
    df["Processed"] = df.index.map(
        lambda i: "✅" if i in st.session_state.processed_questions else ""
    )

    questions = list(df[QUESTION_COL_NAME].values)

    # Show results
    # to update the dataframe in place

    with col1:
        st.header(translate("Questions:", lang))

        dataframe_placeholder = st.empty()

        dataframe_placeholder.dataframe(df)

        progress_bar = st.progress(0)

        info_placeholder = st.empty()
        info_placeholder.info(translate("Processing started!", lang))

    with col2:
        st.header(translate("Answers:", lang))

    answers = []

    # show the collection chosen
    logger.info("Collection chosen: %s", selected_collection)
    logger.info("")

    if is_debug:
        show_books(selected_collection)

    #
    # loop to process all questions
    #
    for i, question in enumerate(questions):

        logger.info("Processing: %s ...", question)

        # call the GenAI
        if enable_hyde:
            response = hyde_rag(
                question,
                llm_model,
                add_reranker=add_reranker,
                lang=lang,
                selected_collection=selected_collection,
            )
        else:
            response = classic_rag(
                question,
                llm_model,
                add_reranker=add_reranker,
                lang=lang,
                selected_collection=selected_collection,
            )

        answer = get_text_from_response(response)

        if ("cohere" in llm_model) and enable_citations:
            # handle citations
            # modify answer to add citations
            answer = add_citations_to_answer(answer, response)

        if is_debug:
            logger.info("Answer: %s", answer)
            logger.info("")

        # add here, because it has been eventually modified for citations
        answers.append(answer)

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

    # when all questions have been processed, handle output
    dict_out = {"Answers": answers}
    df_out = pd.DataFrame(dict_out)

    with col2:
        st.dataframe(df_out)

    # save output file
    create_output_file(questions, answers, uploaded_file.name)

else:
    st.write(translate("Load the xls file with the questions.", lang))

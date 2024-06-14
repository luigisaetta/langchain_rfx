"""
UI for HYDE test
"""

import streamlit as st
import pandas as pd

from factory_hyde import hyde_step1_2
from translations import translations
from utils import get_console_logger

# the name of the column with all the questions
QUESTION_COL_NAME = "Question"
LIMITS = 6


# to handle multilingual use the dictionary in translations.py
def translate(text, v_lang):
    """
    to handle labels in different lang
    """
    return translations.get(v_lang, {}).get(text, text)


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

lang = st.sidebar.selectbox("Select Language", ["en", "es", "fr", "it"])

add_reranker = st.sidebar.checkbox("Add reranker")

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

    for i, question in enumerate(questions):

        logger.info("Processing: %s ...", question)

        # call the GenAI
        answer = hyde_step1_2(question, add_reranker=add_reranker, lang=lang)

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

    # handle output
    dict_out = {"Answers": answers}

    df_out = pd.DataFrame(dict_out)

    with col2:
        st.dataframe(df_out)

else:
    st.write(translate("Load the xls file with the questions.", lang))

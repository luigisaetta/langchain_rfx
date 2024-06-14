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


# to handle multilingual use the dictiornary in translations.py
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

    input_df = input_df[:3]

    return input_df


logger = get_console_logger()

lang = st.sidebar.selectbox("Select Language", ["en", "es", "fr", "it"])

# Titolo dell'applicazione
st.title("RFx AI Assistant")

# to handle progress status and green tick
if "processed_questions" not in st.session_state:
    st.session_state.processed_questions = set()

# Caricamento del file
uploaded_file = st.sidebar.file_uploader(
    translate("Choose an xls file", lang), type=["xls", "xlsx"]
)

if uploaded_file is not None:

    # Carica il file e mostra il contenuto
    df = process_file(uploaded_file)

    # init the dataframe with no green tick
    df["Processed"] = df.index.map(
        lambda i: "✅" if i in st.session_state.processed_questions else ""
    )

    questions = list(df[QUESTION_COL_NAME].values)

    # Show results
    st.write(translate("Questions:", lang))

    # to update the dataframe in place
    dataframe_placeholder = st.empty()

    dataframe_placeholder.dataframe(df)

    progress_bar = st.progress(0)

    info_placeholder = st.empty()
    info_placeholder.info(translate("Processing started!", lang))

    answers = []

    for i, question in enumerate(questions):

        logger.info("Processing: %s ...", question)

        # call the GenAI
        answer = hyde_step1_2(question)

        answers.append(answer)

        # register it has been processed
        st.session_state.processed_questions.add(i)
        df["Processed"] = df.index.map(
            lambda i: "✅" if i in st.session_state.processed_questions else ""
        )

        dataframe_placeholder.dataframe(df)

        # update the progress bar
        progress_bar.progress(int(i + 1) / len(questions))

    info_placeholder.success(translate("Processing completed!", lang))

    # handle output
    dict_out = {"Answers": answers}

    df_out = pd.DataFrame(dict_out)

    st.dataframe(df_out)

else:
    st.write(translate("Load the xls file with the questions.", lang))

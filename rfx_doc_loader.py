""""
Document Loader for RAG

for now it manages only 23AI as Vector Store
"""

import pandas as pd
import streamlit as st

from rfx_doc_loader_backend import (
    load_uploaded_file_in_vector_store,
    get_list_collections,
    get_books,
    delete_documents_in_collection,
)

from utils import get_console_logger, remove_path_from_ref
from config import VERBOSE, CHUNK_SIZE, CHUNK_OVERLAP

# configs
DOC_NAME_COL = "Document name"

logger = get_console_logger()


def on_change_callback():
    """
    to handle on change on the collections' listbox
    """
    logger.info("Selected collection: %s", st.session_state["listbox_collections"])


def get_df_of_books(selected_collection):
    """
    create the dataframe with list of books
    """
    if VERBOSE:
        logger.info("Collection is: %s...", selected_collection)

    list_books = get_books(selected_collection)

    df_dict = {DOC_NAME_COL: list_books}
    result_df = pd.DataFrame(df_dict)

    return result_df


def manage_dataframe_with_selections(v_df, v_col, v_ph_title2, v_ph_df2):
    """
    This function display df of books and
    supports the selection of documents to drop

    col is the Streamlit col where to display
    """
    with v_col:
        v_ph_title2.markdown("#### Documents in collection:")

        df_with_selections = v_df.copy()
        df_with_selections.insert(0, "Select", False)

        # Get dataframe row-selections from user with st.data_editor
        edited_df = v_ph_df2.data_editor(
            df_with_selections,
            hide_index=True,
            column_config={"Select": st.column_config.CheckboxColumn(required=True)},
            disabled=v_df.columns,
            use_container_width=True,
        )

        # Filter the dataframe using the temporary column, then drop the column
        selected_rows = edited_df[edited_df.Select]

    return selected_rows


# Main
#
st.set_page_config(layout="wide")

st.title("AI Vector Search KB Manager")

col1, col2 = st.columns(2)

# inizialize session
if "checkbox_new_collection" not in st.session_state:
    st.session_state.checkbox_new_collection = False
if "selected_collection" not in st.session_state:
    st.session_state.selected_collection = None

# for deletion
sel_rows = None

with col1:
    # for the list of books in the collection
    # ph: placeholde, to keep the position of the element
    ph_title1 = st.empty()
    ph_df1 = st.empty()
    ph_butt1 = st.empty()


st.session_state.checkbox_new_collection = st.sidebar.checkbox(
    "Create new collection", value=st.session_state.checkbox_new_collection
)

if st.session_state.checkbox_new_collection:
    # want to create a new collection
    new_collection_name = st.sidebar.text_input("Insert the name of the new collection")

    if new_collection_name in get_list_collections():
        logger.info("Collection: %s already existing!", new_collection_name)
        st.error("Collection already existing!")
else:
    # select existing collection
    oraclecs_collections_list = get_list_collections()

    st.session_state.selected_collection = st.sidebar.selectbox(
        "Select documents collection",
        key="listbox_collections",
        options=oraclecs_collections_list,
        on_change=on_change_callback,
    )

    # show books in collection
    if st.session_state.selected_collection is not None:
        df = get_df_of_books(st.session_state.selected_collection)

        with col1:
            ph_title1.markdown(
                f"#### Documents in collection {st.session_state.selected_collection}"
            )

            sel_rows = manage_dataframe_with_selections(df, col1, ph_title1, ph_df1)

if st.session_state.checkbox_new_collection:
    collection_name = new_collection_name
else:
    collection_name = st.session_state.selected_collection

# 06/07 added chunks parameter
chunk_size = st.sidebar.slider(
    "Chunk size (in char)", 500, 5000, value=CHUNK_SIZE, step=50
)
chunk_overlap = st.sidebar.slider(
    "Chunk overlap (in char)", 50, 300, value=CHUNK_OVERLAP, step=50
)

# loading file
st.session_state.uploaded_file = st.sidebar.file_uploader(
    "Choose a pdf file to add", type=["pdf"]
)

if st.session_state.uploaded_file is not None:
    # remove path before loading
    new_file_name = remove_path_from_ref(st.session_state.uploaded_file.name)

    if VERBOSE:
        logger.info("Selected collection: %s...", collection_name)

    if (st.session_state.checkbox_new_collection) or (
        new_file_name not in get_books(collection_name)
    ):
        logger.info("Loading file: %s", new_file_name)

        with st.spinner("Loading in progress.."):
            STATUS = load_uploaded_file_in_vector_store(
                st.session_state.uploaded_file,
                collection_name,
                chunk_size,
                chunk_overlap,
            )

        if STATUS == "OK":
            # better reading from the DB, there is the state
            st.session_state.selected_collection = collection_name
            df = get_df_of_books(collection_name)

            with col1:
                # shows df
                sel_rows = manage_dataframe_with_selections(df, col1, ph_title1, ph_df1)

            # after creating it the collection is not anymore new
            st.session_state.checkbox_new_collection = False

        else:
            logger.info(
                "Error: file %s not loaded...", st.session_state.uploaded_file.name
            )
    else:
        if VERBOSE:
            logger.info("File: %s already loaded", st.session_state.uploaded_file.name)
        st.info("File already loaded !")

# handle deletion
if ph_butt1.button("Drop selected docs."):
    if len(sel_rows) > 0:

        delete_documents_in_collection(
            st.session_state.selected_collection, sel_rows[DOC_NAME_COL].values
        )

        # refresh
        df = get_df_of_books(st.session_state.selected_collection)

        with col1:
            sel_rows = manage_dataframe_with_selections(df, col1, ph_title1, ph_df1)

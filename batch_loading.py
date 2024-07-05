"""
Batch loading

Create a new collection and load a set of pdf
Can be used ONLY for a new collection.
"""

import sys
from glob import glob

from chunk_index_utils import (
    load_book_and_split,
    create_collection_and_add_docs_to_23ai,
)
from rfx_doc_loader_backend import get_list_collections
from factory_rfx import get_embed_model

from utils import get_console_logger

from batch_loading_config import BOOKS_DIR, NEW_COLLECTION_NAME

#
# Main
#

logger = get_console_logger()

logger.info("")
logger.info("Batch loading books in collection %s ...", NEW_COLLECTION_NAME)
logger.info("")

# init models
embed_model = get_embed_model()

# check that the collection doens't exist yet
collection_list = get_list_collections()

if NEW_COLLECTION_NAME in collection_list:
    logger.info("")
    logger.error("Collection %s already exist!", NEW_COLLECTION_NAME)
    logger.error("Exiting !")
    logger.info("")

    sys.exit(-1)

logger.info("")

books_list = glob(BOOKS_DIR + "/*.pdf")

docs = []

for book in books_list:
    logger.info("Chunking: %s", book)
    docs += load_book_and_split(book)

if len(docs) > 0:
    logger.info("Loading documents in collection %s", NEW_COLLECTION_NAME)

    create_collection_and_add_docs_to_23ai(docs, embed_model, NEW_COLLECTION_NAME)

    logger.info("Loading completed.")
    logger.info("")
else:
    logger.info("No document to load!")
    logger.info("")

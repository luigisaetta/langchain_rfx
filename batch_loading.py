"""
Batch loading

Create a new collection and load a set of pdf
Can be used ONLY for a new collection.
"""

import sys
import argparse
from glob import glob

from chunk_index_utils import (
    load_book_and_split,
    create_collection_and_add_docs_to_23ai,
)
from rfx_doc_loader_backend import get_list_collections
from factory_rfx import get_embed_model

from utils import get_console_logger

from batch_loading_config import BOOKS_DIR

from config import CHUNK_SIZE, CHUNK_OVERLAP

#
# Main
#

# handle input for new_collection_name from command line
parser = argparse.ArgumentParser(description="Document batch loading.")

parser.add_argument("new_collection_name", type=str, help="New collection name.")
parser.add_argument("books_dir", type=str, help="Dir with the books to load.")

args = parser.parse_args()

new_collection_name = args.new_collection_name
BOOKS_DIR = args.books_dir

logger = get_console_logger()

logger.info("")
logger.info("Batch loading books in collection %s ...", new_collection_name)
logger.info("")

# init models
embed_model = get_embed_model()

# check that the collection doesn't exist yet
collection_list = get_list_collections()

if new_collection_name in collection_list:
    logger.info("")
    logger.error("Collection %s already exist!", new_collection_name)
    logger.error("Exiting !")
    logger.info("")

    sys.exit(-1)

logger.info("")

# the list of books to be loaded
books_list = glob(BOOKS_DIR + "/*.pdf")

docs = []

for book in books_list:
    logger.info("Chunking: %s", book)
    docs += load_book_and_split(book, CHUNK_SIZE, CHUNK_OVERLAP)

if len(docs) > 0:
    logger.info("")
    logger.info("Loading documents in collection %s", new_collection_name)

    create_collection_and_add_docs_to_23ai(docs, embed_model, new_collection_name)

    logger.info("Loading completed.")
    logger.info("")
else:
    logger.info("No document to load!")
    logger.info("")

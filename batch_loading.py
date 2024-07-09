"""
Batch loading

Create a new collection and load a set of pdf
Can be used ONLY for a new collection.
"""

import sys
import argparse
from glob import glob
import numpy as np

from chunk_index_utils import (
    load_book_and_split,
    create_collection_and_add_docs_to_23ai,
)
from rfx_doc_loader_backend import get_list_collections
from factory_rfx import get_embed_model

from utils import get_console_logger

from batch_loading_config import BOOKS_DIR

from config import CHUNK_SIZE, CHUNK_OVERLAP


def compute_stats(list_docs):
    """
    Compute stats for the distribution of chunks' lengths

    list_docs: LangChain list of Documents
    """
    lengths = [len(d.page_content) for d in list_docs]

    mean_length = int(round(np.mean(lengths), 0))

    std_dev = int(round(np.std(lengths), 0))

    perc_75_len = int(round(np.percentile(lengths, 75), 0))

    return mean_length, std_dev, perc_75_len


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

logger.info("These books will be loaded:")
for book in books_list:
    logger.info(book)

logger.info("")

logger.info("These are the parameters used for chunking:")
logger.info("Chunk size: %s", CHUNK_SIZE)
logger.info("Chunk overlap: %s", CHUNK_OVERLAP)
logger.info("")

docs = []

for book in books_list:
    logger.info("Chunking: %s", book)
    docs += load_book_and_split(book, CHUNK_SIZE, CHUNK_OVERLAP)

if len(docs) > 0:
    logger.info("")
    logger.info("Embedding and loading documents in collection %s", new_collection_name)

    create_collection_and_add_docs_to_23ai(docs, embed_model, new_collection_name)

    logger.info("Loading completed.")
    logger.info("")

    mean, stdev, perc_75 = compute_stats(docs)

    logger.info("")
    logger.info("Statistics on the distribution of chunk lengths:")
    logger.info("Total num. of chunks loaded: %s", len(docs))
    logger.info("Avg. length : %s (chars)", mean)
    logger.info("Std dev: %s (chars)", stdev)
    logger.info("75-perc. : %s (chars)", perc_75)
    logger.info("")

else:
    logger.info("No document to load!")
    logger.info("")

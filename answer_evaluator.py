"""
Answer evaluation tool:

"""

import argparse
import pandas as pd

from factory_vector_store import get_vector_store
from factory_rfx import get_embed_model, get_llm, format_docs_for_cohere

from utils import get_console_logger

# configs
PREAMBLE = """

## Task
Based on the provided criteria, evaluate and compare the two answers. 
Do not answer the question itself. Instead, follow these instructions:

1. Comparison and Analysis:
Highlight the strengths and weaknesses of each answer in relation to the provided 
documentation and criteria.

2. Scoring:
Assign a score for each criterion (Accuracy, Completeness, Relevance, Clarity) 
on a scale from 0 to 10 for each answer.
Summarize the scores for each criterion in a table format.
Organize the table with a column for each answer and rows representing each criterion.

## Comparison criteria
Accuracy: Which answer more accurately reflects the information in the documentation?
Completeness: Which answer provides a more comprehensive response?
Relevance: Which answer is more relevant to the question asked?
Clarity: Which answer is clearer and easier to understand?

## Formatting rules
Ensure that the table is neatly formatted, 
with aligned columns and uniform field widths.
The column "Criteria" must be 14 chars wide.
Use space for formatting.
This is an example of the formatting for table to be produced:

## Scoring

| Criteria     | Answer 1 | Answer 2 |
|--------------|----------|----------|
| Accuracy     |    7     |    9     |
| Completeness |    7     |   10     |
| Relevance    |    8     |    9     |
| Clarity      |    8     |    9     |

The 'Criteria' column should be wide enough to accommodate the longest text, with values left-aligned.

"""

# the model we're using as Judge
COMPARATOR_MODEL = "cohere.command-r-plus"

# number of docs retrieved
TOP_K = 8
COLLECTION = "CLOUD_COACHING"

#
# Main
#
logger = get_console_logger()

# handle input for new_collection_name from command line
parser = argparse.ArgumentParser(description="Answers' evaluation.")

parser.add_argument("input_file", type=str, help="Input file (xlsx).")

args = parser.parse_args()

INPUT_FILE = args.input_file

# input (3 cols: question, answer1, answer2)
# INPUT_FILE = "./data/answer_comp_commandr_llama3.xlsx"

# setup models
embed_model = get_embed_model()

v_store = get_vector_store("23AI", embed_model, COLLECTION)

chat = get_llm(COMPARATOR_MODEL, temperature=0.0)
chat.preamble_override = PREAMBLE
# make it fully deterministic
chat.top_k = 1
chat.top_p = 1
# want streaming mode
chat.is_streaming = True

# read input file
answ_df = pd.read_excel(INPUT_FILE)

print("")
print("-" * 50)
print("              Evaluation Report")
print("-" * 50)
print("")

# process
for index, row in enumerate(answ_df.itertuples(index=False), start=1):

    QUERY = row.question
    ANSW1 = row.answer1
    ANSW2 = row.answer2

    # the request with the task for the model
    REQUEST = (
        f"Question:\n"
        f"{QUERY}\n\n"
        f"Answer1:\n"
        f"{ANSW1}\n\n"
        f"Answer2:\n"
        f"{ANSW2}\n"
    )

    print("-" * 50)
    print("        Evaluation results for query n. ", index)
    print("-" * 50)
    print("")
    print("Query: ", QUERY)
    print("")

    # here we do the search
    docs = v_store.similarity_search(query=QUERY, k=TOP_K)

    # format the docs in the format expected by Cohere
    c_docs = format_docs_for_cohere(docs)

    # invoke the chat model for the comparison
    # the preamble has been set, with the instructions
    response = chat.invoke(query=REQUEST, chat_history=[], documents=c_docs)

    chat.print_response(response)

    print("")

"""
Answer evaluation tool:

"""

import pandas as pd

from factory_vector_store import get_vector_store
from factory_rfx import get_embed_model, get_llm, format_docs_for_cohere

from utils import get_console_logger

# configs
PREAMBLE = """
## Comparison criteria
Accuracy: Which answer more accurately reflects the information in the documentation?
Completeness: Which answer provides a more comprehensive response?
Relevance: Which answer is more relevant to the question asked?
Clarity: Which answer is clearer and easier to understand?

## Task
Based on the provided criteria, compare the two answers. 
Highlight the strengths and weaknesses of each answer in relation to the documentation and criteria.
Provide a score, in the range 0-10, for each of the criteria (accuracy, completeness, relevance, clarity)
and provide also an overall score.

Summarize the scores for the criteria and the two answers in a table organized
in a column for each answer and using as rows the comparison criteria.
Ensure that the table is neatly formatted, 
with aligned columns and uniform field widths.

This is an example of the table to be produced with the required formatting
| Criteria     | Answer 1 | Answer 2 |
|--------------|----------|----------|
| Accuracy     |    8     |    9     |
| Completeness |    7     |   10     |
| Relevance    |    8     |    9     |
| Clarity      |    8     |    9     |
| Overall      |  7.75    |  9.25    |
The 'Criteria' column should be wide enough to accommodate the longest text, with values left-aligned.

"""

COMPARATOR_MODEL = "cohere.command-r-plus"

# number of docs retrieved
TOP_K = 8
COLLECTION = "MY_BOOKS"

#
# Main
#
logger = get_console_logger()

# setup models
embed_model = get_embed_model()

v_store = get_vector_store("23AI", embed_model, COLLECTION)

chat = get_llm(COMPARATOR_MODEL, temperature=0.0)
chat.preamble_override = PREAMBLE
# make it fully deterministic
chat.top_k = 1
chat.top_p = 1

# input (3 cols: question, answer1, answer2)
INPUT_FILE = "./data/answer_comp_commandr_llama3.xlsx"

answ_df = pd.read_excel(INPUT_FILE)

print("")
print("")

for index, row in enumerate(answ_df.itertuples(index=False), start=1):
    logger.info("Evaluating n. %s", index)
    QUERY = row.question
    ANSW1 = row.answer1
    ANSW2 = row.answer2

    # the requst with the task for the model
    REQUEST = (
        f"Question:\n"
        f"{QUERY}\n\n"
        f"Answer1:\n"
        f"{ANSW1}\n\n"
        f"Answer2:\n"
        f"{ANSW2}\n"
    )

    # here we do the search
    docs = v_store.similarity_search(query=QUERY, k=TOP_K)

    # format the docs in the format expected by Cohere
    c_docs = format_docs_for_cohere(docs)

    # invoke the chat model for the comparison
    result = chat.invoke(query=REQUEST, chat_history=[], documents=c_docs)

    print("-" * 50)
    print("          Evaluation results ")
    print("-" * 50)
    print(result.data.chat_response.text)
    print("")

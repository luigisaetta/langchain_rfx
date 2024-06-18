"""
Test hyde
"""

from pathlib import Path

# to read, write xls
import pandas as pd
from tqdm import tqdm

from factory_hyde import hyde_rag
from utils import get_console_logger
from pdf_utils import PDF

# the input file is here...
INPUT_DIR = Path(".")
# output file in this directory
OUTPUT_DIR = Path(".")

# Input: we take the questions from this file
QUERY_FILE = "rfp01.xlsx"

# the name of the column with all questions
QUESTION_COL_NAME = "Question"

# full path of questions file
QUERY_PATH_NAME = INPUT_DIR / QUERY_FILE

# we write results to this file
OUTPUT_FILE_NAME = "answers01.xlsx"
OUTPUT_PATH_NAME = OUTPUT_DIR / OUTPUT_FILE_NAME
input_df = pd.read_excel(QUERY_PATH_NAME)

questions = list(input_df[QUESTION_COL_NAME].values)

# reduce the number to test
questions = questions[:3]

logger = get_console_logger()

# a look at the first five questions

logger.info(f"There are {len(questions)} questions...")
logger.info("")
logger.info("Questions:")
logger.info("")
for question in questions:
    logger.info(question)
logger.info("")

logger.info("Processing...")
logger.info("")

answers = []
for question in tqdm(questions):
    answer = hyde_rag(question)

    answers.append(answer)

# show results
logger.info("Results:")
logger.info("")
for question, answer in zip(questions, answers):
    print("")
    print("Query:", question)
    print("")
    print(answer)
    print("")

logger.info("Creating pdf..")

pdf = PDF()

# Aggiungere domande e risposte al PDF
for i, (question, answer) in enumerate(zip(questions, answers)):
    pdf.add_question_answer(question, answer, i)

# Salvare il PDF
pdf.output("aswers01.pdf")

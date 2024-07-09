"""
Utilities to handle production of output
"""

import pandas as pd

TITLE = "# Report with Requests and Answers\n\n"


def generate_xlsx_file(all_questions, all_answers, input_file_name):
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


def generate_markdown_file(questions, answers, input_file_name):
    """
    This function can be used to generate the final report in MD format
    """

    only_name = input_file_name.split(".")[0]
    new_name = only_name + "_out.md"

    with open(new_name, "w", encoding="utf-8") as file:
        file.write(TITLE)

        for question, answer in zip(questions, answers):
            file.write(f"## Request: {question}\n")
            file.write(f"{answer}\n\n\n")

    return new_name

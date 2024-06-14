"""
To create the pdf
"""

from fpdf import FPDF


class PDF(FPDF):
    font_name = "Helvetica"

    def header(self):
        self.set_font(self.font_name, "B", 12)
        self.cell(0, 10, "Questions and answers", 0, 1, "C")

    def footer(self):
        self.set_y(-15)
        self.set_font(self.font_name, "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    def chapter_title(self, question, chapter_num):
        self.set_font(self.font_name, "B", 12)
        self.cell(0, 10, f"Question {chapter_num+1}: {question}", 0, 1, "L")
        self.ln(5)

    def chapter_body(self, answer):
        self.set_font(self.font_name, "", 12)
        self.multi_cell(0, 10, answer)
        self.ln()

    def add_question_answer(self, question, answer, chapter_num):
        self.add_page()
        self.chapter_title(question, chapter_num)
        self.chapter_body(answer)

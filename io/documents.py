""" This module contains functions for reading files. """ ""

import xlrd
import PyPDF2


def read_plain_text(path):
    """Reads a plain text file and returns the text as a string."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def read_pdf(path):
    """Reads a pdf file and returns the text as a string."""
    file_obj = open(path, "rb")
    reader = PyPDF2.PdfFileReader(file_obj)
    text = ""
    for page in range(reader.numPages):
        page_obj = reader.getPage(page)
        text += page_obj.extractText()
    return text


def read_xlsx(path):
    """Reads an xlsx file and returns the text as a string."""
    wb = xlrd.open_workbook(path)
    text = ""
    for sheet in wb.sheets():
        for row in range(sheet.nrows):
            for col in range(sheet.ncols):
                text += str(sheet.cell(row, col).value)
    return text

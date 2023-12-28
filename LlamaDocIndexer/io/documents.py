""" This module contains functions for reading files. """ ""

import os

import pypdf
import xlrd
from llama_index import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)


def make_dirs(folder):
    """Creates a folder if it does not exist."""
    if not os.path.isdir(folder):
        os.makedirs(folder)


def read_plain_text(path):
    """Reads a plain text file and returns the text as a string."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf(path):
    """Reads a pdf file and returns the text as a string."""
    with open(path, "rb") as file:
        reader = pypdf.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
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


def text_to_index(text_path):
    reader = SimpleDirectoryReader(input_files=[text_path])
    documents = reader.load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index


def save_index(index, index_path):
    index.storage_context.persist(index_path)


def load_index(index_path):
    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=index_path)
    # load index
    return load_index_from_storage(storage_context)

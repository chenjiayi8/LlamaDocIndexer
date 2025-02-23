""" This module contains functions for reading files. """ ""

import os

import pypdf
import xlrd
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)


def make_dirs(folder):
    """Creates a folder if it does not exist."""
    if not os.path.isdir(folder):
        os.makedirs(folder)


def is_plain_text(path, encoding="utf-8"):
    """Check if a file is likely a plain text file, considering UTF-8 encoding."""
    try:
        with open(path, "rb") as f:
            # Try reading the file and decoding it as UTF-8 (or specified encoding)
            # This will raise a UnicodeDecodeError if the file contains characters
            # that can't be decoded with the specified encoding.
            content = f.read()
            content.decode(encoding)
    except UnicodeDecodeError:
        # If a UnicodeDecodeError is encountered, it's likely not a text file
        return False
    except Exception as e:
        # Handle other potential errors (like file not found)
        print(f"Error reading file: {e}")
        return False
    return True


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

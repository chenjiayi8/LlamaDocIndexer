import hashlib
import json
import os
import time

from llama_index import ComposableGraph, ListIndex, StorageContext

from LlamaDocIndexer.io.documents import (load_index, make_dirs, read_pdf,
                                          read_plain_text, read_xlsx,
                                          save_index, text_to_index)


class Indexer:
    """Indexes a folder of documents and saves the index to a folder."""

    def __init__(
        self, folder_path, index_path, types=None, ignored_files=None, depth=3
    ):
        self.folder_path = folder_path
        self.index_path = index_path
        self.types = types
        self.ignored_files = ignored_files
        self.depth = depth
        self.indices = {"menu": {}}
        self.query_engine = None
        self.initiate()

    def initiate(self):
        """Initiates the indexer."""
        if self.types is None:
            self.types = [
                ".txt",
                ".pdf",
                ".xlsx",
                ".tex",
            ]
        if self.ignored_files is None:
            self.ignored_files = []
        make_dirs(self.index_path)
        # load menu
        menu_path = os.path.join(self.index_path, "menu.json")
        if os.path.isfile(menu_path):
            with open(menu_path, "r", encoding="utf-8") as f:
                self.indices["menu"] = json.load(f)
        else:
            self.indices["menu"] = {}
            with open(menu_path, "w", encoding="utf-8") as f:
                json.dump(self.indices["menu"], f, indent=4)

        self.load_indices()

    def generate_summary(self, index):
        """Generates a summary from an index."""
        engine = index.as_query_engine()
        summary = engine.query("Please summarise this document.")
        return str(summary)

    def load_indices(self):
        """Loads the indices from the index folder."""
        for path_hash in self.indices["menu"]:
            data_path = os.path.join(self.index_path, path_hash, "data.json")
            if not os.path.isfile(data_path):
                del self.indices["menu"][path_hash]
                continue

            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                index_path = os.path.join(self.index_path, path_hash, "index")
                self.indices[path_hash] = {
                    "summary": data["path"],
                    "index": load_index(index_path),
                }

    def is_supported_file(self, file):
        """Checks if a file is supported."""
        file_basename = os.path.splitext(file)[0]
        if file_basename.lower() in self.ignored_files:
            return False
        file_extension = os.path.splitext(file)[1]
        if file_extension.lower() in self.types:
            return True
        return False

    def read_text(self, root, file):
        """Reads the text from a file."""
        file_path = os.path.join(root, file)
        # get file extension
        file_extension = os.path.splitext(file)[1]
        text = None
        if file_extension not in self.types:
            return text
        if file_extension.lower() in [".txt", ".tex", ".json"]:
            text = read_plain_text(file_path)
        elif file_extension.lower() == ".pdf":
            text = read_pdf(file_path)
        elif file_extension.lower() == ".xlsx":
            text = read_xlsx(file_path)
        else:
            raise ValueError("Unsupported file type: " + file_extension)
        return text

    def build(self):
        """Builds the index."""
        update = False
        # loop through all files in folder recursively
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                # check if file is supported
                if not self.is_supported_file(file):
                    continue
                file_path = os.path.join(root, file)
                depth = file_path.replace(self.folder_path, "").count(os.sep)
                # check depth
                if depth > self.depth:
                    continue
                # get relative path
                relative_path = os.path.relpath(file_path, self.folder_path)
                # get path hash
                path_hash = hashlib.md5(
                    relative_path.encode("utf-8")
                ).hexdigest()

                # create summary object
                summary = {
                    "name": file,
                    "path": relative_path,
                    "text": None,
                }
                modified = os.path.getmtime(file_path)

                # check if file is already indexed
                if path_hash not in self.indices:
                    self.indices["menu"][path_hash] = {
                        "name": summary["name"],
                        "path": summary["path"],
                        "modified": -1,
                    }
                    self.indices[path_hash] = {
                        "summary": summary["path"],
                        "index": None,
                    }
                # check if file has been modified
                if modified == self.indices["menu"][path_hash]["modified"]:
                    continue
                self.indices["menu"][path_hash]["modified"] = modified
                # read text
                text = self.read_text(root, file)
                if text is None:
                    raise ValueError("Cannot read text from " + file)
                summary["text"] = text

                # create index folder
                index_folder = os.path.join(self.index_path, path_hash)
                make_dirs(index_folder)

                # save summary
                summary_path = os.path.join(index_folder, "data.json")
                index_path = os.path.join(index_folder, "index")
                with open(summary_path, "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=4)

                # add to index
                print("Indexing " + relative_path)
                self.indices[path_hash] = {
                    "summary": summary["path"],
                    "index": text_to_index(summary_path),
                }

                # save index
                save_index(self.indices[path_hash]["index"], index_path)
                update = True

        # save menu
        if update:
            menu_path = os.path.join(self.index_path, "menu.json")
            with open(menu_path, "w", encoding="utf-8") as f:
                json.dump(self.indices["menu"], f, indent=4)
        return update

    def create_query_engine(self):
        """Returns a query engine."""
        if self.query_engine is not None:
            return

        indices_list = []
        indices_summary = []
        for path_hash in self.indices["menu"]:
            value = self.indices[path_hash]
            indices_list.append(value["index"])
            indices_summary.append(value["summary"])

        storage_context = StorageContext.from_defaults()
        combined_index = ComposableGraph.from_indices(
            ListIndex,
            indices_list,
            index_summaries=indices_summary,
            storage_context=storage_context,
        )
        self.query_engine = combined_index.as_query_engine()

    def query(self, query):
        if self.build() or self.query_engine is None:
            self.create_query_engine()
        response = self.query_engine.query(query)
        return response

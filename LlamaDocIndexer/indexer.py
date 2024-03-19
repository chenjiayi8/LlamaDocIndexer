import hashlib
import json
import os
import time

from llama_index import (
    ComposableGraph,
    Document,
    ListIndex,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)

from LlamaDocIndexer.io.documents import (
    is_plain_text,
    load_index,
    make_dirs,
    read_pdf,
    read_plain_text,
    read_xlsx,
    save_index,
    text_to_index,
)


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

    def generate_index(self, text):
        """Generates an index from text."""
        document = Document(text=text)
        index = VectorStoreIndex.from_documents([document])
        return index

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
                    "summary": data["summary"],
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
        if is_plain_text(file_path):
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

                # create summary data object
                data = {
                    "name": file,
                    "path": relative_path,
                    "text": None,
                    "summary": "",
                }
                modified = os.path.getmtime(file_path)

                # check if file is already indexed
                if path_hash not in self.indices:
                    self.indices["menu"][path_hash] = {
                        "name": data["name"],
                        "path": data["path"],
                        "modified": -1,
                    }
                    self.indices[path_hash] = {
                        "summary": data["summary"],
                        "index": None,
                    }
                # check if file has been modified
                if modified == self.indices["menu"][path_hash]["modified"]:
                    continue
                self.indices["menu"][path_hash]["modified"] = modified

                # read text
                text = self.read_text(root, file)
                if text is None or len(text) == 0:
                    del self.indices["menu"][path_hash]
                    del self.indices[path_hash]
                    continue

                # announce indexing
                print("Indexing " + relative_path)
                data["text"] = text

                # create index folder
                index_folder = os.path.join(self.index_path, path_hash)
                make_dirs(index_folder)

                # create index
                index_path = os.path.join(index_folder, "index")
                index = self.generate_index(data["text"])

                # generate summary
                data["summary"] = self.generate_summary(index)

                # save summary data
                data_path = os.path.join(index_folder, "data.json")
                with open(data_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)

                # add to index
                self.indices[path_hash] = {
                    "summary": data["summary"],
                    "index": index,
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

    def get_file_list(self):
        """Returns a list of indexed files."""
        files = [v["path"] for k, v in self.indices["menu"].items()]
        return files

    def get_file_engine(self, file_path):
        """Returns a query engine for a file."""
        path_hash = hashlib.md5(file_path.encode("utf-8")).hexdigest()
        if path_hash not in self.indices["menu"]:
            raise ValueError("File not indexed: " + file_path)
        index_path = os.path.join(self.index_path, path_hash, "index")
        storage_context = StorageContext.from_defaults(persist_dir=index_path)
        index = load_index_from_storage(storage_context)
        file_engine = index.as_query_engine()
        return file_engine

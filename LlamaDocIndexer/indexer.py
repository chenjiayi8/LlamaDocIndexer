import concurrent.futures
import hashlib
import json
import os

from llama_index.core import (
    Document,
    StorageContext,
    SummaryIndex,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.schema import IndexNode

from LlamaDocIndexer.io.documents import (
    is_plain_text,
    load_index,
    make_dirs,
    read_pdf,
    read_plain_text,
    read_xlsx,
    save_index,
)
from LlamaDocIndexer.utilities.patterns import ignored_files_to_patterns


class Indexer:
    """Indexes a folder of documents and saves the index to a folder."""

    def __init__(
        self,
        folder_path,
        index_path=None,
        ignored_folders=None,
        ignored_files=None,
        depth=3,
        types=None,
    ):
        self.folder_path = folder_path
        self.index_path = index_path
        self.ignored_folders = ignored_folders
        self.ignored_files = ignored_files
        self.depth = depth
        self.types = types
        self.menu = {}
        self.indices = {}
        self.query_engine = None
        self.initiate()

    def initiate(self):
        """Initiates the indexer."""
        if self.index_path is None:
            self.index_path = os.path.join(self.folder_path, ".indices")
        if self.ignored_folders is None:
            self.ignored_folders = [".indices"]
        else:
            self.ignored_folders.append(".indices")
        if self.ignored_files is None:
            self.ignored_files = []
        self.ignored_patterns = ignored_files_to_patterns(self.ignored_files)
        make_dirs(self.index_path)
        # load menu
        menu_path = os.path.join(self.index_path, "menu.json")
        if os.path.isfile(menu_path):
            with open(menu_path, "r", encoding="utf-8") as f:
                self.menu = json.load(f)
        else:
            self.menu = {}
            with open(menu_path, "w", encoding="utf-8") as f:
                json.dump(self.menu, f, indent=4)

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
        for path_hash in self.menu:
            data_path = os.path.join(self.index_path, path_hash, "data.json")
            if not os.path.isfile(data_path):
                del self.menu[path_hash]
                continue

            with open(data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                index_path = os.path.join(self.index_path, path_hash, "index")
                self.indices[path_hash] = {
                    "summary": data.get("summary", ""),
                    "index": load_index(index_path),
                }

    def is_supported_file(self, file_path):
        """Checks if a file is supported."""
        if self.types is not None:
            file_extension = os.path.splitext(file_path)[1]
            if file_extension not in self.types:
                return False
        file_name = os.path.basename(file_path)
        for pattern in self.ignored_patterns:
            if pattern.match(file_name) is not None:
                return False
        if is_plain_text(file_path):
            return True

        return False

    def read_text(self, root, file):
        """Reads the text from a file."""
        file_path = os.path.join(root, file)
        # get file extension
        file_extension = os.path.splitext(file)[1]
        text = None
        if is_plain_text(file_path):
            text = read_plain_text(file_path)
        elif file_extension.lower() == ".pdf":
            text = read_pdf(file_path)
        elif file_extension.lower() == ".xlsx":
            text = read_xlsx(file_path)
        else:
            raise ValueError("Unsupported file type: " + file_extension)
        return text

    def has_ignore_folder(self, path):
        """Checks if a path contains an ignored folder."""
        if self.folder_path in path:
            path = path.replace(self.folder_path, "")
        folders = path.split(os.sep)
        if any(folder in self.ignored_folders for folder in folders):
            return True
        return False

    def build(self, num_workers=8):
        """Builds the index."""
        update = False
        # loop through all files in folder recursively
        tasks = []
        for root, _, files in os.walk(self.folder_path):
            if self.has_ignore_folder(root):
                continue
            for file in files:
                # check if file is supported
                file_path = os.path.join(root, file)
                if not self.is_supported_file(file_path):
                    continue

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
                    self.menu[path_hash] = {
                        "name": data["name"],
                        "path": data["path"],
                        "modified": -1,
                    }
                    self.indices[path_hash] = {
                        "summary": data.get("summary", ""),
                        "index": None,
                    }
                # check if file has been modified
                if modified == self.menu[path_hash]["modified"]:
                    continue
                self.menu[path_hash]["modified"] = modified

                # read text
                text = self.read_text(root, file)
                if text is None or len(text) == 0:
                    del self.menu[path_hash]
                    del self.indices[path_hash]
                    continue

                data["text"] = text

                # create index folder
                index_folder = os.path.join(self.index_path, path_hash)
                make_dirs(index_folder)

                # create index
                task = {
                    "path_hash": path_hash,
                    "index_folder": index_folder,
                    "data": data,
                }
                tasks.append(task)
                update = True

        if len(tasks) == 0:
            return False

        # run tasks in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers
        ) as executor:
            results = executor.map(self.run_embedding_task, tasks)

        for result in results:
            self.save_embedding_data(result)

        # save menu
        if update:
            menu_path = os.path.join(self.index_path, "menu.json")
            with open(menu_path, "w", encoding="utf-8") as f:
                json.dump(self.menu, f, indent=4)
        return update

    def run_embedding_task(self, task):
        """Runs an embedding task."""
        print("Indexing " + task["data"]["path"])
        index = self.generate_index(task["data"]["text"])
        task["summary"] = self.generate_summary(index)
        task["index"] = index
        task["data"] = task["data"]
        return task

    def save_embedding_data(self, task):
        """Saves data to the index."""
        path_hash = task["path_hash"]
        index_folder = task["index_folder"]
        index = task["index"]
        # save summary data
        data_path = os.path.join(index_folder, "data.json")
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(task["data"], f, indent=4)

        # add to index
        self.indices[path_hash] = {
            "summary": task["summary"],
            "index": index,
        }

        # save index
        index_path = os.path.join(index_folder, "index")
        save_index(self.indices[path_hash]["index"], index_path)

    def create_query_engine(self, paths=None, top_k=5):
        """Returns a combined index as engine."""
        if paths is None:
            paths = self.get_file_list()

        path_hashes = [
            hashlib.md5(path.encode("utf-8")).hexdigest() for path in paths
        ]
        indices_nodes = []
        for path_hash in path_hashes:
            value = self.indices[path_hash]
            index = value.get("index")
            if index is None:
                continue
            vector_retriever = index.as_retriever(similarity_top_k=top_k)
            indices_nodes.append(
                IndexNode(
                    index_id=path_hash,
                    obj=vector_retriever,
                    text=value["summary"],
                )
            )
        summary_index = SummaryIndex(objects=indices_nodes)
        return summary_index.as_query_engine()

    def query(self, query, top_k=5):
        """Queries the index."""
        if self.build() or self.query_engine is None:
            self.query_engine = self.create_query_engine(top_k=top_k)
        response = self.query_engine.query(query)
        return response

    def get_file_list(self):
        """Returns a list of indexed files."""
        files = [v["path"] for k, v in self.menu.items()]
        # filter out ignored folders
        files = [file for file in files if not self.has_ignore_folder(file)]
        # filter out ignored files
        files = [
            file
            for file in files
            if self.is_supported_file(os.path.join(self.folder_path, file))
        ]

        return files

    def get_file_engine(self, file_path, top_k=5):
        """Returns a query engine for a file."""
        path_hash = hashlib.md5(file_path.encode("utf-8")).hexdigest()
        if path_hash not in self.menu:
            raise ValueError("File not indexed: " + file_path)
        file_engine = self.create_query_engine(paths=[file_path], top_k=top_k)
        return file_engine

    def get_folder_engine(self, folder_path, top_k=5):
        """Returns a query engine for a subfolder."""
        all_paths = [v["path"] for k, v in self.menu.items()]
        paths = [path for path in all_paths if path.startswith(folder_path)]
        return self.create_query_engine(paths=paths, top_k=top_k)

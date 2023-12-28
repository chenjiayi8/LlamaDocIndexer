# LlamaDocIndexer

LlamaDocIndexer is a dynamic and efficient repository designed to seamlessly integrate with LlamaIndex, a powerful data framework for LLM-based applications. This repository specializes in recursively indexing a folder of documents, ensuring that every file, whether it's in a SQL database, trapped in PDFs, or embedded in slide decks, is meticulously cataloged. It excels in automatically detecting any changes within these documents, triggering a swift and precise re-indexing process. This feature ensures that the LLMs have the most up-to-date and relevant information at their disposal, enhancing their ability to provide accurate and contextually relevant responses. Ideal for handling private or domain-specific data, LlamaDocIndexer bridges the gap between vast LLM knowledge bases and your unique data sets, making it an indispensable tool for anyone looking to leverage the full potential of language model applications in specialized fields.



# Usage
```
documents_folder = "./documents/tutorials/"
indices_folder = "./indices/tutorials/"
file_types = [".txt", ".tex"]
ignored_files = ["temp", "template]
recursive_depth = 3
indexer = LlamaDocIndexer(
    documents_folder,
    indices_folder,
    types=file_types,
    ignored_files=ignored_files,
    depth=recursive_depth,
)

indexer.query("What is the best way to cook a steak?")
```

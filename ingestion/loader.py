from langchain_community.document_loaders import PyPDFLoader
import os

def load_documents(data_path="data/factsheets"):
    documents = []

    for file in os.listdir(data_path):
        if file.endswith(".pdf"):
            full_path = os.path.join(data_path, file)

            loader = PyPDFLoader(full_path)
            docs = loader.load()

            for d in docs:
                d.metadata["source_file"] = file

            documents.extend(docs)

    return documents
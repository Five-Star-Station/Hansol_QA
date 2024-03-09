import pandas as pd
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
import chromadb


# embedding_model = "osung-station/deco_embedding"
embedding_model = "jhgan/ko-sroberta-multitask"
chromadb_store = "vector_store/"


documents = []
loader = CSVLoader(file_path="data/answers.csv", encoding="UTF-8")
documents.extend(loader.load())
# loader = CSVLoader(file_path="data/train_combined.csv", encoding="UTF-8")
# documents.extend(loader.load())
client = chromadb.Client()
if client.list_collections():
    consent_collection = client.create_collection("consent_collection")
else:
    print("Collection already exists")
vectordb = Chroma.from_documents(
    documents=documents,
    embedding=HuggingFaceEmbeddings(
        model_name=embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    ),
    persist_directory=chromadb_store,
)

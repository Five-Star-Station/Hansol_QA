# Import necessary libraries
import pandas as pd
import numpy as np
import torch
from kiwipiepy import Kiwi
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer

# Initialize Kiwi for Korean sentence splitting
kiwi = Kiwi()

# Load the datasets
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")


# Define a function to split queries into sentences using Kiwi
def split_into_sentences(query):
    return [item[0] for item in kiwi.split_into_sents(query)]


# Apply the function to the test dataset
test_df["split_queries"] = test_df["질문"].map(split_into_sentences)

# Template for generating prompts
PROMPT_TEMPLATE = """<s>[INST] <<SYS>>
You are an helpful assistant for construction and interior. Your task is to generate a valid answer based on the given information:

{context}

<</SYS>>
{query} [/INST]
"""

# Initialize embedding and reranking models
embedding_model_name = "sentence-transformers/distiluse-base-multilingual-cased-v1"
rerank_model_name = "Dongjin-kr/ko-reranker"
final_embedding_model_name = (
    "sentence-transformers/distiluse-base-multilingual-cased-v1"
)
final_embedding_model = SentenceTransformer(final_embedding_model_name)

# Set up the vector store for embeddings
chromadb_store = "vector_store/"
vectordb = Chroma(
    persist_directory=chromadb_store,
    embedding_function=HuggingFaceEmbeddings(model_name=embedding_model_name),
)

# Load reranking model and tokenizer
rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_name)
rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model_name)
rerank_model.eval()


# Function to normalize scores
def exp_normalize(scores):
    b = scores.max()
    y = np.exp(scores - b)
    return y / y.sum()


# Function to get the best matching document
def get_first_place_doc(query, matching_docs):
    pairs = [[query, doc.page_content.split("answer: ")[1]] for doc in matching_docs]

    with torch.no_grad():
        inputs = rerank_tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        logits = rerank_model(**inputs, return_dict=True).logits.view(-1).float()
        scores = exp_normalize(logits.numpy())
        max_idx = scores.argmax()
        print(scores[max_idx])

    return matching_docs[max_idx]


# Processing the queries from the test data
embeddings = []
generated_answers = []

for queries in test_df["split_queries"]:
    generated_answer = []
    for query in queries:
        # Get matching documents for the query
        matching_docs = vectordb.similarity_search(query, k=7)
        # Find the best document
        first_place_doc = get_first_place_doc(query, matching_docs)
        # Generate prompt
        prompt = PROMPT_TEMPLATE.format(
            context=first_place_doc.page_content, query=query
        )

        # Generate Text
        ### CODE ###
        # generated_texts.append(text)

        generated_answer.append(first_place_doc.page_content.split("answer: ")[1])

    generated_answer = " ".join(generated_answer)
    embeddings.append(final_embedding_model.encode(generated_answer))
    generated_answers.append(generated_answer)


test_df["generated_answer"] = generated_answers

ids = test_df["id"]
columns = ["id"] + [f"vec_{i}" for i in range(512)]
data = [[id] + list(embedding) for id, embedding in zip(ids, embeddings)]

submission = pd.DataFrame(data, columns=columns)
submission.to_csv("submission.csv", index=False)

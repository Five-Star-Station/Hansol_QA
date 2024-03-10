import numpy as np
import pandas as pd

import torch
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Custom Kiwi tokenizer
class CustomKiwi():
    def __init__(self, user_words, allow_tokens):
        self.kiwi = Kiwi()
        for tag, words in user_words.items():
            for word in words:
                self.kiwi.add_user_word(word, tag=tag, score=10, orig_word=None)
        self.allow_tokens = allow_tokens

    def __call__(self, text):
        return self.tokenize(text)

    def tokenize(self, text):
        tokens = self.kiwi.tokenize(text)
        selected_tokens = [(t.form, t.tag) for t in tokens if t.tag in self.allow_tokens]
        keywords = self.kiwi.join(selected_tokens)
        return keywords.split(' ')

# Function to normalize scores
def exp_normalize(scores):
    b = scores.max()
    y = np.exp(scores - b)
    return y / y.sum()


user_words = {'NNG' : ['면진', '결로', '몰딩', '사이딩', '징크', '판재'],
              'XPN' : ['준']}
allow_tokens = ['NNG', 'XSA-I', 'XPN', 'XSN', 'SL', 'NNP']

rerank_model_name = "Dongjin-kr/ko-reranker"
final_embedding_model_name = (
    "sentence-transformers/distiluse-base-multilingual-cased-v1"
)

kiwi = CustomKiwi(user_words, allow_tokens)

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

## TF-IDF Retrieval
##
documents = train_df['답변_2'].values
queries = test_df['질문'].values

query_corpus = train_df.apply(lambda x: f"{x['질문_1']} {x['질문_2']}", axis=1)

# 질문 1+질문 2 기반 vocab 생성
base_vectorizer = TfidfVectorizer(ngram_range=(1, 3), 
                                  sublinear_tf=True,
                                  tokenizer=kiwi)
base_vectorizer.fit(query_corpus)
vocab = base_vectorizer.get_feature_names_out()

# 답변 2 벡터화
vectorizer = TfidfVectorizer(ngram_range=(1, 3), 
                             sublinear_tf=True, 
                             vocabulary=vocab,
                             tokenizer=kiwi)
document_vector = vectorizer.fit_transform(documents)

# test 쿼리 벡터화
test_query_vector = vectorizer.transform(queries)

# 유사도 계산 및 상위 5개 문서 검색
scores = (test_query_vector * document_vector.T).toarray()
top_indices = np.argsort(scores, axis=1)[:, -5:]
top_scores = np.sort(scores, axis=1)[:, -5:]
retrieved_context = train_df['답변_2'][top_indices.flatten()].values.reshape(-1, 5)

## Reranking
## 
# Load reranking model and tokenizer
rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_name)
rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model_name)
rerank_model.eval()

# Function to get the best matching document
def get_first_place_doc(query, matching_docs):
    pairs = [[query, doc] for doc in matching_docs]

    with torch.no_grad():
        inputs = rerank_tokenizer(
            pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        logits = rerank_model(**inputs, return_dict=True).logits.view(-1).float()
        # scores = exp_normalize(logits.numpy())
        max_idx = logits.argmax()

    # return sorted(zip(matching_docs, scores), key=lambda x: x[1], reverse=True)
    return matching_docs[max_idx], logits[max_idx]

# Processing the queries from the test data
final_embedding_model = SentenceTransformer(final_embedding_model_name)

embeddings = []
generated_answers = []
answers_scores = []

for query, matching_docs in zip(queries, retrieved_context):
    first_place_doc, max_score = get_first_place_doc(query, matching_docs)
    print(max_score, query, first_place_doc)

    answers_scores.append(max_score)
    generated_answers.append(first_place_doc)
    embeddings.append(final_embedding_model.encode(first_place_doc))

## Save the results
## 
test_df["generated_answer"] = generated_answers
test_df["scores"] = answers_scores
test_df.to_csv("generated_test.csv", index=False)

ids = test_df["id"]
columns = ["id"] + [f"vec_{i}" for i in range(512)]
data = [[id] + list(embedding) for id, embedding in zip(ids, embeddings)]

submission = pd.DataFrame(data, columns=columns)
submission.to_csv("submission.csv", index=False)
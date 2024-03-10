import numpy as np
import pandas as pd
from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer

def kiwi_init(user_words):
    kiwi = Kiwi()
    for tag, words in user_words.items():
        for word in words:
            kiwi.add_user_word(word, tag=tag, score=10, orig_word=None)
    return kiwi

def extract_keyword(text, tokenizer, allow_tokens=None):
    tokens = tokenizer.tokenize(text)
    if allow_tokens is not None:
        selected_tokens = [t for t in tokens if t.tag in allow_tokens]
    keywords = tokenizer.join(selected_tokens)
    return keywords


user_words = {'NNG' : ['면진', '결로', '몰딩', '사이딩', '징크', '판재'],
              'XPN' : ['준']}

allow_tokens = ['NNG', 'XSA-I', 'XPN', 'XSN', 'SL', 'NNP']

kiwi = kiwi_init(user_words)

df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')

train_query_concat = df_train.apply(lambda x: f"{x['질문_1']} {x['질문_2']}", axis=1)
train_corpus = train_query_concat.map(lambda x: extract_keyword(x, kiwi, allow_tokens))

# 질문 1+질문 2 합쳐서 키워드 추출
base_vectorizer = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True)
base_vectorizer.fit(train_corpus)
vocab = base_vectorizer.get_feature_names_out()

# 답변 2 키워드 추출
train_context = df_train['답변_2'].map(lambda x: extract_keyword(x, kiwi, allow_tokens)).values
vectorizer = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, vocabulary=vocab)
context_vector = vectorizer.fit_transform(train_context)

test_query_keywords = df_test['질문'].map(lambda x: extract_keyword(x, kiwi, allow_tokens)).values
test_query_vector = vectorizer.transform(test_query_keywords)

scores = (test_query_vector * context_vector.T).toarray()
top_indices = np.argsort(scores, axis=1)[:, -5:].flatten()

retrieved_context = df_train['답변_2'][top_indices].values.reshape(-1, 5)
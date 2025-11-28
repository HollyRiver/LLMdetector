import pandas as pd
import re
import string
import nltk
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.corpora import Dictionary

import json


## Warning 이상의 문제 발생 시 출력
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

## nltk set download
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

try:
    nltk.data.find("corpora/stopwords")
except:
    nltk.download("stopwords")

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except:
    nltk.download("averaged_perceptron_tagger")

stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text) ## 구두점 전처리
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return tokens

def tokens_to_sequence(tokens, dictionary, max_len):
    ids = dictionary.doc2idx(tokens, unknown_word_index=-1) ## unk word는 일단 -1로 한 다음 1을 더해줌
    
    ## 패딩을 위해 인덱스 1부터 시작 (unknown도 padding으로 취급)
    ids = [idx + 1 for idx in ids] 
    
    ## 패딩 및 자르기 (Truncation)
    if len(ids) < max_len:
        ## 짧으면 0으로 채움 (Padding)
        ids = ids + [0] * (max_len - len(ids))
    else:
        ## 길면 자름 (Truncation)
        ids = ids[:max_len]
        
    return ids


if __name__ == "__main__":
    file_path = "../combined_data_NLP.xlsx"
    df = pd.read_excel(file_path)
    df = df.dropna(subset = ["answer"])

    df["tokens"] = df["answer"].apply(preprocess_text)
    tokens_list = df["tokens"].tolist()

    ## max token length 설정 (For Truncation)
    token_len = df["tokens"].map(lambda x : len(x)).tolist()
    token_len.sort()
    max_len = token_len[round(len(token_len)*0.95)]
    print(f"max token length(truncated): {max_len}")

    dictionary = Dictionary(tokens_list)
    dictionary.save("vocab.dict")
    X_sequences = [tokens_to_sequence(t, dictionary, max_len) for t in tokens_list]

    metadata = {
        "vocab_size": len(dictionary) + 1,
        "max_len": max_len
    }

    with open("metadata.json", "w") as f:
        json.dump(metadata, f)

    pd.concat([df[["model"]], pd.DataFrame(X_sequences)], axis = 1).to_csv("llm_embedded_token_vector.csv", index = False)
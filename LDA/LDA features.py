import pandas as pd
import numpy as np
import re
import string
import nltk
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from kneed import KneeLocator
from tqdm import tqdm

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

def extract_english_tokens(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text) ## 구두점 전처리
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words and len(t) > 2]
    return tokens

def select_optimal_topic(coherence_scores, topic_range):
    knee = KneeLocator(topic_range, coherence_scores, curve = "concave", direction = "decreasing")
    
    if knee.knee is not None:
        print(f"KneeLocator detected optimal topic number: {knee.knee}")
        return knee.knee
    
    best_topic = topic_range[np.argmax(coherence_scores)]
    print(f"Using highest coherence: {best_topic}")
    return best_topic

def main():
    file_path = "combined_data_NLP.xlsx"
    df = pd.read_excel(file_path)
    df = df.dropna(subset = ["answer"])
    df["text"] = df["answer"]

    ## 불용어 및 토크나이즈 설정
    df["tokens"] = df["text"].apply(extract_english_tokens)
    tokens_list = df["tokens"].tolist()

    dictionary = Dictionary(tokens_list)
    corpus = [dictionary.doc2bow(tokens) for tokens in tokens_list]

    ## 최적 토픽 수 계산
    topic_range = range(2, 6)
    coherence_scores = []

    print("-----토픽 개수 산정을 위한 coherence score 계산 중-----")

    for n in topic_range:
        lda_model = LdaModel(corpus = corpus, id2word = dictionary, num_topics = n, passes = 10, random_state = 42, alpha = "auto")
        cm = CoherenceModel(model = lda_model, texts = tokens_list, dictionary = dictionary, coherence = "c_v")
        coherence_scores.append(cm.get_coherence())
        print(f"Topic {n} - Coherence: {coherence_scores[-1]:.4f}")

    optimal_topics = select_optimal_topic(coherence_scores, topic_range)
    print(f"최적 토픽 수: {optimal_topics}")

    ## 메인 토픽 모델 학습
    main_model = LdaModel(corpus = corpus, id2word = dictionary, num_topics = optimal_topics,
                          passes = 50,
                          random_state = 42,
                          update_every = 1,
                          eval_every = 1,
                          alpha = "auto")
    
    ## 서브토픽 분석 : 생략?
    ##------------------------------------------##
    main_model[corpus]

    ## 결과 저장 : 일단 반환만...
    return main_model, corpus


main_model, corpus = main()
main_model[corpus]
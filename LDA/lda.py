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
    
    ## 서브토픽 분석
    subtopic_records = []
    subtopic_labels = []
    subtopic_words = []

    ## 각 최적의 토픽별로 서브토픽 추출
    for topic_id in range(optimal_topics):
        print(f"\n[Subtopic Analysis for Topic {topic_id}]")
        topic_docs = []

        ## 메인 토픽과 관련된 문서 추출
        for i, dist in enumerate(main_model[corpus]):
            for t_id, prob in dist:
                if t_id == topic_id:
                    topic_docs.append((i, prob))

        topic_docs = sorted(topic_docs, key=lambda x: x[1], reverse=True)[:20]  ## 토픽을 대표하는 상위 20개 문서 추출
        doc_indices = [idx for idx, _ in topic_docs]
        filtered_texts = [tokens_list[i] for i in doc_indices]

        ## 메인 토픽에 해당하는 문서가 하나도 없으면 다음 반복문으로 이동... 방어용 코드
        if len(filtered_texts) == 0:
            print(f"⚠️ Topic {topic_id} has no representative documents.")
            continue

        sub_dict = Dictionary(filtered_texts)
        sub_corpus = [sub_dict.doc2bow(text) for text in tokens_list]  ## 전체 문서를 sub_dict 공간으로 사영?

        # 최적 서브토픽 수 선택
        scores = []
        for n in range(2, 10):
            try:
                temp_model = LdaModel(corpus=[sub_dict.doc2bow(t) for t in filtered_texts], id2word=sub_dict, num_topics=n, passes=15, random_state=42)
                cm = CoherenceModel(model=temp_model, texts=filtered_texts, dictionary=sub_dict, coherence='c_v')
                scores.append(cm.get_coherence())
            except:
                scores.append(0)
        best_n = range(2, 10)[np.argmax(scores)]

        # 서브모델 학습
        sub_model = LdaModel(corpus=sub_corpus, id2word=sub_dict, num_topics=best_n,
                             passes=50,
                             random_state=42,
                             update_every=1,
                             eval_every=1,
                             alpha='auto')

        # 키워드 저장
        for t_id in range(best_n):
            terms = sub_model.show_topic(t_id, topn=20)
            keywords = ", ".join([f"{word} ({weight:.3f})" for word, weight in terms])
            subtopic_words.append({
                "Model": f"LDA {topic_id}",
                "Subtopic": f"{topic_id}-{t_id+1}",
                "Keywords": keywords
            })

        # 전체 문서에 대한 분포 계산
        for i, doc in enumerate(tokens_list):
            bow = sub_dict.doc2bow(doc)
            dist = sub_model.get_document_topics(bow, minimum_probability=0) #해당 문서가 서브토픽들에 대해 가지는 확률 분포
            record = {'doc_id': i}
            for t_id, prob in dist:
                label = f"{topic_id}-{t_id+1}"
                record[label] = prob
                if label not in subtopic_labels:
                    subtopic_labels.append(label)
            subtopic_records.append(record)

    # 5. 결과 저장
    result_df = pd.DataFrame(subtopic_records)
    result_df = result_df.groupby("doc_id").mean().reset_index()
    result_df["llm_name"] = df["model"]

    # 누락된 서브토픽 컬럼 0으로 채우기
    for col in subtopic_labels:
        if col not in result_df.columns:
            result_df[col] = 0.0

    result_df.to_excel("llm_subtopic_distribution_v2.xlsx", index=False)
    print("저장 완료: llm_subtopic_distribution.xlsx")

    words_df = pd.DataFrame(subtopic_words)
    words_df.to_excel("subtopic_keywords_v2.xlsx", index=False)
    print("저장 완료: subtopic_keywords.xlsx")

# 6. 실행 시작
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
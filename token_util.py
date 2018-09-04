import jieba
import time
import multiprocessing
import pandas as pd
import threading

def tokenzie(lines, preprocessed_documents):
    for line in lines:
        results = list(jieba.cut(line))
        preprocessed_documents.append(results)

def perform_tokenzie(documents, parallel=False):
    threads = []
    preprocessed_documents = []

    start = time.time()
    if parallel:
        jieba.enable_parallel(2)
        t1 = threading.Thread(target=tokenzie, args=(documents[0:len(documents)//2], preprocessed_documents))
        t2 = threading.Thread(target=tokenzie, args=(documents[len(documents)//2:], preprocessed_documents))
        threads.append(t1)
        threads.append(t2)
        t1.start()
        t2.start()
    for t in threads:
        t.join()
    print('parallel:%s, time elapsed:%f second' % (parallel, time.time() - start))
    return preprocessed_documents

def thread_execute():
    jieba.set_dictionary("data/jieba_dict/dict.txt.big")
    jieba.initialize()
    raw_data = pd.read_csv("data/big_data/3c_comment_0.csv", usecols=["POST_CONTENT"])
    raw_data = raw_data.dropna()
    raw_data = raw_data["POST_CONTENT"].values.tolist()
    corpus = perform_tokenzie(raw_data[:1000], parallel=True)
    print(corpus)

if __name__ == "__main__":
    thread_execute()

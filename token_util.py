import jieba
import time
import threading
import pandas as pd

class SegmentThread(threading.Thread):

    def __init__(self, lines, preprocessed_documents):
        threading.Thread.__init__(self)
        self.lines = lines
        self.preprocessed_documents = preprocessed_documents

    def run(self):
        for line in self.lines:
            results = jieba.cut(line)
            self.preprocessed_documents.append(list(results))

def test_segment(documents, parallel=False):
    threads = []
    preprocessed_document = []
    start = time.time()
    if parallel:
        jieba.enable_parallel(2)
        t1 = SegmentThread(documents[0:len(documents)//2], preprocessed_document)
        t2 = SegmentThread(documents[len(documents)//2:], preprocessed_document)
        threads.append(t1)
        threads.append(t2)
        t1.start()
        t2.start()
    else:
        t1 = SegmentThread(documents, preprocessed_document)
        t1.start()
        threads.append(t1)
    for t in threads:
        t.join()
    print('parallel:%s, time elapsed:%f second' % (parallel, time.time() - start))
    return preprocessed_document

def thread_execute():
    jieba.set_dictionary("data/jieba_dict/dict.txt.big")
    jieba.initialize()
    raw_data = pd.read_csv("data/big_data/3c_comment_0.csv", usecols=["POST_CONTENT"])
    raw_data = raw_data.dropna()
    raw_data = raw_data["POST_CONTENT"].values.tolist()
    test_segment(raw_data, parallel=True)

if __name__ == "__main__":
    thread_execute()

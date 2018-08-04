import logging
import math

logging.basicConfig(level=logging.INFO)

class TFIDFService:
    def __init__(self):
        self.documents = None

    def set_documents(self, documents):
        self.documents = documents

    def _document_to_freq_dict(self, document):
        return dict([(word, document.count(word)) for word in document])
    
    def _count_doc_freq_by_word(self, word):
        doc_count = 0
        for document in self.documents:
            if word in document:
                doc_count += 1
        return doc_count
    
    def compute_tf_idf(self, word, document):
        freq_dict = self._document_to_freq_dict(document)
        logging.info("freq_dict: {}".format(freq_dict))
        tf = freq_dict.get(word, 0) / sum(freq_dict.values())
        logging.info("tf: {}".format(tf))
        idf = math.log10(len(self.documents) / (1 + self._count_doc_freq_by_word(word)))
        logging.info("idf: {}".format(idf))
        return tf * idf

def tokenize(documents):
    documents = [document.split(" ") for document in documents]
    return documents

def main():
    documents = [
        "I want to adept the dog",
        "a apple a day keeps doctor away",
        "I have a pen I have an apple",
        "who is your daddy",
        "daddy daddy daddy daddy daddy"
    ]
    # 分詞前處理
    documents = tokenize(documents)

    tf_idf = TFIDFService()
    tf_idf.set_documents(documents)
    for index, doc in enumerate(documents, 1):
        tf_idf_value = tf_idf.compute_tf_idf("daddy", doc)
        logging.info("document{}'s tf-idf: {}".format(index, tf_idf_value))

if __name__ == "__main__":
    main()

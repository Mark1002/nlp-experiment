import time
import logging
import pandas as pd
import jieba
import jieba.posseg as pseg
import re
from gensim.models import word2vec

logging.basicConfig(level=logging.INFO)

class SimilarTagExperiment:
    def __init__(self):
        self.documents = []
        # 分詞與詞性對應
        self.part_speech_mapping = {}
        self.model = None

    def load_data(self, file_path):
        self.documents = pd.read_csv(file_path, usecols=["POST_CONTENT"])
    
    def get_documents(self):
        return self.documents
    
    def get_part_speech_mapping(self):
        return self.part_speech_mapping

    def preprocess_data(self):
        self.documents = self.documents.dropna()
        self.documents = self.documents["POST_CONTENT"].values.tolist()
        # 用來存放分詞後的結果
        preprocessed_documents = []
        # stopword
        with open("data/jieba_dict/stopwords.txt") as stop_words:
            stop_word_list = [stop_word.strip() for stop_word in stop_words]
        # 支援繁體中文較好的詞庫
        jieba.set_dictionary("data/jieba_dict/dict.txt.big")

        try:
            for index, document in enumerate(self.documents, 0):
                # 只取中文
                document = "".join(re.findall(r"[\u4e00-\u9fa5]+", document))
                if index % 200 == 0:
                    logging.info("current document index:{}".format(index))
                part_speech_list = list(pseg.cut(document))
                # 去除保留字
                part_speech_list = list(filter(lambda x: x.word not in stop_word_list, part_speech_list))
                # 篩選字詞
                part_speech_list = self.filter_part_speech(['n', 'x', 'n', 'ng', 'nr', 'ns'], part_speech_list)
                preprocessed_document = []
                for part_speech in part_speech_list:
                    preprocessed_document.append(part_speech.word)
                    self.part_speech_mapping[part_speech.word] = part_speech.flag
                preprocessed_documents.append(preprocessed_document)
            self.documents = preprocessed_documents
        except Exception as e:
            logging.info("{}, index {}".format(str(e), index))
                     
    def train_word_to_vec(self):
        self.model = word2vec.Word2Vec(self.documents, min_count=3, window=20, sg=1)
    
    # 指定濾掉的詞性，並過濾掉其他詞
    def filter_part_speech(self, pos_list, part_speech_list):
        return list(filter(lambda x: x.flag not in pos_list, part_speech_list))
    
    def get_query_expansion(self, key_terms):
        return self.model.wv.most_similar(key_terms, topn=10)

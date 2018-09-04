import logging
import time
from get_similar_tag import SimilarTagExperiment

logging.basicConfig(level=logging.INFO)

def main():
    similar_tag = SimilarTagExperiment()
    # load data
    similar_tag.load_data("data/big_data/3c_comment_0.csv")
    # 前處理
    start_time = time.time()
    similar_tag.preprocess_data()
    end_time = time.time()
    logging.info("preprocessed time: {}".format(end_time - start_time))
    logging.info(len(similar_tag.get_documents()))
    # 訓練模型
    similar_tag.train_word_to_vec()
    # query expansion
    logging.info(similar_tag.get_query_expansion("划算"))

if __name__ == "__main__":
    main()

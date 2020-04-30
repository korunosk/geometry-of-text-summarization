
from gensim.parsing.preprocessing import *

from encoders import *
from config import *
from helpers import *


BASE_DATA_DIR = '/scratch/korunosk/data'
EMBEDDINGS_DIR = os.path.join(BASE_DATA_DIR, 'LSA')


if __name__ == '__main__':
    vocab, embs = load_embeddings(os.path.join(EMBEDDINGS_DIR, 'tac-100d'))
    
    # LSA
    sentences = read_sentences(BASE_DATA_DIR, DATASET_IDS, TOPIC_IDS)
    vectorizer = make_vectorizer(sentences)

    encode = lambda documents: encode_bigrams_lsa(documents, vocab, embs, vectorizer=vectorizer)
    
    # # fasttext or GloVe
    # filters = [
    #     lambda s: s.lower(),
    #     strip_punctuation,
    #     strip_multiple_whitespaces,
    #     remove_stopwords,
    # ]

    # encode = lambda documents: encode_words_glove_fasttext(documents, vocab, embs, filters=filters)

    for dataset_id in DATASET_IDS:
        print(dataset_id)
        dataset = load_data(BASE_DATA_DIR, dataset_id, encoded=False)
        
        for topic_id in TOPIC_IDS[dataset_id]:
            print('   {}'.format(topic_id))

            topic = make_topic(dataset[topic_id], encode)
            store_data(os.path.join(EMBEDDINGS_DIR, dataset_id), topic_id, topic)

# Main script for encodding the topics data. We only use it
# for LSA, fasttext and GloVE. We embed the data with BERT using
# a Google Colab notebook that needs faster machine to complete
# in reasonable times.
#
# We need the data on disk for faster execution of the scripts.

from gensim.parsing.preprocessing import *

from encoders import *
from config import *
from helpers import *


# Globals
BASE_DATA_DIR = '/scratch/korunosk/data'
# Change the embedding method as a second parameter
# to this function call.
EMBEDDINGS_DIR = os.path.join(BASE_DATA_DIR, 'LSA')


if __name__ == '__main__':
    # Load the vocabulary and embeddings of the given method
    vocab, embs = load_embeddings(os.path.join(EMBEDDINGS_DIR, 'tac-100d'))
    
    # *** LSA ***
    # For LSA we need to read all the sentences from the topics
    # and create the vectorizer.
    sentences = read_sentences(BASE_DATA_DIR, DATASET_IDS, TOPIC_IDS)
    vectorizer = make_vectorizer(sentences)

    # Per sentence encoder
    encode = lambda documents: encode_bigrams_lsa(documents, vocab, embs, vectorizer=vectorizer)
    
    # #  *** fasttext or GloVe ***
    # # For fasttext or GloVe we need a simple preprocessing pipeline
    # filters = [
    #     lambda s: s.lower(),
    #     strip_punctuation,
    #     strip_multiple_whitespaces,
    #     remove_stopwords,
    # ]

    # # Per sentence encoder
    # encode = lambda documents: encode_words_glove_fasttext(documents, vocab, embs, filters=filters)

    # Iterate over the datasets
    for dataset_id in DATASET_IDS:
        print(dataset_id)

        # Load the data
        dataset = load_data(BASE_DATA_DIR, dataset_id, encoded=False)
        
        for topic_id in TOPIC_IDS[dataset_id]:
            print('   {}'.format(topic_id))

            # Embed the data and store it back to disk.
            topic = make_topic(dataset[topic_id], encode)
            store_data(os.path.join(EMBEDDINGS_DIR, dataset_id), topic_id, topic)
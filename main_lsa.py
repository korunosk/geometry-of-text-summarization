import os
import numpy as np
from sklearn.utils.extmath import randomized_svd

from config import *
from helpers import read_sentences, make_vectorizer


BASE_DATA_DIR = '/scratch/korunosk/data'
EMBEDDINGS_DIR = os.path.join(BASE_DATA_DIR, 'LSA')


if __name__ == '__main__':
    sentences = read_sentences(BASE_DATA_DIR, DATASET_IDS, TOPIC_IDS)
    vectorizer = make_vectorizer(sentences)
    X = vectorizer.transform(sentences)

    U, Sigma, VT = randomized_svd(X, n_components=300, random_state=42)

    V = VT.T
    bigrams = vectorizer.get_feature_names()

    print(V.shape, len(bigrams))

    with open(os.path.join(EMBEDDINGS_DIR, f'tac-100d.npy'), mode='wb') as fp:
        np.save(fp, V)

    with open(os.path.join(EMBEDDINGS_DIR, f'tac-100d.vocab'), mode='w') as fp:
        fp.write('\n'.join(bigrams))
        
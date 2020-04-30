import os
import json
import orjson
import numpy as np
from operator import itemgetter
from itertools import chain
import time
import datetime

from sklearn.feature_extraction.text import CountVectorizer


def format_time(elapsed):
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def load_embeddings(embeddings_path):
    with open(embeddings_path + '.vocab', mode='r') as fp:
        vocab = { line.strip(): i for i, line in enumerate(fp.readlines()) }
 
    embs = np.load(embeddings_path + '.npy', allow_pickle=True)
    
    return vocab, embs


def load_data(data_dir, fname, encoded):
    ext = '_encoded.json' if encoded else '.json'
    with open(os.path.join(data_dir, fname + ext), mode='r') as fp:
        return orjson.loads(fp.read())


def store_data(data_dir, fname, data):
    with open(os.path.join(data_dir, fname + '_encoded.json'), mode='w') as fp:
        json.dump(data, fp, indent=4)


def make_annotations(summ_ids, pyr_scores, embeddings):
    # Keep the format for easier parsing of different files
    return [ {
        'summ_id': summ_id,
        'pyr_score': pyr_score,
        'text': embedding
    } for summ_id, pyr_score, embedding in zip(summ_ids, pyr_scores, embeddings) ]


def make_topic(topic, encode):
    documents = topic['documents']
    annotations = topic['annotations']

    summary_ids = list(map(itemgetter('summ_id'), annotations))
    pyr_scores = list(map(itemgetter('pyr_score'), annotations))
    summaries = list(map(itemgetter('text'), annotations))

    return {
        'documents': encode(documents),
        'annotations': make_annotations(summary_ids, pyr_scores, encode(summaries))
    }


def make_tac(data, encode):
    tac = {}
    for topic_id, topic in data.items():
        print('   {}'.format(topic_id))
        tac[topic_id] = make_topic(topic, encode)
    return tac


def extract(topic):
    documents = list(chain(*topic['documents']))
    annotations = topic['annotations']
    
    summary_ids = [annotations[0]['summ_id']]
    pyr_scores = [annotations[0]['pyr_score']]
    summaries = annotations[0]['text']
    indices = [[0, len(summaries)]]
    
    for o in annotations[1:]:
        summary_ids.append(o['summ_id'])
        pyr_scores.append(o['pyr_score'])
        summaries.extend(o['text'])
        start = indices[-1][1]
        indices.append([start, start + len(o['text'])])
    
    return documents, summaries, indices, pyr_scores, summary_ids


def read_sentences(data_dir, dataset_ids, topic_ids) -> list:
    ''' Reads all the sentences from all topics and
    returns a list of them.
    
    :return: Sentences
    '''
    sentences = []

    for dataset_id in dataset_ids:
        print(dataset_id)
        dataset = load_data(data_dir, dataset_id, encoded=False)

        for topic_id in topic_ids[dataset_id]:
            documents, _, _, _, _ = extract(dataset[topic_id])

            sentences.extend(documents)
    
    return sentences


def make_vectorizer(sentences: list) -> CountVectorizer:
    ''' Factory method for generating count vectorizer.
    The same initialization is used in several places.
    
    :param sentences: Sentences to vectorize
    
    :return: Count vectorizer
    '''
    from gensim.parsing.preprocessing import STOPWORDS
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(2,2), stop_words=list(STOPWORDS))
    vectorizer.fit(sentences)
    return vectorizer

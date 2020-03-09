import os
import json
import numpy as np
from operator import itemgetter
from itertools import chain


def load_data(data_dir, fname, encoded):
    ext = '_encoded.json' if encoded else '.json'
    with open(os.path.join(data_dir, fname + ext), mode='r') as fp:
        return json.load(fp)


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
        print('  ', topic_id)
        tac[topic_id] = make_topic(topic, encode)
    
    return tac


def extract(topic):
    documents = np.array(list(chain(*topic['documents'])))
    annotations = topic['annotations']

    summaries_tmp = list(map(itemgetter('text'), annotations))
    indices_tmp = np.cumsum([0] + list(map(len, summaries_tmp)))
    summaries = np.array(list(chain(*summaries_tmp)))
    indices = np.array(list(zip(indices_tmp[:-1], indices_tmp[1:])))
    
    pyr_scores = np.array(list(map(itemgetter('pyr_score'), annotations)))
    summary_ids = np.array(list(map(itemgetter('summ_id'), annotations)))
    
    return documents, summaries, indices, pyr_scores, summary_ids

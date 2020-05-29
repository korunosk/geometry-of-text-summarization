# Contains relevant preprocessing functions

import numpy as np
from sklearn.model_selection import train_test_split


def stratified_sampling(data, test_size=0.3):
    ''' Splits the data into train and test set using
    the topic ID for stratification. '''
    train, test = train_test_split(data, test_size=test_size, random_state=42, stratify=data[:,0])
    # Sort the data by topic_id since we will load each topic separatelly
    train = train[train[:,0].argsort(kind='mergesort')]
    test = test[test[:,0].argsort(kind='mergesort')]
    return train, test


def leave_n_out(data, test_size=0.3):
    ''' Leave N topics out for train (the last ones) '''
    topics = np.unique(data[:,0]) # sorts the data
    n = int(test_size * len(topics))
    train_topics = topics[:-n]
    test_topics = topics[-n:]
    train = data[np.isin(data[:,0], train_topics)]
    test = data[np.isin(data[:,0], test_topics)]
    return train, test
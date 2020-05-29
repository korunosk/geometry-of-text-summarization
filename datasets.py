# Contains all the Pytorch datasets required for training.
# Some of these classes will be overriden by notebooks to
# support batching or different loading modes.
#
# Initially the datasets loaded the whole topic from disk,
# and cached them in memory. The loading process takes time.
# We wanted to support random topic loading. For that, we
# splitted topic data and serialized each part on disk as
# numpy arrays. With this, we can load each part spearatelly.

import os
import numpy as np
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import preprocess_documents
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import normalize

from helpers import load_data, extract


class TACDataset(Dataset):
    ''' Base Dataset class '''

    def _load_topic(self, topic_id):
        ''' Loads topic data given the topic ID '''

        # Check if the topic is already loaded
        if self.topic_id == topic_id:
            return
        
        # Otherwise, load the topic
        self.topic_id = topic_id
        
        # Here we check if we want the data encodded or not
        if self.encoded:
            topic = load_data(os.path.join(self.base_data_dir, self.dataset_id), topic_id, encoded=True)
        else:
            topic = load_data(self.base_data_dir, self.dataset_id, encoded=False)[self.topic_id]
        
        # Extract the topic data
        (self.documents,
         self.summaries,
         self.indices,
         self.pyr_scores,
         self.summary_ids) = extract(topic)
        
        print(f'Loaded data from topic {topic_id}')
    
    def _load_item(self, topic_id, item):
        ''' Loads part of the topic data, already stored on disk.
        If the topic data is large and takes time to load, we will use
        this method to load serialized matrices from disk.
        '''
        self.topic_id = topic_id
        directory = os.path.join(self.embeddings_dir, self.dataset_id, self.topic_id, item)
        return np.load(directory)

    def __init__(self, base_data_dir, embeddings_dir, dataset_id, data, encoded):
        ''' Constructor.
        
        Expects data parameter for regression or classification.
        Based on the elements of this data array, it will load the
        appropriate matrices accordingly.
        '''
        self.base_data_dir = base_data_dir
        self.embeddings_dir = embeddings_dir
        self.dataset_id = dataset_id
        self.data = data
        self.encoded = encoded
        self.topic_id = ''        
        
    def __len__(self):
        ''' Returns data length '''
        return len(self.data)


class TACDatasetRegression(TACDataset):
    ''' Extends Dataset base class for regression '''

    def __init__(self, base_data_dir, embeddings_dir, dataset_id, data):
        ''' Constructor '''
        super().__init__(base_data_dir, embeddings_dir, dataset_id, data, encoded=True)

        # Shuffle the data.
        np.random.shuffle(data)

#     def __getitem__(self, idx):
#         ''' Loads data sequentially. '''
#         self._load_topic(self.data[idx][0])
        
#         i = self.indices[int(self.data[idx][1])]
#         x = (self.documents,
#              self.summaries[i[0]:i[1]]),
#         y = float(self.data[idx][3])
        
#         return (x, y)

    def __getitem__(self, idx):
        ''' Loads data randomly. '''
        topic_id = self.data[idx][0]
        
        i = int(self.data[idx][1])
        x = (self._load_item(topic_id, 'document_embs.npy'),
             self._load_item(topic_id, f'summary_{i}_embs.npy'))
        y = float(self.data[idx][2])
        
        return (x, y)


class TACDatasetClassification(TACDataset):
    ''' Extends Dataset base class for classification '''

    def __init__(self, base_data_dir, embeddings_dir, dataset_id, data):
        super().__init__(base_data_dir, embeddings_dir, dataset_id, data, encoded=True)

        # Shuffle the data
        np.random.shuffle(data)
    
    def __getitem__(self, idx):
        ''' Loads data randomly. '''
        topic_id = self.data[idx][0]
        
        i1 = int(self.data[idx][1])
        i2 = int(self.data[idx][2])
        x = (self._load_item(topic_id, 'document_embs.npy'),
             self._load_item(topic_id, f'summary_{i1}_embs.npy'),
             self._load_item(topic_id, f'summary_{i2}_embs.npy'))
        y = float(self.data[idx][3])
        
        return (x, y)


class Normalize():
    ''' Normalizes embeddings to unit length '''
    def __call__(self, sample):
        x, y = sample
        return (tuple(normalize(x_i, axis=1) for x_i in x), y)


class ToTensor():
    ''' Converts numpy arrays to Pytorch tensors '''
    def __call__(self, sample):
        x, y = sample
        return (tuple(torch.tensor(x_i, dtype=torch.float) for x_i in x), torch.tensor(y, dtype=torch.float))

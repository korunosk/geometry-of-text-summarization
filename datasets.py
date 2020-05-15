import os
import numpy as np
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import preprocess_documents
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import normalize

from helpers import load_data, extract


class TACDataset(Dataset):
    def _load_topic(self, topic_id):
        if self.topic_id == topic_id:
            return
        
        self.topic_id = topic_id
        
        if self.encoded:
            topic = load_data(os.path.join(self.base_data_dir, self.dataset_id), topic_id, encoded=True)
        else:
            topic = load_data(self.base_data_dir, self.dataset_id, encoded=False)[self.topic_id]
            
        (self.documents,
         self.summaries,
         self.indices,
         self.pyr_scores,
         self.summary_ids) = extract(topic)
        
        print(f'Loaded data from topic {topic_id}')
    
    def _load_item(self, topic_id, item):
        self.topic_id = topic_id
        directory = os.path.join(self.embeddings_dir, self.dataset_id, self.topic_id, item)
        return np.load(directory)

    def __init__(self, base_data_dir, embeddings_dir, dataset_id, data, encoded):
        self.base_data_dir = base_data_dir
        self.embeddings_dir = embeddings_dir
        self.dataset_id = dataset_id
        self.data = data
        self.encoded = encoded
        self.topic_id = ''        
        
    def __len__(self):
        return len(self.data)


class TACDatasetRegression(TACDataset):
    def __init__(self, base_data_dir, embeddings_dir, dataset_id, data):
        super().__init__(base_data_dir, embeddings_dir, dataset_id, data, encoded=True)

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
    def __init__(self, base_data_dir, embeddings_dir, dataset_id, data):
        super().__init__(base_data_dir, embeddings_dir, dataset_id, data, encoded=True)

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
    def __call__(self, sample):
        x, y = sample
        return (tuple(normalize(x_i, axis=1) for x_i in x), y)


class ToTensor():
    def __call__(self, sample):
        x, y = sample
        return (tuple(torch.tensor(x_i, dtype=torch.float) for x_i in x), torch.tensor(y, dtype=torch.float))

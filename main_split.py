import os
from config import *
from helpers import *


BASE_DATA_DIR = '/scratch/korunosk/data'
EMBEDDINGS_DIR = os.path.join(BASE_DATA_DIR, 'BERT_sent')


if __name__ == '__main__':
    for dataset_id in DATASET_IDS:
        print(dataset_id)
        
        dataset = load_data(BASE_DATA_DIR, dataset_id, encoded=False)
        
        for topic_id in TOPIC_IDS[dataset_id]:
            print('   {}'.format(topic_id))
            
            topic = load_data(os.path.join(EMBEDDINGS_DIR, dataset_id), topic_id, encoded=True)
            document_embs, summary_embs, indices, pyr_scores, summary_ids = extract(topic)

            directory = os.path.join(EMBEDDINGS_DIR, dataset_id, topic_id)

            if not os.path.exists(directory):
                os.makedirs(directory)

            np.save(os.path.join(directory, f'document_embs.npy'), document_embs)

            for i, idx in enumerate(indices):
                np.save(os.path.join(directory, f'summary_{i}_embs.npy'), summary_embs[idx[0]:idx[1]])
            
            documents, summaries, indices, pyr_scores, summary_ids = extract(dataset[topic_id])
            
            np.save(os.path.join(directory, f'documents.npy'), documents)

            for i, idx in enumerate(indices):
                np.save(os.path.join(directory, f'summary_{i}.npy'), summaries[idx[0]:idx[1]])

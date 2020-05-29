# This script splits topic's data and serializes it to disk.
# If we need random access to different parts of the topic's
# data, we have to load each topic constantly which takes time.
# This way, we only load the relevant part of each topic.

import os
from config import *
from helpers import *


# Globals
BASE_DATA_DIR = '/scratch/korunosk/data'
EMBEDDINGS_DIR = os.path.join(BASE_DATA_DIR, 'BERT_sent')


if __name__ == '__main__':
    # Iterate over each dataset
    for dataset_id in DATASET_IDS:
        print(dataset_id)
        
        # Load the original textual data
        dataset = load_data(BASE_DATA_DIR, dataset_id, encoded=False)
        
        for topic_id in TOPIC_IDS[dataset_id]:
            print('   {}'.format(topic_id))
            
            # Load and extract the embedded data
            topic = load_data(os.path.join(EMBEDDINGS_DIR, dataset_id), topic_id, encoded=True)
            document_embs, summary_embs, indices, pyr_scores, summary_ids = extract(topic)

            # Get the directory handler
            directory = os.path.join(EMBEDDINGS_DIR, dataset_id, topic_id)

            # If the directory doesn't exist, create it
            if not os.path.exists(directory):
                os.makedirs(directory)

            # Save the document embeddings
            np.save(os.path.join(directory, f'document_embs.npy'), document_embs)

            # Save the summary embeddings
            for i, idx in enumerate(indices):
                np.save(os.path.join(directory, f'summary_{i}_embs.npy'), summary_embs[idx[0]:idx[1]])
            
            # Extract the textual data
            documents, summaries, indices, pyr_scores, summary_ids = extract(dataset[topic_id])
            
            # Save the documents
            np.save(os.path.join(directory, f'documents.npy'), documents)

            # Save the summaries
            for i, idx in enumerate(indices):
                np.save(os.path.join(directory, f'summary_{i}.npy'), summaries[idx[0]:idx[1]])

import os
import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau
from src.lexrank import degree_centrality_scores
import ray # Interferes with linalg.det causing semantic_spread to output different results

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks')

from typing import Callable

from config import *
from helpers import *
from visualization import *
from redundancy import *
from relevance import *


class MetricsExperimentExecutor():
    ''' This class serves as experiment executor for the proposed metrics. '''
    
    def __init__(self, base_data_dir: str, dataset_id: str, topic_ids: list, embedding_met: str):
        ''' Constructor.
        
        The data directory will be generated as:
        
            base_data_dir + embedding_met + dataset_id
        
        :param base_data_dir: Base data directory
        :param dataset_id:    Dataset ID (TAC2008 or TAC2009)
        :param topic_ids:     Topic IDs for the particular dataset
        :param embedding_met: Embedding method
        '''
        self.dataset_id    = dataset_id
        self.embedding_met = embedding_met
        self.data_dir      = os.path.join(base_data_dir, self.embedding_met, self.dataset_id)
        self.topic_ids     = topic_ids
        
        # Define list of experiments to execute.
        # Every entry needs to contain a label - the experiment name,
        # and a procedure - the method that will be executed.
        self.experiments = [{
                'label': 'Average Pairwise Distance',
                'procedure': self.experiment_average_pairwise_distance 
            }, {
                'label': 'Semantic Volume',
                'procedure': self.experiment_semantic_volume 
            }, {
                'label': 'Semantic Spread',
                'procedure': self.experiment_semantic_spread 
            }, {
                'label': 'Word Mover Distance',
                'procedure': self.experiment_word_mover_distance 
            }, {
                'label': 'LexRank',
                'procedure': self.experiment_lex_rank 
            }]
    
    @staticmethod
    @ray.remote
    def load_and_extract_topic(data_dir: str, topic_id: str) -> tuple:
        ''' Encapsulates the loading and extracting procedure.
        
        :param data_dir: Data directory
        :param topic_id: Topic ID
        
        :return: Tuple as recieved from extract()
        '''
        topic = load_data(data_dir, topic_id, encoded=True)
        return extract(topic)
    
    @staticmethod
    @ray.remote
    def experiment_average_pairwise_distance(data: tuple) -> float:
        document_embs, summary_embs, indices, pyr_scores, summary_ids = data
        document_embs = np.array(document_embs)
        metric = lambda i: average_pairwise_distance(np.array(summary_embs[i[0]:i[1]]))
        return kendalltau(pyr_scores, np.array([metric(i) for i in indices]))[0]

    @staticmethod
    @ray.remote
    def experiment_semantic_volume(data: tuple) -> float:
        document_embs, summary_embs, indices, pyr_scores, summary_ids = data
        document_embs = np.array(document_embs)
        document_pts, summary_pts = project_pca(np.concatenate((document_embs, summary_embs)), document_embs.shape[0])
        metric = lambda i: semantic_volume(np.array(summary_pts[i[0]:i[1]]))
        return kendalltau(pyr_scores, np.array([metric(i) for i in indices]))[0]

    @staticmethod
    @ray.remote
    def experiment_semantic_spread(data: tuple) -> float:
        document_embs, summary_embs, indices, pyr_scores, summary_ids = data
        document_embs = np.array(document_embs)
        metric = lambda i: semantic_spread(np.array(summary_embs[i[0]:i[1]]))
        return kendalltau(pyr_scores, np.array([metric(i) for i in indices]))[0]

    @staticmethod
    @ray.remote
    def experiment_word_mover_distance(data: tuple) -> float:
        document_embs, summary_embs, indices, pyr_scores, summary_ids = data
        document_embs = np.array(document_embs)
        metric = lambda i: word_mover_distance(document_embs, np.array(summary_embs[i[0]:i[1]]))
        return kendalltau(pyr_scores, np.array([metric(i) for i in indices]))[0]

    @staticmethod
    @ray.remote
    def experiment_lex_rank(data: tuple) -> float:
        document_embs, summary_embs, indices, pyr_scores, summary_ids = data
        document_embs = np.array(document_embs)
        lr_scores = degree_centrality_scores(cdist(document_embs, document_embs, metric='cosine'))
        metric = lambda i: lex_rank(document_embs, np.array(summary_embs[i[0]:i[1]]), lr_scores)
        return kendalltau(pyr_scores, np.array([metric(i) for i in indices]))[0]
    
    def __execute_experiment(self, experiment: Callable) -> np.array:
        ''' Main method that executes an experiment.
        
        :param experiment: Experiment to execute
        
        :return: Array of values, one per topic
        '''
        # Pass 1: Collect the topics
        dataset = [ self.load_and_extract_topic.remote(self.data_dir, topic_id)
                       for topic_id in self.topic_ids ]
        # Pass 2: Execute the experiment
        scores  = [ experiment.remote(topic)
                       for topic in dataset ]

        return np.array(ray.get(scores))
    
    def __generate_plots(self):
        ''' Utility method for generating plots.
        
        Assumes that first three experiments are redundancy metrics.
        '''
        fig = plt.figure(figsize=(17.5,10))
        # Redundancy
        ax1 = fig.add_subplot(2,1,1)
        plot_corr_coeff(ax1, self.topic_ids, self.experiments[:3])
        ax1.set_xlabel('')
        # Relevance
        ax2 = fig.add_subplot(2,1,2)
        plot_corr_coeff(ax2, self.topic_ids, self.experiments[3:])
        fig.savefig(os.path.join(PLOTS_DIR, f'{self.dataset_id}_{self.embedding_met}.png'), dpi=fig.dpi, bbox_inches='tight')
          
    def execute(self):
        ''' Entry point.
        
        Executes each experiment one by one, prints execution times,
        results, and generates correlation coefficient plots.
        '''
        result = ''
        print(f'=== Experiment "{self.dataset_id}" - Embeddings "{self.embedding_met}" ===\n')

        for i, experiment in enumerate(self.experiments):
            label = experiment['label']
            procedure = experiment['procedure']
            
            print('Executing "{}"\n'.format(label))
            
            start = time.time()
            values = np.nan_to_num(self.__execute_experiment(procedure), nan=0)
            end = time.time()
            
            print('   *** Elapsed: {:}\n'.format(format_time(end - start)))
            
            result += '{:30} {:.4}\n'.format(label, np.mean(values))
            
            self.experiments[i]['values'] = values

        print('\n=== Results ===\n')
        print(result)
        
        self.__generate_plots()

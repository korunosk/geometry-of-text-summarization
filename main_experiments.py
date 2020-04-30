import ray

from experiment_executors import MetricsExperimentExecutor
from config import *


BASE_DATA_DIR = f'/scratch/korunosk/data'


if __name__ == '__main__':
    ex0 = MetricsExperimentExecutor(BASE_DATA_DIR, DATASET_IDS[0], TOPIC_IDS[DATASET_IDS[0]], EMBEDDING_METS[0])
    ex1 = MetricsExperimentExecutor(BASE_DATA_DIR, DATASET_IDS[1], TOPIC_IDS[DATASET_IDS[1]], EMBEDDING_METS[0])
    ex2 = MetricsExperimentExecutor(BASE_DATA_DIR, DATASET_IDS[0], TOPIC_IDS[DATASET_IDS[0]], EMBEDDING_METS[1])
    ex3 = MetricsExperimentExecutor(BASE_DATA_DIR, DATASET_IDS[1], TOPIC_IDS[DATASET_IDS[1]], EMBEDDING_METS[1])
    ex4 = MetricsExperimentExecutor(BASE_DATA_DIR, DATASET_IDS[0], TOPIC_IDS[DATASET_IDS[0]], EMBEDDING_METS[2])
    ex5 = MetricsExperimentExecutor(BASE_DATA_DIR, DATASET_IDS[1], TOPIC_IDS[DATASET_IDS[1]], EMBEDDING_METS[2])
    ex6 = MetricsExperimentExecutor(BASE_DATA_DIR, DATASET_IDS[0], TOPIC_IDS[DATASET_IDS[0]], EMBEDDING_METS[3])
    ex7 = MetricsExperimentExecutor(BASE_DATA_DIR, DATASET_IDS[1], TOPIC_IDS[DATASET_IDS[1]], EMBEDDING_METS[3])
    ex8 = MetricsExperimentExecutor(BASE_DATA_DIR, DATASET_IDS[0], TOPIC_IDS[DATASET_IDS[0]], EMBEDDING_METS[4])
    ex9 = MetricsExperimentExecutor(BASE_DATA_DIR, DATASET_IDS[1], TOPIC_IDS[DATASET_IDS[1]], EMBEDDING_METS[4])

    ray.init(num_cpus=30)
    ex0.execute()
    ray.shutdown()
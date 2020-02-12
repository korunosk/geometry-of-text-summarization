wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
mkdir models
unzip uncased_L-12_H-768_A-12.zip -d models

jupyter nbextension enable --py widgetsnbextension

bert-serving-start -model_dir models/uncased_L-12_H-768_A-12/ -num_worker=4 -max_seq_len=160 -max_batch_size=50 -show_tokens_to_client

tensorboard --logdir runs/
"""

To get the name of the desired BERT model, check the following:
https://github.com/huggingface/transformers/blob/master/src/transformers/configuration_bert.py

"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="data/")
parser.add_argument("--save_path", type=str, default='save/')
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--plot_path", type=str, default='save/plot')
parser.add_argument("--bert_model", type=str, default='bert-base-cased')
parser.add_argument("--gpu", action='store_true')

parser.add_argument("--load_frompretrain", action='store_true')
parser.add_argument("--model_state_path", type=str, default="None")
parser.add_argument("--load_dataset_from_pickle", action='store_true')
parser.add_argument("--model_config_path", type=str, default="None")


parser.add_argument("--anomaly_ratio", type = float, default = 1.0)
parser.add_argument("--val_ratio", type = float, default = 0.2)
parser.add_argument("--test_ratio", type = float, default = 0.2)

# Zero shot setting
parser.add_argument("--zero_shot_data_path", type = str, default = None)

arg = parser.parse_args()
print(arg)

TRAIN_FILE = 'df.tsv'
MAX_SEQ_LENGTH = 100

data_path = arg.data_path
save_path = arg.save_path
lr = arg.lr
batch_size = arg.batch_size
plot_path = arg.plot_path
bert_model = arg.bert_model
load_dataset_from_pickle = arg.load_dataset_from_pickle
epochs = arg.epochs
USE_GPU = arg.gpu
load_frompretrain = arg.load_frompretrain
model_config_path = arg.model_config_path
model_state_path = arg.model_state_path
anomaly_ratio = arg.anomaly_ratio
val_ratio = arg.val_ratio
test_ratio = arg.test_ratio
zero_shot_data_path = arg.zero_shot_data_path

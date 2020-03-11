import numpy as np
import pandas as pd
import os
import pickle
from utils import config
import codecs
from sklearn.model_selection import train_test_split



class BertInputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id, actual_input_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.actual_input_id = actual_input_id


# added by Soroush
def save_obj(obj, name):
    with open('./data/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# added by Soroush
def load_obj(name):
    with open('./data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def to_lower_case(txt):
    return txt.lower()


def load_dataset(path, anomaly_ratio = 1, val_ratio = 0.2, test_ratio = 0.2):
    if not os.path.exists(path + config.TRAIN_FILE):
        raise ValueError('DataSet path does not exist ')
        return

    if config.load_dataset_from_pickle == True:

        train = load_obj('train')
        print(len(train))
        test = load_obj('test')
        print(len(test))
        valid = load_obj('valid')
        print(len(valid))

    else:
        data = pd.read_csv(path + config.TRAIN_FILE, sep='\t')

        data['txt'] = data['txt'].apply(to_lower_case)
        data['labels'] = data['labels'] # label 0 => anomaly, label 1 => normal

        print(len(data))

        # split data intro train and test
        train, test = train_test_split(data, test_size = test_ratio + val_ratio, stratify = data['labels'])
        test, valid = train_test_split(test, test_size = val_ratio, stratify = test['labels'])

        # controlling the ratio of anomalies in the training data
        if anomaly_ratio != 1:
            """
            Test realised: 
            
            import pandas as pd 

            df = pd.DataFrame()
            y = [0 for _ in range(10)] + [1 for _ in range(10)]
            x = ['english' for _ in range(10)] + ['francais' for _ in range(10)]
            
            df['txt'] = x 
            df['labels'] = y 
            
            ratio_anomaly = 0.2
            N_nor = 10 
            N_an = int(N_nor * ratio_anomaly / (1-ratio_anomaly))
            
            df.sort_values(by = 'labels', ascending = True, inplace = True)
            
            df.tail(N_an + N_nor)
            """
            N_nor = train[train.labels == 1].shape[0] # Number of normal observations in training data
            # We know anomaly_ratio = N_anomaly/(N_anomaly + N_normal)
            N_an = int(anomaly_ratio*N_nor/(1-anomaly_ratio))
            # Sort dataframe by values
            # First values are those with labels 0 ie anomalies and then we have normal observations
            train.sort_values(by = 'labels', ascending = True, inplace = True)
            # We only N_an + N_nor observations and doing it that way, we kept correctly N_an
            train = train.tail(N_an + N_nor)

            # shufle again df so that we don't have observations on top that are anomalies
            # and observations at the bottom that are normal
            train = train.sample(frac = 1)

        # test = data.sample(1000)
        # data = data.drop(test.index)
        # valid = data.sample(100)
        # data = data.drop(valid.index)

        print(len(train))
        print(len(test))
        print(len(valid))

        save_obj(train, 'train')
        save_obj(test, 'test')
        save_obj(valid, 'valid')

    return train, test, valid


def convert_examples_to_features(pandas, max_seq_length, tokenizer):
    features = []

    for i, r in pandas.iterrows():

        first_tokens = tokenizer.tokenize(r['txt'])
        if len(first_tokens) > max_seq_length - 2:
            first_tokens = first_tokens[: max_seq_length - 2]
        tokens = ["[CLS]"] + first_tokens + ["[SEP]"]

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        features.append(
            BertInputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=r['labels'],
                actual_input_id=r.index)) # r['id'] should be r.index I think
    return features
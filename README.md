# Multilingual BERT classifier for Supervised Anomaly Detection 

### Introduction 


We used mBERT for multiclass classification on an anomaly detection setting. 

The parameters we use from now are the following. For training:  
 
* --gpu : If you want to use GPU, this will set it to true. 
* --data_path **data/**  : Folder where your data is. Data should be named df.tsv 
* --save_path **save/** :  Folder where your model is saved.  
* --lr **5e-5**  : Learning rate. Recommendations from (this paper)[https://arxiv.org/abs/1810.04805] : learning_rate in { 5e-5, 2e-5, 3e-5 }
* --batch_size **16**. Recommendations from (this paper)[https://arxiv.org/abs/1810.04805] : batch_size in { 16, 32 }
* --epochs **4**  Recommendations from (this paper)[https://arxiv.org/abs/1810.04805] : epochs in {3, 4}
* --plot_path **save/plot/** : Path where plot of training loss is stored. 
* --bert_model **bert-base-multilingual-cased** : Bert model to be used. We used mBERT for Multilingual Book Corpus. 
* --anomaly_ratio **0.005** : Anomaly ratio. In this example, we have 0.5% anomalies in training set. 


For testing, check save/model/ and take notes of which one performs best (**should be done automatically in the future**). 
Then update parameters to get test performance:
* --gpu 
* --load_frompretrain 
* --model_state_path  **/content/multilingual_bert_classification/save/model/BEST_MODEL_PERFORMANCE**
* --model_config_path **/content/multilingual_bert_classification/save/model/BEST_MODEL_PERFORMANCE/config.json**
* --data_path **data/** 
* --save_path **save/** 
* --lr **5e-5** 
* --batch_size **16** 
* --epochs **5** 
* --plot_path **save/plot/**
* --bert_model **bert-base-multilingual-cased**
* --anomaly_ratio **0.005** 

You can do Zero-Shot test as well: 

```bash
python zero_shot_test.py --zero_shot_data_path /content/multilingual_bert_classification/data/df_multilingual.tsv --gpu --load_frompretrain --model_state_path  /content/multilingual_bert_classification/save/model/epoch-0-0.7631693513278189-0.733057051566141-0.047560894953811064  --model_config_path /content/multilingual_bert_classification/save/model/epoch-0-0.7631693513278189-0.733057051566141-0.047560894953811064/config.json --data_path data/ --save_path save/ --lr 5e-5 --batch_size 16 --epochs 5 --plot_path save/plot/ --bert_model bert-base-multilingual-cased --anomaly_ratio 0.10
```
### Description
----------

This code is highly inspired from https://github.com/soroushjavdan/OffensiveBertClassifier/blob/master/utils/config.py 


How to:
-------

#### First
install the requirements
```console
❱❱❱ pip install -r requirements.txt
```
Now we are good to go !!

#### Fine-tuning phase
just run the below command
```console
❱❱❱ python train.py --gpu --data_path data/ --save_path save/ --lr 5e-5 --batch_size 32 --epochs 4 --plot_path save/plot/ --bert_model bert-base-cased
```

#### Try it on Google Colab

In folder examples/mBERT_classification.ipynb, 
one can reproduce the results on Multilingual Book Corpus. The dataset can be found [here](https://github.com/MastafaF/multilingual_book_corpus). 

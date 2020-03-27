# Multilingual BERT classifier for Supervised Anomaly Detection 


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

#### Try it on Google Colab:

In folder examples/mBERT_classification.ipynb, 
one can reproduce the results on Multilingual Book Corpus. The dataset can be found [here](https://github.com/MastafaF/multilingual_book_corpus). 
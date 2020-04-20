import bert_classifier
from utils import config
from sklearn.metrics import classification_report


config.USE_GPU = config.USE_GPU

cm = bert_classifier.ClassificationModel(gpu=config.USE_GPU, seed=0)
if config.load_frompretrain == True:
    cm.load_model(config.model_state_path, config.model_config_path)
else:
    cm.new_model()

actual_to_save, predictions_to_save = cm.create_test_predictions(path=None)

from sklearn.metrics import f1_score

print("BERT classifier, F1 score is {}".format(f1_score(actual_to_save,predictions_to_save,average='macro')))

# Classification report
target_names = ['anomaly', 'normal']
classification_report_df = classification_report(actual_to_save, predictions_to_save, target_names=target_names)
print(classification_report_df)


if __name__ == '__main__':
    print('running')
    config.data_path = 'data/'
    # config.USE_GPU = False
    config.save_path = 'save/'

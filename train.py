import bert_classifier
from utils import config



cm = bert_classifier.ClassificationModel(gpu=config.USE_GPU, seed=0)
if config.load_frompretrain == True:
    cm.load_model(config.model_state_path, config.model_config_path)
else:
    cm.new_model()

# cm.save_model(config.save_path + '/model',config.save_path + '/config')

cm.train(config.epochs, config.batch_size, config.lr, config.plot_path,  config.save_path + '/model',
         config.save_path + 'config')


# cm.create_test_predictions("./pred.csv")
cm.create_test_predictions(path=None)

if __name__ == '__main__':
    print('running')
    if config.USE_GPU:
        print("Using GPU to make things faster :)")
    # config.USE_GPU = False

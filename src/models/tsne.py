import time
import random
from models import utils
from config import cfg
from MulticoreTSNE import MulticoreTSNE as TSNE


class MultiCoreTSNE(object):

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te

    def train(self, parameters):

        tsne = TSNE(**parameters)
        tsne_outputs = tsne.fit_transform(self.x_train)

        utils.save_data_to_pkl(tsne_outputs, cfg.tsne_outputs_path + 'tsne_outputs.p')


def tsne_train(seed):

    x_train, y_train, w_train, e_train, x_test, id_test = utils.load_preprocessed_data(cfg.preprocessed_data_path)

    parameters = {'n_components': 2,
                  'perplexity': 30.0,
                  'early_exaggeration': 12.0,
                  'learning_rate': 200.0,
                  'n_iter': 500,
                  'n_iter_without_progress': 300,
                  'min_grad_norm': 1e-07,
                  'metric': 'euclidean',
                  'init': 'random',
                  'verbose': 1,
                  'random_state': seed,
                  'method': 'barnes_hut',
                  'angle': 0.5,
                  'n_jobs': 8}

    MTSNE = MultiCoreTSNE(x_train, y_train, w_train, e_train, x_test, id_test)

    MTSNE.train(parameters=parameters)

if __name__ == "__main__":

    start_time = time.time()

    utils.check_dir([cfg.tsne_outputs_path])

    # Create Global Seed for Training and Cross Validation
    global_seed = random.randint(0, 300)

    print('======================================================')
    print('Start Training...')
    print('======================================================')

    tsne_train(global_seed)

    print('======================================================')
    print('All Task Done!')
    print('Global Seed: {}'.format(global_seed))
    print('Total Time: {}s'.format(time.time() - start_time))
    print('======================================================')

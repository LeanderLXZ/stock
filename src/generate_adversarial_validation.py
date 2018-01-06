import random
import time
from models import utils
from models.adversarial_validation import AdversarialValidation
from config import cfg


class GenerateValidation(object):

    def __init__(self):
        pass

    @staticmethod
    def train(train_path=None, test_path=None, similarity_prob_path=None,
              load_preprocessed_data=False, gan_preprocess_path=None,
              train_seed=None, global_epochs=1, return_similarity_prob=False):

        if train_seed is None:
            train_seed = random.randint(0, 500)

        parameters = {'learning_rate': 0.001,
                      'epochs': 200,
                      'n_discriminator_units': [64, 32, 16],
                      'n_generator_units': [48, 72, 128],
                      'z_dim': 32,
                      'beta1': 0.9,
                      'batch_size': 128,
                      'd_epochs': 1,
                      'g_epochs': 1,
                      'keep_prob': 0.9,
                      'display_step': 100,
                      'show_step': 2000,
                      'train_seed': train_seed}

        AV = AdversarialValidation(parameters=parameters, train_path=train_path,
                                   load_preprocessed_data=load_preprocessed_data,
                                   test_path=test_path, gan_preprocess_path=gan_preprocess_path)

        if return_similarity_prob:
            similarity_prob = AV.train(similarity_prob_path=similarity_prob_path, global_epochs=global_epochs,
                                       return_similarity_prob=return_similarity_prob)
            return similarity_prob
        else:
            AV.train(similarity_prob_path=similarity_prob_path, global_epochs=global_epochs)


if __name__ == '__main__':

    print('======================================================')
    print('Start Training...')

    start_time = time.time()

    utils.check_dir([cfg.gan_prob_path, cfg.gan_preprocessed_data_path])

    global_train_seed = random.randint(0, 500)

    GenerateValidation.train(train_path=cfg.train_csv_path, test_path=cfg.test_csv_path, global_epochs=1,
                             similarity_prob_path=cfg.gan_prob_path, load_preprocessed_data=False,
                             gan_preprocess_path=cfg.gan_preprocessed_data_path, train_seed=global_train_seed)

    print('======================================================')
    print('All Tasks Done!')
    print('Global Train Seed: {}'.format(global_train_seed))
    print('Total Time: {}s'.format(time.time() - start_time))
    print('======================================================')

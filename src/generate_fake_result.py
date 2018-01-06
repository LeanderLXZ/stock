import random
import numpy as np
import pandas as pd
from models import utils

fake_pred_path = './results/fake_results/'


def generate_fake_result(seed, fake_std, label_std):

    utils.check_dir([fake_pred_path])

    np.random.seed(seed)

    index = pd.read_pickle('./data/preprocessed_data/id_test.p')

    label = np.zeros_like(index)
    for i in range(len(label)):
        if i < (label_std*len(index)):
            label[i] = 1
        else:
            label[i] = 0
    np.random.shuffle(label)

    # loc_list = []
    # loss_list = []
    # for i in range(0, 1000, 2):
    #     loc = 0.001 * i
    #     prob = np.random.normal(loc=loc, size=len(index), scale=0.005)
    #     prob = [1 if ii > loc else ii for ii in prob]
    #     loc_list.append(loc)
    #     loss_list.append(utils.log_loss(prob, label))
    #
    # sort_idx = np.argsort(loss_list)[:10]
    # loc_list = np.array(loc_list)[sort_idx]
    # loss_list = np.array(loss_list)[sort_idx]
    #
    # for loc_i, loss_i in zip(loc_list, loss_list):
    #     print(loc_i, ': ', loss_i)

    prob = np.random.normal(loc=fake_std, size=len(index), scale=0.0000005)
    # prob[0:10000] = 1
    # prob = [0.999 if ii > loc else ii for ii in prob]
    print(utils.log_loss(prob, label))

    # utils.save_pred_to_csv(fake_pred_path + str(fake_std) + '_fake_', index, prob)

if __name__ == '__main__':

    global_seed = random.randint(500, 1000)

    generate_fake_result(global_seed, fake_std=0.5, label_std=0.5)

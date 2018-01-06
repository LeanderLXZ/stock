import random
import tqdm
import time
import numpy as np
import pandas as pd
import preprocess
from models import utils

base_fake_result_path = './results/fake_results/'
tampered_pred_path = './results/tampered_results/'
preprocessed_data_path = preprocess.preprocessed_path
test_path = preprocess.test_csv_path


class GenerateTamperedData(object):

    def __init__(self, seed):

        utils.check_dir([tampered_pred_path])
        np.random.seed(seed)

        base_result = pd.read_csv(base_fake_result_path + '0.5_fake_result.csv', header=0, dtype=np.float64)
        self.prob = np.array(base_result['proba'], dtype=np.float64)
        self.id_test = utils.load_pkl_to_data(preprocessed_data_path + 'id_test.p')
        self.same_idx_list = utils.load_pkl_to_data(preprocessed_data_path + 'same_test_idx_pairs.p')
        self.code_id_train, self.code_id_test = utils.load_preprocessed_code_id(preprocessed_data_path)
        same_test_df = pd.read_csv(preprocessed_data_path + 'same_test_pairs.csv', header=0, dtype=np.float64)
        self.same_test_code_id = same_test_df['code_id']
        self.same_test_id = same_test_df['id']

        self.test_id_to_idx_dict = {}
        for idx, id_ in enumerate(self.id_test):
            self.test_id_to_idx_dict[id_] = idx

    def generate_fake_results_by_batch(self, loc):

        tampered_idx_ = np.concatenate(np.array(self.same_idx_list), axis=0).tolist()

        others_idx = []
        for i in range(len(self.id_test)):
            if i not in tampered_idx_:
                others_idx.append(i)
        tampered_idx = np.random.choice(others_idx, len(tampered_idx_)).tolist()

        for ii in tampered_idx:
            if ii in tampered_idx_:
                raise ValueError

        # tampered_id_list_ = list(range(start_id, stop_id))
        #
        # tampered_idx = []
        # for i, code_id in enumerate(code_id_test):
        #     if code_id in tampered_id_list:
        #         tampered_idx.append(i)

        # tampered_idx = tampered_idx_

        tampered_prob = np.random.normal(loc=loc, size=len(tampered_idx), scale=0.0002)
        for i, idx in enumerate(tampered_idx):
            self.prob[idx] = tampered_prob[i]

        # loc = 0.60
        # prob = np.random.normal(loc=loc, size=len(index), scale=0.0002)

        utils.save_pred_to_csv(tampered_pred_path + 'tampered_', self.id_test, self.prob)

    def save_same_pair_tampered_results(self, i, prob, x1, x2, idx1, idx2, data_path, info):

        tampered_prob = prob.copy()
        tampered_prob[idx1] = x1
        tampered_prob[idx2] = x2
        id1, id2 = self.id_test[idx1], self.id_test[idx2]
        data_path += '{}_tampered_results/'.format(info)
        utils.check_dir([data_path])
        data_path += '{}_{}_{}/'.format(i + 1, id1, id2)
        utils.check_dir([data_path])

        df = pd.DataFrame({'id': self.id_test, 'proba': tampered_prob})
        df.to_csv(data_path + '{}_{}-{}_{}-{}_result.csv'.format(i+1, id1, x1, id2, x2), sep=',', index=False)

    @staticmethod
    def save_same_pairs_test_csv(data_path, idx_pair_list):

        print('------------------------------------------------------')
        print('Saving {} ...'.format(data_path))
        big_weight_idx = np.concatenate(np.array(idx_pair_list)).tolist()
        test_f = pd.read_csv(test_path, header=0, dtype=np.float64)
        df = test_f.iloc[big_weight_idx]
        cols = ['code_id', *['feature{}'.format(i) for i in range(97)], 'group1', 'group2', 'id']
        df = df.loc[:, cols]
        df.to_csv(data_path, sep=',', index=False)

    def get_test_idx(self, id_):

        return self.test_id_to_idx_dict[id_]

    def get_pair_list(self, n_pairs, code_id_list, use_weight=False, w_train_list=None):

        w_train_col = []
        code_id_col = []
        id_col = []
        idx_pair_list = []
        exit_flag = False

        for i, code_id in enumerate(code_id_list):
            if use_weight:
                w_train = w_train_list[i]
            else:
                w_train = None
            idx_pair = []
            is_first = True
            for idx_s, code_id_s in enumerate(self.same_test_code_id):
                if code_id == code_id_s:
                    # print('is_first: {} | {}'.format(is_first, same_test_df.iloc[idx_s]['feature0']))
                    id_ = self.same_test_id[idx_s]
                    test_idx = self.get_test_idx(id_)
                    idx_pair.append(test_idx)
                    if is_first:
                        is_first = False
                    else:
                        idx_pair_list.append(idx_pair)
                        idx_pair = []
                        is_first = True
                    if use_weight:
                        w_train_col.append(w_train)
                    code_id_col.append(code_id)
                    id_col.append(id_)
                    if len(idx_pair_list) >= n_pairs:
                        exit_flag = True
                        break
            if exit_flag:
                break

        if use_weight:
            return idx_pair_list, w_train_col, code_id_col, id_col
        else:
            return idx_pair_list, code_id_col, id_col

    def tamper_result(self, idx_pair_list, info):

        print('------------------------------------------------------')
        print('Generating Tampered Results...')

        for i in tqdm.trange(len(idx_pair_list)):

            idx1, idx2 = idx_pair_list[i]
            self.save_same_pair_tampered_results(i, self.prob, 1, 0, idx1, idx2, tampered_pred_path, info)
            self.save_same_pair_tampered_results(i, self.prob, 0, 1, idx1, idx2, tampered_pred_path, info)
            self.save_same_pair_tampered_results(i, self.prob, 1, 1, idx1, idx2, tampered_pred_path, info)

    def generate_all_same_tampered_results(self):

        print('------------------------------------------------------')
        print('Generating All Tampered Results...')

        self.tamper_result(self.same_idx_list, 'all_same')

    def generate_tampered_results_by_range(self, start_code_id, n_pairs, reverse=False):

        print('------------------------------------------------------')
        print('Generating All Tampered Results...')
        stop_code_id = start_code_id+(n_pairs*3)
        range_code_id = range(start_code_id, stop_code_id)
        if reverse:
            range_code_id = range_code_id[::-1]

        print('Generating Pairs...')
        idx_pair_list, code_id_col, id_col = self.get_pair_list(n_pairs, range_code_id,)

        print('------------------------------------------------------')
        print('Number of Range Same Pairs: {}'.format(len(idx_pair_list)))
        pickle_path = preprocessed_data_path + 'range-{}-{}_idx_pairs.p'.format(start_code_id, n_pairs)
        utils.save_data_to_pkl(idx_pair_list, pickle_path)

        index = []
        for i in range(1, len(idx_pair_list) + 1):
            index.extend([i, i])
        df_log = pd.DataFrame({'index': np.array(index, dtype=int),
                               'code_id': np.array(code_id_col, dtype=int),
                               'id': np.array(id_col, dtype=int)})
        cols = ['index', 'code_id', 'id']
        df_log = df_log.loc[:, cols]
        tampered_pred_path_ = tampered_pred_path + 'range-{}-{}_tampered_log.csv'.format(start_code_id, n_pairs)

        print('------------------------------------------------------')
        print('Saving {} ...'.format(tampered_pred_path_))
        df_log.to_csv(tampered_pred_path_, sep=',', index=False)

        # Save Same Pairs csv file
        test_csv_path = preprocessed_data_path + 'range-{}-{}_same_pairs.csv'.format(start_code_id, n_pairs)
        self.save_same_pairs_test_csv(test_csv_path, idx_pair_list)

        # Generate Tampered Results
        self.tamper_result(idx_pair_list, 'range-{}-{}'.format(start_code_id, n_pairs))

    def generate_tampered_results_by_weight(self, n_pairs):

        w_train = utils.load_pkl_to_data(preprocessed_data_path + 'w_train.p')

        print('------------------------------------------------------')
        print('Calculating Big Weight Same Pairs...')
        print('------------------------------------------------------')
        print('Sorting...')
        sorted_by_weight_idx = np.argsort(w_train)[:-n_pairs*3:-1]
        sorted_w_train = w_train[sorted_by_weight_idx]
        sorted_code_id_train = self.code_id_train[sorted_by_weight_idx]

        print('Deduplicating...')
        big_weight_code_id = []
        big_weight_w_train = []
        for idx, code_id in enumerate(sorted_code_id_train):
            if code_id not in big_weight_code_id:
                if code_id in set(self.same_test_code_id):
                    big_weight_code_id.append(code_id)
                    big_weight_w_train.append(sorted_w_train[idx])

        print('Generating Pairs...')
        idx_pair_list, w_train_col, code_id_col, id_col = \
            self.get_pair_list(n_pairs, big_weight_code_id, use_weight=True, w_train_list=big_weight_w_train)

        print('------------------------------------------------------')
        print('Number of Big Weight Same Pairs: {}'.format(len(idx_pair_list)))
        utils.save_data_to_pkl(idx_pair_list, preprocessed_data_path + 'big_weight_idx_pairs.p')

        index = []
        for i in range(1, len(idx_pair_list)+1):
            index.extend([i, i])
        df_log = pd.DataFrame({'index': np.array(index, dtype=int),
                               'weight': np.array(w_train_col),
                               'code_id': np.array(code_id_col, dtype=int),
                               'id': np.array(id_col, dtype=int)})
        cols = ['index', 'weight', 'code_id', 'id']
        df_log = df_log.loc[:, cols]
        tampered_pred_path_ = tampered_pred_path + 'big_weight_tampered_log.csv'

        print('------------------------------------------------------')
        print('Saving {} ...'.format(tampered_pred_path_))
        df_log.to_csv(tampered_pred_path_, sep=',', index=False)

        # Save Same Pairs csv file
        self.save_same_pairs_test_csv(preprocessed_data_path + 'big_weight_same_pairs.csv', idx_pair_list)

        # Generate Tampered Results
        self.tamper_result(idx_pair_list, 'big_weight')

    def generate_tampered_results_by_absence(self, n_pairs):

        diff_code_id_test = utils.load_pkl_to_data(preprocessed_data_path + 'diff_code_id_test.p')

        print('------------------------------------------------------')
        print('Calculating Absent Same Pairs...')
        print('------------------------------------------------------')
        print('Sorting...')
        diff_code_id_test = np.sort(diff_code_id_test)
        absent_code_id = diff_code_id_test[:-n_pairs*3:-1]

        print('Generating Pairs...')
        idx_pair_list, code_id_col, id_col = self.get_pair_list(n_pairs, absent_code_id)

        print('------------------------------------------------------')
        print('Number of Absent Same Pairs: {}'.format(len(idx_pair_list)))
        utils.save_data_to_pkl(idx_pair_list, preprocessed_data_path + 'absent_idx_pairs.p')

        index = []
        for i in range(1, len(idx_pair_list)+1):
            index.extend([i, i])
        df_log = pd.DataFrame({'index': np.array(index, dtype=int),
                               'code_id': np.array(code_id_col, dtype=int),
                               'id': np.array(id_col, dtype=int)})
        cols = ['index', 'code_id', 'id']
        df_log = df_log.loc[:, cols]
        tampered_pred_path_ = tampered_pred_path + 'absent_tampered_log.csv'

        print('------------------------------------------------------')
        print('Saving {} ...'.format(tampered_pred_path_))
        df_log.to_csv(tampered_pred_path_, sep=',', index=False)

        # Save Same Pairs csv file
        self.save_same_pairs_test_csv(preprocessed_data_path + 'absent_same_pairs.csv', idx_pair_list)

        # Generate Tampered Results
        self.tamper_result(idx_pair_list, 'absent')

    @staticmethod
    def compare_is_same(x1, x2):
        is_same = True
        for i_x1, i_x2 in zip(x1, x2):
            if i_x1 != i_x2:
                is_same = False
        return is_same

    @staticmethod
    def value_is_right(tamper_prob, leaderboard):
        is_right = True
        if leaderboard > 0.69315:
            if tamper_prob != 0:
                is_right = False
        else:
            if tamper_prob != 1:
                is_right = False
        return is_right

    def generate_custom_tempered_result(self, tamper_list_path=None, base_result_path=None,
                                        append_info=None, check_same=False, check_value=False):

        print('Loading {}...'.format(tamper_list_path))
        tamper_list_df = pd.read_csv(tamper_list_path, header=0, dtype=np.float64)
        tamper_id = tamper_list_df['id']
        tamper_prob = tamper_list_df['proba']
        tamper_lb = None
        if check_value:
            if 'leaderboard' not in tamper_list_df.keys():
                raise ValueError("Header of 'tamper_list' must have column: 'leaderboard'!")
            else:
                tamper_lb = tamper_list_df['leaderboard']

        if check_same:
            print('Loading {}...'.format(test_path))
            test_df = pd.read_csv(test_path, header=0, dtype=np.float64)

        if base_result_path is None:
            tampered_prob = self.prob.copy()
        else:
            base_result = pd.read_csv(base_result_path, header=0, dtype=np.float64)
            tampered_prob = np.array(base_result['proba'], dtype=np.float64)

        print('------------------------------------------------------')
        print('Generating Custom Tampered Result...')
        is_first = True
        check_feature_list = [1, 47, 57, 82]
        check_list_1 = []
        check_list_2 = []
        prob1 = 0
        leaderboard_score = 0

        for i, (id_, prob_) in enumerate(zip(tamper_id, tamper_prob)):

            test_idx = self.get_test_idx(id_)
            tampered_prob[test_idx] = prob_

            if is_first:
                prob1 = prob_
                is_first = False
                if check_same:
                    for feature_idx in check_feature_list:
                        check_list_1.append(test_df.loc[test_idx, 'feature'+str(feature_idx)])
                if check_value:
                    leaderboard_score = tamper_lb[i]
                    if not self.value_is_right(prob_, leaderboard_score):
                        raise ValueError("[E] Tampered Prob Value is Wrong! (ID: {})".format(id_))

            else:
                prob2 = prob_
                is_first = True
                if check_same:
                    for feature_idx in check_feature_list:
                        check_list_2.append(test_df.loc[test_idx, 'feature'+str(feature_idx)])
                    if not self.compare_is_same(check_list_1, check_list_2):
                        raise ValueError("[E] Test Sample Not Same! (ID: {}-{})".format(tamper_id[i-1], id_))
                    if prob1 != prob2:
                        raise ValueError("[E] Tampered Prob Value Not Same! (ID: {}-{})".format(tamper_id[i - 1], id_))
                if check_value:
                    if not self.value_is_right(prob_, leaderboard_score):
                        raise ValueError("[E] Tampered Prob Value is Wrong! (ID: {})".format(id_))
                check_list_1 = []
                check_list_2 = []

        print('------------------------------------------------------')
        tampered_pred_path_ = './results/custom_tampered_results/'
        utils.check_dir([tampered_pred_path_])
        if append_info is not None:
            tampered_pred_path_ += append_info + '_'
        print('Saving {}...'.format(tampered_pred_path_ + 'tampered_result.csv'))
        df = pd.DataFrame({'id': self.id_test, 'proba': tampered_prob})
        df.to_csv(tampered_pred_path_ + 'tampered_result.csv', sep=',', index=False)


if __name__ == '__main__':

    start_time = time.time()

    print('======================================================')

    global_seed = random.randint(0, 500)
    GTD = GenerateTamperedData(global_seed)

    """Generate Tampered Result by Batch"""
    # GTD.generate_fake_results_by_batch(loc=0.1)

    """Generate Tampered Result by All Same List"""
    # GTD.generate_all_same_tampered_results()

    """Generate Tampered Result by Range"""
    GTD.generate_tampered_results_by_range(2000, 300, reverse=False)

    """Generate Tampered Result by Weight"""
    # GTD.generate_tampered_results_by_weight(300)

    """Generate Tampered Result by Absence"""
    # GTD.generate_tampered_results_by_absence(300)

    """Generate Custom Tampered Result"""
    # base_result_path_ = './results/fake_results/0.5_fake_result.csv'
    # tamper_list_path_ = './tamper_list.csv'
    # GTD.generate_custom_tempered_result(tamper_list_path=tamper_list_path_, append_info='custom',
    #                                     base_result_path=base_result_path_, check_same=True, check_value=True)

    print('------------------------------------------------------')
    print('Done!')
    print('Total Time: {:.2f}s'.format(time.time() - start_time))
    print('======================================================')

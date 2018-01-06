import time
import numpy as np
from models import classifiers
from models import utils
from models.cross_validation import CrossValidation


class PrejudgeBinary(object):
    """
        Prejudge - Training by Binary Class of Split Era sign
    """
    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_g_tr, x_tr_p, y_tr_p, w_tr_p, e_tr_p, x_g_tr_p,
                 x_tr_n, y_tr_n, w_tr_n, e_tr_n, x_g_tr_n, x_te, id_te, x_g_te,
                 pred_path, prejudged_data_path, loss_log_path, csv_log_path,
                 models_parameters, hyper_parameters):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_g_train = x_g_tr
        self.x_train_p = x_tr_p
        self.y_train_p = y_tr_p
        self.w_train_p = w_tr_p
        self.e_train_p = e_tr_p
        self.x_g_train_p = x_g_tr_p
        self.x_train_n = x_tr_n
        self.y_train_n = y_tr_n
        self.w_train_n = w_tr_n
        self.e_train_n = e_tr_n
        self.x_g_train_n = x_g_tr_n
        self.x_test = x_te
        self.id_test = id_te
        self.x_g_test = x_g_te
        self.pred_path = pred_path
        self.prejudged_data_path = prejudged_data_path
        self.loss_log_path = loss_log_path
        self.csv_log_path = csv_log_path
        self.parameters_e = models_parameters[0]
        self.parameters_p = models_parameters[1]
        self.parameters_n = models_parameters[2]
        self.cv_seed = hyper_parameters['cv_seed']
        self.train_seed = hyper_parameters['train_seed']
        self.n_splits_e = hyper_parameters['n_splits_e']
        self.num_boost_round_e = hyper_parameters['num_boost_round_e']
        self.n_cv_e = hyper_parameters['n_cv_e']
        self.n_valid_p = hyper_parameters['n_valid_p']
        self.n_cv_p = hyper_parameters['n_cv_p']
        self.n_era_p = hyper_parameters['n_era_p']
        self.num_boost_round_p = hyper_parameters['num_boost_round_p']
        self.era_list_p = hyper_parameters['era_list_p']
        self.n_valid_n = hyper_parameters['n_valid_n']
        self.n_cv_n = hyper_parameters['n_cv_n']
        self.n_era_n = hyper_parameters['n_era_n']
        self.num_boost_round_n = hyper_parameters['num_boost_round_n']
        self.era_list_n = hyper_parameters['era_list_n']
        self.force_convert_era = hyper_parameters['force_convert_era']
        self.use_weight = hyper_parameters['use_weight']
        self.show_importance = hyper_parameters['show_importance']
        self.show_accuracy = hyper_parameters['show_accuracy']

    @staticmethod
    def load_era_sign_csv(pred_path):
        """
            Load era sign from existed csv file
        """
        f = np.loadtxt(pred_path, dtype=np.float64, skiprows=1, delimiter=",")
        era_prob_test = f[:, -1]

        era_sign_test = np.array([0 if era_prob < 0.5 else 1 for era_prob in era_prob_test])

        return era_sign_test

    def era_prejudge_model_initializer(self, era_sign_train):
        """
            Initialize model for era prejudging
        """
        LGB_E = classifiers.LightGBM(self.x_g_train, era_sign_train, self.w_train, self.e_train,
                                     self.x_g_test, self.id_test, num_boost_round=self.num_boost_round_e)
        # XGB_E = models.XGBoost(self.x_train, era_sign_train, self.w_train, self.e_train,
        #                        self.x_test, self.id_test, num_boost_round=self.num_boost_round_e)
        # DNN_E = models.DeepNeuralNetworks(self.x_train, era_sign_train, self.w_train,
        #                                   self.e_train, self.x_test, self.id_test, self.dnn_l1_params)

        model_e = LGB_E

        return model_e

    def positive_model_initializer(self, x_test_p, x_g_test_p, id_test_p):
        """
            Initialize model for positive eras
        """
        LGB_P = classifiers.LightGBM(self.x_g_train_p, self.y_train_p, self.w_train_p, self.e_train_p,
                                     x_g_test_p, id_test_p, num_boost_round=self.num_boost_round_p)
        # XGB_P = models.XGBoost(self.x_train_p, self.y_train_p, self.w_train_p, self.e_train_p,
        #                        x_test_p, id_test_p, num_boost_round=self.num_boost_round_p)
        # DNN_P = models.DeepNeuralNetworks(self.x_train_p, self.y_train_p, self.w_train_p,
        #                                   self.e_train_p, x_test_p, id_test_p, self.dnn_l1_params)

        model_p = LGB_P

        return model_p

    def negative_model_initializer(self, x_test_n, x_g_test_n, id_test_n):
        """
            Initialize model for negative eras
        """
        LGB_N = classifiers.LightGBM(self.x_g_train_n, self.y_train_n, self.w_train_n, self.e_train_n,
                                     x_g_test_n, id_test_n, num_boost_round=self.num_boost_round_n)
        # XGB_N = models.XGBoost(self.x_train_n, self.y_train_n, self.w_train_n, self.e_train_n,
        #                        x_test_n, id_test_n, num_boost_round=self.num_boost_round_n)
        # DNN_N = models.DeepNeuralNetworks(self.x_train_n, self.y_train_n, self.w_train_n,
        #                                   self.e_train_n, x_test_n, id_test_n, self.dnn_l1_params)

        model_n = LGB_N

        return model_n

    def predict_era_sign(self):
        """
            Training and predict era signs of instances
        """
        print('======================================================')
        print('Positive Era List: {}'.format(self.era_list_p))
        print('Negative Era List: {}'.format(self.era_list_n))
        print('======================================================')
        print('Training Era Sign...')

        # Convert Eras of Training Data to 0 and 1
        era_sign_train = np.array([0 if era in self.era_list_n else 1 for era in self.e_train])

        # Init Model
        print('------------------------------------------------------')
        print('Initializing Model...')
        model = self.era_prejudge_model_initializer(era_sign_train)

        # Training and Get Probabilities of Test Era Being Positive
        era_prob_test = model.prejudge_train_binary(self.pred_path + 'pred_era/', n_splits=self.n_splits_e,
                                                    n_cv=self.n_cv_e, cv_seed=self.cv_seed, use_weight=self.use_weight,
                                                    parameters=self.parameters_e, show_importance=self.show_importance,
                                                    show_accuracy=self.show_accuracy)

        # Convert Probabilities of Test Era to 0 and 1
        if self.force_convert_era:
            era_sign_test = np.array([0 if era_prob < 0.5 else 1 for era_prob in era_prob_test])
        else:
            era_sign_test = era_prob_test

        return era_sign_test

    def split_test_set_by_era_sign(self, era_sign_test):
        """
            Split whole data set to positive and negative set using era sign
        """

        era_idx_test_p = []
        era_idx_test_n = []

        for idx, era_sign in enumerate(era_sign_test):
            if era_sign == 1:
                era_idx_test_p.append(idx)
            else:
                era_idx_test_n.append(idx)

        np.random.shuffle(era_idx_test_p)
        np.random.shuffle(era_idx_test_n)

        # Split Set of Test Set
        x_test_p = self.x_test[era_idx_test_p]
        x_g_test_p = self.x_test[era_idx_test_p]
        id_test_p = self.id_test[era_idx_test_p]
        x_test_n = self.x_test[era_idx_test_n]
        x_g_test_n = self.x_test[era_idx_test_n]
        id_test_n = self.id_test[era_idx_test_n]

        return x_test_p, x_g_test_p, id_test_p, era_idx_test_p, x_test_n, x_g_test_n, id_test_n, era_idx_test_n

    def train_models_by_era_sign(self, x_test_p, x_g_test_p, id_test_p, era_idx_test_p,
                                 x_test_n, x_g_test_n, id_test_n, era_idx_test_n):
        """
            Training positive and negative model using data set respectively
        """
        print('======================================================')
        print('Training Models by Era Sign...')

        print('------------------------------------------------------')
        print('Initializing Model...')
        model_p = self.positive_model_initializer(x_test_p, x_g_test_p, id_test_p)
        # model_p = self.negative_model_initializer(x_test_p, x_g_test_p, id_test_p)
        model_n = self.negative_model_initializer(x_test_n, x_g_test_n, id_test_n)
        # model_n = self.positive_model_initializer(x_test_n, x_g_test_n, id_test_n)

        print('======================================================')
        print('Training Models of Positive Era Sign...')
        prob_test_p = model_p.train(self.pred_path + 'positive/', self.loss_log_path + 'positive/',
                                    csv_log_path=self.csv_log_path, n_valid=self.n_valid_p, n_cv=self.n_cv_p,
                                    n_era=self.n_era_p, train_seed=self.train_seed, cv_seed=self.cv_seed,
                                    parameters=self.parameters_p, return_prob_test=True, era_list=self.era_list_p,
                                    show_importance=self.show_importance, show_accuracy=self.show_accuracy,
                                    save_csv_log=True, csv_idx='prejudge_p')

        # prob_test_p = model_p.train(self.pred_path + 'negative/', self.loss_log_path + 'negative/',
        #                             csv_log_path=self.csv_log_path, n_valid=self.n_valid_n, n_cv=self.n_cv_n,
        #                             n_era=self.n_era_n, train_seed=self.train_seed, cv_seed=self.cv_seed,
        #                             parameters=self.parameters_n, return_prob_test=True, era_list=self.era_list_n,
        #                             show_importance=self.show_importance, show_accuracy=self.show_accuracy,
        #                             save_csv_log=True, csv_idx='prejudge_n')

        print('======================================================')
        print('Training Models of Negative Era Sign...')
        prob_test_n = model_n.train(self.pred_path + 'negative/', self.loss_log_path + 'negative/',
                                    csv_log_path=self.csv_log_path, n_valid=self.n_valid_n, n_cv=self.n_cv_n,
                                    n_era=self.n_era_n, train_seed=self.train_seed, cv_seed=self.cv_seed,
                                    parameters=self.parameters_n, return_prob_test=True, era_list=self.era_list_n,
                                    show_importance=self.show_importance, show_accuracy=self.show_accuracy,
                                    save_csv_log=True, csv_idx='prejudge_n')

        # prob_test_n = model_n.train(self.pred_path + 'positive/', self.loss_log_path + 'positive/',
        #                             csv_log_path=self.csv_log_path, n_valid=self.n_valid_p, n_cv=self.n_cv_p,
        #                             n_era=self.n_era_p, train_seed=self.train_seed, cv_seed=self.cv_seed,
        #                             parameters=self.parameters_p, return_prob_test=True, era_list=self.era_list_p,
        #                             show_importance=self.show_importance, show_accuracy=self.show_accuracy,
        #                             save_csv_log=True, csv_idx='prejudge_p')

        prob_test = np.zeros_like(self.id_test, dtype=np.float64)

        for idx_p, prob_p in zip(era_idx_test_p, prob_test_p):
            prob_test[idx_p] = prob_p

        for idx_n, prob_n in zip(era_idx_test_n, prob_test_n):
            prob_test[idx_n] = prob_n

        return prob_test

    def train(self, load_pickle=False, load_pickle_path=None):
        """
            Training the model
        """
        start_time = time.time()

        path_list = [self.pred_path + 'positive/',
                     self.pred_path + 'negative/',
                     self.pred_path + 'pred_era/',
                     self.pred_path + 'pred_era/final_results/',
                     self.pred_path + 'final_results/',
                     self.loss_log_path + 'positive/',
                     self.loss_log_path + 'negative/']
        utils.check_dir(path_list)

        print('======================================================')
        print('Start Training PrejudgeBinary...')

        if load_pickle:

            # Load era_sign_test
            if load_pickle_path is None:
                era_sign_test = utils.load_pkl_to_data(self.prejudged_data_path + 'binary_era_sign_test.p')
            else:
                era_sign_test = utils.load_pkl_to_data(load_pickle_path)

        else:

            # Training Era Sign
            era_sign_test = self.predict_era_sign()

            # era_sign_test = self.load_era_sign_csv(self.pred_path + 'pred_era/final_results/lgb_result.csv')

            # Save era_sign_test to Pickle File
            utils.save_data_to_pkl(era_sign_test, self.prejudged_data_path + 'binary_era_sign_test.p')

        # Print Prediction of Positive Era Rate
        utils.print_positive_rate_test(era_sign_test)

        # Get Split Data
        x_test_p, x_g_test_p, id_test_p, era_idx_test_p, x_test_n, \
            x_g_test_n, id_test_n, era_idx_test_n = self.split_test_set_by_era_sign(era_sign_test)

        # Training Models by Era Sign
        prob_test = \
            self.train_models_by_era_sign(x_test_p, x_g_test_p, id_test_p, era_idx_test_p,
                                          x_test_n, x_g_test_n, id_test_n, era_idx_test_n)

        # Save Predictions
        utils.save_pred_to_csv(self.pred_path + 'final_results/prejudge_', self.id_test, prob_test)

        # Print Prediction of Positive Era Rate
        utils.print_positive_rate_test(era_sign_test)

        total_time = time.time() - start_time
        print('======================================================')
        print('Training Done!')
        print('Total Time: {}s'.format(total_time))
        print('======================================================')


class PrejudgeMultiClass(object):
    """
        Prejudge - Training by Multi Class of Split Era sign
    """
    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_g_tr, x_te, id_te, x_g_te,
                 pred_path, prejudged_data_path, loss_log_path, csv_log_path,
                 models_parameters, hyper_parameters):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_g_train = x_g_tr
        self.x_test = x_te
        self.id_test = id_te
        self.x_g_test = x_g_te
        self.pred_path = pred_path
        self.prejudged_data_path = prejudged_data_path
        self.loss_log_path = loss_log_path
        self.csv_log_path = csv_log_path
        self.parameters_e = models_parameters[0]
        self.parameters_m = models_parameters[1]
        self.cv_seed = hyper_parameters['cv_seed']
        self.train_seed = hyper_parameters['train_seed']
        self.n_splits_e = hyper_parameters['n_splits_e']
        self.num_boost_round_e = hyper_parameters['num_boost_round_e']
        self.n_cv_e = hyper_parameters['n_cv_e']
        self.n_era = hyper_parameters['n_era']
        self.n_valid_m = hyper_parameters['n_valid_m']
        self.n_cv_m = hyper_parameters['n_cv_m']
        self.num_boost_round_m = hyper_parameters['num_boost_round_m']
        self.use_weight = hyper_parameters['use_weight']
        self.show_importance = hyper_parameters['show_importance']
        self.show_accuracy = hyper_parameters['show_accuracy']

    def era_prejudge_model_initializer(self, e_train):
        """
            Initialize model for era prejudging
        """
        LGB_E = classifiers.LightGBM(self.x_g_train, e_train, self.w_train, self.e_train,
                                     self.x_g_test, self.id_test, num_boost_round=self.num_boost_round_e)
        # XGB_E = models.XGBoost(self.x_train, e_train, self.w_train, self.e_train,
        #                        self.x_test, self.id_test, num_boost_round=self.num_boost_round_e)
        # DNN_E = models.DeepNeuralNetworks(self.x_train, e_train, self.w_train,
        #                                   self.e_train, self.x_test, self.id_test, self.dnn_l1_params)

        model_e = LGB_E

        return model_e

    def multiclass_model_initializer(self, x_train_era, x_g_train_era, y_train_era, w_train_era,
                                     e_train_era, x_test_era, x_g_test_era, id_test_era):
        """
            Initialize model for positive eras
        """
        LGB_M = classifiers.LightGBM(x_g_train_era, y_train_era, w_train_era, e_train_era,
                                     x_g_test_era, id_test_era, num_boost_round=self.num_boost_round_m)
        # XGB_M = models.XGBoost(x_train_era, y_train_era, w_train_era, e_train_era,
        #                        x_test_era, id_test_era, num_boost_round=self.num_boost_round_m)
        # DNN_M = models.DeepNeuralNetworks(x_train_era, y_train_era, w_train_era, e_train_era,
        #                                   x_test_era, id_test_era, self.dnn_l1_params)

        model_m = LGB_M

        return model_m

    def predict_era_sign(self):
        """
            Training and predict era signs of instances
        """
        print('======================================================')
        print('Training Era Sign...')

        # Init Model
        print('------------------------------------------------------')
        print('Initializing Model...')

        # Shift e_train to (0-19)
        e_train_shifted = self.e_train - 1

        model = self.era_prejudge_model_initializer(e_train_shifted)

        # Training and Get Probabilities of Test Era Being Positive
        # n_test_sample x n_era_class
        era_prob_test = \
            model.prejudge_train_multiclass(self.pred_path + 'pred_era/', n_splits=self.n_splits_e, n_cv=self.n_cv_e,
                                            n_era=self.n_era, cv_seed=self.cv_seed, use_weight=self.use_weight,
                                            parameters=self.parameters_e, show_importance=self.show_importance)

        # Generate Index of Most Probably Era
        era_sign_test = np.argsort(era_prob_test, axis=1)[:, :-4:-1]

        return era_sign_test

    def split_test_set_by_era_sign(self, era_sign_test):
        """
            Split test data set to n_era sets using era signs
        """
        if self.cv_seed is not None:
            np.random.seed(self.cv_seed)

        x_test_idx = []
        x_test = []
        x_g_test = []
        id_test = []
        for e in range(self.n_era):
            x_test_idx_e = []
            for i, era_idx_list in enumerate(era_sign_test):
                if e in era_idx_list:
                    x_test_idx_e.append(i)

            # Shuffle
            np.random.shuffle(x_test_idx_e)

            x_test.append(self.x_test[x_test_idx_e])
            x_g_test.append(self.x_g_test[x_test_idx_e])
            id_test.append(self.id_test[x_test_idx_e])
            x_test_idx.append(x_test_idx_e)

        return x_test, x_g_test, id_test, x_test_idx

    def split_train_set_by_era(self):
        """
            Split train data set to n_era sets
        """
        x_train_e = []
        x_g_train_e = []
        y_train_e = []
        w_train_e = []
        e_train_e = []
        x_train = []
        x_g_train = []
        y_train = []
        w_train = []
        e_train = []
        era_tag = 0

        for idx, era in enumerate(self.e_train):

            if idx == len(self.e_train)-1:
                x_train_e.append(self.x_train[idx])
                x_g_train_e.append(self.x_g_train[idx])
                y_train_e.append(self.y_train[idx])
                w_train_e.append(self.w_train[idx])
                e_train_e.append(self.e_train[idx])
                x_train.append(x_train_e)
                x_g_train.append(x_g_train_e)
                y_train.append(y_train_e)
                w_train.append(w_train_e)
                e_train.append(e_train_e)
            elif era_tag == era:
                x_train_e.append(self.x_train[idx])
                x_g_train_e.append(self.x_g_train[idx])
                y_train_e.append(self.y_train[idx])
                w_train_e.append(self.w_train[idx])
                e_train_e.append(self.e_train[idx])
            else:
                era_tag = era
                x_train.append(x_train_e)
                x_g_train.append(x_g_train_e)
                y_train.append(y_train_e)
                w_train.append(w_train_e)
                e_train.append(e_train_e)
                x_train_e = [self.x_train[idx]]
                x_g_train_e = [self.x_g_train[idx]]
                y_train_e = [self.y_train[idx]]
                w_train_e = [self.w_train[idx]]
                e_train_e = [self.e_train[idx]]

        return x_train, x_g_train, y_train, w_train, e_train

    def train_models_by_era_sign(self, x_train, x_g_train, y_train, w_train, e_train,
                                 x_test, x_g_test, id_test, x_test_idx):
        """
            Training Models for Different Eras
        """
        print('======================================================')
        print('Training Models by Era Sign...')

        prob_test = np.zeros_like(self.id_test, dtype=np.float64).reshape(-1, 1).tolist()

        for model_iter in range(self.n_era):

            print('======================================================')
            print('Training Models of Era: {}/{}'.format(model_iter+1, self.n_era))

            x_train_era = np.array(x_train[model_iter])
            x_g_train_era = np.array(x_g_train[model_iter])
            y_train_era = np.array(y_train[model_iter])
            w_train_era = np.array(w_train[model_iter])
            e_train_era = np.array(e_train[model_iter])
            x_test_era = np.array(x_test[model_iter])
            x_g_test_era = np.array(x_g_test[model_iter])
            id_test_era = np.array(id_test[model_iter])
            x_test_idx_era = np.array(x_test_idx[model_iter])

            print('------------------------------------------------------')
            print('Initializing Model...')
            model = self.multiclass_model_initializer(x_train_era, x_g_train_era, y_train_era, w_train_era,
                                                      e_train_era, x_test_era, x_g_test_era, id_test_era)

            cv_generator = CrossValidation.random_split
            prob_test_era = model.train(self.pred_path + 'multiclass/', self.loss_log_path + 'multiclass/',
                                        csv_log_path=self.csv_log_path, n_valid=self.n_valid_m, n_cv=self.n_cv_m,
                                        train_seed=self.train_seed, cv_seed=self.cv_seed,
                                        parameters=self.parameters_m, return_prob_test=True,
                                        show_importance=self.show_importance, show_accuracy=self.show_accuracy,
                                        save_csv_log=True, csv_idx='era_{}'.format(model_iter+1),
                                        cv_generator=cv_generator)

            utils.save_data_to_pkl(prob_test, self.prejudged_data_path + 'multi_prob_test_era_{}.p'.format(model_iter + 1))

            for idx_era, prob_era in zip(x_test_idx_era, prob_test_era):
                if prob_test[idx_era][0] == 0.:
                    prob_test[idx_era][0] = prob_era
                else:
                    prob_test[idx_era].append(prob_era)

        utils.save_data_to_pkl(prob_test, self.prejudged_data_path + 'multi_prob_test.p')

        # Calculate Mean of prob_test
        prob_test = np.mean(prob_test, axis=1, dtype=np.float64)

        return prob_test

    def train(self, load_pickle=False, load_pickle_path=None):
        """
            Training the model
        """
        start_time = time.time()

        path_list = [self.pred_path + 'multiclass/',
                     self.pred_path + 'pred_era/',
                     self.pred_path + 'pred_era/final_results/',
                     self.pred_path + 'final_results/']
        utils.check_dir(path_list)

        print('======================================================')
        print('Start Training PrejudgeMultiClass...')

        if load_pickle:

            # Load era_sign_test
            if load_pickle_path is None:
                era_sign_test = utils.load_pkl_to_data(self.prejudged_data_path + 'multiclass_era_sign_test.p')
            else:
                era_sign_test = utils.load_pkl_to_data(load_pickle_path)

        else:

            # Training Era Sign
            era_sign_test = self.predict_era_sign()

            # era_sign_test = self.load_era_sign_csv(self.pred_path + 'pred_era/final_results/lgb_result.csv')

            # Save era_sign_test to Pickle File
            utils.save_data_to_pkl(era_sign_test, self.prejudged_data_path + 'multiclass_era_sign_test.p')

        # Get Split Data
        x_test, x_g_test, id_test, x_test_idx = self.split_test_set_by_era_sign(era_sign_test)
        x_train, x_g_train, y_train, w_train, e_train = self.split_train_set_by_era()

        # Training Models by Era Sign
        prob_test = \
            self.train_models_by_era_sign(x_train, x_g_train, y_train, w_train, e_train,
                                          x_test, x_g_test, id_test, x_test_idx)

        # Save Predictions
        utils.save_pred_to_csv(self.pred_path + 'final_results/prejudge_', self.id_test, prob_test)

        total_time = time.time() - start_time
        print('======================================================')
        print('Training Done!')
        print('Total Time: {}s'.format(total_time))
        print('======================================================')

from models.classifiers import *
from models import stacking
from models import prejudge
from models.sk_grid_search import SKLearnGridSearch
from models.cross_validation import CrossValidation
from models.dnn import DeepNeuralNetworks
from config import cfg


class SingleModel(object):
    """
        Train single model
    """
    def __init__(self, reduced_feature_list=None, train_args=None, cv_args=None, use_multi_group=False, mode=None):

        self.x_train, self.y_train, self.w_train, self.e_train, self.x_test, self.pct_test, self.id_test = \
            utils.load_preprocessed_data(cfg.preprocessed_data_path)
        if preprocess.group_list is not None:
            self.x_g_train, self.x_g_test = \
                utils.load_preprocessed_data_g(cfg.preprocessed_data_path)
        else:
            self.x_g_train, self.x_g_test = self.x_train, self.x_test

        if train_args['use_global_valid']:
            self.x_gl_valid, self.x_g_gl_valid, self.y_gl_valid, self.w_gl_valid, self.e_gl_valid = \
                utils.load_global_valid_data(cfg.preprocessed_data_path)

        # Choose Useful features
        if reduced_feature_list is not None:

            reduced_feature_list_g = reduced_feature_list + [-1]
            reduced_feature_list.extend(list(range(-1, -29, -1)))
            self.x_train = self.x_train[:, reduced_feature_list]
            self.x_g_train = self.x_g_train[:, reduced_feature_list_g]
            self.x_test = self.x_test[:, reduced_feature_list]
            self.x_g_test = self.x_g_test[:, reduced_feature_list_g]
            if train_args['use_global_valid']:
                self.x_gl_valid = self.x_gl_valid[reduced_feature_list]
                self.x_g_gl_valid = self.x_g_gl_valid[reduced_feature_list_g]

        self.loss_log_path = cfg.loss_log_path
        self.boost_round_log_path = cfg.boost_round_out_path
        self.train_args = train_args
        self.cv_args = cv_args
        self.use_multi_group = use_multi_group
        self.mode = mode

        # Different Mode
        if mode == 'auto_grid_search':
            self.csv_log_path = cfg.grid_search_out_path
            self.pred_path = cfg.single_model_pred_path
        elif mode == 'auto_train':
            self.csv_log_path = cfg.auto_train_pred_path
            self.pred_path = cfg.auto_train_pred_path
        else:
            self.csv_log_path = cfg.csv_log_path
            self.pred_path = cfg.single_model_pred_path

    def train_model(self, model=None, grid_search_tuple_list=None):

        if grid_search_tuple_list is not None:
            param_name_list = []
            param_value_list = []
            for grid_search_tuple in grid_search_tuple_list:
                param_name = grid_search_tuple[0]
                param_value = grid_search_tuple[1]
                if param_name in ['n_valid', 'n_cv', 'n_era', 'valid_rate', 'window_size', 'cv_weights']:
                    self.cv_args[param_name] = param_value
                else:
                    self.train_args['parameters'][param_name] = param_value
                param_name_list.append(param_name)
                param_value_list.append(param_value)

        else:
            param_name_list = None
            param_value_list = None

        # Parameters for Train
        model.train(pred_path=self.pred_path, loss_log_path=self.loss_log_path, csv_log_path=self.csv_log_path,
                    boost_round_log_path=self.boost_round_log_path, mode=self.mode, param_name_list=param_name_list,
                    param_value_list=param_value_list, cv_args=self.cv_args, **self.train_args)

    def lr_train(self, train_seed, cv_seed, parameters=None, grid_search_tuple_list=None):
        """
            Logistic Regression
        """
        if parameters is None:
            parameters = {'C': 1.0,
                          'class_weight': None,
                          'dual': False,
                          'fit_intercept': True,
                          'intercept_scaling': 1,
                          'max_iter': 100,
                          # 'multi_class': 'multinomial',
                          'multi_class': 'ovr',
                          'n_jobs': -1,
                          'penalty': 'l2',
                          'solver': 'sag',
                          'tol': 0.0001,
                          'random_state': train_seed,
                          'verbose': 1,
                          'warm_start': False}
        else:
            parameters['random_state'] = train_seed

        self.train_args['parameters'] = parameters
        self.train_args['train_seed'] = train_seed
        self.cv_args['cv_seed'] = cv_seed

        if self.train_args['use_global_valid']:
            model = LRegression(self.x_train, self.y_train, self.w_train, self.e_train,
                                self.x_test, self.pct_test, self.id_test, self.x_gl_valid, self.y_gl_valid,
                                self.w_gl_valid, self.e_gl_valid, use_multi_group=self.use_multi_group)
        else:
            model = LRegression(self.x_train, self.y_train, self.w_train, self.e_train,
                                self.x_test, self.pct_test, self.id_test, use_multi_group=self.use_multi_group)

        self.train_model(model=model, grid_search_tuple_list=grid_search_tuple_list)

    def rf_train(self, train_seed, cv_seed, parameters=None, grid_search_tuple_list=None):
        """
            Random Forest
        """
        if parameters is None:
            parameters = {'bootstrap': True,
                          'class_weight': None,
                          'criterion': 'gini',
                          'max_depth': 2,
                          'max_features': 7,
                          'max_leaf_nodes': None,
                          'min_impurity_decrease': 0.0,
                          'min_samples_leaf': 286,
                          'min_samples_split': 3974,
                          'min_weight_fraction_leaf': 0.0,
                          'n_estimators': 32,
                          'n_jobs': -1,
                          'oob_score': True,
                          'random_state': train_seed,
                          'verbose': 2,
                          'warm_start': False}
        else:
            parameters['random_state'] = train_seed

        self.train_args['parameters'] = parameters
        self.train_args['train_seed'] = train_seed
        self.cv_args['cv_seed'] = cv_seed

        if self.train_args['use_global_valid']:
            model = RandomForest(self.x_train, self.y_train, self.w_train, self.e_train,
                                 self.x_test, self.pct_test, self.id_test, self.x_gl_valid, self.y_gl_valid,
                                 self.w_gl_valid, self.e_gl_valid, use_multi_group=self.use_multi_group)
        else:
            model = RandomForest(self.x_train, self.y_train, self.w_train, self.e_train,
                                 self.x_test, self.pct_test, self.id_test, use_multi_group=self.use_multi_group)

        self.train_model(model=model, grid_search_tuple_list=grid_search_tuple_list)

    def et_train(self, train_seed, cv_seed, parameters=None, grid_search_tuple_list=None):
        """
            Extra Trees
        """
        if parameters is None:
            parameters = {'bootstrap': True,
                          'class_weight': None,
                          'criterion': 'gini',
                          'max_depth': 2,
                          'max_features': 7,
                          'max_leaf_nodes': None,
                          'min_impurity_decrease': 0.0,
                          'min_samples_leaf': 357,
                          'min_samples_split': 4909,
                          'min_weight_fraction_leaf': 0.0,
                          'n_estimators': 20,
                          'n_jobs': -1,
                          'oob_score': True,
                          'random_state': train_seed,
                          'verbose': 2,
                          'warm_start': False}
        else:
            parameters['random_state'] = train_seed

        self.train_args['parameters'] = parameters
        self.train_args['train_seed'] = train_seed
        self.cv_args['cv_seed'] = cv_seed

        if self.train_args['use_global_valid']:
            model = ExtraTrees(self.x_train, self.y_train, self.w_train, self.e_train,
                               self.x_test, self.pct_test, self.id_test, self.x_gl_valid, self.y_gl_valid,
                               self.w_gl_valid, self.e_gl_valid, use_multi_group=self.use_multi_group)
        else:
            model = ExtraTrees(self.x_train, self.y_train, self.w_train, self.e_train,
                               self.x_test, self.pct_test, self.id_test, use_multi_group=self.use_multi_group)

        self.train_model(model=model, grid_search_tuple_list=grid_search_tuple_list)

    def ab_train(self, train_seed, cv_seed, parameters=None, grid_search_tuple_list=None):
        """
            AdaBoost
        """
        et_parameters = {'bootstrap': True,
                         'class_weight': None,
                         'criterion': 'gini',
                         'max_depth': 2,
                         'max_features': 7,
                         'max_leaf_nodes': None,
                         'min_impurity_decrease': 0.0,
                         'min_samples_leaf': 357,
                         'min_samples_split': 4909,
                         'min_weight_fraction_leaf': 0.0,
                         'n_estimators': 20,
                         'n_jobs': -1,
                         'oob_score': True,
                         'random_state': train_seed,
                         'verbose': 2,
                         'warm_start': False}

        clf_et = ExtraTreesClassifier(**et_parameters)

        if parameters is None:
            parameters = {'algorithm': 'SAMME.R',
                          'base_estimator': clf_et,
                          'learning_rate': 0.0051,
                          'n_estimators': 9,
                          'random_state': train_seed}
        else:
            parameters['random_state'] = train_seed

        self.train_args['parameters'] = parameters
        self.train_args['train_seed'] = train_seed
        self.cv_args['cv_seed'] = cv_seed

        if self.train_args['use_global_valid']:
            model = AdaBoost(self.x_train, self.y_train, self.w_train, self.e_train,
                             self.x_test, self.pct_test, self.id_test, self.x_gl_valid, self.y_gl_valid,
                             self.w_gl_valid, self.e_gl_valid, use_multi_group=self.use_multi_group)
        else:
            model = AdaBoost(self.x_train, self.y_train, self.w_train, self.e_train,
                             self.x_test, self.pct_test, self.id_test, use_multi_group=self.use_multi_group)

        self.train_model(model=model, grid_search_tuple_list=grid_search_tuple_list)

    def gb_train(self, train_seed, cv_seed, parameters=None, grid_search_tuple_list=None):
        """
            GradientBoosting
        """
        if parameters is None:
            parameters = {'criterion': 'friedman_mse',
                          'init': None,
                          'learning_rate': 0.05,
                          'loss': 'deviance',
                          'max_depth': 25,
                          'max_features': 'auto',
                          'max_leaf_nodes': None,
                          'min_impurity_decrease': 0.0,
                          'min_impurity_split': None,
                          'min_samples_leaf': 50,
                          'min_samples_split': 1000,
                          'min_weight_fraction_leaf': 0.0,
                          'n_estimators': 200,
                          'presort': 'auto',
                          'random_state': train_seed,
                          'subsample': 0.8,
                          'verbose': 2,
                          'warm_start': False}
        else:
            parameters['random_state'] = train_seed

        self.train_args['parameters'] = parameters
        self.train_args['train_seed'] = train_seed
        self.cv_args['cv_seed'] = cv_seed

        if self.train_args['use_global_valid']:
            model = GradientBoosting(self.x_train, self.y_train, self.w_train, self.e_train,
                                     self.x_test, self.pct_test, self.id_test, self.x_gl_valid, self.y_gl_valid,
                                     self.w_gl_valid, self.e_gl_valid, use_multi_group=self.use_multi_group)
        else:
            model = GradientBoosting(self.x_train, self.y_train, self.w_train, self.e_train,
                                     self.x_test, self.pct_test, self.id_test, use_multi_group=self.use_multi_group)

        self.train_model(model=model, grid_search_tuple_list=grid_search_tuple_list)

    def xgb_train(self, train_seed, cv_seed, parameters=None, grid_search_tuple_list=None, num_boost_round=None):
        """
            XGBoost
        """
        if parameters is None:
            parameters = {'learning_rate': 0.003,
                          'gamma': 0.001,                   # 如果loss function小于设定值，停止产生子节点
                          'max_depth': 10,                  # default=6
                          'min_child_weight': 12,           # default=1，建立每个模型所需最小样本权重和
                          'subsample': 0.92,                # 建立树模型时抽取子样本占整个样本的比例
                          'colsample_bytree': 0.85,         # 建立树时对特征随机采样的比例
                          'colsample_bylevel': 0.7,
                          'lambda': 0,
                          'alpha': 0,
                          'early_stopping_rounds': 10000,
                          'n_jobs': -1,
                          'objective': 'binary:logistic',
                          'eval_metric': 'logloss',
                          'seed': train_seed}
        else:
            parameters['seed'] = train_seed

        file_name_params = ['num_boost_round', 'learning_rate', 'max_depth', 'subsample',
                            'colsample_bytree', 'colsample_bylevel', 'gamma']

        self.train_args['parameters'] = parameters
        self.train_args['train_seed'] = train_seed
        self.cv_args['cv_seed'] = cv_seed
        self.train_args['file_name_params'] = file_name_params

        if num_boost_round is None:
            num_boost_round = 150

        if self.train_args['use_global_valid']:
            model = XGBoost(self.x_train, self.y_train, self.w_train, self.e_train,
                            self.x_test, self.pct_test, self.id_test,
                            self.x_gl_valid, self.y_gl_valid, self.w_gl_valid, self.e_gl_valid,
                            use_multi_group=self.use_multi_group, num_boost_round=num_boost_round)
        else:
            model = XGBoost(self.x_train, self.y_train, self.w_train, self.e_train,
                            self.x_test, self.pct_test, self.id_test,
                            use_multi_group=self.use_multi_group, num_boost_round=num_boost_round)

        self.train_model(model=model, grid_search_tuple_list=grid_search_tuple_list)

    def xgb_train_sklearn(self, train_seed, cv_seed, parameters=None, grid_search_tuple_list=None):
        """
            XGBoost using scikit-learn module
        """
        if parameters is None:
            parameters = {'max_depth': 3,
                          'learning_rate': 0.1,
                          'n_estimators': 100,
                          'silent': True,
                          'objective': "binary:logistic",
                          #  'booster': 'gbtree',
                          #  'n_jobs':  1,
                          'nthread': -1,
                          'gamma': 0,
                          'min_child_weight': 1,
                          'max_delta_step': 0,
                          'subsample': 1,
                          'colsample_bytree': 1,
                          'colsample_bylevel': 1,
                          'reg_alpha': 0,
                          'reg_lambda': 1,
                          'scale_pos_weight': 1,
                          'base_score': 0.5,
                          #  'random_state': train_seed,
                          'seed': train_seed,
                          'missing': None}
        else:
            parameters['seed'] = train_seed

        self.train_args['parameters'] = parameters
        self.train_args['train_seed'] = train_seed
        self.cv_args['cv_seed'] = cv_seed

        if self.train_args['use_global_valid']:
            model = SKLearnXGBoost(self.x_train, self.y_train, self.w_train, self.e_train,
                                   self.x_test, self.pct_test, self.id_test, self.x_gl_valid, self.y_gl_valid,
                                   self.w_gl_valid, self.e_gl_valid, use_multi_group=self.use_multi_group)
        else:
            model = SKLearnXGBoost(self.x_train, self.y_train, self.w_train, self.e_train,
                                   self.x_test, self.pct_test, self.id_test, use_multi_group=self.use_multi_group)

        self.train_model(model=model, grid_search_tuple_list=grid_search_tuple_list)

    def lgb_train(self, train_seed, cv_seed, parameters=None, grid_search_tuple_list=None, num_boost_round=None):
        """
            LightGBM
        """
        if parameters is None:
            parameters = {'application': 'binary',
                          'boosting': 'gbdt',                   # gdbt,rf,dart,goss
                          'learning_rate': 0.003,               # default=0.1
                          'num_leaves': 88,                     # default=31       <2^(max_depth)
                          'max_depth': 9,                       # default=-1
                          'min_data_in_leaf': 2500,             # default=20       reduce over-fit
                          'min_sum_hessian_in_leaf': 1e-3,      # default=1e-3     reduce over-fit
                          'feature_fraction': 0.5,              # default=1
                          'feature_fraction_seed': 19,          # default=2
                          'bagging_fraction': 0.8,              # default=1
                          'bagging_freq': 2,                    # default=0        perform bagging every k iteration
                          'bagging_seed': 1,                    # default=3
                          'lambda_l1': 0,                       # default=0
                          'lambda_l2': 0,                       # default=0
                          'min_gain_to_split': 0,               # default=0
                          'max_bin': 225,                       # default=255
                          'min_data_in_bin': 5,                 # default=5
                          'metric': 'binary_logloss',
                          'num_threads': -1,
                          'verbosity': 1,
                          'early_stopping_rounds': 10000}
        else:
            parameters['seed'] = train_seed

        file_name_params = ['num_boost_round', 'learning_rate', 'num_leaves', 'max_depth',
                            'feature_fraction', 'bagging_fraction', 'bagging_freq', 'min_data_in_bin']

        self.train_args['parameters'] = parameters
        self.train_args['train_seed'] = train_seed
        self.cv_args['cv_seed'] = cv_seed
        self.train_args['file_name_params'] = file_name_params

        if num_boost_round is None:
            num_boost_round = 100

        if self.train_args['use_global_valid']:
            model = LightGBM(self.x_g_train, self.y_train, self.w_train, self.e_train,
                             self.x_g_test, self.pct_test, self.id_test,
                             self.x_g_gl_valid, self.y_gl_valid, self.w_gl_valid, self.e_gl_valid,
                             use_multi_group=self.use_multi_group, num_boost_round=num_boost_round)
        else:
            model = LightGBM(self.x_g_train, self.y_train, self.w_train, self.e_train,
                             self.x_g_test, self.pct_test, self.id_test,
                             use_multi_group=self.use_multi_group, num_boost_round=num_boost_round)

        self.train_model(model=model, grid_search_tuple_list=grid_search_tuple_list)

    def lgb_train_sklearn(self, train_seed, cv_seed, parameters=None, grid_search_tuple_list=None):
        """
            LightGBM using scikit-learn module
        """
        if parameters is None:
            parameters = {'learning_rate': 0.003,
                          'boosting_type': 'gbdt',        # traditional Gradient Boosting Decision Tree.
                          'num_leaves': 80,               # <2^(max_depth)
                          'max_depth': 7,                 # default=-1
                          'n_estimators': 50,
                          'max_bin': 1005,
                          'subsample_for_bin': 1981,
                          'objective': 'binary',
                          'min_split_gain': 0.,
                          'min_child_weight': 1,
                          'min_child_samples': 0,
                          'subsample': 0.8,
                          'subsample_freq': 5,
                          'colsample_bytree': 0.8,
                          'reg_alpha': 0.5,
                          'reg_lambda': 0.5,
                          'silent': False,
                          'seed': train_seed}
        else:
            parameters['seed'] = train_seed

        self.train_args['parameters'] = parameters
        self.train_args['train_seed'] = train_seed
        self.cv_args['cv_seed'] = cv_seed

        if self.train_args['use_global_valid']:
            model = SKLearnLightGBM(self.x_g_train, self.y_train, self.w_train, self.e_train,
                                    self.x_g_test, self.pct_test, self.id_test, self.x_g_gl_valid, self.y_gl_valid,
                                    self.w_gl_valid, self.e_gl_valid, use_multi_group=self.use_multi_group)
        else:
            model = SKLearnLightGBM(self.x_g_train, self.y_train, self.w_train, self.e_train,
                                    self.x_g_test, self.pct_test, self.id_test, use_multi_group=self.use_multi_group)

        self.train_model(model=model, grid_search_tuple_list=grid_search_tuple_list)

    def cb_train(self, train_seed, cv_seed, parameters=None, grid_search_tuple_list=None, num_boost_round=None):
        """
            CatBoost
        """
        if parameters is None:
            parameters = {'iterations': 70,
                          'learning_rate': 0.004,
                          'depth': 9,                            # Depth of the tree.
                          'l2_leaf_reg': 0.01,                   # L2 regularization coefficient.
                          'rsm': 0.8,                            # The percentage of features to use at each iteration.
                          'bagging_temperature': 0.7,            # Controls intensity of Bayesian bagging. The higher the temperature the more aggressive bagging is.
                          'loss_function': 'Logloss',
                          'border': 0.5,
                          'border_count': 128,
                          'feature_border_type': 'MinEntropy',
                          'fold_permutation_block_size': 1,
                          'od_pval': None,                       # Use overfitting detector to stop training when reaching a specified threshold.
                          'od_wait': None,                       # Number of iterations which overfitting detector will wait after new best error.
                          'od_type': 'IncToDec',                 # Type of overfitting detector which will be used in program.
                          'gradient_iterations': None,           # The number of gradient steps when calculating the values in leaves.
                          'leaf_estimation_method': 'Gradient',  # The method used to calculate the values in leaves.
                          'thread_count': None,                  # Number of parallel threads used to run CatBoost.
                          'random_seed': train_seed,
                          'use_best_model': False,               # To limit the number of trees in predict() using information about the optimal value of the error function.
                          'verbose': True,
                          'ctr_description': None,               # Binarization settings for categorical features.
                          'ctr_border_count': 16,                # The number of partitions for Categ features.
                          'ctr_leaf_count_limit': None,          # The maximum number of leafs with categorical features.
                          'priors': None,                        # Use priors when training.
                          'has_time': False,                     # To use the order in which objects are represented in the input data.
                          'name': 'experiment',
                          'ignored_features': None,
                          'train_dir': None,
                          'custom_loss': None,
                          'eval_metric': 'Logloss',
                          'class_weights': None}
        else:
            parameters['random_seed'] = train_seed

        if num_boost_round is not None:
            parameters['iterations'] = num_boost_round

        file_name_params = ['iterations', 'learning_rate', 'depth', 'rsm', 'bagging_temperature']

        self.train_args['parameters'] = parameters
        self.train_args['train_seed'] = train_seed
        self.cv_args['cv_seed'] = cv_seed
        self.train_args['file_name_params'] = file_name_params

        if self.train_args['use_global_valid']:
            model = CatBoost(self.x_g_train, self.y_train, self.w_train, self.e_train,
                             self.x_g_test, self.pct_test, self.id_test, self.x_g_gl_valid, self.y_gl_valid,
                             self.w_gl_valid, self.e_gl_valid, use_multi_group=self.use_multi_group)
        else:
            model = CatBoost(self.x_g_train, self.y_train, self.w_train, self.e_train,
                             self.x_g_test, self.pct_test, self.id_test, use_multi_group=self.use_multi_group)

        self.train_model(model=model, grid_search_tuple_list=grid_search_tuple_list)

    def dnn_tf_train(self, train_seed, cv_seed, parameters=None, epochs=None, grid_search_tuple_list=None):
        """
            Deep Neural Networks
        """
        if parameters is None:
            parameters = {'version': '1.0',
                          'epochs': 2,
                          'unit_number': [256, 128, 64],
                          'learning_rate': 0.0001,
                          'keep_probability': 0.5,
                          'batch_size': 128,
                          'seed': train_seed,
                          'display_step': 100,
                          'save_path': cfg.dnn_checkpoint_path,
                          'log_path': cfg.dnn_log_path}
        else:
            parameters['seed'] = train_seed

        if epochs is not None:
            parameters['epochs'] = epochs

        file_name_params = ['epochs', 'unit_number', 'learning_rate', 'keep_probability', 'batch_size']

        self.train_args['parameters'] = parameters
        self.train_args['train_seed'] = train_seed
        self.cv_args['cv_seed'] = cv_seed
        self.train_args['file_name_params'] = file_name_params

        if self.train_args['use_global_valid']:
            model = DeepNeuralNetworks(self.x_train, self.y_train, self.w_train, self.e_train, self.x_test,
                                       self.id_test, self.pct_test, self.x_gl_valid, self.y_gl_valid, self.w_gl_valid,
                                       self.e_gl_valid, use_multi_group=self.use_multi_group, parameters=parameters)
        else:
            model = DeepNeuralNetworks(self.x_train, self.y_train, self.w_train, self.e_train,
                                       self.x_test, self.pct_test, self.id_test,
                                       use_multi_group=self.use_multi_group, parameters=parameters)

        self.train_model(model=model, grid_search_tuple_list=grid_search_tuple_list)

    def stack_lgb_train(self, train_seed, cv_seed, parameters=None, auto_idx=None,
                        num_boost_round=None, grid_search_tuple_list=None):
        """
            LightGBM for stack layer
        """
        blender_x_tree, blender_test_tree, blender_x_g_tree, blender_test_g_tree\
            = utils.load_stacked_data(cfg.stack_output_path + 'auto_stack_{}_l2_'.format(auto_idx))

        g_train = self.x_g_train[:, -1]
        g_test = self.x_g_test[:, -1]

        x_train_reuse = self.x_train[:, :87]
        x_test_reuse = self.x_test[:, :87]

        print('------------------------------------------------------')
        print('Stacking Reused Features of Train Set...')
        # n_sample * (n_feature + n_reuse)
        blender_x_tree = np.concatenate((blender_x_tree, x_train_reuse), axis=1)
        # n_sample * (n_feature + n_reuse + 1)
        blender_x_g_tree = np.column_stack((blender_x_tree, g_train))

        print('------------------------------------------------------')
        print('Stacking Reused Features of Test Set...')
        # n_sample * (n_feature + n_reuse)
        blender_test_tree = np.concatenate((blender_test_tree, x_test_reuse), axis=1)
        # n_sample * (n_feature + n_reuse + 1)
        blender_test_g_tree = np.column_stack((blender_test_tree, g_test))

        if parameters is None:
            parameters = {'application': 'binary',
                          'boosting': 'gbdt',                   # gdbt,rf,dart,goss
                          'learning_rate': 0.003,               # default=0.1
                          'num_leaves': 85,                     # default=31       <2^(max_depth)
                          'max_depth': 9,                       # default=-1
                          'min_data_in_leaf': 2500,             # default=20       reduce over-fit
                          'min_sum_hessian_in_leaf': 1e-3,      # default=1e-3     reduce over-fit
                          'feature_fraction': 0.5,              # default=1
                          'feature_fraction_seed': train_seed,  # default=2
                          'bagging_fraction': 0.8,              # default=1
                          'bagging_freq': 2,                    # default=0        perform bagging every k iteration
                          'bagging_seed': train_seed,           # default=3
                          'lambda_l1': 0,                       # default=0
                          'lambda_l2': 0,                       # default=0
                          'min_gain_to_split': 0,               # default=0
                          'max_bin': 225,                       # default=255
                          'min_data_in_bin': 9,                 # default=5
                          'metric': 'binary_logloss',
                          'num_threads': -1,
                          'verbosity': 1,
                          'early_stopping_rounds': 50,          # default=0
                          'seed': train_seed}
        else:
            parameters['seed'] = train_seed

        # Grid Search
        if grid_search_tuple_list is not None:
            parameters[grid_search_tuple_list[0]] = grid_search_tuple_list[1]

        self.train_args['parameters'] = parameters
        self.train_args['train_seed'] = train_seed
        self.cv_args['cv_seed'] = cv_seed

        if num_boost_round is None:
            num_boost_round = 100

        LGB = LightGBM(blender_x_g_tree, self.y_train, self.w_train, self.e_train, blender_test_g_tree,
                       self.pct_test, self.id_test, use_multi_group=self.use_multi_group,
                       num_boost_round=num_boost_round)

        # cv_generator = CrossValidation.era_k_fold_with_weight_balance

        LGB.train(self.pred_path, self.loss_log_path, csv_log_path=self.csv_log_path + 'stack_final_',
                  mode=self.mode, cv_args=self.cv_args, **self.train_args)


class ChampionModel(object):

    def __init__(self, reduced_feature_list=None, save_auto_train_results=True,
                 train_args=None, cv_args=None, use_multi_group=False, mode=None):

        self.x_train, self.y_train, self.w_train, self.e_train, self.x_test, self.pct_test, self.id_test\
            = utils.load_preprocessed_data(cfg.preprocessed_data_path)
        self.x_g_train, self.x_g_test\
            = utils.load_preprocessed_data_g(cfg.preprocessed_data_path)

        # Choose Useful features
        if reduced_feature_list is not None:
            useful_feature_list_g = reduced_feature_list + [-1]
            reduced_feature_list.extend(list(range(-1, -29, -1)))
            self.x_train = self.x_train[:, reduced_feature_list]
            self.x_g_train = self.x_g_train[:, useful_feature_list_g]
            self.x_test = self.x_test[:, reduced_feature_list]
            self.x_g_test = self.x_g_test[:, useful_feature_list_g]

        self.single_model_pred_path = cfg.single_model_pred_path
        self.loss_log_path = cfg.loss_log_path
        self.csv_log_path = cfg.csv_log_path + 'christar1991_'
        self.save_auto_train_results = save_auto_train_results
        self.train_args = train_args
        self.cv_args = cv_args
        self.use_multi_group = use_multi_group
        self.mode = mode

        # Different Mode
        if mode == 'auto_grid_search':
            self.csv_log_path = cfg.grid_search_out_path
            self.pred_path = cfg.single_model_pred_path
        elif mode == 'auto_train':
            self.csv_log_path = cfg.auto_train_pred_path
            self.pred_path = cfg.auto_train_pred_path
        else:
            self.csv_log_path = cfg.csv_log_path + 'single_'
            self.pred_path = cfg.single_model_pred_path

    def train_model(self, model=None, grid_search_tuple_list=None):

        if grid_search_tuple_list is not None:
            param_name_list = []
            param_value_list = []
            for grid_search_tuple in grid_search_tuple_list:
                param_name = grid_search_tuple[0]
                param_value = grid_search_tuple[1]
                self.train_args['parameters'][param_name] = param_value
                param_name_list.append(param_name)
                param_value_list.append(param_value)
        else:
            param_name_list = None
            param_value_list = None

        # Parameters for Train
        model.train(pred_path=self.single_model_pred_path, loss_log_path=self.loss_log_path,
                    csv_log_path=self.csv_log_path, mode=self.mode, param_name_list=param_name_list,
                    param_value_list=param_value_list, cv_args=self.cv_args, **self.train_args)

    def Christar1991(self, train_seed, cv_seed, parameters=None, grid_search_tuple_list=None):
        """
            Model of week3 champion
        """
        if parameters is None:
            parameters = {'application': 'binary',
                          'learning_rate': 0.002,
                          'num_leaves': 128,              # <2^(max_depth)
                          'tree_learner': 'serial',
                          'max_depth': 8,                 # default=-1
                          'min_data_in_leaf': 20,         # default=20
                          'feature_fraction': 0.8,        # default=1
                          'bagging_fraction': 0.8,        # default=1
                          'bagging_freq': 5,              # default=0 perform bagging every k iteration
                          'bagging_seed': 6,              # default=3
                          'early_stopping_rounds': 50,
                          'max_bin': 255,
                          'metric': 'binary_logloss',
                          'verbosity': 1,
                          'seed': train_seed}
        else:
            parameters['seed'] = train_seed

        cv_generator = CrossValidation.era_k_fold_all_random

        self.train_args['parameters'] = parameters
        self.cv_args['cv_generator'] = cv_generator
        self.train_args['train_seed'] = train_seed
        self.cv_args['cv_seed'] = cv_seed

        model = LightGBM(self.x_g_train, self.y_train, self.w_train, self.e_train, self.x_g_test, self.pct_test,
                         self.id_test, use_multi_group=self.use_multi_group, num_boost_round=65)

        self.train_model(model=model, grid_search_tuple_list=grid_search_tuple_list)


class SKGridSearch(object):
    """
        Grid Search
    """
    def __init__(self):
        pass

    @staticmethod
    def lr_grid_search(train_seed, cv_seed):
        """
            Logistic Regression
        """
        _log_path = cfg.grid_search_log_path + 'lr_'

        x_train, y_train, w_train, e_train, x_test, pct_test, id_test = \
            utils.load_preprocessed_data(cfg.preprocessed_data_path)

        parameters = {'C': 1.0,
                      'class_weight': None,
                      'dual': False,
                      'fit_intercept': 'True',
                      'intercept_scaling': 1,
                      'max_iter': 100,
                      'multi_class': 'multinomial',
                      'n_jobs': -1,
                      'penalty': 'l2',
                      'solver': 'sag',
                      'tol': 0.0001,
                      'random_state': train_seed,
                      'verbose': 2,
                      'warm_start': False}

        LR = LRegression(x_train, y_train, w_train, e_train, x_test, pct_test, id_test)

        clf = LR.get_clf(parameters)

        # parameters_grid = None

        parameters_grid = {
                           'C': (0.2, 0.5, 1),
                           'max_iter': (50, 100, 200),
                           'tol': (0.001, 0.005, 0.01)
                           }

        SKLearnGridSearch.grid_search(_log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20, n_era=20,
                                      cv_seed=cv_seed, params=parameters, params_grid=parameters_grid)

        utils.print_grid_info('Logistic Regression', parameters, parameters_grid)

    @staticmethod
    def rf_grid_search(train_seed, cv_seed):
        """
            Random Forest
        """
        _log_path = cfg.grid_search_log_path + 'rf_'

        x_train, y_train, w_train, e_train, x_test, pct_test, id_test = \
            utils.load_preprocessed_data(cfg.preprocessed_data_path)

        parameters = {'n_estimators': 32,
                      'bootstrap': True,
                      'class_weight': None,
                      'criterion': 'gini',
                      'max_depth': 6,
                      'max_features': 7,
                      'max_leaf_nodes': None,
                      'min_impurity_decrease': 0.0,
                      'min_samples_leaf': 300,
                      'min_samples_split': 4000,
                      'min_weight_fraction_leaf': 0.0,
                      'n_jobs': -1,
                      'oob_score': True,
                      'random_state': train_seed,
                      'verbose': 2,
                      'warm_start': False}

        RF = RandomForest(x_train, y_train, w_train, e_train, x_test, pct_test, id_test)

        clf = RF.get_clf(parameters)

        # parameters_grid = None

        parameters_grid = {
                           # 'n_estimators': (30, 31, 32),
                           'max_depth': (2, 3),
                           # 'max_features': (6, 7),
                           'min_samples_leaf': (286, 287),
                           'min_samples_split': (3972, 3974, 3976, 3978)
                           }

        SKLearnGridSearch.grid_search(_log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20, n_era=20,
                                      cv_seed=cv_seed, params=parameters, params_grid=parameters_grid)

        utils.print_grid_info('Random Forest', parameters, parameters_grid)

    @staticmethod
    def et_grid_search(train_seed, cv_seed):
        """
            Extra Trees
        """
        _log_path = cfg.grid_search_log_path + 'et_'

        x_train, y_train, w_train, e_train, x_test, pct_test, id_test = \
            utils.load_preprocessed_data(cfg.preprocessed_data_path)

        parameters = {'bootstrap': True,
                      'n_estimators': 50,
                      'class_weight': None,
                      'criterion': 'gini',
                      'max_depth': 25,
                      'max_features': 'auto',
                      'max_leaf_nodes': None,
                      'min_impurity_decrease': 0.0,
                      'min_samples_leaf': 50,
                      'min_samples_split': 1000,
                      'min_weight_fraction_leaf': 0.0,
                      'n_jobs': -1,
                      'oob_score': True,
                      'random_state': train_seed,
                      'verbose': 2,
                      'warm_start': False}

        ET = ExtraTrees(x_train, y_train, w_train, e_train, x_test, pct_test, id_test)

        clf = ET.get_clf(parameters)

        # parameters_grid = None

        parameters_grid = {
                           'n_estimators': (30, 40, 50),
                           'max_depth': (5, 6),
                           'max_features': (6, 7),
                           'min_samples_leaf': (200, 250, 300),
                           'min_samples_split': (3000, 3500, 4000)
                           }

        SKLearnGridSearch.grid_search(_log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20, n_era=20,
                                      cv_seed=cv_seed, params=parameters, params_grid=parameters_grid)

        utils.print_grid_info('Extra Trees', parameters, parameters_grid)

    @staticmethod
    def ab_grid_search(train_seed, cv_seed):
        """
            AdaBoost
        """
        _log_path = cfg.grid_search_log_path + 'ab_'

        x_train, y_train, w_train, e_train, x_test, pct_test, id_test = \
            utils.load_preprocessed_data(cfg.preprocessed_data_path)

        et_for_ab_params = {'bootstrap': True,
                            'class_weight': None,
                            'criterion': 'gini',
                            'max_depth': 2,
                            'max_features': 7,
                            'max_leaf_nodes': None,
                            'min_impurity_decrease': 0.0,
                            'min_samples_leaf': 357,
                            'min_samples_split': 4909,
                            'min_weight_fraction_leaf': 0.0,
                            'n_estimators': 20,
                            'n_jobs': -1,
                            'oob_score': True,
                            'random_state': train_seed,
                            'verbose': 2,
                            'warm_start': False}
        clf_et_for_ab = ExtraTreesClassifier(**et_for_ab_params)

        parameters = {'algorithm': 'SAMME.R',
                      'base_estimator': clf_et_for_ab,
                      'learning_rate': 0.005,
                      'n_estimators': 100,
                      'random_state': train_seed}

        AB = AdaBoost(x_train, y_train, w_train, e_train, x_test, pct_test, id_test)

        clf = AB.get_clf(parameters)

        # parameters_grid = None

        parameters_grid = {
                           'learning_rate': (0.002, 0.003, 0.005),
                           'n_estimators': (50, 100),
                           #  'algorithm': 'SAMME.R',
                           #  'base_estimator': clf_et,
                           }

        SKLearnGridSearch.grid_search(_log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20, n_era=20,
                                      cv_seed=cv_seed, params=parameters, params_grid=parameters_grid)

        utils.print_grid_info('AdaBoost', parameters, parameters_grid)

    @staticmethod
    def gb_grid_search(train_seed, cv_seed):
        """
            GradientBoosting
        """
        _log_path = cfg.grid_search_log_path + 'gb_'

        x_train, y_train, w_train, e_train, x_test, pct_test, id_test = \
            utils.load_preprocessed_data(cfg.preprocessed_data_path)

        parameters = {'criterion': 'friedman_mse',
                      'init': None,
                      'learning_rate': 0.05,
                      'loss': 'deviance',
                      'max_depth': 25,
                      'max_features': 'auto',
                      'max_leaf_nodes': None,
                      'min_impurity_decrease': 0.0,
                      'min_impurity_split': None,
                      'min_samples_leaf': 50,
                      'min_samples_split': 1000,
                      'min_weight_fraction_leaf': 0.0,
                      'n_estimators': 200,
                      'presort': 'auto',
                      'random_state': train_seed,
                      'subsample': 0.8,
                      'verbose': 2,
                      'warm_start': False}

        GB = GradientBoosting(x_train, y_train, w_train, e_train, x_test, pct_test, id_test)

        clf = GB.get_clf(parameters)

        # parameters_grid = None

        parameters_grid = {
                           'n_estimators': (20, 50, 100),
                           'learning_rate': (0.05, 0.2, 0.5),
                           'max_depth': (5, 10, 15),
                           'max_features': (6, 8, 10),
                           'min_samples_leaf': (300, 400, 500),
                           'min_samples_split': (3000, 4000, 5000),
                           'subsample': (0.6, 0.8, 1)
                           }

        SKLearnGridSearch.grid_search(_log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20, n_era=20,
                                      cv_seed=cv_seed, params=parameters, params_grid=parameters_grid)

        utils.print_grid_info('GradientBoosting', parameters, parameters_grid)

    @staticmethod
    def xgb_grid_search(train_seed, cv_seed):
        """
            XGBoost
        """
        _log_path = cfg.grid_search_log_path + 'xgb_'

        x_train, y_train, w_train, e_train, x_test, pct_test, id_test = \
            utils.load_preprocessed_data(cfg.preprocessed_data_path)

        parameters = {'objective': 'binary:logistic',
                      'learning_rate': 0.002,
                      'n_estimators': 100,
                      'max_depth': 9,
                      'min_child_weight': 5,
                      'max_delta_step': 0,
                      'silent': False,
                      'subsample': 0.8,
                      'colsample_bytree': 0.8,
                      'colsample_bylevel': 1,
                      'base_score': 0.5,
                      'gamma': 0,
                      'reg_alpha': 0,
                      'reg_lambda': 0,
                      'nthread': -1,
                      'seed': train_seed
                      # 'missing': None,
                      # 'nthread': -1,
                      # 'scale_pos_weight': 1,
                      }

        XGB = SKLearnXGBoost(x_train, y_train, w_train, e_train, x_test, pct_test, id_test)

        clf = XGB.get_clf(parameters)

        # parameters_grid = None

        parameters_grid = {'learning_rate': (0.002, 0.005, 0.01),
                           'n_estimators': (20, 50, 100, 150),
                           'max_depth': (5, 7, 9),
                           # 'subsample': 0.8,
                           # 'colsample_bytree': 0.8,
                           # 'colsample_bylevel': 1,
                           # 'gamma': 0,
                           # 'min_child_weight': 1,
                           # 'max_delta_step': 0,
                           # 'base_score': 0.5,
                           # 'reg_alpha': 0,
                           # 'reg_lambda': 0,
                           }

        SKLearnGridSearch.grid_search(_log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20, n_era=20,
                                      cv_seed=cv_seed, params=parameters, params_grid=parameters_grid)

        utils.print_grid_info('XGBoost', parameters, parameters_grid)

    @staticmethod
    def lgb_grid_search(train_seed, cv_seed):
        """
            LightGBM
        """
        _log_path = cfg.grid_search_log_path + 'lgb_'

        x_train, y_train, w_train, e_train, x_test, pct_test, id_test = \
            utils.load_preprocessed_data(cfg.preprocessed_data_path)
        if preprocess.group_list is not None:
            x_g_train, x_g_test = \
                utils.load_preprocessed_data_g(cfg.preprocessed_data_path)
        else:
            x_g_train, x_g_test = x_train, x_test

        parameters = {'learning_rate': 0.006,
                      'boosting_type': 'gbdt',        # traditional Gradient Boosting Decision Tree.
                      # 'boosting_type': 'dart',        # Dropouts meet Multiple Additive Regression Trees.
                      # 'boosting_type': 'goss',        # Gradient-based One-Side Sampling.
                      # 'boosting_type': 'rf',          # Random Forest.
                      'num_leaves': 3,                # <2^(max_depth)
                      'max_depth': 8,                 # default=-1
                      'n_estimators': 79,
                      'max_bin': 1005,
                      'subsample_for_bin': 1981,
                      'objective': 'binary',
                      'min_split_gain': 0.,
                      'min_child_weight': 1,
                      'min_child_samples': 0,
                      'subsample': 0.723,
                      'subsample_freq': 3,
                      'colsample_bytree': 0.11,
                      'reg_alpha': 0.,
                      'reg_lambda': 0.,
                      'silent': False,
                      'seed': train_seed}

        LGB = SKLearnLightGBM(x_g_train, y_train, w_train, e_train, x_g_test, pct_test, id_test)

        clf = LGB.get_clf(parameters)

        # parameters_grid = None

        parameters_grid = {
                           'learning_rate': (0.002, 0.005, 0.01),
                           'n_estimators': (30, 60, 90),
                           'num_leaves': (32, 64, 128),             # <2^(max_depth)
                           'colsample_bytree': (0.6, 0.8, 0.1),
                           'max_depth': (6, 8, 10),                 # default=-1
                           # 'min_data_in_leaf': 20,                  # default=20
                           # 'bagging_fraction': (0.5, 0.7, 0.9),
                           # 'feature_fraction': (0.5, 0.7, 0.9),
                           # 'subsample_for_bin': (50000, 100000, 150000),
                           # 'subsample_freq': (4, 6, 8),
                           # 'subsample': (0.6, 0.8, 1.0),
                           # 'max_bin': (255, 355, 455)
                           }

        SKLearnGridSearch.grid_search(_log_path, x_train, y_train, e_train, clf, n_valid=4, n_cv=20, n_era=20,
                                      cv_seed=cv_seed, params=parameters, params_grid=parameters_grid)

        utils.print_grid_info('LightGBM', parameters, parameters_grid)

    # Stacking Layer LightGBM
    @staticmethod
    def stack_lgb_grid_search(train_seed, cv_seed):

        _log_path = cfg.grid_search_log_path + 'stk_lgb_'

        x_train, y_train, w_train, e_train, x_test, pct_test, id_test = \
            utils.load_preprocessed_data(cfg.preprocessed_data_path)
        x_g_train, x_g_test = \
            utils.load_preprocessed_data_g(cfg.preprocessed_data_path)
        blender_x_tree, blender_test_tree, blender_x_g_tree, blender_test_g_tree = \
            utils.load_stacked_data(cfg.stack_output_path + 'l2_')

        g_train = x_g_train[:, -1]
        x_train_reuse = x_train[:, :88]

        print('------------------------------------------------------')
        print('Stacking Reused Features of Train Set...')
        blender_x_tree = np.concatenate((blender_x_tree, x_train_reuse), axis=1)
        blender_x_g_tree = np.column_stack((blender_x_tree, g_train))

        parameters = {'learning_rate': 0.006,
                      'boosting_type': 'gbdt',        # traditional Gradient Boosting Decision Tree.
                      # 'boosting_type': 'dart',        # Dropouts meet Multiple Additive Regression Trees.
                      # 'boosting_type': 'goss',        # Gradient-based One-Side Sampling.
                      # 'boosting_type': 'rf',          # Random Forest.
                      'num_leaves': 3,  # <2^(max_depth)
                      'max_depth': 8,  # default=-1
                      'n_estimators': 79,
                      'max_bin': 1005,
                      'subsample_for_bin': 1981,
                      'objective': 'binary',
                      'min_split_gain': 0.,
                      'min_child_weight': 1,
                      'min_child_samples': 0,
                      'subsample': 0.723,
                      'subsample_freq': 3,
                      'colsample_bytree': 0.11,
                      'reg_alpha': 0.,
                      'reg_lambda': 0.,
                      'silent': False,
                      'random_state': train_seed}

        LGB = SKLearnLightGBM(blender_x_g_tree, y_train, w_train, e_train, blender_test_g_tree, pct_test, id_test)

        clf = LGB.get_clf(parameters)

        # parameters_grid = None

        parameters_grid = {
            'learning_rate': (0.002, 0.005, 0.01),
            'n_estimators': (30, 60, 90),
            'num_leaves': (32, 64, 128),             # <2^(max_depth)
            'colsample_bytree': (0.6, 0.8, 0.1),
            'max_depth': (6, 8, 10),                 # default=-1
            # 'min_data_in_leaf': 20,                  # default=20
            # 'bagging_fraction': (0.5, 0.7, 0.9),
            # 'feature_fraction': (0.5, 0.7, 0.9),
            # 'subsample_for_bin': (50000, 100000, 150000),
            # 'subsample_freq': (4, 6, 8),
            # 'subsample': (0.6, 0.8, 1.0),
            # 'max_bin': (255, 355, 455)
        }

        SKLearnGridSearch.grid_search(_log_path, blender_x_tree, y_train, e_train, clf, n_valid=4, n_cv=20, n_era=20,
                                      cv_seed=cv_seed, params=parameters, params_grid=parameters_grid)

        utils.print_grid_info('LightGBM', parameters, parameters_grid)


class PrejudgeTraining(object):
    """
        Prejudge - Training by Split Era sign
    """
    def __init__(self, reduced_feature_list=None, load_pickle=False, train_args=None):
        
        self.x_train, self.y_train, self.w_train, self.e_train, self.x_test, self.pct_test, self.id_test\
            = utils.load_preprocessed_data(cfg.preprocessed_data_path)
        if preprocess.group_list is not None:
            self.x_g_train, self.x_g_test = \
                utils.load_preprocessed_data_g(cfg.preprocessed_data_path)
        else:
            self.x_g_train, self.x_g_test = self.x_train, self.x_test
        self.x_train_p, self.y_train_p, self.w_train_p, self.e_train_p, self.x_g_train_p \
            = utils.load_preprocessed_positive_data(cfg.preprocessed_data_path)
        self.x_train_n, self.y_train_n, self.w_train_n, self.e_train_n, self.x_g_train_n \
            = utils.load_preprocessed_negative_data(cfg.preprocessed_data_path)
        self.load_pickle = load_pickle
        self.show_importance = train_args['show_importance']
        self.show_accuracy = train_args['show_accuracy']

        # Choose Useful features
        if reduced_feature_list is not None:
            useful_feature_list_g = reduced_feature_list + [-1]
            reduced_feature_list.extend(list(range(-1, -29, -1)))
            self.x_train = self.x_train[:, reduced_feature_list]
            self.x_g_train = self.x_g_train[:, useful_feature_list_g]
            self.x_test = self.x_test[:, reduced_feature_list]
            self.x_g_test = self.x_g_test[:, useful_feature_list_g]

    @staticmethod
    def get_binary_models_parameters(train_seed):
        """
            Set Parameters for models of PrejudgeBinary
        """
        era_training_params = {'application': 'binary',
                               'boosting': 'gbdt',                   # gdbt,rf,dart,goss
                               'learning_rate': 0.1,                 # default=0.1
                               'num_leaves': 75,                     # default=31       <2^(max_depth)
                               'max_depth': 9,                       # default=-1
                               'min_data_in_leaf': 2500,             # default=20       reduce over-fit
                               'min_sum_hessian_in_leaf': 1e-3,      # default=1e-3     reduce over-fit
                               'feature_fraction': 0.5,              # default=1
                               'feature_fraction_seed': train_seed,  # default=2
                               'bagging_fraction': 0.8,              # default=1
                               'bagging_freq': 2,                    # default=0        perform bagging every k iteration
                               'bagging_seed': train_seed,           # default=3
                               'lambda_l1': 0,                       # default=0
                               'lambda_l2': 0,                       # default=0
                               'min_gain_to_split': 0,               # default=0
                               'max_bin': 225,                       # default=255
                               'min_data_in_bin': 9,                 # default=5
                               'metric': 'binary_logloss',
                               'num_threads': -1,
                               'verbosity': 1,
                               'early_stopping_rounds': 50,          # default=0
                               'seed': train_seed}

        positive_params = {'application': 'binary',
                           'boosting': 'gbdt',                   # gdbt,rf,dart,goss
                           'learning_rate': 0.003,               # default=0.1
                           'num_leaves': 75,                     # default=31       <2^(max_depth)
                           'max_depth': 9,                       # default=-1
                           'min_data_in_leaf': 2500,             # default=20       reduce over-fit
                           'min_sum_hessian_in_leaf': 1e-3,      # default=1e-3     reduce over-fit
                           'feature_fraction': 0.5,              # default=1
                           'feature_fraction_seed': train_seed,  # default=2
                           'bagging_fraction': 0.8,              # default=1
                           'bagging_freq': 2,                    # default=0        perform bagging every k iteration
                           'bagging_seed': train_seed,           # default=3
                           'lambda_l1': 0,                       # default=0
                           'lambda_l2': 0,                       # default=0
                           'min_gain_to_split': 0,               # default=0
                           'max_bin': 225,                       # default=255
                           'min_data_in_bin': 9,                 # default=5
                           'metric': 'binary_logloss',
                           'num_threads': -1,
                           'verbosity': 1,
                           'early_stopping_rounds': 50,          # default=0
                           'seed': train_seed}

        negative_params = {'application': 'binary',
                           'boosting': 'gbdt',                   # gdbt,rf,dart,goss
                           'learning_rate': 0.003,               # default=0.1
                           'num_leaves': 75,                     # default=31       <2^(max_depth)
                           'max_depth': 9,                       # default=-1
                           'min_data_in_leaf': 2500,             # default=20       reduce over-fit
                           'min_sum_hessian_in_leaf': 1e-3,      # default=1e-3     reduce over-fit
                           'feature_fraction': 0.5,              # default=1
                           'feature_fraction_seed': train_seed,  # default=2
                           'bagging_fraction': 0.8,              # default=1
                           'bagging_freq': 2,                    # default=0        perform bagging every k iteration
                           'bagging_seed': train_seed,           # default=3
                           'lambda_l1': 0,                       # default=0
                           'lambda_l2': 0,                       # default=0
                           'min_gain_to_split': 0,               # default=0
                           'max_bin': 225,                       # default=255
                           'min_data_in_bin': 9,                 # default=5
                           'metric': 'binary_logloss',
                           'num_threads': -1,
                           'verbosity': 1,
                           'early_stopping_rounds': 50,          # default=0
                           'seed': train_seed}

        models_parameters = [era_training_params, positive_params, negative_params]

        return models_parameters

    @staticmethod
    def get_multiclass_models_parameters(train_seed):
        """
            Set Parameters for models of PrejudgeMultiClass
        """
        era_training_params = {'application': 'multiclass',
                               'num_class': 20,
                               'learning_rate': 0.2,
                               'num_leaves': 80,            # <2^(max_depth)
                               'tree_learner': 'serial',
                               'max_depth': 7,              # default=-1
                               'min_data_in_leaf': 2000,    # default=20
                               'feature_fraction': 0.5,     # default=1
                               'bagging_fraction': 0.6,     # default=1
                               'bagging_freq': 5,           # default=0 perform bagging every k iteration
                               'bagging_seed': 1,           # default=3
                               'early_stopping_rounds': 50,
                               'max_bin': 50,
                               'metric': 'multi_logloss',
                               'verbosity': 0,
                               'seed': train_seed}

        multiclass_params = {'application': 'binary',
                             'learning_rate': 0.002,
                             'num_leaves': 80,              # <2^(max_depth)
                             'tree_learner': 'serial',
                             'max_depth': 7,                # default=-1
                             'min_data_in_leaf': 2000,      # default=20
                             'feature_fraction': 0.5,       # default=1
                             'bagging_fraction': 0.6,       # default=1
                             'bagging_freq': 5,             # default=0 perform bagging every k iteration
                             'bagging_seed': 1,             # default=3
                             'early_stopping_rounds': 65,
                             'max_bin': 50,
                             'metric': 'binary_logloss',
                             'verbosity': 1,
                             'seed': train_seed}

        models_parameters = [era_training_params, multiclass_params]

        return models_parameters

    def binary_train(self, train_seed, cv_seed):
        """
            Training model of PrejudgeBinary
        """
        
        models_parameters = PrejudgeTraining.get_binary_models_parameters(train_seed)

        positive_era_list = preprocess.positive_era_list
        negative_era_list = preprocess.negative_era_list

        hyper_parameters = {'cv_seed': cv_seed,
                            'train_seed': train_seed,
                            'n_splits_e': 10,
                            'num_boost_round_e': 3000,
                            'n_cv_e': 10,
                            'n_valid_p': 2,
                            'n_cv_p': 12,
                            'n_era_p': len(positive_era_list),
                            'num_boost_round_p': 80,
                            'era_list_p': positive_era_list,
                            'n_valid_n': 1,
                            'n_cv_n': 8,
                            'n_era_n': len(negative_era_list),
                            'num_boost_round_n': 80,
                            'era_list_n': negative_era_list,
                            'force_convert_era': True,
                            'use_weight': True,
                            'show_importance': self.show_importance,
                            'show_accuracy': self.show_accuracy}

        utils.check_dir([cfg.prejudge_pred_path, cfg.prejudged_data_path,
                         cfg.prejudge_loss_log_path, cfg.csv_log_path])

        PB = prejudge.PrejudgeBinary(self.x_train, self.y_train, self.w_train, self.e_train, self.x_g_train, 
                                     self.x_train_p, self.y_train_p, self.w_train_p, self.e_train_p, self.x_g_train_p, 
                                     self.x_train_n, self.y_train_n, self.w_train_n, self.e_train_n, self.x_g_train_n, 
                                     self.x_test, self.id_test, self.x_g_test,
                                     pred_path=cfg.prejudge_pred_path,
                                     prejudged_data_path=cfg.prejudged_data_path,
                                     loss_log_path=cfg.prejudge_loss_log_path,
                                     csv_log_path=cfg.csv_log_path + 'prejudge_',
                                     models_parameters=models_parameters,
                                     hyper_parameters=hyper_parameters)

        PB.train(load_pickle=self.load_pickle, load_pickle_path=None)

    def multiclass_train(self, train_seed, cv_seed):
        """
            Training model of PrejudgeMultiClass
        """

        models_parameters = PrejudgeTraining.get_multiclass_models_parameters(train_seed)

        hyper_parameters = {'cv_seed': cv_seed,
                            'train_seed': train_seed,
                            'n_splits_e': 10,
                            'num_boost_round_e': 500,
                            'n_cv_e': 10,
                            'n_era': 20,
                            'n_valid_m': 4,
                            'n_cv_m': 20,
                            'num_boost_round_m': 65,
                            'use_weight': True,
                            'show_importance': self.show_importance,
                            'show_accuracy': self.show_accuracy}

        utils.check_dir([cfg.prejudge_pred_path, cfg.prejudged_data_path,
                         cfg.prejudge_loss_log_path, cfg.csv_log_path])

        PB = prejudge.PrejudgeMultiClass(self.x_train, self.y_train, self.w_train, self.e_train, self.x_g_train,
                                         self.x_test, self.id_test, self.x_g_test,
                                         pred_path=cfg.prejudge_pred_path,
                                         prejudged_data_path=cfg.prejudged_data_path,
                                         loss_log_path=cfg.prejudge_loss_log_path,
                                         csv_log_path=cfg.csv_log_path + 'prejudge_',
                                         models_parameters=models_parameters,
                                         hyper_parameters=hyper_parameters)

        PB.train(load_pickle=self.load_pickle, load_pickle_path=None)


class ModelStacking(object):
    """
        Stacking
    """
    def __init__(self, reduced_feature_list=None, save_auto_train_results=True, train_args=None, mode=None):

        self.x_train, self.y_train, self.w_train, self.e_train, self.x_test, self.id_test\
            = utils.load_preprocessed_data(cfg.preprocessed_data_path)
        if preprocess.group_list is not None:
            self.x_g_train, self.x_g_test = \
                utils.load_preprocessed_data_g(cfg.preprocessed_data_path)
        else:
            self.x_g_train, self.x_g_test = self.x_train, self.x_test

        # Choose Useful features
        if reduced_feature_list is not None:
            useful_feature_list_g = reduced_feature_list + [-1]
            reduced_feature_list.extend(list(range(-1, -29, -1)))
            self.x_train = self.x_train[:, reduced_feature_list]
            self.x_g_train = self.x_g_train[:, useful_feature_list_g]
            self.x_test = self.x_test[:, reduced_feature_list]
            self.x_g_test = self.x_g_test[:, useful_feature_list_g]

        self.save_auto_train_results = save_auto_train_results
        self.train_args = train_args
        self.train_options = {'show_importance': train_args['show_importance'],
                              'show_accuracy': train_args['show_accuracy'],
                              'save_final_pred': train_args['save_final_pred'],
                              'save_final_prob_train': train_args['save_final_prob_train'],
                              'save_cv_pred': train_args['save_cv_pred'],
                              'save_cv_prob_train': train_args['save_cv_prob_train'],
                              'save_csv_log': train_args['save_csv_log']}
        self.mode = mode

    @staticmethod
    def get_layer1_params(train_seed):
        """
            Set Parameters for models of layer1
        """
        # Parameters of LightGBM
        lgb_params = {'application': 'binary',
                      'boosting': 'gbdt',                   # gdbt,rf,dart,goss
                      'learning_rate': 0.003,               # default=0.1
                      'num_leaves': 83,                     # default=31       <2^(max_depth)
                      'max_depth': 9,                       # default=-1
                      'min_data_in_leaf': 2500,             # default=20       reduce over-fit
                      'min_sum_hessian_in_leaf': 1e-3,      # default=1e-3     reduce over-fit
                      'feature_fraction': 0.5,              # default=1
                      'feature_fraction_seed': train_seed,  # default=2
                      'bagging_fraction': 0.8,              # default=1
                      'bagging_freq': 2,                    # default=0        perform bagging every k iteration
                      'bagging_seed': train_seed,           # default=3
                      'lambda_l1': 0,                       # default=0
                      'lambda_l2': 0,                       # default=0
                      'min_gain_to_split': 0,               # default=0
                      'max_bin': 225,                       # default=255
                      'min_data_in_bin': 9,                 # default=5
                      'metric': 'binary_logloss',
                      'num_threads': -1,
                      'verbosity': 1,
                      'early_stopping_rounds': 10000,          # default=0
                      'seed': train_seed}

        # Parameters of XGBoost
        xgb_params = {'eta': 0.003,
                      'gamma': 0,                       # 如果loss function小于设定值，停止产生子节点
                      'max_depth': 9,                   # default=6
                      'min_child_weight': 18,           # default=1，建立每个模型所需最小样本权重和
                      'subsample': 0.9,                 # 建立树模型时抽取子样本占整个样本的比例
                      'colsample_bytree': 0.8,          # 建立树时对特征随机采样的比例
                      'colsample_bylevel': 0.7,
                      'lambda': 0,
                      'alpha': 0,
                      'early_stopping_rounds': 10000,
                      'n_jobs': -1,
                      'objective': 'binary:logistic',
                      'eval_metric': 'logloss',
                      'seed': train_seed}

        # # Parameters of XGBoost
        # xgb_params = {'eta': 0.008,
        #               'gamma': 0,                           # 如果loss function小于设定值，停止产生子节点
        #               'max_depth': 7,                       # default=6
        #               'min_child_weight': 15,               # default=1，建立每个模型所需最小样本权重和
        #               'subsample': 0.8,                     # 建立树模型时抽取子样本占整个样本的比例
        #               'colsample_bytree': 0.7,              # 建立树时对特征随机采样的比例
        #               'colsample_bylevel': 0.6,
        #               'lambda': 2500,
        #               'alpha': 0,
        #               'early_stopping_rounds': 30,
        #               'nthread': -1,
        #               'objective': 'binary:logistic',
        #               'eval_metric': 'logloss',
        #               'seed': train_seed}

        # # Parameters of AdaBoost
        # et_for_ab_params = {'bootstrap': True,
        #                     'class_weight': None,
        #                     'criterion': 'gini',
        #                     'max_depth': 2,
        #                     'max_features': 7,
        #                     'max_leaf_nodes': None,
        #                     'min_impurity_decrease': 0.0,
        #                     'min_samples_leaf': 357,
        #                     'min_samples_split': 4909,
        #                     'min_weight_fraction_leaf': 0.0,
        #                     'n_estimators': 20,
        #                     'n_jobs': -1,
        #                     'oob_score': True,
        #                     'random_state': train_seed,
        #                     'verbose': 2,
        #                     'warm_start': False}
        # clf_et_for_ab = models.ExtraTreesClassifier(**et_for_ab_params)
        # ab_params = {'algorithm': 'SAMME.R',
        #              'base_estimator': clf_et_for_ab,
        #              'learning_rate': 0.0051,
        #              'n_estimators': 9,
        #              'random_state': train_seed}
        #
        # # Parameters of Random Forest
        # rf_params = {'bootstrap': True,
        #              'class_weight': None,
        #              'criterion': 'gini',
        #              'max_depth': 2,
        #              'max_features': 7,
        #              'max_leaf_nodes': None,
        #              'min_impurity_decrease': 0.0,
        #              'min_samples_leaf': 286,
        #              'min_samples_split': 3974,
        #              'min_weight_fraction_leaf': 0.0,
        #              'n_estimators': 32,
        #              'n_jobs': -1,
        #              'oob_score': True,
        #              'random_state': train_seed,
        #              'verbose': 2,
        #              'warm_start': False}
        #
        # # Parameters of Extra Trees
        # et_params = {'bootstrap': True,
        #              'class_weight': None,
        #              'criterion': 'gini',
        #              'max_depth': 2,
        #              'max_features': 7,
        #              'max_leaf_nodes': None,
        #              'min_impurity_decrease': 0.0,
        #              'min_samples_leaf': 357,
        #              'min_samples_split': 4909,
        #              'min_weight_fraction_leaf': 0.0,
        #              'n_estimators': 20,
        #              'n_jobs': -1,
        #              'oob_score': True,
        #              'random_state': train_seed,
        #              'verbose': 2,
        #              'warm_start': False}
        #
        # # Parameters of Gradient Boost
        # gb_params = {'criterion': 'friedman_mse',
        #              'init': None,
        #              'learning_rate': 0.002,
        #              'loss': 'deviance',
        #              'max_depth': 5,
        #              'max_features': 'auto',
        #              'max_leaf_nodes': None,
        #              'min_impurity_decrease': 0.0,
        #              'min_impurity_split': None,
        #              'min_samples_leaf': 50,
        #              'min_samples_split': 1000,
        #              'min_weight_fraction_leaf': 0.0,
        #              'n_estimators': 200,
        #              'presort': 'auto',
        #              'random_state': train_seed,
        #              'subsample': 0.8,
        #              'verbose': 2,
        #              'warm_start': False}

        # Parameters of Deep Neural Network
        dnn_params = {'version': '1.0',
                      'epochs': 4,
                      'unit_number': [48, 24, 12],
                      'learning_rate': 0.0001,
                      'keep_probability': 0.4,
                      'batch_size': 256,
                      'seed': train_seed,
                      'display_step': 100,
                      'save_path': cfg.dnn_checkpoint_path,
                      'log_path': cfg.dnn_log_path}

        # List of parameters for layer1
        layer1_params = [
                        lgb_params,
                        xgb_params,
                        # ab_params,
                        # rf_params,
                        # et_params,
                        # gb_params,
                        dnn_params
                        ]

        return layer1_params

    @staticmethod
    def get_layer2_params(train_seed):
        """
            Set Parameters for models of layer2
        """
        # Parameters of LightGBM
        lgb_params = {'application': 'binary',
                      'boosting': 'gbdt',                   # gdbt,rf,dart,goss
                      'learning_rate': 0.003,               # default=0.1
                      'num_leaves': 83,                     # default=31       <2^(max_depth)
                      'max_depth': 9,                       # default=-1
                      'min_data_in_leaf': 2500,             # default=20       reduce over-fit
                      'min_sum_hessian_in_leaf': 1e-3,      # default=1e-3     reduce over-fit
                      'feature_fraction': 0.5,              # default=1
                      'feature_fraction_seed': train_seed,  # default=2
                      'bagging_fraction': 0.8,              # default=1
                      'bagging_freq': 2,                    # default=0        perform bagging every k iteration
                      'bagging_seed': train_seed,           # default=3
                      'lambda_l1': 0,                       # default=0
                      'lambda_l2': 0,                       # default=0
                      'min_gain_to_split': 0,               # default=0
                      'max_bin': 225,                       # default=255
                      'min_data_in_bin': 9,                 # default=5
                      'metric': 'binary_logloss',
                      'num_threads': -1,
                      'verbosity': 1,
                      'early_stopping_rounds': 50,          # default=0
                      'seed': train_seed}

        # Parameters of Deep Neural Network
        # dnn_params = {'version': '1.0',
        #               'epochs': 10,
        #               'unit_number': [4, 2],
        #               'learning_rate': 0.0001,
        #               'keep_probability': 0.8,
        #               'batch_size': 256,
        #               'seed': train_seed,
        #               'display_step': 100,
        #               'save_path': dnn_save_path,
        #               'log_path': dnn_log_path}

        # List of parameters for layer2
        layer2_params = [
                        lgb_params,
                        # dnn_params
                        ]

        return layer2_params

    @staticmethod
    def get_final_layer_params(train_seed):
        """
            Set Parameters for models of final layer
        """
        # Parameters of LightGBM
        lgb_params = {'application': 'binary',
                      'boosting': 'gbdt',                   # gdbt,rf,dart,goss
                      'learning_rate': 0.003,               # default=0.1
                      'num_leaves': 85,                     # default=31       <2^(max_depth)
                      'max_depth': 9,                       # default=-1
                      'min_data_in_leaf': 2500,             # default=20       reduce over-fit
                      'min_sum_hessian_in_leaf': 1e-3,      # default=1e-3     reduce over-fit
                      'feature_fraction': 0.5,              # default=1
                      'feature_fraction_seed': train_seed,  # default=2
                      'bagging_fraction': 0.8,              # default=1
                      'bagging_freq': 2,                    # default=0        perform bagging every k iteration
                      'bagging_seed': train_seed,           # default=3
                      'lambda_l1': 0,                       # default=0
                      'lambda_l2': 0,                       # default=0
                      'min_gain_to_split': 0,               # default=0
                      'max_bin': 225,                       # default=255
                      'min_data_in_bin': 9,                 # default=5
                      'metric': 'binary_logloss',
                      'num_threads': -1,
                      'verbosity': 1,
                      'early_stopping_rounds': 10000,          # default=0
                      'seed': train_seed}

        # Parameters of Deep Neural Network
        # dnn_params = {'version': '1.0',
        #               'epochs': 10,
        #               'unit_number': [4, 2],
        #               'learning_rate': 0.0001,
        #               'keep_probability': 0.8,
        #               'batch_size': 256,
        #               'seed': train_seed,
        #               'display_step': 100,
        #               'save_path': dnn_save_path,
        #               'log_path': dnn_log_path}

        return lgb_params

    @staticmethod
    def deep_stack_train(train_seed, cv_seed):
        """
            Training model using DeepStack model
        """
        hyper_params = {'n_valid': (4, 4),
                        'n_era': (20, 20),
                        'n_epoch': (3, 1),
                        'train_seed': train_seed,
                        'cv_seed': cv_seed,
                        'num_boost_round_lgb_l1': 65,
                        'num_boost_round_xgb_l1': 30,
                        'num_boost_round_lgb_l2': 65,
                        'num_boost_round_final': 65,
                        'show_importance': False,
                        'show_accuracy': True,
                        'save_epoch_results': False}

        layer1_prams = ModelStacking.get_layer1_params(train_seed)
        layer2_prams = ModelStacking.get_layer2_params(train_seed)
        # layer3_prams = ModelStacking.get_layer3_params(train_seed)

        layers_params = [layer1_prams,
                         layer2_prams,
                         # layer3_prams
                         ]

        x_train, y_train, w_train, e_train, x_test, id_test = \
            utils.load_preprocessed_data(cfg.preprocessed_data_path)
        x_g_train, x_g_test = \
            utils.load_preprocessed_data_g(cfg.preprocessed_data_path)

        utils.check_dir([cfg.stack_pred_path, cfg.loss_log_path, cfg.stack_output_path])

        STK = stacking.DeepStack(x_train, y_train, w_train, e_train, x_test, id_test, x_g_train, x_g_test,
                                 pred_path=cfg.stack_pred_path, loss_log_path=cfg.loss_log_path,
                                 stack_output_path=cfg.stack_output_path, hyper_params=hyper_params,
                                 layers_params=layers_params)

        STK.stack()

    def prejudge_stack_tree_train(self, train_seed, cv_seed):
        """
            Training model using StackTree model
        """
        if self.save_auto_train_results:
            auto_train_path = cfg.auto_train_pred_path
            utils.check_dir([auto_train_path])
        else:
            auto_train_path = None

        # useful_feature_list_l1 = [2, 7, 8, 10, 15, 23, 26, 32, 51, 53, 64, 69, 70, 73, 77, 86]
        useful_feature_list_l1 = None
        reuse_feature_list_final = range(87)

        era_list_n = preprocess.negative_era_list
        era_sign_train = np.array([0 if era in era_list_n else 1 for era in self.e_train])
        era_sign_test = utils.load_pkl_to_data(cfg.prejudged_data_path + 'binary_era_sign_test.p')

        hyper_params = {'n_valid': (4, 4),
                        'n_era': (20, 20),
                        'n_epoch': (1, 8),
                        'final_n_cv': 20,
                        'train_seed': train_seed,
                        'cv_seed': cv_seed,
                        'models_l1': ('lgb', 'xgb', 'dnn'),
                        'models_l2': (),
                        'model_final': 'lgb',
                        'num_boost_round_lgb_l1': 108,
                        'num_boost_round_xgb_l1': 110,
                        'num_boost_round_final': 88,
                        'useful_feature_list_l1': useful_feature_list_l1,
                        'reuse_feature_list_final': reuse_feature_list_final,
                        'scale_blender_final': True,
                        'save_epoch_results': False}

        layer1_params = ModelStacking.get_layer1_params(train_seed)
        # layer2_params = ModelStacking.get_layer2_params(train_seed)
        final_layer_params = ModelStacking.get_final_layer_params(train_seed)

        layers_params = [layer1_params,
                         # layer2_params,
                         final_layer_params]

        idx = self.train_args['idx']

        STK = stacking.PrejudgeStackTree(self.x_train, self.y_train, self.w_train, self.e_train,
                                         self.x_test, self.id_test, self.x_g_train, self.x_g_test,
                                         layers_params=layers_params, hyper_params=hyper_params,
                                         options=self.train_options)

        utils.check_dir([cfg.stack_pred_path, cfg.loss_log_path, cfg.stack_output_path, cfg.csv_log_path])

        STK.stack(pred_path=cfg.stack_pred_path, auto_train_pred_path=auto_train_path, loss_log_path=cfg.loss_log_path,
                  stack_output_path=cfg.stack_output_path, csv_log_path=cfg.csv_log_path+'stack_final_',
                  era_list_n=era_list_n, era_sign_train=era_sign_train, era_sign_test=era_sign_test, csv_idx=idx)

    def stack_tree_train(self, train_seed, cv_seed):
        """
            Training model using StackTree model
        """
        if self.save_auto_train_results:
            auto_train_path = cfg.auto_train_pred_path
            utils.check_dir([auto_train_path])
        else:
            auto_train_path = None

        # useful_feature_list_l1 = [2, 7, 8, 10, 15, 23, 26, 32, 51, 53, 64, 69, 70, 73, 77, 86]
        useful_feature_list_l1 = None
        reuse_feature_list_final = range(87)

        hyper_params = {'n_valid': (4, 4),
                        'n_era': (20, 20),
                        'n_epoch': (1, 8),
                        'final_n_cv': 20,
                        'train_seed': train_seed,
                        'cv_seed': cv_seed,
                        'models_l1': ('lgb', 'xgb', 'dnn'),
                        'models_l2': (),
                        'model_final': 'lgb',
                        'num_boost_round_lgb_l1': 108,
                        'num_boost_round_xgb_l1': 110,
                        'num_boost_round_final': 88,
                        'useful_feature_list_l1': useful_feature_list_l1,
                        'reuse_feature_list_final': reuse_feature_list_final,
                        'scale_blender_final': True,
                        'save_epoch_results': False}

        layer1_params = ModelStacking.get_layer1_params(train_seed)
        # layer2_params = ModelStacking.get_layer2_params(train_seed)
        final_layer_params = ModelStacking.get_final_layer_params(train_seed)

        layers_params = [layer1_params,
                         # layer2_params,
                         final_layer_params]

        idx = self.train_args['idx']

        STK = stacking.StackTree(self.x_train, self.y_train, self.w_train, self.e_train,
                                 self.x_test, self.id_test, self.x_g_train, self.x_g_test,
                                 layers_params=layers_params, hyper_params=hyper_params, options=self.train_options)

        utils.check_dir([cfg.stack_pred_path, cfg.loss_log_path, cfg.stack_output_path, cfg.csv_log_path])

        STK.stack(pred_path=cfg.stack_pred_path, auto_train_pred_path=auto_train_path, loss_log_path=cfg.loss_log_path,
                  stack_output_path=cfg.stack_output_path, csv_log_path=cfg.csv_log_path+'stack_final_', csv_idx=idx)

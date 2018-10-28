import time
import os
import re
import sys
import copy
import numpy as np
import tensorflow as tf
from os.path import isdir
from models.classifiers import ModelBase
from models import utils
from models.cross_validation import CrossValidation


class DeepNeuralNetworks(ModelBase):
    """
        Deep Neural Networks
    """
    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, pct_te, id_te,
                 x_va=None, y_va=None, w_va=None, e_va=None,
                 use_multi_group=False, parameters=None):

        super(DeepNeuralNetworks, self).__init__(x_tr, y_tr, w_tr, e_tr, x_te, pct_te, id_te,
                                  x_va, y_va, w_va, e_va, use_multi_group)

        # Hyperparameters
        self.parameters = parameters
        self.version = parameters['version']
        self.epochs = parameters['epochs']
        self.unit_number = parameters['unit_number']
        self.learning_rate = parameters['learning_rate']
        self.keep_probability = parameters['keep_probability']
        self.batch_size = parameters['batch_size']
        self.train_seed = parameters['seed']
        self.display_step = parameters['display_step']
        self.save_path = parameters['save_path']
        self.log_path = parameters['log_path']

        utils.check_dir([self.save_path, self.log_path])

    def print_start_info(self):

        print('------------------------------------------------------')
        print('Training Deep Neural Networks...')
        print('------------------------------------------------------')

        self.model_name = 'dnn'

    def get_pattern(self):

        # [0] | CV: 10 | Epoch: 4/4 | Batch: 7300 | Time: 352.85s | Train_Loss: 0.71237314 | Valid_Loss: 0.72578128
        return re.compile(r'\[(\d*)\].*Train_Loss: (.*) \| Valid_Loss: (.*)')

    # Input Tensors
    def input_tensor(self):

        feature_num = self.x_train.shape[1]

        inputs_ = tf.placeholder(tf.float32, [None, feature_num], name='inputs')
        labels_ = tf.placeholder(tf.float32, None, name='labels')
        loss_weights_ = tf.placeholder(tf.float32, None, name='loss_weights')
        learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')
        keep_prob_ = tf.placeholder(tf.float32, name='keep_prob')
        is_training_ = tf.placeholder(tf.bool, name='is_training')

        return inputs_, labels_, loss_weights_, learning_rate_, keep_prob_, is_training_

    # Full Connected Layer
    def fc_layer(self, x_tensor, layer_name, num_outputs, keep_prob, is_training):

        if is_training is True:
            print('Using Batch Normalization')

        with tf.name_scope(layer_name):

            # x_shape = features.get_shape().as_list()
            # weights_initializer = tf.truncated_normal_initializer(stddev=2.0 / math.sqrt(x_shape[1])),
            # weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN',
            #                                                                      seed=self.train_seed)
            weights_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32, seed=self.train_seed)
            weights_reg = tf.contrib.layers.l2_regularizer(1e-3)
            normalizer_fn = tf.contrib.layers.batch_norm
            normalizer_params = {'is_training': is_training}

            fc = tf.contrib.layers.fully_connected(inputs=x_tensor,
                                                   num_outputs=num_outputs,
                                                   activation_fn=tf.nn.sigmoid,
                                                   weights_initializer=weights_initializer,
                                                   weights_regularizer=weights_reg,
                                                   normalizer_fn=normalizer_fn,
                                                   normalizer_params=normalizer_params,
                                                   biases_initializer=tf.zeros_initializer(dtype=tf.float32))

            tf.summary.histogram('fc_layer', fc)

            fc = tf.nn.dropout(fc, keep_prob)

        return fc

    # Output Layer
    def output_layer(self, x_tensor, layer_name, num_outputs):

        with tf.name_scope(layer_name):

            # weights_initializer = tf.truncated_normal_initializer(stddev=2.0 / math.sqrt(x_shape[1])),
            # weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN',
            #                                                                      seed=self.train_seed)
            weights_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32, seed=self.train_seed)

            out = tf.contrib.layers.fully_connected(inputs=x_tensor,
                                                    num_outputs=num_outputs,
                                                    activation_fn=None,
                                                    weights_initializer=weights_initializer,
                                                    biases_initializer=tf.zeros_initializer(dtype=tf.float32))

        return out

    # Model
    def model(self, x, n_unit, keep_prob, is_training):

        #  fc1 = fc_layer(x, 'fc1', n_unit[0], keep_prob)
        #  fc2 = fc_layer(fc1, 'fc2', n_unit[1], keep_prob)
        #  fc3 = fc_layer(fc2, 'fc3', n_unit[2], keep_prob)
        #  fc4 = fc_layer(fc3, 'fc4', n_unit[3], keep_prob)
        #  fc5 = fc_layer(fc4, 'fc5', n_unit[4], keep_prob)
        #  logit_ = self.output_layer(fc5, 'output', 1)

        fc = [x]

        for i in range(len(n_unit)):
            fc.append(self.fc_layer(fc[i], 'fc{}'.format(i + 1), n_unit[i], keep_prob, is_training))

        logit_ = self.output_layer(fc[len(n_unit)], 'output', 1)

        return logit_

    # LogLoss
    @staticmethod
    def log_loss(logit, w, y):

        with tf.name_scope('prob'):
            logit = tf.squeeze(logit)
            prob = tf.nn.sigmoid(logit)

        with tf.name_scope('log_loss'):

            w = w / tf.reduce_sum(w)
            ones = tf.ones_like(y, dtype=tf.float32)
            loss = - tf.reduce_sum(w * (y * tf.log(prob) + (ones-y) * tf.log(ones-prob)))
            # loss = tf.losses.log_loss(labels=y, predictions=prob, weights=w)

        tf.summary.scalar('log_loss', loss)

        return loss

    # Get Batches
    @staticmethod
    def get_batches(x, y, w, batch_num):

        n_batches = len(x) // batch_num

        for ii in range(0, n_batches * batch_num + 1, batch_num):

            if ii != n_batches * batch_num - 1:
                batch_x, batch_y, batch_w = x[ii: ii + batch_num], y[ii: ii + batch_num], w[ii: ii + batch_num]
            else:
                batch_x, batch_y, batch_w = x[ii:], y[ii:], w[ii:]

            yield batch_x, batch_y, batch_w

    # Get Batches for Prediction
    @staticmethod
    def get_batches_for_predict(x, batch_num):

        n_batches = len(x) // batch_num

        for ii in range(0, n_batches * batch_num + 1, batch_num):

            if ii != n_batches * batch_num - 1:
                batch_x = x[ii: ii + batch_num]
            else:
                batch_x = x[ii:]

            yield batch_x

    # Get Probabilities
    def get_prob(self, sess, logits, x, batch_num, inputs, keep_prob, is_training):

        logits_pred = np.array([])

        for x_batch in self.get_batches_for_predict(x, batch_num):
            logits_pred_batch = sess.run(logits, {inputs: x_batch, keep_prob: 1.0, is_training: False})
            logits_pred_batch = logits_pred_batch.flatten()
            logits_pred = np.concatenate((logits_pred, logits_pred_batch))

        prob = 1.0 / (1.0 + np.exp(-logits_pred))

        return prob

    # Trainer
    def trainer(self, sess, cv_counter, x_train, y_train, w_train, x_valid, y_valid, w_valid,
                optimizer, merged, cost_, inputs, labels, weights, lr, keep_prob, is_training, start_time):

        train_log_path = self.log_path + self.version + '/cv_{}/train'.format(cv_counter)
        valid_log_path = self.log_path + self.version + '/cv_{}/valid'.format(cv_counter)

        if not isdir(train_log_path):
            os.makedirs(train_log_path)
        if not isdir(valid_log_path):
            os.makedirs(valid_log_path)

        train_writer = tf.summary.FileWriter(train_log_path, sess.graph)
        valid_writer = tf.summary.FileWriter(valid_log_path)

        sess.run(tf.global_variables_initializer())

        batch_counter = 0
        idx = 0

        for epoch_i in range(self.epochs):

            for batch_i, (batch_x, batch_y, batch_w) in enumerate(self.get_batches(x_train,
                                                                                   y_train,
                                                                                   w_train,
                                                                                   self.batch_size)):

                batch_counter += 1

                _, cost = sess.run([optimizer, cost_],
                                   {inputs: batch_x,
                                    labels: batch_y,
                                    weights: batch_w,
                                    lr: self.learning_rate,
                                    keep_prob: self.keep_probability,
                                    is_training: True})

                if str(cost) == 'nan':
                    raise ValueError('NaN BUG!!! Try Another Seed!!!')

                if batch_counter % self.display_step == 0 and batch_i > 0:

                    idx += 1

                    summary_train, cost_train = sess.run([merged, cost_],
                                                         {inputs: batch_x,
                                                          labels: batch_y,
                                                          weights: batch_w,
                                                          keep_prob: 1.0,
                                                          is_training: False})
                    train_writer.add_summary(summary_train, batch_counter)

                    cost_valid_all = []

                    for iii, (valid_batch_x,
                              valid_batch_y,
                              valid_batch_w) in enumerate(self.get_batches(x_valid,
                                                                           y_valid,
                                                                           w_valid,
                                                                           self.batch_size)):
                        summary_valid_i, cost_valid_i = sess.run([merged, cost_],
                                                                 {inputs: valid_batch_x,
                                                                  labels: valid_batch_y,
                                                                  weights: valid_batch_w,
                                                                  keep_prob: 1.0,
                                                                  is_training: False})

                        valid_writer.add_summary(summary_valid_i, batch_counter)

                        cost_valid_all.append(cost_valid_i)

                    cost_valid = sum(cost_valid_all) / len(cost_valid_all)

                    total_time = time.time() - start_time

                    print('[{}] |'.format(idx),
                          'CV: {} |'.format(cv_counter),
                          'Epoch: {}/{} |'.format(epoch_i + 1, self.epochs),
                          'Batch: {} |'.format(batch_counter),
                          'Time: {:3.2f}s |'.format(total_time),
                          'Train_Loss: {:.8f} |'.format(cost_train),
                          'Valid_Loss: {:.8f}'.format(cost_valid))

    def train_with_round_log(self, boost_round_log_path, sess, cv_counter, x_train, y_train, w_train,
                             x_valid, y_valid, w_valid, optimizer, merged, cost_, inputs, labels, weights,
                             lr, keep_prob, is_training, start_time, param_name_list, param_value_list, append_info=''):

        boost_round_log_path, _ = utils.get_boost_round_log_path(boost_round_log_path, self.model_name,
                                                                 param_name_list, param_value_list, append_info)
        boost_round_log_path += 'cv_cache/'
        utils.check_dir([boost_round_log_path])
        boost_round_log_path += self.model_name + '_cv_{}_log.txt'.format(cv_counter)

        print('Saving Outputs to:', boost_round_log_path)
        print('------------------------------------------------------')

        open(boost_round_log_path, 'w+').close()

        with open(boost_round_log_path, 'a') as f:
            __console__ = sys.stdout
            sys.stdout = f
            self.trainer(sess, cv_counter, x_train, y_train, w_train, x_valid, y_valid, w_valid,
                         optimizer, merged, cost_, inputs, labels, weights, lr, keep_prob, is_training, start_time)
            sys.stdout = __console__

        with open(boost_round_log_path) as f:
            lines = f.readlines()
            idx_round_cv = []
            train_loss_round_cv = []
            valid_loss_round_cv = []
            pattern = self.get_pattern()
            for line in lines:
                if pattern.match(line) is not None:
                    idx_round_cv.append(int(pattern.match(line).group(1)))
                    train_loss_round_cv.append(float(pattern.match(line).group(2)))
                    valid_loss_round_cv.append(float(pattern.match(line).group(3)))

        return idx_round_cv, train_loss_round_cv, valid_loss_round_cv

    # Training
    def train(self, pred_path=None, loss_log_path=None, csv_log_path=None, boost_round_log_path=None,
              train_seed=None, cv_args=None, parameters=None, show_importance=False, show_accuracy=False,
              save_cv_pred=True, save_cv_prob_train=False, save_final_pred=True, save_final_prob_train=False,
              save_csv_log=True, csv_idx=None, prescale=False, postscale=False, use_global_valid=False,
              return_prob_test=False, mode=None, param_name_list=None, param_value_list=None, strategy_args=None,
              use_custom_obj=False, use_scale_pos_weight=False, file_name_params=None, append_info=None):

        # Check if directories exit or not
        utils.check_dir_model(pred_path, loss_log_path)

        # Global Validation
        self.use_global_valid = use_global_valid

        # Cross Validation Arguments
        cv_args_copy, n_valid, n_cv, n_era, cv_seed = utils.get_cv_args(cv_args, append_info)

        # Append Information
        if append_info is None:
            append_info = 'v-' + str(n_valid) + '_c-' + str(n_cv) + '_e-' + str(n_era)
            if 'window_size' in cv_args_copy:
                append_info += '_w-' + str(cv_args_copy['window_size'])

        if csv_idx is None:
            csv_idx = self.model_name

        # Build Network
        tf.reset_default_graph()
        train_graph = tf.Graph()

        with train_graph.as_default():

            # Inputs
            inputs, labels, weights, lr, keep_prob, is_training = self.input_tensor()

            # Logits
            logits = self.model(inputs, self.unit_number, keep_prob, is_training)
            logits = tf.identity(logits, name='logits')

            # Loss
            with tf.name_scope('Loss'):
                # cost_ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
                cost_ = self.log_loss(logits, weights, labels)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr).minimize(cost_)

        # Training
        self.print_start_info()

        if use_global_valid:
            print('------------------------------------------------------')
            print('[W] Using Global Validation...')

        with tf.Session(graph=train_graph) as sess:

            # Merge all the summaries
            merged = tf.summary.merge_all()

            start_time = time.time()
            cv_counter = 0

            prob_test_total = []
            prob_train_total = []
            loss_train_total = []
            loss_valid_total = []
            loss_train_w_total = []
            loss_valid_w_total = []
            idx_round = []
            train_loss_round_total = []
            valid_loss_round_total = []
            prob_global_valid_total = []
            loss_global_valid_total = []
            loss_global_valid_w_total = []

            # Get Cross Validation Generator
            if 'cv_generator' in cv_args_copy:
                cv_generator = cv_args_copy['cv_generator']
                if cv_generator is None:
                    cv_generator = CrossValidation.era_k_fold
                cv_args_copy.pop('cv_generator')
            else:
                cv_generator = CrossValidation.era_k_fold
            print('------------------------------------------------------')
            print('[W] Using CV Generator: {}'.format(getattr(cv_generator, '__name__')))

            if 'era_list' in cv_args_copy:
                print('Era List: ', cv_args_copy['era_list'])
            if 'window_size' in cv_args_copy:
                print('Window Size: ', cv_args_copy['window_size'])
            if 'cv_weights' in cv_args_copy:
                cv_weights = cv_args_copy['cv_weights']
                cv_args_copy.pop('cv_weights')
            else:
                cv_weights = None

            for x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, e_valid, valid_era \
                    in cv_generator(self.x_train, self.y_train, self.w_train, self.e_train, **cv_args_copy):

                # CV Start Time
                cv_start_time = time.time()

                cv_counter += 1

                # Get Positive Rate of Train Set and postscale Rate
                positive_rate_train, postscale_rate = self.get_postscale_rate(y_train)
                positive_rate_valid, _ = self.get_postscale_rate(y_valid)

                print('------------------------------------------------------')
                print('Number of Features: ', x_train.shape[1])
                print('Validation Set Era: ', valid_era)
                print('------------------------------------------------------')
                print('Positive Rate of Train Set: ', positive_rate_train)
                print('Positive Rate of Valid Set: ', positive_rate_valid)
                print('------------------------------------------------------')

                # prescale
                if prescale:
                    x_train, y_train, w_train, e_train = self.prescale(x_train, y_train, w_train, e_train)

                # Training
                if mode == 'auto_train_boost_round':
                    idx_round_cv, train_loss_round_cv, valid_loss_round_cv = \
                        self.train_with_round_log(boost_round_log_path, sess, cv_counter, x_train, y_train,
                                                  w_train, x_valid, y_valid, w_valid, optimizer, merged, cost_,
                                                  inputs, labels, weights, lr, keep_prob, is_training, start_time,
                                                  param_name_list, param_value_list, append_info=append_info)
                    idx_round = idx_round_cv
                    train_loss_round_total.append(train_loss_round_cv)
                    valid_loss_round_total.append(valid_loss_round_cv)
                else:
                    self.trainer(sess, cv_counter, x_train, y_train, w_train, x_valid, y_valid, w_valid, optimizer,
                                 merged, cost_, inputs, labels, weights, lr, keep_prob, is_training, start_time)

                # Save Model
                # print('Saving model...')
                # saver = tf.train.Saver()
                # saver.save(sess, self.save_path + 'model.' + self.version + '.ckpt')

                # Prediction
                print('------------------------------------------------------')
                print('Predicting Probabilities...')
                prob_train = self.get_prob(sess, logits, x_train, self.batch_size, inputs, keep_prob, is_training)
                prob_train_all = self.get_prob(sess, logits, self.x_train, self.batch_size, inputs, keep_prob,
                                               is_training)
                prob_valid = self.get_prob(sess, logits, x_valid, self.batch_size, inputs, keep_prob, is_training)
                prob_test = self.get_prob(sess, logits, self.x_test, self.batch_size, inputs, keep_prob, is_training)

                # Predict Global Validation Set
                if use_global_valid:
                    prob_global_valid = self.get_prob(sess, logits, self.x_global_valid,
                                                      self.batch_size, inputs, keep_prob, is_training)
                else:
                    prob_global_valid = np.array([])

                # postscale
                if postscale:
                    print('------------------------------------------------------')
                    print('[W] PostScaling Results...')
                    print('PostScale Rate: {:.6f}'.format(postscale_rate))
                    prob_test *= postscale_rate
                    prob_train *= postscale_rate
                    prob_valid *= postscale_rate
                    if use_global_valid:
                        prob_global_valid *= postscale_rate

                # Print Losses of CV
                loss_train, loss_valid, loss_train_w, loss_valid_w = \
                    utils.print_loss(prob_train, y_train, w_train, prob_valid, y_valid, w_valid)

                prob_test_total.append(prob_test)
                prob_train_total.append(prob_train_all)
                loss_train_total.append(loss_train)
                loss_valid_total.append(loss_valid)
                loss_train_w_total.append(loss_train_w)
                loss_valid_w_total.append(loss_valid_w)

                # Print and Get Accuracies of CV
                acc_train_cv, acc_valid_cv, acc_train_cv_era, acc_valid_cv_era = \
                    utils.print_and_get_accuracy(prob_train, y_train, e_train,
                                                 prob_valid, y_valid, e_valid, show_accuracy)

                # Print Loss and Accuracy of Global Validation Set
                if use_global_valid:
                    loss_global_valid, loss_global_valid_w, acc_global_valid = \
                        utils.print_global_valid_loss_and_acc(prob_global_valid, self.y_global_valid,
                                                              self.w_global_valid)
                    prob_global_valid_total.append(prob_global_valid)
                    loss_global_valid_total.append(loss_global_valid)
                    loss_global_valid_w_total.append(loss_global_valid_w)

                # Save losses
                utils.save_loss_log(loss_log_path + self.model_name + '_', cv_counter, self.parameters, n_valid, n_cv,
                                    valid_era, loss_train, loss_valid, loss_train_w, loss_valid_w, train_seed, cv_seed,
                                    acc_train_cv, acc_valid_cv, acc_train_cv_era, acc_valid_cv_era)

                if save_cv_pred:
                    utils.save_pred_to_csv(pred_path + 'cv_results/' + self.model_name + '_cv_{}_'.format(cv_counter),
                                           self.id_test, prob_test)

                # CV End Time
                print('------------------------------------------------------')
                print('CV Done! Using Time: {}s'.format(time.time() - cv_start_time))

            # Final Result
            print('======================================================')
            print('Calculating Final Result...')

            # Calculate Means of prob and losses
            prob_test_mean, prob_train_mean, loss_train_mean, loss_valid_mean, loss_train_w_mean, loss_valid_w_mean = \
                utils.calculate_means(prob_test_total, prob_train_total, loss_train_total, loss_valid_total,
                                      loss_train_w_total, loss_valid_w_total, weights=cv_weights)

            # Calculate Profit
            profit = self.apply_strategy(
                strategy_args, prob_test_mean, mode, csv_log_path,
                boost_round_log_path, csv_idx, train_seed, cv_seed, param_name_list,
                param_value_list, parameters, file_name_params, append_info)

            # Save Logs of num_boost_round
            if mode == 'auto_train_boost_round':
                l = len(train_loss_round_total[0])
                for train_loss_cv in train_loss_round_total:
                    if l > len(train_loss_cv):
                        l = len(train_loss_cv)
                idx_round = idx_round[:l]
                train_loss_round_total = [train_loss[:l] for train_loss in train_loss_round_total]
                valid_loss_round_total = [valid_loss[:l] for valid_loss in valid_loss_round_total]
                train_loss_round_mean, valid_loss_round_mean = \
                    utils.calculate_boost_round_means(train_loss_round_total, valid_loss_round_total, weights=cv_weights)
                self.save_boost_round_log(boost_round_log_path, idx_round, train_loss_round_mean, valid_loss_round_mean,
                                          train_seed, cv_seed, csv_idx, parameters, param_name_list, param_value_list,
                                          append_info=append_info, profit=profit)

            # Save Final Result
            if save_final_pred:
                self.save_final_pred(mode, prob_test_mean, pred_path, parameters, csv_idx, train_seed,
                                     cv_seed, boost_round_log_path, param_name_list, param_value_list,
                                     file_name_params=None, append_info=append_info)

            # Save Final prob_train
            if save_final_prob_train:
                utils.save_prob_train_to_csv(pred_path + 'final_prob_train/' + self.model_name + '_',
                                             prob_train_mean, self.y_train)

            # Print Total Losses
            utils.print_total_loss(loss_train_mean, loss_valid_mean, loss_train_w_mean, loss_valid_w_mean)

            # Print and Get Accuracies of CV of All Train Set
            acc_train, acc_train_era = \
                utils.print_and_get_train_accuracy(prob_train_mean, self.y_train, self.e_train, show_accuracy)

            # Save Final Losses to File
            utils.save_final_loss_log(loss_log_path + self.model_name + '_', self.parameters, n_valid, n_cv,
                                      loss_train_mean, loss_valid_mean, loss_train_w_mean, loss_valid_w_mean,
                                      train_seed, cv_seed, acc_train, acc_train_era)

            # Print Global Validation Information and Save
            if use_global_valid:
                # Calculate Means of Probabilities and Losses
                prob_global_valid_mean, loss_global_valid_mean, loss_global_valid_w_mean = \
                    utils.calculate_global_valid_means(prob_global_valid_total, loss_global_valid_total,
                                                       loss_global_valid_w_total, weights=cv_weights)
                # Print Loss and Accuracy
                acc_total_global_valid = \
                    utils.print_total_global_valid_loss_and_acc(prob_global_valid_mean, self.y_global_valid,
                                                                loss_global_valid_mean, loss_global_valid_w_mean)
                # Save csv log
                if save_csv_log:
                    self.save_csv_log(mode, csv_log_path, param_name_list, param_value_list, csv_idx, loss_train_w_mean,
                                      loss_valid_w_mean, acc_train, train_seed, cv_seed, n_valid, n_cv, parameters,
                                      boost_round_log_path=boost_round_log_path, file_name_params=file_name_params,
                                      append_info=append_info, loss_global_valid=loss_global_valid_w_mean,
                                      acc_global_valid=acc_total_global_valid, profit=profit)

            # Save Loss Log to csv File
            if save_csv_log:
                if not use_global_valid:
                    self.save_csv_log(mode, csv_log_path, param_name_list, param_value_list, csv_idx,
                                      loss_train_w_mean, loss_valid_w_mean, acc_train, train_seed,
                                      cv_seed, n_valid, n_cv, parameters, file_name_params=file_name_params,
                                      append_info=append_info, profit=profit)

            # Return Final Result
            if return_prob_test:
                return prob_test_mean

    def stack_train(self, x_train, y_train, w_train, x_g_train, x_valid, y_valid,
                    w_valid, x_g_valid, x_test, x_g_test, parameters=None, show_importance=False):

        # Print Start Information
        self.print_start_info()
        print('Number of Features: ', x_train.shape[1])
        print('------------------------------------------------------')

        # Build Network
        tf.reset_default_graph()
        train_graph = tf.Graph()

        with train_graph.as_default():

            # Inputs
            inputs, labels, weights, lr, keep_prob, is_train = self.input_tensor()

            # Logits
            logits = self.model(inputs, self.unit_number, keep_prob, is_train)
            logits = tf.identity(logits, name='logits')

            # Loss
            with tf.name_scope('Loss'):
                # cost_ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
                cost_ = self.log_loss(logits, weights, labels)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr).minimize(cost_)

        with tf.Session(graph=train_graph) as sess:

            start_time = time.time()
            sess.run(tf.global_variables_initializer())

            for epoch_i in range(self.epochs):

                batch_counter = 0

                for batch_i, (batch_x, batch_y, batch_w) in enumerate(self.get_batches(x_train,
                                                                                       y_train,
                                                                                       w_train,
                                                                                       self.batch_size)):

                    batch_counter += 1

                    _, cost_train = sess.run([optimizer, cost_],
                                             {inputs: batch_x,
                                              labels: batch_y,
                                              weights: batch_w,
                                              lr: self.learning_rate,
                                              keep_prob: self.keep_probability,
                                              is_train: True})

                    if str(cost_train) == 'nan':
                        raise ValueError('NaN BUG!!! Try Another Seed!!!')

                    if batch_counter % self.display_step == 0 and batch_i > 0:

                        cost_valid_all = []

                        for iii, (valid_batch_x,
                                  valid_batch_y,
                                  valid_batch_w) in enumerate(self.get_batches(x_valid,
                                                                               y_valid,
                                                                               w_valid,
                                                                               self.batch_size)):
                            cost_valid_i = sess.run(cost_, {inputs: valid_batch_x,
                                                            labels: valid_batch_y,
                                                            weights: valid_batch_w,
                                                            keep_prob: 1.0,
                                                            is_train: False})

                            cost_valid_all.append(cost_valid_i)

                        cost_valid = sum(cost_valid_all) / len(cost_valid_all)

                        total_time = time.time() - start_time

                        print('Epoch: {}/{} |'.format(epoch_i + 1, self.epochs),
                              'Batch: {} |'.format(batch_counter),
                              'Time: {:>3.2f}s |'.format(total_time),
                              'Train_Loss: {:>.8f} |'.format(cost_train),
                              'Valid_Loss: {:>.8f}'.format(cost_valid))

            # Prediction
            print('------------------------------------------------------')
            print('Predicting Probabilities...')

            logits_pred_train = sess.run(logits, {inputs: x_train, keep_prob: 1.0, is_train: False})
            logits_pred_valid = sess.run(logits, {inputs: x_valid, keep_prob: 1.0, is_train: False})
            logits_pred_test = sess.run(logits, {inputs: x_test, keep_prob: 1.0, is_train: False})

            logits_pred_train = logits_pred_train.flatten()
            logits_pred_valid = logits_pred_valid.flatten()
            logits_pred_test = logits_pred_test.flatten()

            prob_train = 1.0 / (1.0 + np.exp(-logits_pred_train))
            prob_valid = 1.0 / (1.0 + np.exp(-logits_pred_valid))
            prob_test = 1.0 / (1.0 + np.exp(-logits_pred_test))

            loss_train, loss_valid, loss_train_w, loss_valid_w = \
                utils.print_loss(prob_train, y_train, w_train, prob_valid, y_valid, w_valid)

            losses = [loss_train, loss_valid, loss_train_w, loss_valid_w]

            return prob_valid, prob_test, losses

    def prejudge_stack_train(self, x_train, x_g_train, y_train, w_train, e_train, x_valid,
                             x_g_valid, y_valid, w_valid, e_valid, x_test, x_g_test,
                             pred_path=None, loss_log_path=None, csv_log_path=None, parameters=None, cv_args=None,
                             train_seed=None, show_importance=False, show_accuracy=False, save_final_pred=True,
                             save_csv_log=True, csv_idx=None, mode=None, file_name_params=None,
                             param_name_list=None, param_value_list=None, append_info=None):

        # Check if directories exit or not
        utils.check_dir_model(pred_path, loss_log_path)

        cv_args_copy = copy.deepcopy(cv_args)
        if 'n_valid' in cv_args:
            n_valid = cv_args_copy['n_valid']
        elif 'valid_rate' in cv_args:
            n_valid = cv_args_copy['valid_rate']
        else:
            n_valid = ''
        n_cv = cv_args_copy['n_cv']
        cv_seed = cv_args_copy['cv_seed']

        if csv_idx is None:
            csv_idx = self.model_name

        # Print Start Information and Get Model Name
        self.print_start_info()

        print('======================================================')
        print('Number of Features: ', x_train.shape[1])
        print('------------------------------------------------------')

        # Build Network
        tf.reset_default_graph()
        train_graph = tf.Graph()

        with train_graph.as_default():

            # Inputs
            inputs, labels, weights, lr, keep_prob, is_train = self.input_tensor()

            # Logits
            logits = self.model(inputs, self.unit_number, keep_prob, is_train)
            logits = tf.identity(logits, name='logits')

            # Loss
            with tf.name_scope('Loss'):
                # cost_ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
                cost_ = self.log_loss(logits, weights, labels)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr).minimize(cost_)

        with tf.Session(graph=train_graph) as sess:

            start_time = time.time()
            sess.run(tf.global_variables_initializer())

            for epoch_i in range(self.epochs):

                batch_counter = 0

                for batch_i, (batch_x, batch_y, batch_w) in enumerate(self.get_batches(x_train,
                                                                                       y_train,
                                                                                       w_train,
                                                                                       self.batch_size)):

                    batch_counter += 1

                    _, cost_train = sess.run([optimizer, cost_],
                                             {inputs: batch_x,
                                              labels: batch_y,
                                              weights: batch_w,
                                              lr: self.learning_rate,
                                              keep_prob: self.keep_probability,
                                              is_train: True})

                    if str(cost_train) == 'nan':
                        raise ValueError('NaN BUG!!! Try Another Seed!!!')

                    if batch_counter % self.display_step == 0 and batch_i > 0:

                        cost_valid_all = []

                        for iii, (valid_batch_x,
                                  valid_batch_y,
                                  valid_batch_w) in enumerate(self.get_batches(x_valid,
                                                                               y_valid,
                                                                               w_valid,
                                                                               self.batch_size)):
                            cost_valid_i = sess.run(cost_, {inputs: valid_batch_x,
                                                            labels: valid_batch_y,
                                                            weights: valid_batch_w,
                                                            keep_prob: 1.0,
                                                            is_train: False})

                            cost_valid_all.append(cost_valid_i)

                        cost_valid = sum(cost_valid_all) / len(cost_valid_all)

                        total_time = time.time() - start_time

                        print('Epoch: {}/{} |'.format(epoch_i + 1, self.epochs),
                              'Batch: {} |'.format(batch_counter),
                              'Time: {:>3.2f}s |'.format(total_time),
                              'Train_Loss: {:>.8f} |'.format(cost_train),
                              'Valid_Loss: {:>.8f}'.format(cost_valid))

            # Prediction
            print('------------------------------------------------------')
            print('Predicting Train Probabilities...')

            logits_pred_train = sess.run(logits, {inputs: x_train, keep_prob: 1.0, is_train: False})
            logits_pred_valid = sess.run(logits, {inputs: x_valid, keep_prob: 1.0, is_train: False})
            logits_pred_test = sess.run(logits, {inputs: x_test, keep_prob: 1.0, is_train: False})

            logits_pred_train = logits_pred_train.flatten()
            logits_pred_valid = logits_pred_valid.flatten()
            logits_pred_test = logits_pred_test.flatten()

            prob_train = 1.0 / (1.0 + np.exp(-logits_pred_train))
            prob_valid = 1.0 / (1.0 + np.exp(-logits_pred_valid))
            prob_test = 1.0 / (1.0 + np.exp(-logits_pred_test))

            loss_train, loss_valid, loss_train_w, loss_valid_w = \
                utils.print_loss(prob_train, y_train, w_train, prob_valid, y_valid, w_valid)

            # Save Final Result
            if save_final_pred:
                self.save_final_pred(mode, save_final_pred, prob_test, pred_path,
                                     parameters, csv_idx, train_seed, cv_seed, file_name_params=file_name_params)

            # Print Total Losses
            utils.print_total_loss(loss_train, loss_valid, loss_train_w, loss_valid_w)
            losses = [loss_train, loss_valid, loss_train_w, loss_valid_w]

            # Print and Get Accuracies of CV
            acc_train, acc_valid, acc_train_era, acc_valid_era = \
                utils.print_and_get_accuracy(prob_train, y_train, e_train,
                                             prob_valid, y_valid, e_valid, show_accuracy)

            # Save Final Losses to File
            utils.save_final_loss_log(loss_log_path + self.model_name + '_', parameters, n_valid, n_cv,
                                      loss_train, loss_valid, loss_train_w, loss_valid_w,
                                      train_seed, cv_seed, acc_train, acc_train_era)

            # Save Loss Log to csv File
            if save_csv_log:
                self.save_csv_log(mode, csv_log_path, param_name_list, param_value_list, csv_idx,
                                  loss_train_w, loss_valid_w, acc_train, train_seed,
                                  cv_seed, n_valid, n_cv, parameters, file_name_params=file_name_params)

            return prob_valid, prob_test, losses


class RNN(ModelBase):
    """
        Recurrent Neural Networks
    """
    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, pct_te, id_te,
                 x_va=None, y_va=None, w_va=None, e_va=None,
                 use_multi_group=False, parameters=None):

        super(RNN, self).__init__(x_tr, y_tr, w_tr, e_tr, x_te, pct_te, id_te,
                                  x_va, y_va, w_va, e_va, use_multi_group)

        # Hyperparameters
        self.parameters = parameters
        self.version = parameters['version']
        self.epochs = parameters['epochs']
        self.learning_rate = parameters['learning_rate']
        self.keep_probability = parameters['keep_probability']
        self.batch_size = parameters['batch_size']
        self.train_seed = parameters['seed']
        self.display_step = parameters['display_step']
        self.save_path = parameters['save_path']
        self.log_path = parameters['log_path']
        self.rnn_size = parameters['rnn_size']
        self.num_lstm_cell = parameters['num_lstm_cell']
        self.fc_unit_number = parameters['fc_unit_number']  # format: [128, 64, 32] or None
        self.target_sequence_length = None
        self.max_target_sequence_length = None

        utils.check_dir([self.save_path, self.log_path])

    def print_start_info(self):

        print('------------------------------------------------------')
        print('Training Deep Neural Networks...')
        print('------------------------------------------------------')

        self.model_name = 'dnn'

    def get_pattern(self):

        # [0] | CV: 10 | Epoch: 4/4 | Batch: 7300 | Time: 352.85s | Train_Loss: 0.71237314 | Valid_Loss: 0.72578128
        return re.compile(r'\[(\d*)\].*Train_Loss: (.*) \| Valid_Loss: (.*)')

    # Input Tensors
    def input_tensor(self):

        feature_num = self.x_train.shape[1]

        inputs_ = tf.placeholder(tf.float32, [None, feature_num], name='inputs')
        labels_ = tf.placeholder(tf.float32, None, name='labels')
        loss_weights_ = tf.placeholder(tf.float32, None, name='loss_weights')
        learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')
        keep_prob_ = tf.placeholder(tf.float32, name='keep_prob')
        is_training_ = tf.placeholder(tf.bool, name='is_training')

        return inputs_, labels_, loss_weights_, learning_rate_, keep_prob_, is_training_

    # Full Connected Layer
    def fc_layer(self, x_tensor, layer_name, num_outputs, keep_prob, is_training):

        if is_training is True:
            print('Using Batch Normalization')

        with tf.name_scope(layer_name):

            # x_shape = features.get_shape().as_list()
            # weights_initializer = tf.truncated_normal_initializer(stddev=2.0 / math.sqrt(x_shape[1])),
            # weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN',
            #                                                                      seed=self.train_seed)
            weights_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32, seed=self.train_seed)
            weights_reg = tf.contrib.layers.l2_regularizer(1e-3)
            normalizer_fn = tf.contrib.layers.batch_norm
            normalizer_params = {'is_training': is_training}

            fc = tf.contrib.layers.fully_connected(inputs=x_tensor,
                                                   num_outputs=num_outputs,
                                                   activation_fn=tf.nn.sigmoid,
                                                   weights_initializer=weights_initializer,
                                                   weights_regularizer=weights_reg,
                                                   normalizer_fn=normalizer_fn,
                                                   normalizer_params=normalizer_params,
                                                   biases_initializer=tf.zeros_initializer(dtype=tf.float32))

            tf.summary.histogram('fc_layer', fc)

            fc = tf.nn.dropout(fc, keep_prob)

        return fc

    def encoder(self, enc_inputs, keep_prob, source_seq_length):

        with tf.variable_scope('encoder'):

            def _single_lstm(lstm_size):
                lstm = tf.contrib.rnn.LSTMCell(
                    lstm_size,
                    initializer=tf.random_uniform_initializer(-0.1, 0.1))
                # Add dropout to the cell
                return tf.contrib.rnn.DropoutWrapper(
                    lstm, output_keep_prob=keep_prob)

            multi_lstm = tf.contrib.rnn.MultiRNNCell(
                [_single_lstm(self.rnn_size) for _ in range(self.num_lstm_cell)])

            enc_output, enc_state = tf.nn.dynamic_rnn(
                multi_lstm,
                enc_inputs,
                sequence_length=source_seq_length,
                dtype=tf.float32
            )

            return enc_output, enc_state

    # Decoding - Training
    def decoder_train(self, encoder_state, dec_cell, dec_inputs, output_layer):

        with tf.name_scope('train_helper'):
            train_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=dec_inputs,
                sequence_length=self.target_sequence_length,
                time_major=False
            )
        with tf.name_scope('train_decoder'):
            train_decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=dec_cell,
                helper=train_helper,
                initial_state=encoder_state,
                output_layer=output_layer
            )
        with tf.name_scope('train_decoder_output'):
            train_dec_output, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=train_decoder,
                impute_finished=True,
                maximum_iterations=self.max_target_sequence_length
            )
        return train_dec_output

    # Decoding - Inference
    def decoder_infer(self, encoder_state, dec_cell, ddec_inputs,
                      start_of_sequence_id, end_of_sequence_id,
                      max_target_sequence_length,
                             output_layer):

        start_tokens = tf.tile(
            tf.constant([start_of_sequence_id], dtype=tf.int32), [batch_size],
            name='start_tokens')

        with tf.name_scope('infer_helper'):
            infer_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                embedding=dec_embeddings,
                start_tokens=start_tokens,
                end_token=end_of_sequence_id)

        with tf.name_scope('infer_decoder'):
            infer_decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell,
                                                            helper=infer_helper,
                                                            initial_state=encoder_state,
                                                            output_layer=output_layer)

        with tf.name_scope('infer_decoder_output'):
            infer_dec_output, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=infer_decoder,
                impute_finished=True,
                maximum_iterations=max_target_sequence_length)

        return infer_dec_output

    # Output Layer
    def output_layer(self, x_tensor, layer_name, num_outputs):

        with tf.name_scope(layer_name):

            # weights_initializer = tf.truncated_normal_initializer(stddev=2.0 / math.sqrt(x_shape[1])),
            # weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_IN',
            #                                                                      seed=self.train_seed)
            weights_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32, seed=self.train_seed)

            out = tf.contrib.layers.fully_connected(inputs=x_tensor,
                                                    num_outputs=num_outputs,
                                                    activation_fn=None,
                                                    weights_initializer=weights_initializer,
                                                    biases_initializer=tf.zeros_initializer(dtype=tf.float32))

        return out

    # Model
    def model(self, cell, x, keep_prob, is_training):

        rnn_output, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
        final_state = tf.identity(state, name='final_state')

        if self.fc_unit_number is not None:
            fc = [rnn_output]
            for i in range(len(self.fc_unit_number)):
                fc.append(
                    self.fc_layer(fc[i], 'fc{}'.format(i + 1), self.fc_unit_number[i], keep_prob,
                                  is_training))
            logit = self.output_layer(fc[len(self.fc_unit_number)], 'output', 1)
        else:
            logit = self.output_layer(rnn_output, 'output', 1)

        return logit, final_state

    # LogLoss
    @staticmethod
    def log_loss(logit, w, y):

        with tf.name_scope('prob'):
            logit = tf.squeeze(logit)
            prob = tf.nn.sigmoid(logit)

        with tf.name_scope('log_loss'):

            w = w / tf.reduce_sum(w)
            ones = tf.ones_like(y, dtype=tf.float32)
            loss = - tf.reduce_sum(w * (y * tf.log(prob) + (ones-y) * tf.log(ones-prob)))
            # loss = tf.losses.log_loss(labels=y, predictions=prob, weights=w)

        tf.summary.scalar('log_loss', loss)

        return loss

    # Get Batches
    @staticmethod
    def get_batches(x, y, w, batch_num):

        n_batches = len(x) // batch_num

        for ii in range(0, n_batches * batch_num + 1, batch_num):

            if ii != n_batches * batch_num - 1:
                batch_x, batch_y, batch_w = x[ii: ii + batch_num], y[ii: ii + batch_num], w[ii: ii + batch_num]
            else:
                batch_x, batch_y, batch_w = x[ii:], y[ii:], w[ii:]

            yield batch_x, batch_y, batch_w

    # Get Batches for Prediction
    @staticmethod
    def get_batches_for_predict(x, batch_num):

        n_batches = len(x) // batch_num

        for ii in range(0, n_batches * batch_num + 1, batch_num):

            if ii != n_batches * batch_num - 1:
                batch_x = x[ii: ii + batch_num]
            else:
                batch_x = x[ii:]

            yield batch_x

    # Get Probabilities
    def get_prob(self, sess, logits, x, batch_num, inputs, keep_prob, is_training):

        logits_pred = np.array([])

        for x_batch in self.get_batches_for_predict(x, batch_num):
            logits_pred_batch = sess.run(logits, {inputs: x_batch, keep_prob: 1.0, is_training: False})
            logits_pred_batch = logits_pred_batch.flatten()
            logits_pred = np.concatenate((logits_pred, logits_pred_batch))

        prob = 1.0 / (1.0 + np.exp(-logits_pred))

        return prob

    # Trainer
    def trainer(self, sess, cv_counter, x_train, y_train, w_train,
                x_valid, y_valid, w_valid, optimizer, merged, cost,
                inputs, labels, weights, lr, keep_prob, is_training,
                initial_state, final_state, start_time):

        train_log_path = self.log_path + self.version + '/cv_{}/train'.format(cv_counter)
        valid_log_path = self.log_path + self.version + '/cv_{}/valid'.format(cv_counter)

        if not isdir(train_log_path):
            os.makedirs(train_log_path)
        if not isdir(valid_log_path):
            os.makedirs(valid_log_path)

        train_writer = tf.summary.FileWriter(train_log_path, sess.graph)
        valid_writer = tf.summary.FileWriter(valid_log_path)

        sess.run(tf.global_variables_initializer())

        batch_counter = 0
        idx = 0

        for epoch_i in range(self.epochs):

            state = sess.run(initial_state, {
                inputs: next(self.get_batches(x_train, y_train, w_train, self.batch_size))[0][0]})

            for batch_i, (batch_x, batch_y, batch_w) in enumerate(self.get_batches(x_train,
                                                                                   y_train,
                                                                                   w_train,
                                                                                   self.batch_size)):

                batch_counter += 1

                _, final_state_, cost_ = sess.run(
                    [optimizer, final_state, cost],
                    {inputs: batch_x,
                     labels: batch_y,
                     weights: batch_w,
                     lr: self.learning_rate,
                     keep_prob: self.keep_probability,
                     initial_state: state,
                     is_training: True}
                )

                if str(cost_) == 'nan':
                    raise ValueError('NaN BUG!!! Try Another Seed!!!')

                if batch_counter % self.display_step == 0 and batch_i > 0:

                    idx += 1

                    summary_train, cost_train = sess.run(
                        [merged, cost],
                        {inputs: batch_x,
                         labels: batch_y,
                         weights: batch_w,
                         keep_prob: 1.0,
                         initial_state: final_state_,
                         is_training: False}
                    )
                    train_writer.add_summary(summary_train, batch_counter)

                    cost_valid_all = []

                    for iii, (valid_batch_x,
                              valid_batch_y,
                              valid_batch_w) in enumerate(self.get_batches(x_valid,
                                                                           y_valid,
                                                                           w_valid,
                                                                           self.batch_size)):
                        summary_valid_i, cost_valid_i = sess.run(
                            [merged, cost_],
                            {inputs: valid_batch_x,
                             labels: valid_batch_y,
                             weights: valid_batch_w,
                             keep_prob: 1.0,
                             initial_state: final_state_,
                             is_training: False})

                        valid_writer.add_summary(summary_valid_i, batch_counter)

                        cost_valid_all.append(cost_valid_i)

                    cost_valid = sum(cost_valid_all) / len(cost_valid_all)

                    total_time = time.time() - start_time

                    print('[{}] |'.format(idx),
                          'CV: {} |'.format(cv_counter),
                          'Epoch: {}/{} |'.format(epoch_i + 1, self.epochs),
                          'Batch: {} |'.format(batch_counter),
                          'Time: {:3.2f}s |'.format(total_time),
                          'Train_Loss: {:.8f} |'.format(cost_train),
                          'Valid_Loss: {:.8f}'.format(cost_valid))

    def train_with_round_log(self, boost_round_log_path, sess, cv_counter, x_train, y_train, w_train,
                             x_valid, y_valid, w_valid, optimizer, merged, cost_, inputs, labels, weights,
                             lr, keep_prob, is_training, initial_state, final_state, start_time,
                             param_name_list, param_value_list, append_info=''):

        boost_round_log_path, _ = utils.get_boost_round_log_path(boost_round_log_path, self.model_name,
                                                                 param_name_list, param_value_list, append_info)
        boost_round_log_path += 'cv_cache/'
        utils.check_dir([boost_round_log_path])
        boost_round_log_path += self.model_name + '_cv_{}_log.txt'.format(cv_counter)

        print('Saving Outputs to:', boost_round_log_path)
        print('------------------------------------------------------')

        open(boost_round_log_path, 'w+').close()

        with open(boost_round_log_path, 'a') as f:
            __console__ = sys.stdout
            sys.stdout = f
            self.trainer(sess, cv_counter, x_train, y_train, w_train, x_valid, y_valid,
                         w_valid, optimizer, merged, cost_, inputs, labels, weights,
                         lr, keep_prob, is_training, initial_state, final_state, start_time)
            sys.stdout = __console__

        with open(boost_round_log_path) as f:
            lines = f.readlines()
            idx_round_cv = []
            train_loss_round_cv = []
            valid_loss_round_cv = []
            pattern = self.get_pattern()
            for line in lines:
                if pattern.match(line) is not None:
                    idx_round_cv.append(int(pattern.match(line).group(1)))
                    train_loss_round_cv.append(float(pattern.match(line).group(2)))
                    valid_loss_round_cv.append(float(pattern.match(line).group(3)))

        return idx_round_cv, train_loss_round_cv, valid_loss_round_cv

    # Training
    def train(self, pred_path=None, loss_log_path=None, csv_log_path=None, boost_round_log_path=None,
              train_seed=None, cv_args=None, parameters=None, show_importance=False, show_accuracy=False,
              save_cv_pred=True, save_cv_prob_train=False, save_final_pred=True, save_final_prob_train=False,
              save_csv_log=True, csv_idx=None, prescale=False, postscale=False, use_global_valid=False,
              return_prob_test=False, mode=None, param_name_list=None, param_value_list=None, strategy_args=None,
              use_custom_obj=False, use_scale_pos_weight=False, file_name_params=None, append_info=None):

        # Check if directories exit or not
        utils.check_dir_model(pred_path, loss_log_path)

        # Global Validation
        self.use_global_valid = use_global_valid

        # Cross Validation Arguments
        cv_args_copy, n_valid, n_cv, n_era, cv_seed = utils.get_cv_args(cv_args, append_info)

        # Append Information
        if append_info is None:
            append_info = 'v-' + str(n_valid) + '_c-' + str(n_cv) + '_e-' + str(n_era)
            if 'window_size' in cv_args_copy:
                append_info += '_w-' + str(cv_args_copy['window_size'])

        if csv_idx is None:
            csv_idx = self.model_name

        # Build Network
        tf.reset_default_graph()
        train_graph = tf.Graph()

        with train_graph.as_default():

            # Inputs
            inputs, labels, weights, lr, keep_prob, is_training = self.input_tensor()

            cell, initial_state = self.get_init_cell(keep_prob)

            # Logits
            logits, final_state = self.model(cell, inputs, keep_prob, is_training)
            logits = tf.identity(logits, name='logits')

            # Loss
            with tf.name_scope('Loss'):
                # TODO: 这里定义你的loss
                # cost_ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
                cost = self.log_loss(logits, weights, labels)

            # Optimizer
            train_op = tf.train.AdamOptimizer(lr)

            # Gradient Clipping
            gradients = train_op.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for
                                grad, var in gradients if grad is not None]
            optimizer = train_op.apply_gradients(capped_gradients)

        # Training
        self.print_start_info()

        if use_global_valid:
            print('------------------------------------------------------')
            print('[W] Using Global Validation...')

        with tf.Session(graph=train_graph) as sess:

            # Merge all the summaries
            merged = tf.summary.merge_all()

            start_time = time.time()
            cv_counter = 0

            prob_test_total = []
            prob_train_total = []
            loss_train_total = []
            loss_valid_total = []
            loss_train_w_total = []
            loss_valid_w_total = []
            idx_round = []
            train_loss_round_total = []
            valid_loss_round_total = []
            prob_global_valid_total = []
            loss_global_valid_total = []
            loss_global_valid_w_total = []

            # Get Cross Validation Generator
            if 'cv_generator' in cv_args_copy:
                cv_generator = cv_args_copy['cv_generator']
                if cv_generator is None:
                    cv_generator = CrossValidation.era_k_fold
                cv_args_copy.pop('cv_generator')
            else:
                cv_generator = CrossValidation.era_k_fold
            print('------------------------------------------------------')
            print('[W] Using CV Generator: {}'.format(getattr(cv_generator, '__name__')))

            if 'era_list' in cv_args_copy:
                print('Era List: ', cv_args_copy['era_list'])
            if 'window_size' in cv_args_copy:
                print('Window Size: ', cv_args_copy['window_size'])
            if 'cv_weights' in cv_args_copy:
                cv_weights = cv_args_copy['cv_weights']
                cv_args_copy.pop('cv_weights')
            else:
                cv_weights = None

            for x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, e_valid, valid_era \
                    in cv_generator(self.x_train, self.y_train, self.w_train, self.e_train, **cv_args_copy):

                # CV Start Time
                cv_start_time = time.time()

                cv_counter += 1

                # Get Positive Rate of Train Set and postscale Rate
                positive_rate_train, postscale_rate = self.get_postscale_rate(y_train)
                positive_rate_valid, _ = self.get_postscale_rate(y_valid)

                print('------------------------------------------------------')
                print('Number of Features: ', x_train.shape[1])
                print('Validation Set Era: ', valid_era)
                print('------------------------------------------------------')
                print('Positive Rate of Train Set: ', positive_rate_train)
                print('Positive Rate of Valid Set: ', positive_rate_valid)
                print('------------------------------------------------------')

                # prescale
                if prescale:
                    x_train, y_train, w_train, e_train = self.prescale(x_train, y_train, w_train, e_train)

                # Training
                if mode == 'auto_train_boost_round':
                    idx_round_cv, train_loss_round_cv, valid_loss_round_cv = \
                        self.train_with_round_log(
                            boost_round_log_path, sess, cv_counter, x_train,
                            y_train, w_train, x_valid, y_valid, w_valid,
                            optimizer, merged, cost, inputs, labels, weights,
                            lr, keep_prob, is_training, initial_state,
                            final_state, start_time, param_name_list,
                            param_value_list, append_info=append_info)
                    idx_round = idx_round_cv
                    train_loss_round_total.append(train_loss_round_cv)
                    valid_loss_round_total.append(valid_loss_round_cv)
                else:
                    self.trainer(sess, cv_counter, x_train, y_train, w_train,
                                 x_valid, y_valid, w_valid, optimizer,
                                 merged, cost, inputs, labels, weights,
                                 lr, keep_prob, is_training, initial_state,
                                 final_state, start_time)

                # Save Model
                # print('Saving model...')
                # saver = tf.train.Saver()
                # saver.save(sess, self.save_path + 'model.' + self.version + '.ckpt')

                # Prediction
                print('------------------------------------------------------')
                print('Predicting Probabilities...')
                prob_train = self.get_prob(sess, logits, x_train, self.batch_size, inputs, keep_prob, is_training)
                prob_train_all = self.get_prob(sess, logits, self.x_train, self.batch_size, inputs, keep_prob,
                                               is_training)
                prob_valid = self.get_prob(sess, logits, x_valid, self.batch_size, inputs, keep_prob, is_training)
                prob_test = self.get_prob(sess, logits, self.x_test, self.batch_size, inputs, keep_prob, is_training)

                # Predict Global Validation Set
                if use_global_valid:
                    prob_global_valid = self.get_prob(sess, logits, self.x_global_valid,
                                                      self.batch_size, inputs, keep_prob, is_training)
                else:
                    prob_global_valid = np.array([])

                # postscale
                if postscale:
                    print('------------------------------------------------------')
                    print('[W] PostScaling Results...')
                    print('PostScale Rate: {:.6f}'.format(postscale_rate))
                    prob_test *= postscale_rate
                    prob_train *= postscale_rate
                    prob_valid *= postscale_rate
                    if use_global_valid:
                        prob_global_valid *= postscale_rate

                # Print Losses of CV
                loss_train, loss_valid, loss_train_w, loss_valid_w = \
                    utils.print_loss(prob_train, y_train, w_train, prob_valid, y_valid, w_valid)

                prob_test_total.append(prob_test)
                prob_train_total.append(prob_train_all)
                loss_train_total.append(loss_train)
                loss_valid_total.append(loss_valid)
                loss_train_w_total.append(loss_train_w)
                loss_valid_w_total.append(loss_valid_w)

                # Print and Get Accuracies of CV
                acc_train_cv, acc_valid_cv, acc_train_cv_era, acc_valid_cv_era = \
                    utils.print_and_get_accuracy(prob_train, y_train, e_train,
                                                 prob_valid, y_valid, e_valid, show_accuracy)

                # Print Loss and Accuracy of Global Validation Set
                if use_global_valid:
                    loss_global_valid, loss_global_valid_w, acc_global_valid = \
                        utils.print_global_valid_loss_and_acc(prob_global_valid, self.y_global_valid,
                                                              self.w_global_valid)
                    prob_global_valid_total.append(prob_global_valid)
                    loss_global_valid_total.append(loss_global_valid)
                    loss_global_valid_w_total.append(loss_global_valid_w)

                # Save losses
                utils.save_loss_log(loss_log_path + self.model_name + '_', cv_counter, self.parameters, n_valid, n_cv,
                                    valid_era, loss_train, loss_valid, loss_train_w, loss_valid_w, train_seed, cv_seed,
                                    acc_train_cv, acc_valid_cv, acc_train_cv_era, acc_valid_cv_era)

                if save_cv_pred:
                    utils.save_pred_to_csv(pred_path + 'cv_results/' + self.model_name + '_cv_{}_'.format(cv_counter),
                                           self.id_test, prob_test)

                # CV End Time
                print('------------------------------------------------------')
                print('CV Done! Using Time: {}s'.format(time.time() - cv_start_time))

            # Final Result
            print('======================================================')
            print('Calculating Final Result...')

            # Calculate Means of prob and losses
            prob_test_mean, prob_train_mean, loss_train_mean, loss_valid_mean, loss_train_w_mean, loss_valid_w_mean = \
                utils.calculate_means(prob_test_total, prob_train_total, loss_train_total, loss_valid_total,
                                      loss_train_w_total, loss_valid_w_total, weights=cv_weights)

            # Calculate Profit
            profit = self.apply_strategy(
                strategy_args, prob_test_mean, mode, csv_log_path,
                boost_round_log_path, csv_idx, train_seed, cv_seed, param_name_list,
                param_value_list, parameters, file_name_params, append_info)

            # Save Logs of num_boost_round
            if mode == 'auto_train_boost_round':
                l = len(train_loss_round_total[0])
                for train_loss_cv in train_loss_round_total:
                    if l > len(train_loss_cv):
                        l = len(train_loss_cv)
                idx_round = idx_round[:l]
                train_loss_round_total = [train_loss[:l] for train_loss in train_loss_round_total]
                valid_loss_round_total = [valid_loss[:l] for valid_loss in valid_loss_round_total]
                train_loss_round_mean, valid_loss_round_mean = \
                    utils.calculate_boost_round_means(train_loss_round_total, valid_loss_round_total, weights=cv_weights)
                self.save_boost_round_log(boost_round_log_path, idx_round, train_loss_round_mean, valid_loss_round_mean,
                                          train_seed, cv_seed, csv_idx, parameters, param_name_list, param_value_list,
                                          append_info=append_info, profit=profit)

            # Save Final Result
            if save_final_pred:
                self.save_final_pred(mode, prob_test_mean, pred_path, parameters, csv_idx, train_seed,
                                     cv_seed, boost_round_log_path, param_name_list, param_value_list,
                                     file_name_params=None, append_info=append_info)

            # Save Final prob_train
            if save_final_prob_train:
                utils.save_prob_train_to_csv(pred_path + 'final_prob_train/' + self.model_name + '_',
                                             prob_train_mean, self.y_train)

            # Print Total Losses
            utils.print_total_loss(loss_train_mean, loss_valid_mean, loss_train_w_mean, loss_valid_w_mean)

            # Print and Get Accuracies of CV of All Train Set
            acc_train, acc_train_era = \
                utils.print_and_get_train_accuracy(prob_train_mean, self.y_train, self.e_train, show_accuracy)

            # Save Final Losses to File
            utils.save_final_loss_log(loss_log_path + self.model_name + '_', self.parameters, n_valid, n_cv,
                                      loss_train_mean, loss_valid_mean, loss_train_w_mean, loss_valid_w_mean,
                                      train_seed, cv_seed, acc_train, acc_train_era)

            # Print Global Validation Information and Save
            if use_global_valid:
                # Calculate Means of Probabilities and Losses
                prob_global_valid_mean, loss_global_valid_mean, loss_global_valid_w_mean = \
                    utils.calculate_global_valid_means(prob_global_valid_total, loss_global_valid_total,
                                                       loss_global_valid_w_total, weights=cv_weights)
                # Print Loss and Accuracy
                acc_total_global_valid = \
                    utils.print_total_global_valid_loss_and_acc(prob_global_valid_mean, self.y_global_valid,
                                                                loss_global_valid_mean, loss_global_valid_w_mean)
                # Save csv log
                if save_csv_log:
                    self.save_csv_log(mode, csv_log_path, param_name_list, param_value_list, csv_idx, loss_train_w_mean,
                                      loss_valid_w_mean, acc_train, train_seed, cv_seed, n_valid, n_cv, parameters,
                                      boost_round_log_path=boost_round_log_path, file_name_params=file_name_params,
                                      append_info=append_info, loss_global_valid=loss_global_valid_w_mean,
                                      acc_global_valid=acc_total_global_valid, profit=profit)

            # Save Loss Log to csv File
            if save_csv_log:
                if not use_global_valid:
                    self.save_csv_log(mode, csv_log_path, param_name_list, param_value_list, csv_idx,
                                      loss_train_w_mean, loss_valid_w_mean, acc_train, train_seed,
                                      cv_seed, n_valid, n_cv, parameters, file_name_params=file_name_params,
                                      append_info=append_info, profit=profit)

            # Return Final Result
            if return_prob_test:
                return prob_test_mean

    def stack_train(self, x_train, y_train, w_train, x_g_train, x_valid, y_valid,
                    w_valid, x_g_valid, x_test, x_g_test, parameters=None, show_importance=False):

        # Print Start Information
        self.print_start_info()
        print('Number of Features: ', x_train.shape[1])
        print('------------------------------------------------------')

        # Build Network
        tf.reset_default_graph()
        train_graph = tf.Graph()

        with train_graph.as_default():

            # Inputs
            inputs, labels, weights, lr, keep_prob, is_train = self.input_tensor()

            # Logits
            logits = self.model(inputs, self.unit_number, keep_prob, is_train)
            logits = tf.identity(logits, name='logits')

            # Loss
            with tf.name_scope('Loss'):
                # cost_ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
                cost_ = self.log_loss(logits, weights, labels)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr).minimize(cost_)

        with tf.Session(graph=train_graph) as sess:

            start_time = time.time()
            sess.run(tf.global_variables_initializer())

            for epoch_i in range(self.epochs):

                batch_counter = 0

                for batch_i, (batch_x, batch_y, batch_w) in enumerate(self.get_batches(x_train,
                                                                                       y_train,
                                                                                       w_train,
                                                                                       self.batch_size)):

                    batch_counter += 1

                    _, cost_train = sess.run([optimizer, cost_],
                                             {inputs: batch_x,
                                              labels: batch_y,
                                              weights: batch_w,
                                              lr: self.learning_rate,
                                              keep_prob: self.keep_probability,
                                              is_train: True})

                    if str(cost_train) == 'nan':
                        raise ValueError('NaN BUG!!! Try Another Seed!!!')

                    if batch_counter % self.display_step == 0 and batch_i > 0:

                        cost_valid_all = []

                        for iii, (valid_batch_x,
                                  valid_batch_y,
                                  valid_batch_w) in enumerate(self.get_batches(x_valid,
                                                                               y_valid,
                                                                               w_valid,
                                                                               self.batch_size)):
                            cost_valid_i = sess.run(cost_, {inputs: valid_batch_x,
                                                            labels: valid_batch_y,
                                                            weights: valid_batch_w,
                                                            keep_prob: 1.0,
                                                            is_train: False})

                            cost_valid_all.append(cost_valid_i)

                        cost_valid = sum(cost_valid_all) / len(cost_valid_all)

                        total_time = time.time() - start_time

                        print('Epoch: {}/{} |'.format(epoch_i + 1, self.epochs),
                              'Batch: {} |'.format(batch_counter),
                              'Time: {:>3.2f}s |'.format(total_time),
                              'Train_Loss: {:>.8f} |'.format(cost_train),
                              'Valid_Loss: {:>.8f}'.format(cost_valid))

            # Prediction
            print('------------------------------------------------------')
            print('Predicting Probabilities...')

            logits_pred_train = sess.run(logits, {inputs: x_train, keep_prob: 1.0, is_train: False})
            logits_pred_valid = sess.run(logits, {inputs: x_valid, keep_prob: 1.0, is_train: False})
            logits_pred_test = sess.run(logits, {inputs: x_test, keep_prob: 1.0, is_train: False})

            logits_pred_train = logits_pred_train.flatten()
            logits_pred_valid = logits_pred_valid.flatten()
            logits_pred_test = logits_pred_test.flatten()

            prob_train = 1.0 / (1.0 + np.exp(-logits_pred_train))
            prob_valid = 1.0 / (1.0 + np.exp(-logits_pred_valid))
            prob_test = 1.0 / (1.0 + np.exp(-logits_pred_test))

            loss_train, loss_valid, loss_train_w, loss_valid_w = \
                utils.print_loss(prob_train, y_train, w_train, prob_valid, y_valid, w_valid)

            losses = [loss_train, loss_valid, loss_train_w, loss_valid_w]

            return prob_valid, prob_test, losses

    def prejudge_stack_train(self, x_train, x_g_train, y_train, w_train, e_train, x_valid,
                             x_g_valid, y_valid, w_valid, e_valid, x_test, x_g_test,
                             pred_path=None, loss_log_path=None, csv_log_path=None, parameters=None, cv_args=None,
                             train_seed=None, show_importance=False, show_accuracy=False, save_final_pred=True,
                             save_csv_log=True, csv_idx=None, mode=None, file_name_params=None,
                             param_name_list=None, param_value_list=None, append_info=None):

        # Check if directories exit or not
        utils.check_dir_model(pred_path, loss_log_path)

        cv_args_copy = copy.deepcopy(cv_args)
        if 'n_valid' in cv_args:
            n_valid = cv_args_copy['n_valid']
        elif 'valid_rate' in cv_args:
            n_valid = cv_args_copy['valid_rate']
        else:
            n_valid = ''
        n_cv = cv_args_copy['n_cv']
        cv_seed = cv_args_copy['cv_seed']

        if csv_idx is None:
            csv_idx = self.model_name

        # Print Start Information and Get Model Name
        self.print_start_info()

        print('======================================================')
        print('Number of Features: ', x_train.shape[1])
        print('------------------------------------------------------')

        # Build Network
        tf.reset_default_graph()
        train_graph = tf.Graph()

        with train_graph.as_default():

            # Inputs
            inputs, labels, weights, lr, keep_prob, is_train = self.input_tensor()

            # Logits
            logits = self.model(inputs, self.unit_number, keep_prob, is_train)
            logits = tf.identity(logits, name='logits')

            # Loss
            with tf.name_scope('Loss'):
                # cost_ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
                cost_ = self.log_loss(logits, weights, labels)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr).minimize(cost_)

        with tf.Session(graph=train_graph) as sess:

            start_time = time.time()
            sess.run(tf.global_variables_initializer())

            for epoch_i in range(self.epochs):

                batch_counter = 0

                for batch_i, (batch_x, batch_y, batch_w) in enumerate(self.get_batches(x_train,
                                                                                       y_train,
                                                                                       w_train,
                                                                                       self.batch_size)):

                    batch_counter += 1

                    _, cost_train = sess.run([optimizer, cost_],
                                             {inputs: batch_x,
                                              labels: batch_y,
                                              weights: batch_w,
                                              lr: self.learning_rate,
                                              keep_prob: self.keep_probability,
                                              is_train: True})

                    if str(cost_train) == 'nan':
                        raise ValueError('NaN BUG!!! Try Another Seed!!!')

                    if batch_counter % self.display_step == 0 and batch_i > 0:

                        cost_valid_all = []

                        for iii, (valid_batch_x,
                                  valid_batch_y,
                                  valid_batch_w) in enumerate(self.get_batches(x_valid,
                                                                               y_valid,
                                                                               w_valid,
                                                                               self.batch_size)):
                            cost_valid_i = sess.run(cost_, {inputs: valid_batch_x,
                                                            labels: valid_batch_y,
                                                            weights: valid_batch_w,
                                                            keep_prob: 1.0,
                                                            is_train: False})

                            cost_valid_all.append(cost_valid_i)

                        cost_valid = sum(cost_valid_all) / len(cost_valid_all)

                        total_time = time.time() - start_time

                        print('Epoch: {}/{} |'.format(epoch_i + 1, self.epochs),
                              'Batch: {} |'.format(batch_counter),
                              'Time: {:>3.2f}s |'.format(total_time),
                              'Train_Loss: {:>.8f} |'.format(cost_train),
                              'Valid_Loss: {:>.8f}'.format(cost_valid))

            # Prediction
            print('------------------------------------------------------')
            print('Predicting Train Probabilities...')

            logits_pred_train = sess.run(logits, {inputs: x_train, keep_prob: 1.0, is_train: False})
            logits_pred_valid = sess.run(logits, {inputs: x_valid, keep_prob: 1.0, is_train: False})
            logits_pred_test = sess.run(logits, {inputs: x_test, keep_prob: 1.0, is_train: False})

            logits_pred_train = logits_pred_train.flatten()
            logits_pred_valid = logits_pred_valid.flatten()
            logits_pred_test = logits_pred_test.flatten()

            prob_train = 1.0 / (1.0 + np.exp(-logits_pred_train))
            prob_valid = 1.0 / (1.0 + np.exp(-logits_pred_valid))
            prob_test = 1.0 / (1.0 + np.exp(-logits_pred_test))

            loss_train, loss_valid, loss_train_w, loss_valid_w = \
                utils.print_loss(prob_train, y_train, w_train, prob_valid, y_valid, w_valid)

            # Save Final Result
            if save_final_pred:
                self.save_final_pred(mode, save_final_pred, prob_test, pred_path,
                                     parameters, csv_idx, train_seed, cv_seed, file_name_params=file_name_params)

            # Print Total Losses
            utils.print_total_loss(loss_train, loss_valid, loss_train_w, loss_valid_w)
            losses = [loss_train, loss_valid, loss_train_w, loss_valid_w]

            # Print and Get Accuracies of CV
            acc_train, acc_valid, acc_train_era, acc_valid_era = \
                utils.print_and_get_accuracy(prob_train, y_train, e_train,
                                             prob_valid, y_valid, e_valid, show_accuracy)

            # Save Final Losses to File
            utils.save_final_loss_log(loss_log_path + self.model_name + '_', parameters, n_valid, n_cv,
                                      loss_train, loss_valid, loss_train_w, loss_valid_w,
                                      train_seed, cv_seed, acc_train, acc_train_era)

            # Save Loss Log to csv File
            if save_csv_log:
                self.save_csv_log(mode, csv_log_path, param_name_list, param_value_list, csv_idx,
                                  loss_train_w, loss_valid_w, acc_train, train_seed,
                                  cv_seed, n_valid, n_cv, parameters, file_name_params=file_name_params)

            return prob_valid, prob_test, losses
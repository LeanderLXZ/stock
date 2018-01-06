#  import math
import time
from .. import utils
import os
from os.path import isdir
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# from keras.layers import Dense
# from keras.models import Sequential
# from keras.layers import Dropout
# from keras import initializers
# from keras import optimizers

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import seaborn as sns

sns.set(style="whitegrid", color_codes=True)
sns.set(font_scale=1)
color = sns.color_palette()


# Logistic Regression
class LRegression:

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te
        self.importance = np.array([])
        self.indices = np.array([])
        self.std = np.array([])

    def show(self):

        feature_num = self.x_train.shape[1]

        plt.figure(figsize=(20, 10))
        plt.title('Feature Importance in Logistic Regression')
        plt.bar(range(feature_num), self.importance[self.indices],
                color=color[2], yerr=self.std[self.indices], align="center")
        plt.xticks(range(feature_num), self.indices)
        plt.xlim([-1, feature_num])
        plt.show()

    @staticmethod
    def get_clf(parameters):

        clf = LogisticRegression(**parameters)

        return clf

    def get_importance(self, clf):

        print('------------------------------------------------------')
        print('Feature Importance')
        self.importance = np.abs(clf.coef_)[0]
        indices = np.argsort(self.importance)[::-1]

        feature_num = self.x_train.shape[1]

        for f in range(feature_num):
            print("%d | feature %d | %f" % (f + 1, indices[f], self.importance[indices[f]]))

    def predict(self, clf, x_test, pred_path=None):

        print('Predicting...')

        prob_test = np.array(clf.predict_proba(x_test))[:, 1]

        if pred_path is not None:
            utils.save_pred_to_csv(pred_path, self.id_test, prob_test)

        return prob_test

    def get_prob_train(self, clf, x_train, pred_path=None):

        print('Predicting...')

        prob_train = np.array(clf.predict_proba(x_train))[:, 1]

        if pred_path is not None:
            utils.save_prob_train_to_csv(pred_path, prob_train, self.y_train)

        return prob_train

    def train(self, pred_path, loss_log_path, n_valid, n_cv, n_era, cv_seed,
              era_list=None, parameters=None, show_importance=False):

        # Check if directories exit or not
        utils.check_dir_model(pred_path, loss_log_path)

        print('------------------------------------------------------')
        print('Training Logistic Regression...')
        print('------------------------------------------------------')

        count = 0
        prob_test_total = []
        prob_train_total = []
        loss_train_total = []
        loss_valid_total = []
        loss_train_w_total = []
        loss_valid_w_total = []

        for x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, \
            e_valid, valid_era in CrossValidation.era_k_fold_with_weight(x=self.x_train,
                                                                         y=self.y_train,
                                                                         w=self.w_train,
                                                                         e=self.e_train,
                                                                         n_valid=n_valid,
                                                                         n_cv=n_cv,
                                                                         n_era=n_era,
                                                                         seed=cv_seed,
                                                                         era_list=era_list):
            count += 1

            print('======================================================')
            print('Training on the Cross Validation Set: {}'.format(count))
            print('Validation Set Era: ', valid_era)
            print('------------------------------------------------------')

            # Classifier
            clf = self.get_clf(parameters)

            clf.fit(x_train, y_train, sample_weight=w_train)

            # Prediction
            prob_test = self.predict(clf, self.x_test, pred_path=pred_path + 'cv_results/lr_cv_{}_'.format(count))

            # Save Train Probabilities to CSV File
            prob_train = self.get_prob_train(clf, self.x_train,
                                             pred_path=pred_path + 'cv_prob_train/lr_cv_{}_'.format(count))

            # Get Probabilities of Validation Set
            prob_valid = self.predict(clf, x_valid)

            # Print LogLoss
            print('------------------------------------------------------')
            print('Validation Set Era: ', valid_era)
            loss_train, loss_valid, loss_train_w, loss_valid_w = \
                utils.print_loss_proba(clf, x_train, y_train, w_train, x_valid, y_valid, w_valid)

            # Print and Get Accuracies of CV
            acc_train_cv, acc_valid_cv, acc_train_cv_era, acc_valid_cv_era = \
                utils.print_and_get_accuracy(prob_train, y_train, e_train, prob_valid, y_valid, e_valid)

            # Save Losses to File
            utils.save_loss_log(loss_log_path + 'lr_', count, parameters, n_valid, n_cv, valid_era,
                                loss_train, loss_valid, loss_train_w, loss_valid_w, acc_train_cv, acc_valid_cv,
                                acc_train_cv_era, acc_valid_cv_era)

            # Feature Importance
            if show_importance:
                self.get_importance(clf)

            prob_test_total.append(list(prob_test))
            prob_train_total.append(list(prob_train))
            loss_train_total.append(loss_train)
            loss_valid_total.append(loss_valid)
            loss_train_w_total.append(loss_train_w)
            loss_valid_w_total.append(loss_valid_w)

        print('======================================================')
        print('Calculating Final Result...')

        prob_test_mean = np.mean(np.array(prob_test_total), axis=0)
        prob_train_mean = np.mean(np.array(prob_train_total), axis=0)
        loss_train_mean = np.mean(np.array(loss_train_total), axis=0)
        loss_valid_mean = np.mean(np.array(loss_valid_total), axis=0)
        loss_train_w_mean = np.mean(np.array(loss_train_w_total), axis=0)
        loss_valid_w_mean = np.mean(np.array(loss_valid_w_total), axis=0)

        # Print Total Losses
        utils.print_total_loss(loss_train_mean, loss_valid_mean, loss_train_w_mean, loss_valid_w_mean)

        # Print and Get Accuracies of CV of All Train Set
        acc_train, acc_train_era = utils.print_and_get_train_accuracy(prob_train_mean, self.y_train, self.e_train)

        # Save Final Losses to File
        utils.save_final_loss_log(loss_log_path + 'lr_', parameters, n_valid, n_cv, loss_train_mean,
                                  loss_valid_mean, loss_train_w_mean, loss_valid_w_mean, acc_train, acc_train_era)

        # Save Final Result
        utils.save_pred_to_csv(pred_path + 'final_results/lr_', self.id_test, prob_test_mean)
        utils.save_prob_train_to_csv(pred_path + 'final_prob_train/lr_', prob_train_mean, self.y_train)

    def stack_train(self, x_train, y_train, w_train, x_g_train, x_valid, y_valid,
                    w_valid, x_g_valid, x_test, x_g_test, parameters, show_importance=False):

        print('------------------------------------------------------')
        print('Training Logistic Regression...')
        print('------------------------------------------------------')

        clf = self.get_clf(parameters)

        clf.fit(x_train, y_train, sample_weight=w_train)

        # Feature Importance
        if show_importance:
            self.get_importance(clf)

        # Print LogLoss
        loss_train, loss_valid, \
        loss_train_w, loss_valid_w = utils.print_loss_proba(clf, x_train, y_train, w_train,
                                                            x_valid, y_valid, w_valid)

        losses = [loss_train, loss_valid, loss_train_w, loss_valid_w]

        # Prediction
        prob_valid = self.predict(clf, x_valid)
        prob_test = self.predict(clf, x_test)

        return prob_valid, prob_test, losses


# k-Nearest Neighbor
class KNearestNeighbor:

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te):
        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te

    def train(self, parameters):

        clf = KNeighborsClassifier()
        '''
        KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                             metric_params=None, n_jobs=1, n_neighbors=5, p=2,
                             weights='uniform')
        '''

        clf.fit(self.x_train, self.y_train)  # without parameter sample_weight

        # K-fold cross-validation on Logistic Regression
        scores = cross_val_score(clf, self.x_train, self.y_train, cv=10)
        print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))


# SVM-SVC
class SupportVectorClustering:

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te):
        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te

    def train(self, parameters):

        clf = SVC(random_state=1)

        clf.fit(self.x_train, self.y_train, sample_weight=self.w_train)

        scores = cross_val_score(clf, self.x_train, self.y_train, cv=10)
        print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))


# Gaussian NB
class Gaussian:

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te):
        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te

    def train(self, parameters):

        clf_gnb = GaussianNB()

        clf_gnb.fit(self.x_train, self.y_train, sample_weight=self.w_train)

        scores = cross_val_score(clf_gnb, self.x_train, self.y_train, cv=10)
        print("Accuracy: %0.6f (+/- %0.6f)" % (scores.mean(), scores.std() * 2))


# Decision Tree
class DecisionTree:

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te
        self.importance = np.array([])
        self.indices = np.array([])

    def show(self):

        feature_num = self.x_train.shape[1]

        plt.figure(figsize=(20, 10))
        plt.title('Feature Importance in Decision Tree')
        plt.bar(range(feature_num), self.importance[self.indices], color=color[1], align="center")
        plt.xticks(range(feature_num), self.indices)
        plt.xlim([-1, feature_num])
        plt.show()

    @staticmethod
    def get_clf(parameters):

        clf = DecisionTreeClassifier(**parameters)

        return clf

    def get_importance(self, clf):

        print('------------------------------------------------------')
        print('Feature Importance')

        self.importance = clf.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = self.x_train.shape[1]

        for f in range(feature_num):
            print("%d | feature %d | %f" % (f + 1, self.indices[f], self.importance[self.indices[f]]))

    def predict(self, clf, x_test, pred_path=None):

        print('Predicting...')

        prob_test = np.array(clf.predict_proba(x_test))[:, 1]

        if pred_path is not None:
            utils.save_pred_to_csv(pred_path, self.id_test, prob_test)

        return prob_test

    def get_prob_train(self, clf, x_train, pred_path=None):

        print('Predicting...')

        prob_train = np.array(clf.predict_proba(x_train))[:, 1]

        if pred_path is not None:
            utils.save_prob_train_to_csv(pred_path, prob_train, self.y_train)

        return prob_train

    def train(self, pred_path, loss_log_path, n_valid, n_cv, n_era, cv_seed,
              era_list=None, parameters=None, show_importance=False):

        # Check if directories exit or not
        utils.check_dir_model(pred_path, loss_log_path)

        print('------------------------------------------------------')
        print('Training Decision Tree...')
        print('------------------------------------------------------')

        count = 0
        prob_test_total = []
        prob_train_total = []
        loss_train_total = []
        loss_valid_total = []
        loss_train_w_total = []
        loss_valid_w_total = []

        for x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, \
            e_valid, valid_era in CrossValidation.era_k_fold_with_weight(x=self.x_train,
                                                                         y=self.y_train,
                                                                         w=self.w_train,
                                                                         e=self.e_train,
                                                                         n_valid=n_valid,
                                                                         n_cv=n_cv,
                                                                         n_era=n_era,
                                                                         seed=cv_seed,
                                                                         era_list=era_list):
            count += 1

            print('======================================================')
            print('Training on the Cross Validation Set: {}'.format(count))
            print('Validation Set Era: ', valid_era)
            print('------------------------------------------------------')

            # Classifier
            clf = self.get_clf(parameters)

            clf.fit(x_train, y_train, sample_weight=w_train)

            # Prediction
            prob_test = self.predict(clf, self.x_test, pred_path=pred_path + 'cv_results/dt_cv_{}_'.format(count))

            # Save Train Probabilities to CSV File
            prob_train = self.get_prob_train(clf, self.x_train,
                                             pred_path=pred_path + 'cv_prob_train/dt_cv_{}_'.format(count))

            # Get Probabilities of Validation Set
            prob_valid = self.predict(clf, x_valid)

            # Print LogLoss
            print('------------------------------------------------------')
            print('Validation Set Era: ', valid_era)
            loss_train, loss_valid, loss_train_w, loss_valid_w = \
                utils.print_loss_proba(clf, x_train, y_train, w_train, x_valid, y_valid, w_valid)

            # Print and Get Accuracies of CV
            acc_train_cv, acc_valid_cv, acc_train_cv_era, acc_valid_cv_era = \
                utils.print_and_get_accuracy(prob_train, y_train, e_train, prob_valid, y_valid, e_valid)

            # Save Losses to File
            utils.save_loss_log(loss_log_path + 'dt_', count, parameters, n_valid, n_cv, valid_era,
                                loss_train, loss_valid, loss_train_w, loss_valid_w, acc_train_cv, acc_valid_cv,
                                acc_train_cv_era, acc_valid_cv_era)

            # Feature Importance
            if show_importance:
                self.get_importance(clf)

            prob_test_total.append(list(prob_test))
            prob_train_total.append(list(prob_train))
            loss_train_total.append(loss_train)
            loss_valid_total.append(loss_valid)
            loss_train_w_total.append(loss_train_w)
            loss_valid_w_total.append(loss_valid_w)

        print('======================================================')
        print('Calculating Final Result...')

        prob_test_mean = np.mean(np.array(prob_test_total), axis=0)
        prob_train_mean = np.mean(np.array(prob_train_total), axis=0)
        loss_train_mean = np.mean(np.array(loss_train_total), axis=0)
        loss_valid_mean = np.mean(np.array(loss_valid_total), axis=0)
        loss_train_w_mean = np.mean(np.array(loss_train_w_total), axis=0)
        loss_valid_w_mean = np.mean(np.array(loss_valid_w_total), axis=0)

        # Print Total Losses
        utils.print_total_loss(loss_train_mean, loss_valid_mean, loss_train_w_mean, loss_valid_w_mean)

        # Print and Get Accuracies of CV of All Train Set
        acc_train, acc_train_era = utils.print_and_get_train_accuracy(prob_train_mean, self.y_train, self.e_train)

        # Save Final Losses to File
        utils.save_final_loss_log(loss_log_path + 'dt_', parameters, n_valid, n_cv, loss_train_mean,
                                  loss_valid_mean, loss_train_w_mean, loss_valid_w_mean, acc_train, acc_train_era)

        # Save Final Result
        utils.save_pred_to_csv(pred_path + 'final_results/dt_', self.id_test, prob_test_mean)
        utils.save_prob_train_to_csv(pred_path + 'final_prob_train/dt_', prob_train_mean, self.y_train)

    def stack_train(self, x_train, y_train, w_train, x_g_train, x_valid, y_valid,
                    w_valid, x_g_valid, x_test, x_g_test, parameters, show_importance=False):

        print('------------------------------------------------------')
        print('Training Decision Tree...')
        print('------------------------------------------------------')

        clf = self.get_clf(parameters)

        clf.fit(x_train, y_train, sample_weight=w_train)

        # Feature Importance
        if show_importance:
            self.get_importance(clf)

        # Print LogLoss
        loss_train, loss_valid, \
        loss_train_w, loss_valid_w = utils.print_loss_proba(clf, x_train, y_train, w_train,
                                                            x_valid, y_valid, w_valid)

        losses = [loss_train, loss_valid, loss_train_w, loss_valid_w]

        # Prediction
        prob_valid = self.predict(clf, x_valid)
        prob_test = self.predict(clf, x_test)

        return prob_valid, prob_test, losses


# Random Forest
class RandomForest:

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te
        self.importance = np.array([])
        self.indices = np.array([])
        self.std = np.array([])

    def show(self):

        feature_num = self.x_train.shape[1]

        plt.figure(figsize=(20, 10))
        plt.title('Feature Importance in Random Forest')
        plt.bar(range(feature_num), self.importance[self.indices],
                color=color[2], yerr=self.std[self.indices], align="center")
        plt.xticks(range(feature_num), self.indices)
        plt.xlim([-1, feature_num])
        plt.show()

    @staticmethod
    def get_clf(parameters):

        clf = RandomForestClassifier(**parameters)

        return clf

    def get_importance(self, clf):

        print('------------------------------------------------------')
        print('Feature Importance')

        self.importance = clf.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = self.x_train.shape[1]

        for f in range(feature_num):
            print("%d. feature %d (%f)" % (f + 1, self.indices[f], self.importance[self.indices[f]]))

    def predict(self, clf, x_test, pred_path=None):

        print('Predicting...')

        prob_test = np.array(clf.predict_proba(x_test))[:, 1]

        if pred_path is not None:
            utils.save_pred_to_csv(pred_path, self.id_test, prob_test)

        return prob_test

    def get_prob_train(self, clf, x_train, pred_path=None):

        print('Predicting...')

        prob_train = np.array(clf.predict_proba(x_train))[:, 1]

        if pred_path is not None:
            utils.save_prob_train_to_csv(pred_path, prob_train, self.y_train)

        return prob_train

    def train(self, pred_path, loss_log_path, n_valid, n_cv, n_era, cv_seed,
              era_list=None, parameters=None, show_importance=False):

        # Check if directories exit or not
        utils.check_dir_model(pred_path, loss_log_path)

        print('------------------------------------------------------')
        print('Training Random Forest...')
        print('------------------------------------------------------')

        count = 0
        prob_test_total = []
        prob_train_total = []
        loss_train_total = []
        loss_valid_total = []
        loss_train_w_total = []
        loss_valid_w_total = []

        for x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, \
            e_valid, valid_era in CrossValidation.era_k_fold_with_weight(x=self.x_train,
                                                                         y=self.y_train,
                                                                         w=self.w_train,
                                                                         e=self.e_train,
                                                                         n_valid=n_valid,
                                                                         n_cv=n_cv,
                                                                         n_era=n_era,
                                                                         seed=cv_seed,
                                                                         era_list=era_list):
            count += 1

            print('======================================================')
            print('Training on the Cross Validation Set: {}'.format(count))
            print('Validation Set Era: ', valid_era)
            print('------------------------------------------------------')

            # Classifier
            clf = self.get_clf(parameters)

            clf.fit(x_train, y_train, sample_weight=w_train)

            # Prediction
            prob_test = self.predict(clf, self.x_test, pred_path=pred_path + 'cv_results/rf_cv_{}_'.format(count))

            # Save Train Probabilities to CSV File
            prob_train = self.get_prob_train(clf, self.x_train,
                                             pred_path=pred_path + 'cv_prob_train/rf_cv_{}_'.format(count))

            # Get Probabilities of Validation Set
            prob_valid = self.predict(clf, x_valid)

            # Print LogLoss
            print('------------------------------------------------------')
            print('Validation Set Era: ', valid_era)
            loss_train, loss_valid, loss_train_w, loss_valid_w = \
                utils.print_loss_proba(clf, x_train, y_train, w_train, x_valid, y_valid, w_valid)

            # Print and Get Accuracies of CV
            acc_train_cv, acc_valid_cv, acc_train_cv_era, acc_valid_cv_era = \
                utils.print_and_get_accuracy(prob_train, y_train, e_train, prob_valid, y_valid, e_valid)

            # Save Losses to File
            utils.save_loss_log(loss_log_path + 'rf_', count, parameters, n_valid, n_cv, valid_era,
                                loss_train, loss_valid, loss_train_w, loss_valid_w, acc_train_cv, acc_valid_cv,
                                acc_train_cv_era, acc_valid_cv_era)

            # Feature Importance
            if show_importance:
                self.get_importance(clf)

            prob_test_total.append(list(prob_test))
            prob_train_total.append(list(prob_train))
            loss_train_total.append(loss_train)
            loss_valid_total.append(loss_valid)
            loss_train_w_total.append(loss_train_w)
            loss_valid_w_total.append(loss_valid_w)

        print('======================================================')
        print('Calculating Final Result...')

        prob_test_mean = np.mean(np.array(prob_test_total), axis=0)
        prob_train_mean = np.mean(np.array(prob_train_total), axis=0)
        loss_train_mean = np.mean(np.array(loss_train_total), axis=0)
        loss_valid_mean = np.mean(np.array(loss_valid_total), axis=0)
        loss_train_w_mean = np.mean(np.array(loss_train_w_total), axis=0)
        loss_valid_w_mean = np.mean(np.array(loss_valid_w_total), axis=0)

        # Print Total Losses
        utils.print_total_loss(loss_train_mean, loss_valid_mean, loss_train_w_mean, loss_valid_w_mean)

        # Print and Get Accuracies of CV of All Train Set
        acc_train, acc_train_era = utils.print_and_get_train_accuracy(prob_train_mean, self.y_train, self.e_train)

        # Save Final Losses to File
        utils.save_final_loss_log(loss_log_path + 'rf_', parameters, n_valid, n_cv, loss_train_mean,
                                  loss_valid_mean, loss_train_w_mean, loss_valid_w_mean, acc_train, acc_train_era)

        # Save Final Result
        utils.save_pred_to_csv(pred_path + 'final_results/rf_', self.id_test, prob_test_mean)
        utils.save_prob_train_to_csv(pred_path + 'final_prob_train/rf_', prob_train_mean, self.y_train)

    def stack_train(self, x_train, y_train, w_train, x_g_train, x_valid, y_valid,
                    w_valid, x_g_valid, x_test, x_g_test, parameters, show_importance=False):

        print('------------------------------------------------------')
        print('Training Random Forest...')
        print('------------------------------------------------------')

        clf = self.get_clf(parameters)

        clf.fit(x_train, y_train, sample_weight=w_train)

        # Feature Importance
        if show_importance:
            self.get_importance(clf)

        # Print LogLoss
        loss_train, loss_valid, \
        loss_train_w, loss_valid_w = utils.print_loss_proba(clf, x_train, y_train, w_train,
                                                            x_valid, y_valid, w_valid)

        losses = [loss_train, loss_valid, loss_train_w, loss_valid_w]

        # Prediction
        prob_valid = self.predict(clf, x_valid)
        prob_test = self.predict(clf, x_test)

        return prob_valid, prob_test, losses


# Extra Trees
class ExtraTrees:

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te
        self.importance = np.array([])
        self.indices = np.array([])
        self.std = np.array([])

    def show(self):

        feature_num = self.x_train.shape[1]

        plt.figure(figsize=(20, 10))
        plt.title('Feature Importance in Extra Trees')
        plt.bar(range(feature_num), self.importance[self.indices],
                color=color[3], yerr=self.std[self.indices], align="center")
        plt.xticks(range(feature_num), self.indices)
        plt.xlim([-1, feature_num])
        plt.show()

    @staticmethod
    def get_clf(parameters):

        clf = ExtraTreesClassifier(**parameters)

        return clf

    def get_importance(self, clf):

        print('------------------------------------------------------')
        print('Feature Importance')

        self.importance = clf.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = self.x_train.shape[1]

        for f in range(feature_num):
            print("%d | feature %d | %f" % (f + 1, self.indices[f], self.importance[self.indices[f]]))

    def predict(self, clf, x_test, pred_path=None):

        print('Predicting...')

        prob_test = np.array(clf.predict_proba(x_test))[:, 1]

        if pred_path is not None:
            utils.save_pred_to_csv(pred_path, self.id_test, prob_test)

        return prob_test

    def get_prob_train(self, clf, x_train, pred_path=None):

        print('Predicting...')

        prob_train = np.array(clf.predict_proba(x_train))[:, 1]

        if pred_path is not None:
            utils.save_prob_train_to_csv(pred_path, prob_train, self.y_train)

        return prob_train

    def train(self, pred_path, loss_log_path, n_valid, n_cv, n_era, cv_seed,
              era_list=None, parameters=None, show_importance=False):

        # Check if directories exit or not
        utils.check_dir_model(pred_path, loss_log_path)

        print('------------------------------------------------------')
        print('Training Extra Trees...')
        print('------------------------------------------------------')

        count = 0
        prob_test_total = []
        prob_train_total = []
        loss_train_total = []
        loss_valid_total = []
        loss_train_w_total = []
        loss_valid_w_total = []

        for x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, \
            e_valid, valid_era in CrossValidation.era_k_fold_with_weight(x=self.x_train,
                                                                         y=self.y_train,
                                                                         w=self.w_train,
                                                                         e=self.e_train,
                                                                         n_valid=n_valid,
                                                                         n_cv=n_cv,
                                                                         n_era=n_era,
                                                                         seed=cv_seed,
                                                                         era_list=era_list):
            count += 1

            print('======================================================')
            print('Training on the Cross Validation Set: {}'.format(count))
            print('Validation Set Era: ', valid_era)
            print('------------------------------------------------------')

            # Classifier
            clf = self.get_clf(parameters)

            clf.fit(x_train, y_train, sample_weight=w_train)

            # Prediction
            prob_test = self.predict(clf, self.x_test, pred_path=pred_path + 'cv_results/et_cv_{}_'.format(count))

            # Save Train Probabilities to CSV File
            prob_train = self.get_prob_train(clf, self.x_train,
                                             pred_path=pred_path + 'cv_prob_train/et_cv_{}_'.format(count))

            # Get Probabilities of Validation Set
            prob_valid = self.predict(clf, x_valid)

            # Print LogLoss
            print('------------------------------------------------------')
            print('Validation Set Era: ', valid_era)
            loss_train, loss_valid, loss_train_w, loss_valid_w = \
                utils.print_loss_proba(clf, x_train, y_train, w_train, x_valid, y_valid, w_valid)

            # Print and Get Accuracies of CV
            acc_train_cv, acc_valid_cv, acc_train_cv_era, acc_valid_cv_era = \
                utils.print_and_get_accuracy(prob_train, y_train, e_train, prob_valid, y_valid, e_valid)

            # Save Losses to File
            utils.save_loss_log(loss_log_path + 'et_', count, parameters, n_valid, n_cv, valid_era,
                                loss_train, loss_valid, loss_train_w, loss_valid_w, acc_train_cv, acc_valid_cv,
                                acc_train_cv_era, acc_valid_cv_era)

            # Feature Importance
            if show_importance:
                self.get_importance(clf)

            prob_test_total.append(list(prob_test))
            prob_train_total.append(list(prob_train))
            loss_train_total.append(loss_train)
            loss_valid_total.append(loss_valid)
            loss_train_w_total.append(loss_train_w)
            loss_valid_w_total.append(loss_valid_w)

        print('======================================================')
        print('Calculating Final Result...')

        prob_test_mean = np.mean(np.array(prob_test_total), axis=0)
        prob_train_mean = np.mean(np.array(prob_train_total), axis=0)
        loss_train_mean = np.mean(np.array(loss_train_total), axis=0)
        loss_valid_mean = np.mean(np.array(loss_valid_total), axis=0)
        loss_train_w_mean = np.mean(np.array(loss_train_w_total), axis=0)
        loss_valid_w_mean = np.mean(np.array(loss_valid_w_total), axis=0)

        # Print Total Losses
        utils.print_total_loss(loss_train_mean, loss_valid_mean, loss_train_w_mean, loss_valid_w_mean)

        # Print and Get Accuracies of CV of All Train Set
        acc_train, acc_train_era = utils.print_and_get_train_accuracy(prob_train_mean, self.y_train, self.e_train)

        # Save Final Losses to File
        utils.save_final_loss_log(loss_log_path + 'et_', parameters, n_valid, n_cv, loss_train_mean,
                                  loss_valid_mean, loss_train_w_mean, loss_valid_w_mean, acc_train, acc_train_era)

        # Save Final Result
        utils.save_pred_to_csv(pred_path + 'final_results/et_', self.id_test, prob_test_mean)
        utils.save_prob_train_to_csv(pred_path + 'final_prob_train/et_', prob_train_mean, self.y_train)

    def stack_train(self, x_train, y_train, w_train, x_g_train, x_valid, y_valid,
                    w_valid, x_g_valid, x_test, x_g_test, parameters, show_importance=False):

        print('------------------------------------------------------')
        print('Training Extra Trees...')
        print('------------------------------------------------------')

        clf = self.get_clf(parameters)

        clf.fit(x_train, y_train, sample_weight=w_train)

        # Feature Importance
        if show_importance:
            self.get_importance(clf)

        # Print LogLoss
        loss_train, loss_valid, \
        loss_train_w, loss_valid_w = utils.print_loss_proba(clf, x_train, y_train, w_train,
                                                            x_valid, y_valid, w_valid)

        losses = [loss_train, loss_valid, loss_train_w, loss_valid_w]

        # Prediction
        prob_valid = self.predict(clf, x_valid)
        prob_test = self.predict(clf, x_test)

        return prob_valid, prob_test, losses


# AdaBoost
class AdaBoost:

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te
        self.importance = np.array([])
        self.indices = np.array([])
        self.std = np.array([])

    def show(self):

        feature_num = self.x_train.shape[1]

        plt.figure(figsize=(20, 10))
        plt.title('Feature Importance in AdaBoost')
        plt.bar(range(feature_num), self.importance[self.indices],
                color=color[4], yerr=self.std[self.indices], align="center")
        plt.xticks(range(feature_num), self.indices)
        plt.xlim([-1, feature_num])
        plt.show()

    @staticmethod
    def get_clf(parameters):

        clf = AdaBoostClassifier(**parameters)

        return clf

    def get_importance(self, clf):

        print('------------------------------------------------------')
        print('Feature Importance')

        self.importance = clf.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = self.x_train.shape[1]

        for f in range(feature_num):
            print("%d | feature %d | %f" % (f + 1, self.indices[f], self.importance[self.indices[f]]))

    def predict(self, clf, x_test, pred_path=None):

        print('Predicting...')

        prob_test = np.array(clf.predict_proba(x_test))[:, 1]

        if pred_path is not None:
            utils.save_pred_to_csv(pred_path, self.id_test, prob_test)

        return prob_test

    def get_prob_train(self, clf, x_train, pred_path=None):

        print('Predicting...')

        prob_train = np.array(clf.predict_proba(x_train))[:, 1]

        if pred_path is not None:
            utils.save_prob_train_to_csv(pred_path, prob_train, self.y_train)

        return prob_train

    def train(self, pred_path, loss_log_path, n_valid, n_cv, n_era, cv_seed,
              era_list=None, parameters=None, show_importance=False):

        # Check if directories exit or not
        utils.check_dir_model(pred_path, loss_log_path)

        print('------------------------------------------------------')
        print('Training AdaBoost...')
        print('------------------------------------------------------')

        count = 0
        prob_test_total = []
        prob_train_total = []
        loss_train_total = []
        loss_valid_total = []
        loss_train_w_total = []
        loss_valid_w_total = []

        for x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, \
            e_valid, valid_era in CrossValidation.era_k_fold_with_weight(x=self.x_train,
                                                                         y=self.y_train,
                                                                         w=self.w_train,
                                                                         e=self.e_train,
                                                                         n_valid=n_valid,
                                                                         n_cv=n_cv,
                                                                         n_era=n_era,
                                                                         seed=cv_seed,
                                                                         era_list=era_list):
            count += 1

            print('======================================================')
            print('Training on the Cross Validation Set: {}'.format(count))
            print('Validation Set Era: ', valid_era)
            print('------------------------------------------------------')

            # Classifier
            clf = self.get_clf(parameters)

            clf.fit(x_train, y_train, sample_weight=w_train)

            # Prediction
            prob_test = self.predict(clf, self.x_test, pred_path=pred_path + 'cv_results/ab_cv_{}_'.format(count))

            # Save Train Probabilities to CSV File
            prob_train = self.get_prob_train(clf, self.x_train,
                                             pred_path=pred_path + 'cv_prob_train/ab_cv_{}_'.format(count))

            # Get Probabilities of Validation Set
            prob_valid = self.predict(clf, x_valid)

            # Print LogLoss
            print('------------------------------------------------------')
            print('Validation Set Era: ', valid_era)
            loss_train, loss_valid, loss_train_w, loss_valid_w = \
                utils.print_loss_proba(clf, x_train, y_train, w_train, x_valid, y_valid, w_valid)

            # Print and Get Accuracies of CV
            acc_train_cv, acc_valid_cv, acc_train_cv_era, acc_valid_cv_era = \
                utils.print_and_get_accuracy(prob_train, y_train, e_train, prob_valid, y_valid, e_valid)

            # Save Losses to File
            utils.save_loss_log(loss_log_path + 'ab_', count, parameters, n_valid, n_cv, valid_era,
                                loss_train, loss_valid, loss_train_w, loss_valid_w, acc_train_cv, acc_valid_cv,
                                acc_train_cv_era, acc_valid_cv_era)

            # Feature Importance
            if show_importance:
                self.get_importance(clf)

            prob_test_total.append(list(prob_test))
            prob_train_total.append(list(prob_train))
            loss_train_total.append(loss_train)
            loss_valid_total.append(loss_valid)
            loss_train_w_total.append(loss_train_w)
            loss_valid_w_total.append(loss_valid_w)

        print('======================================================')
        print('Calculating Final Result...')

        prob_test_mean = np.mean(np.array(prob_test_total), axis=0)
        prob_train_mean = np.mean(np.array(prob_train_total), axis=0)
        loss_train_mean = np.mean(np.array(loss_train_total), axis=0)
        loss_valid_mean = np.mean(np.array(loss_valid_total), axis=0)
        loss_train_w_mean = np.mean(np.array(loss_train_w_total), axis=0)
        loss_valid_w_mean = np.mean(np.array(loss_valid_w_total), axis=0)

        # Print Total Losses
        utils.print_total_loss(loss_train_mean, loss_valid_mean, loss_train_w_mean, loss_valid_w_mean)

        # Print and Get Accuracies of CV of All Train Set
        acc_train, acc_train_era = utils.print_and_get_train_accuracy(prob_train_mean, self.y_train, self.e_train)

        # Save Final Losses to File
        utils.save_final_loss_log(loss_log_path + 'ab_', parameters, n_valid, n_cv, loss_train_mean,
                                  loss_valid_mean, loss_train_w_mean, loss_valid_w_mean, acc_train, acc_train_era)

        # Save Final Result
        utils.save_pred_to_csv(pred_path + 'final_results/ab_', self.id_test, prob_test_mean)
        utils.save_prob_train_to_csv(pred_path + 'final_prob_train/ab_', prob_train_mean, self.y_train)

    def stack_train(self, x_train, y_train, w_train, x_g_train, x_valid, y_valid,
                    w_valid, x_g_valid, x_test, x_g_test, parameters, show_importance=False):

        print('------------------------------------------------------')
        print('Training AdaBoost...')
        print('------------------------------------------------------')

        clf = self.get_clf(parameters)

        clf.fit(x_train, y_train, sample_weight=w_train)

        # Feature Importance
        if show_importance:
            self.get_importance(clf)

        # Print LogLoss
        loss_train, loss_valid, \
        loss_train_w, loss_valid_w = utils.print_loss_proba(clf, x_train, y_train, w_train,
                                                            x_valid, y_valid, w_valid)

        losses = [loss_train, loss_valid, loss_train_w, loss_valid_w]

        # Prediction
        prob_valid = self.predict(clf, x_valid)
        prob_test = self.predict(clf, x_test)

        return prob_valid, prob_test, losses


# GradientBoosting
class GradientBoosting:

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te
        self.importance = np.array([])
        self.indices = np.array([])
        self.std = np.array([])

    def show(self):

        feature_num = self.x_train.shape[1]

        plt.figure(figsize=(20, 10))
        plt.title('Feature Importance in GradientBoosting')
        plt.bar(range(feature_num), self.importance[self.indices],
                color=color[5], yerr=self.std[self.indices], align="center")
        plt.xticks(range(feature_num), self.indices)
        plt.xlim([-1, feature_num])
        plt.show()

    @staticmethod
    def get_clf(parameters):

        clf = GradientBoostingClassifier(**parameters)

        return clf

    def get_importance(self, clf):

        print('------------------------------------------------------')
        print('Feature Importance')

        self.importance = clf.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = self.x_train.shape[1]

        for f in range(feature_num):
            print("%d | feature %d | %f" % (f + 1, self.indices[f], self.importance[self.indices[f]]))

    def predict(self, clf, x_test, pred_path=None):

        print('Predicting...')

        prob_test = np.array(clf.predict_proba(x_test))[:, 1]

        if pred_path is not None:
            utils.save_pred_to_csv(pred_path, self.id_test, prob_test)

        return prob_test

    def get_prob_train(self, clf, x_train, pred_path=None):

        print('Predicting...')

        prob_train = np.array(clf.predict_proba(x_train))[:, 1]

        if pred_path is not None:
            utils.save_prob_train_to_csv(pred_path, prob_train, self.y_train)

        return prob_train

    def train(self, pred_path, loss_log_path, n_valid, n_cv, n_era, cv_seed,
              era_list=None, parameters=None, show_importance=False):

        # Check if directories exit or not
        utils.check_dir_model(pred_path, loss_log_path)

        print('------------------------------------------------------')
        print('Training GradientBoosting...')
        print('------------------------------------------------------')

        count = 0
        prob_test_total = []
        prob_train_total = []
        loss_train_total = []
        loss_valid_total = []
        loss_train_w_total = []
        loss_valid_w_total = []

        for x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, \
            e_valid, valid_era in CrossValidation.era_k_fold_with_weight(x=self.x_train,
                                                                         y=self.y_train,
                                                                         w=self.w_train,
                                                                         e=self.e_train,
                                                                         n_valid=n_valid,
                                                                         n_cv=n_cv,
                                                                         n_era=n_era,
                                                                         seed=cv_seed,
                                                                         era_list=era_list):
            count += 1

            print('======================================================')
            print('Training on the Cross Validation Set: {}'.format(count))
            print('Validation Set Era: ', valid_era)
            print('------------------------------------------------------')

            # Classifier
            clf = self.get_clf(parameters)

            clf.fit(x_train, y_train, sample_weight=w_train)

            # Feature Importance
            if show_importance:
                self.get_importance(clf)

            # Prediction
            prob_test = self.predict(clf, self.x_test, pred_path=pred_path + 'cv_results/gb_cv_{}_'.format(count))

            # Save Train Probabilities to CSV File
            prob_train = self.get_prob_train(clf, self.x_train,
                                             pred_path=pred_path + 'cv_prob_train/gb_cv_{}_'.format(count))

            # Get Probabilities of Validation Set
            prob_valid = self.predict(clf, x_valid)

            # Print LogLoss
            print('------------------------------------------------------')
            print('Validation Set Era: ', valid_era)
            loss_train, loss_valid, loss_train_w, loss_valid_w = \
                utils.print_loss_proba(clf, x_train, y_train, w_train, x_valid, y_valid, w_valid)

            # Print and Get Accuracies of CV
            acc_train_cv, acc_valid_cv, acc_train_cv_era, acc_valid_cv_era = \
                utils.print_and_get_accuracy(prob_train, y_train, e_train, prob_valid, y_valid, e_valid)

            # Save Losses to File
            utils.save_loss_log(loss_log_path + 'gb_', count, parameters, n_valid, n_cv, valid_era,
                                loss_train, loss_valid, loss_train_w, loss_valid_w, acc_train_cv, acc_valid_cv,
                                acc_train_cv_era, acc_valid_cv_era)

            prob_test_total.append(list(prob_test))
            prob_train_total.append(list(prob_train))
            loss_train_total.append(loss_train)
            loss_valid_total.append(loss_valid)
            loss_train_w_total.append(loss_train_w)
            loss_valid_w_total.append(loss_valid_w)

        print('======================================================')
        print('Calculating Final Result...')

        prob_test_mean = np.mean(np.array(prob_test_total), axis=0)
        prob_train_mean = np.mean(np.array(prob_train_total), axis=0)
        loss_train_mean = np.mean(np.array(loss_train_total), axis=0)
        loss_valid_mean = np.mean(np.array(loss_valid_total), axis=0)
        loss_train_w_mean = np.mean(np.array(loss_train_w_total), axis=0)
        loss_valid_w_mean = np.mean(np.array(loss_valid_w_total), axis=0)

        # Print Total Losses
        utils.print_total_loss(loss_train_mean, loss_valid_mean, loss_train_w_mean, loss_valid_w_mean)

        # Print and Get Accuracies of CV of All Train Set
        acc_train, acc_train_era = utils.print_and_get_train_accuracy(prob_train_mean, self.y_train, self.e_train)

        # Save Final Losses to File
        utils.save_final_loss_log(loss_log_path + 'gb_', parameters, n_valid, n_cv, loss_train_mean,
                                  loss_valid_mean, loss_train_w_mean, loss_valid_w_mean, acc_train, acc_train_era)

        # Save Final Result
        utils.save_pred_to_csv(pred_path + 'final_results/gb_', self.id_test, prob_test_mean)
        utils.save_prob_train_to_csv(pred_path + 'final_prob_train/gb_', prob_train_mean, self.y_train)

    def stack_train(self, x_train, y_train, w_train, x_g_train, x_valid, y_valid,
                    w_valid, x_g_valid, x_test, x_g_test, parameters, show_importance=False):

        print('------------------------------------------------------')
        print('Training GradientBoosting...')
        print('------------------------------------------------------')

        clf = self.get_clf(parameters)

        clf.fit(x_train, y_train, sample_weight=w_train)

        # Feature Importance
        if show_importance:
            self.get_importance(clf)

        # Print LogLoss
        loss_train, loss_valid, \
        loss_train_w, loss_valid_w = utils.print_loss_proba(clf, x_train, y_train, w_train,
                                                            x_valid, y_valid, w_valid)

        losses = [loss_train, loss_valid, loss_train_w, loss_valid_w]

        # Prediction
        prob_valid = self.predict(clf, x_valid)
        prob_test = self.predict(clf, x_test)

        return prob_valid, prob_test, losses


# XGBoost
class XGBoost:

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te, num_boost_round):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te
        self.num_boost_round = num_boost_round
        self.importance = np.array([])
        self.indices = np.array([])
        self.std = np.array([])

    def show(self):

        feature_num = self.x_train.shape[1]

        plt.figure(figsize=(20, 10))
        plt.title('Feature Importance in XGBoost')
        plt.bar(range(feature_num), self.importance[self.indices],
                color=color[6], yerr=self.std[self.indices], align="center")
        plt.xticks(range(feature_num), self.indices)
        plt.xlim([-1, feature_num])
        plt.show()

    def get_importance(self, model):

        print('------------------------------------------------------')
        print('Feature Importance')

        self.importance = model.get_fscore()
        sorted_importance = sorted(self.importance.items(), key=lambda d: d[1], reverse=True)

        feature_num = len(self.importance)

        for i in range(feature_num):
            print('Importance:')
            print('{} | feature {} | {}'.format(i + 1, sorted_importance[i][0], sorted_importance[i][1]))

        print('\n')

    @staticmethod
    def print_loss(model, x_t, y_t, w_t, x_v, y_v, w_v):

        prob_train = model.predict(xgb.DMatrix(x_t))
        prob_valid = model.predict(xgb.DMatrix(x_v))

        loss_train = utils.log_loss(prob_train, y_t)
        loss_valid = utils.log_loss(prob_valid, y_v)

        loss_train_w = utils.log_loss_with_weight(prob_train, y_t, w_t)
        loss_valid_w = utils.log_loss_with_weight(prob_valid, y_v, w_v)

        print('Train LogLoss: {:>.8f}\n'.format(loss_train),
              'Validation LogLoss: {:>.8f}\n'.format(loss_valid),
              'Train LogLoss with Weight: {:>.8f}\n'.format(loss_train_w),
              'Validation LogLoss with Weight: {:>.8f}\n'.format(loss_valid_w))

        return loss_train, loss_valid, loss_train_w, loss_valid_w

    def predict(self, model, x_test, pred_path=None):

        print('Predicting...')

        prob_test = model.predict(xgb.DMatrix(x_test))

        if pred_path is not None:
            utils.save_pred_to_csv(pred_path, self.id_test, prob_test)

        return prob_test

    def get_prob_train(self, model, x_train, pred_path=None):

        print('Predicting...')

        prob_train = model.predict(xgb.DMatrix(x_train))

        if pred_path is not None:
            utils.save_prob_train_to_csv(pred_path, prob_train, self.y_train)

        return prob_train

    def train(self, pred_path, loss_log_path, n_valid, n_cv, n_era, cv_seed,
              era_list=None, parameters=None, show_importance=False):

        # Check if directories exit or not
        utils.check_dir_model(pred_path, loss_log_path)

        print('------------------------------------------------------')
        print('Training XGBoost...')
        print('------------------------------------------------------')

        count = 0
        prob_test_total = []
        prob_train_total = []
        loss_train_total = []
        loss_valid_total = []
        loss_train_w_total = []
        loss_valid_w_total = []

        for x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, \
            e_valid, valid_era in CrossValidation.era_k_fold_with_weight(x=self.x_train,
                                                                         y=self.y_train,
                                                                         w=self.w_train,
                                                                         e=self.e_train,
                                                                         n_valid=n_valid,
                                                                         n_cv=n_cv,
                                                                         n_era=n_era,
                                                                         seed=cv_seed,
                                                                         era_list=era_list):

            count += 1

            print('======================================================')
            print('Training on the Cross Validation Set: {}/{}'.format(count, n_cv))
            print('Validation Set Era: ', valid_era)
            print('------------------------------------------------------')

            d_train = xgb.DMatrix(x_train, label=y_train, weight=w_train)
            d_valid = xgb.DMatrix(x_valid, label=y_valid, weight=w_valid)

            # Booster
            eval_list = [(d_train, 'Train'), (d_valid, 'Valid')]
            bst = xgb.train(parameters, d_train, num_boost_round=self.num_boost_round, evals=eval_list)

            # Feature Importance
            if show_importance:
                self.get_importance(bst)

            # Prediction
            prob_test = self.predict(bst, self.x_test, pred_path=pred_path + 'cv_results/xgb_cv_{}_'.format(count))

            # Save Train Probabilities to CSV File
            prob_train = self.get_prob_train(bst, self.x_train,
                                             pred_path=pred_path + 'cv_prob_train/xgb_sk_cv_{}_'.format(count))

            # Get Probabilities of Validation Set
            prob_valid = self.predict(bst, x_valid)

            # Print LogLoss
            print('------------------------------------------------------')
            print('Validation Set Era: ', valid_era)
            loss_train, loss_valid, loss_train_w, loss_valid_w = self.print_loss(bst, x_train, y_train, w_train,
                                                                                 x_valid, y_valid, w_valid)

            # Print and Get Accuracies of CV
            acc_train_cv, acc_valid_cv, acc_train_cv_era, acc_valid_cv_era = \
                utils.print_and_get_accuracy(prob_train, y_train, e_train, prob_valid, y_valid, e_valid)

            # Save Losses to File
            utils.save_loss_log(loss_log_path + 'xgb_', count, parameters, n_valid, n_cv, valid_era,
                                loss_train, loss_valid, loss_train_w, loss_valid_w, acc_train_cv, acc_valid_cv,
                                acc_train_cv_era, acc_valid_cv_era)

            prob_test_total.append(list(prob_test))
            prob_train_total.append(list(prob_train))
            loss_train_total.append(loss_train)
            loss_valid_total.append(loss_valid)
            loss_train_w_total.append(loss_train_w)
            loss_valid_w_total.append(loss_valid_w)

        print('======================================================')
        print('Calculating Final Result...')

        prob_test_mean = np.mean(np.array(prob_test_total), axis=0)
        prob_train_mean = np.mean(np.array(prob_train_total), axis=0)
        loss_train_mean = np.mean(np.array(loss_train_total), axis=0)
        loss_valid_mean = np.mean(np.array(loss_valid_total), axis=0)
        loss_train_w_mean = np.mean(np.array(loss_train_w_total), axis=0)
        loss_valid_w_mean = np.mean(np.array(loss_valid_w_total), axis=0)

        # Print Total Losses
        utils.print_total_loss(loss_train_mean, loss_valid_mean, loss_train_w_mean, loss_valid_w_mean)

        # Print and Get Accuracies of CV of All Train Set
        acc_train, acc_train_era = utils.print_and_get_train_accuracy(prob_train_mean, self.y_train, self.e_train)

        # Save Final Losses to File
        utils.save_final_loss_log(loss_log_path + 'xgb_', parameters, n_valid, n_cv, loss_train_mean,
                                  loss_valid_mean, loss_train_w_mean, loss_valid_w_mean, acc_train, acc_train_era)

        # Save Final Result
        utils.save_pred_to_csv(pred_path + 'final_results/xgb_', self.id_test, prob_test_mean)
        utils.save_prob_train_to_csv(pred_path + 'final_prob_train/xgb_', prob_train_mean, self.y_train)

    def stack_train(self, x_train, y_train, w_train, x_g_train, x_valid, y_valid,
                    w_valid, x_g_valid, x_test, x_g_test, parameters, show_importance=False):

        print('------------------------------------------------------')
        print('Training XGBoost...')
        print('------------------------------------------------------')

        d_train = xgb.DMatrix(x_train, label=y_train, weight=w_train)
        d_valid = xgb.DMatrix(x_valid, label=y_valid, weight=w_valid)

        # Booster
        eval_list = [(d_train, 'Train'), (d_valid, 'Valid')]
        bst = xgb.train(parameters, d_train, num_boost_round=self.num_boost_round, evals=eval_list)

        # Feature Importance
        if show_importance:
            self.get_importance(bst)

        # Print LogLoss
        # Print LogLoss
        print('------------------------------------------------------')
        loss_train, loss_valid, loss_train_w, loss_valid_w = self.print_loss(bst, x_train, y_train, w_train,
                                                                             x_valid, y_valid, w_valid)

        losses = [loss_train, loss_valid, loss_train_w, loss_valid_w]

        # Prediction
        prob_valid = self.predict(bst, x_valid)
        prob_test = self.predict(bst, x_test)

        return prob_valid, prob_test, losses


# XGBoost sklearn
class SKLearnXGBoost:

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te
        self.importance = np.array([])
        self.indices = np.array([])
        self.std = np.array([])

    def show(self):

        feature_num = self.x_train.shape[1]

        plt.figure(figsize=(20, 10))
        plt.title('Feature Importance in XGBoost')
        plt.bar(range(feature_num), self.importance[self.indices],
                color=color[6], yerr=self.std[self.indices], align="center")
        plt.xticks(range(feature_num), self.indices)
        plt.xlim([-1, feature_num])
        plt.show()

    @staticmethod
    def get_clf(parameters=None):

        print('Initialize Model...')

        clf = XGBClassifier(**parameters)

        return clf

    def get_importance(self, clf):

        print('------------------------------------------------------')
        print('Feature Importance')

        self.importance = clf.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = len(self.importance)

        for f in range(feature_num):
            print("%d | feature %d | %d" % (f + 1, self.indices[f], self.importance[self.indices[f]]))

    def predict(self, clf, x_test, pred_path=None):

        print('Predicting...')

        prob_test = np.array(clf.predict_proba(x_test))[:, 1]

        if pred_path is not None:
            utils.save_pred_to_csv(pred_path, self.id_test, prob_test)

        return prob_test

    def get_prob_train(self, clf, x_train, pred_path=None):

        print('Predicting...')

        prob_train = np.array(clf.predict_proba(x_train))[:, 1]

        if pred_path is not None:
            utils.save_prob_train_to_csv(pred_path, prob_train, self.y_train)

        return prob_train

    # Using sk-learn API
    def train(self, pred_path, loss_log_path, n_valid, n_cv, n_era, cv_seed,
              era_list=None, parameters=None, show_importance=False):

        # Check if directories exit or not
        utils.check_dir_model(pred_path, loss_log_path)

        print('------------------------------------------------------')
        print('Training XGBoost...')
        print('------------------------------------------------------')

        count = 0
        prob_test_total = []
        prob_train_total = []
        loss_train_total = []
        loss_valid_total = []
        loss_train_w_total = []
        loss_valid_w_total = []

        for x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, \
            e_valid, valid_era in CrossValidation.era_k_fold_with_weight(x=self.x_train,
                                                                         y=self.y_train,
                                                                         w=self.w_train,
                                                                         e=self.e_train,
                                                                         n_valid=n_valid,
                                                                         n_cv=n_cv,
                                                                         n_era=n_era,
                                                                         seed=cv_seed,
                                                                         era_list=era_list):
            count += 1

            print('======================================================')
            print('Training on the Cross Validation Set: {}/{}'.format(count, n_cv))
            print('Validation Set Era: ', valid_era)
            print('------------------------------------------------------')

            clf = self.get_clf(parameters)

            clf.fit(x_train, y_train, sample_weight=w_train,
                    eval_set=[(x_train, y_train), (x_valid, y_valid)],
                    early_stopping_rounds=100, eval_metric='logloss', verbose=True)

            # Feature Importance
            if show_importance:
                self.get_importance(clf)

            # Prediction
            prob_test = self.predict(clf, self.x_test, pred_path=pred_path + 'cv_results/xgb_sk_cv_{}_'.format(count))

            # Save Train Probabilities to CSV File
            prob_train = self.get_prob_train(clf, self.x_train,
                                             pred_path=pred_path + 'cv_prob_train/xgb_sk_cv_{}_'.format(count))

            # Get Probabilities of Validation Set
            prob_valid = self.predict(clf, x_valid)

            # Print LogLoss
            print('------------------------------------------------------')
            print('Validation Set Era: ', valid_era)
            loss_train, loss_valid, loss_train_w, loss_valid_w = utils.print_loss_proba(clf, x_train, y_train, w_train,
                                                                                        x_valid, y_valid, w_valid)

            # Print and Get Accuracies of CV
            acc_train_cv, acc_valid_cv, acc_train_cv_era, acc_valid_cv_era = \
                utils.print_and_get_accuracy(prob_train, y_train, e_train, prob_valid, y_valid, e_valid)

            # Save Losses to File
            utils.save_loss_log(loss_log_path + 'xgb_sk_', count, parameters, n_valid, n_cv, valid_era,
                                loss_train, loss_valid, loss_train_w, loss_valid_w, acc_train_cv, acc_valid_cv,
                                acc_train_cv_era, acc_valid_cv_era)

            prob_test_total.append(list(prob_test))
            prob_train_total.append(list(prob_train))
            loss_train_total.append(loss_train)
            loss_valid_total.append(loss_valid)
            loss_train_w_total.append(loss_train_w)
            loss_valid_w_total.append(loss_valid_w)

        print('======================================================')
        print('Calculating Final Result...')

        prob_test_mean = np.mean(np.array(prob_test_total), axis=0)
        prob_train_mean = np.mean(np.array(prob_train_total), axis=0)
        loss_train_mean = np.mean(np.array(loss_train_total), axis=0)
        loss_valid_mean = np.mean(np.array(loss_valid_total), axis=0)
        loss_train_w_mean = np.mean(np.array(loss_train_w_total), axis=0)
        loss_valid_w_mean = np.mean(np.array(loss_valid_w_total), axis=0)

        # Print Total Losses
        utils.print_total_loss(loss_train_mean, loss_valid_mean, loss_train_w_mean, loss_valid_w_mean)

        # Print and Get Accuracies of CV of All Train Set
        acc_train, acc_train_era = utils.print_and_get_train_accuracy(prob_train_mean, self.y_train, self.e_train)

        # Save Final Losses to File
        utils.save_final_loss_log(loss_log_path + 'xgb_sk_', parameters, n_valid, n_cv, loss_train_mean,
                                  loss_valid_mean, loss_train_w_mean, loss_valid_w_mean, acc_train, acc_train_era)

        # Save Final Result
        utils.save_pred_to_csv(pred_path + 'final_results/xgb_sk_', self.id_test, prob_test_mean)
        utils.save_prob_train_to_csv(pred_path + 'final_prob_train/xgb_sk_', prob_train_mean, self.y_train)

    def stack_train(self, x_train, y_train, w_train, x_g_train, x_valid, y_valid,
                    w_valid, x_g_valid, x_test, x_g_test, parameters, show_importance=False):

        print('------------------------------------------------------')
        print('Training XGBoost...')
        print('------------------------------------------------------')

        clf = self.get_clf(parameters)

        clf.fit(x_train, y_train, sample_weight=w_train,
                eval_set=[(x_train, y_train), (x_valid, y_valid)],
                early_stopping_rounds=10, eval_metric='logloss', verbose=True)

        # Feature Importance
        if show_importance:
            self.get_importance(clf)

        # Print LogLoss
        loss_train, loss_valid, loss_train_w, loss_valid_w = utils.print_loss_proba(clf, x_train, y_train, w_train,
                                                                                    x_valid, y_valid, w_valid)

        losses = [loss_train, loss_valid, loss_train_w, loss_valid_w]

        # Prediction
        prob_valid = self.predict(clf, x_valid)
        prob_test = self.predict(clf, x_test)

        return prob_valid, prob_test, losses


# LightGBM
class LightGBM:

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te, num_boost_round):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te
        self.num_boost_round = num_boost_round
        self.importance = np.array([])
        self.indices = np.array([])
        self.std = np.array([])

    def show(self):

        feature_num = self.x_train.shape[1]

        plt.figure(figsize=(20, 10))
        plt.title('Feature Importance in XGBoost')
        plt.bar(range(feature_num), self.importance[self.indices],
                color=color[6], yerr=self.std[self.indices], align="center")
        plt.xticks(range(feature_num), self.indices)
        plt.xlim([-1, feature_num])
        plt.show()

    @staticmethod
    def logloss_obj(y, pred):

        grad = (pred - y) / ((1 - pred) * pred)
        hess = (pred * pred - 2 * pred * y + y) / ((1 - pred) * (1 - pred) * pred * pred)

        return grad, hess

    def get_importance(self, model):

        print('------------------------------------------------------')
        print('Feature Importance')

        self.importance = model.feature_importance()
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = len(self.importance)

        for f in range(feature_num):
            print("%d | feature %d | %d" % (f + 1, self.indices[f], self.importance[self.indices[f]]))

        print('\n')

    def predict(self, model, x_test, pred_path=None):

        print('Predicting...')

        prob_test = model.predict(x_test)

        if pred_path is not None:
            utils.save_pred_to_csv(pred_path, self.id_test, prob_test)

        return prob_test

    def get_prob_train(self, model, x_train, pred_path=None):

        print('Predicting...')

        prob_train = model.predict(x_train)

        if pred_path is not None:
            utils.save_prob_train_to_csv(pred_path, prob_train, self.y_train)

        return prob_train

    def train(self, pred_path, loss_log_path, n_valid, n_cv, n_era, cv_seed, era_list=None,
              parameters=None, return_prob_test=False, show_importance=False):

        # Check if directories exit or not
        utils.check_dir_model(pred_path, loss_log_path)

        print('------------------------------------------------------')
        print('Training LightGBM...')
        print('------------------------------------------------------')

        count = 0
        prob_test_total = []
        prob_train_total = []
        loss_train_total = []
        loss_valid_total = []
        loss_train_w_total = []
        loss_valid_w_total = []

        # Cross Validation
        for x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, \
            e_valid, valid_era in CrossValidation.era_k_fold_with_weight(x=self.x_train,
                                                                         y=self.y_train,
                                                                         w=self.w_train,
                                                                         e=self.e_train,
                                                                         n_valid=n_valid,
                                                                         n_cv=n_cv,
                                                                         n_era=n_era,
                                                                         seed=cv_seed,
                                                                         era_list=era_list):

            count += 1

            print('======================================================')
            print('Training on the Cross Validation Set: {}/{}'.format(count, n_cv))
            print('Validation Set Era: ', valid_era)
            print('------------------------------------------------------')

            # Use Category
            idx_category = [x_train.shape[1] - 1]
            d_train = lgb.Dataset(x_train, label=y_train, weight=w_train, categorical_feature=idx_category)
            d_valid = lgb.Dataset(x_valid, label=y_valid, weight=w_valid, categorical_feature=idx_category)

            # Booster
            bst = lgb.binary_train(parameters, d_train, num_boost_round=self.num_boost_round,
                                   valid_sets=[d_valid, d_train], valid_names=['eval', 'train'])

            # Feature Importance
            if show_importance:
                self.get_importance(bst)

            # Prediction
            prob_test = self.predict(bst, self.x_test, pred_path=pred_path + 'cv_results/lgb_cv_{}_'.format(count))

            # Save Train Probabilities to CSV File
            prob_train = self.get_prob_train(bst, self.x_train,
                                             pred_path=pred_path + 'cv_prob_train/lgb_cv_{}_'.format(count))

            # Get Probabilities of Validation Set
            prob_valid = self.predict(bst, x_valid)

            # Print LogLoss
            print('------------------------------------------------------')
            print('Validation Set Era: ', valid_era)
            loss_train, loss_valid, loss_train_w, loss_valid_w = utils.print_loss_lgb(bst, x_train, y_train, w_train,
                                                                                      x_valid, y_valid, w_valid)

            # Print and Get Accuracies of CV
            acc_train_cv, acc_valid_cv, acc_train_cv_era, acc_valid_cv_era = \
                utils.print_and_get_accuracy(prob_train, y_train, e_train, prob_valid, y_valid, e_valid)

            # Save Losses to File
            utils.save_loss_log(loss_log_path + 'lgb_', count, parameters, n_valid, n_cv, valid_era,
                                loss_train, loss_valid, loss_train_w, loss_valid_w, acc_train_cv, acc_valid_cv,
                                acc_train_cv_era, acc_valid_cv_era)

            prob_test_total.append(list(prob_test))
            prob_train_total.append(list(prob_train))
            loss_train_total.append(loss_train)
            loss_valid_total.append(loss_valid)
            loss_train_w_total.append(loss_train_w)
            loss_valid_w_total.append(loss_valid_w)

        print('======================================================')
        print('Calculating Final Result...')

        prob_test_mean = np.mean(np.array(prob_test_total), axis=0)
        prob_train_mean = np.mean(np.array(prob_train_total), axis=0)
        loss_train_mean = np.mean(np.array(loss_train_total), axis=0)
        loss_valid_mean = np.mean(np.array(loss_valid_total), axis=0)
        loss_train_w_mean = np.mean(np.array(loss_train_w_total), axis=0)
        loss_valid_w_mean = np.mean(np.array(loss_valid_w_total), axis=0)

        # Print Total Losses
        utils.print_total_loss(loss_train_mean, loss_valid_mean, loss_train_w_mean, loss_valid_w_mean)

        # Print and Get Accuracies of CV of All Train Set
        acc_train, acc_train_era = utils.print_and_get_train_accuracy(prob_train_mean, self.y_train, self.e_train)

        # Save Final Losses to File
        utils.save_final_loss_log(loss_log_path + 'lgb_', parameters, n_valid, n_cv, loss_train_mean,
                                  loss_valid_mean, loss_train_w_mean, loss_valid_w_mean, acc_train, acc_train_era)

        # Save Final Result
        utils.save_pred_to_csv(pred_path + 'final_results/lgb_', self.id_test, prob_test_mean)
        utils.save_prob_train_to_csv(pred_path + 'final_prob_train/lgb_', prob_train_mean, self.y_train)

        # Return Probabilities of Test Set
        if return_prob_test:
            return prob_test_mean

    def stack_train(self, x_train, y_train, w_train, x_g_train, x_valid, y_valid,
                    w_valid, x_g_valid, x_test, x_g_test, parameters, show_importance=False):

        print('------------------------------------------------------')
        print('Training LightGBM...')
        print('------------------------------------------------------')

        idx_category = [x_g_test.shape[1] - 1]
        print('Index of categorical feature: {}'.format(idx_category))

        # Use Category
        d_train = lgb.Dataset(x_g_train, label=y_train, weight=w_train, categorical_feature=idx_category)
        d_valid = lgb.Dataset(x_g_valid, label=y_valid, weight=w_valid, categorical_feature=idx_category)

        # Booster
        bst = lgb.binary_train(parameters, d_train, num_boost_round=self.num_boost_round,
                               valid_sets=[d_valid, d_train], valid_names=['eval', 'train'])

        # Feature Importance
        if show_importance:
            self.get_importance(bst)

        # Print LogLoss
        print('------------------------------------------------------')
        loss_train, loss_valid, loss_train_w, loss_valid_w = utils.print_loss_lgb(bst, x_train, y_train, w_train,
                                                                                  x_valid, y_valid, w_valid)

        losses = [loss_train, loss_valid, loss_train_w, loss_valid_w]

        # Prediction
        prob_valid = self.predict(bst, x_g_valid)
        prob_test = self.predict(bst, x_g_test)

        return prob_valid, prob_test, losses

    def prejudge_train(self, pred_path, n_splits, n_cv, cv_seed,
                       use_weight=True, parameters=None, show_importance=False):

        # Check if directories exit or not
        utils.check_dir_model(pred_path)

        count = 0
        prob_test_total = []
        prob_train_total = []
        loss_train_total = []
        loss_valid_total = []
        loss_train_w_total = []
        loss_valid_w_total = []

        # Cross Validation
        for x_train, y_train, w_train, x_valid, \
            y_valid, w_valid in CrossValidation.sk_k_fold_with_weight(x=self.x_train,
                                                                      y=self.y_train,
                                                                      w=self.w_train,
                                                                      n_splits=n_splits,
                                                                      n_cv=n_cv,
                                                                      seed=cv_seed):
            count += 1

            print('======================================================')
            print('Training on the Cross Validation Set: {}/{}'.format(count, n_cv))
            print('------------------------------------------------------')

            # Use Category
            idx_category = [x_train.shape[1] - 1]
            if use_weight:
                d_train = lgb.Dataset(x_train, label=y_train, weight=w_train, categorical_feature=idx_category)
                d_valid = lgb.Dataset(x_valid, label=y_valid, weight=w_valid, categorical_feature=idx_category)
            else:
                d_train = lgb.Dataset(x_train, label=y_train, categorical_feature=idx_category)
                d_valid = lgb.Dataset(x_valid, label=y_valid, categorical_feature=idx_category)

            # Booster
            bst = lgb.binary_train(parameters, d_train, num_boost_round=self.num_boost_round,
                                   valid_sets=[d_valid, d_train], valid_names=['eval', 'train'])

            # Feature Importance
            if show_importance:
                self.get_importance(bst)

            # Prediction
            prob_test = self.predict(bst, self.x_test, pred_path=pred_path + 'cv_results/lgb_cv_{}_'.format(count))

            # Save Train Probabilities to CSV File
            prob_train = self.get_prob_train(bst, self.x_train,
                                             pred_path=pred_path + 'cv_prob_train/lgb_cv_{}_'.format(count))

            # Print LogLoss
            print('------------------------------------------------------')
            loss_train, loss_valid, loss_train_w, loss_valid_w = utils.print_loss_lgb(bst, x_train, y_train, w_train,
                                                                                      x_valid, y_valid, w_valid)

            prob_test_total.append(list(prob_test))
            prob_train_total.append(list(prob_train))
            loss_train_total.append(loss_train)
            loss_valid_total.append(loss_valid)
            loss_train_w_total.append(loss_train_w)
            loss_valid_w_total.append(loss_valid_w)

        print('======================================================')
        print('Calculating Final Result...')

        prob_test_mean = np.mean(np.array(prob_test_total), axis=0)
        prob_train_mean = np.mean(np.array(prob_train_total), axis=0)
        loss_train_mean = np.mean(np.array(loss_train_total), axis=0)
        loss_valid_mean = np.mean(np.array(loss_valid_total), axis=0)
        loss_train_w_mean = np.mean(np.array(loss_train_w_total), axis=0)
        loss_valid_w_mean = np.mean(np.array(loss_valid_w_total), axis=0)

        # Print Total Losses
        utils.print_total_loss(loss_train_mean, loss_valid_mean, loss_train_w_mean, loss_valid_w_mean)

        # Print and Get Accuracies of CV of All Train Set
        _, _ = utils.print_and_get_train_accuracy(prob_train_mean, self.y_train, self.e_train)

        # Save Final Result
        utils.save_pred_to_csv(pred_path + 'final_results/lgb_', self.id_test, prob_test_mean)

        return prob_test_mean


# LightGBM - sklearn
class SKLearnLightGBM:

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te
        self.importance = np.array([])
        self.indices = np.array([])
        self.std = np.array([])

    def show(self):

        feature_num = self.x_train.shape[1]

        plt.figure(figsize=(20, 10))
        plt.title('Feature Importance in XGBoost')
        plt.bar(range(feature_num), self.importance[self.indices],
                color=color[6], yerr=self.std[self.indices], align="center")
        plt.xticks(range(feature_num), self.indices)
        plt.xlim([-1, feature_num])
        plt.show()

    @staticmethod
    def get_clf(parameters=None):

        print('Initialize Model...')

        clf = LGBMClassifier(**parameters)

        return clf

    def get_importance(self, clf):

        print('------------------------------------------------------')
        print('Feature Importance')

        self.importance = clf.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = len(self.importance)

        for f in range(feature_num):
            print("%d | feature %d | %d" % (f + 1, self.indices[f], self.importance[self.indices[f]]))

    def predict(self, clf, x_g_test, pred_path=None):

        print('Predicting...')

        prob_test = np.array(clf.predict_proba(x_g_test))[:, 1]

        if pred_path is not None:
            utils.save_pred_to_csv(pred_path, self.id_test, prob_test)

        return prob_test

    def get_prob_train(self, clf, x_train, pred_path=None):

        print('Predicting...')

        prob_train = np.array(clf.predict_proba(x_train))[:, 1]

        if pred_path is not None:
            utils.save_prob_train_to_csv(pred_path, prob_train, self.y_train)

        return prob_train

    # Using sk-learn API
    def train(self, pred_path, loss_log_path, n_valid, n_cv, n_era, cv_seed,
              era_list=None, parameters=None, show_importance=False):

        # Check if directories exit or not
        utils.check_dir_model(pred_path, loss_log_path)

        print('------------------------------------------------------')
        print('Training LightGBM...')
        print('------------------------------------------------------')

        count = 0
        prob_test_total = []
        prob_train_total = []
        loss_train_total = []
        loss_valid_total = []
        loss_train_w_total = []
        loss_valid_w_total = []

        # Use Category
        for x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, \
            e_valid, valid_era in CrossValidation.era_k_fold_with_weight(x=self.x_train,
                                                                         y=self.y_train,
                                                                         w=self.w_train,
                                                                         e=self.e_train,
                                                                         n_valid=n_valid,
                                                                         n_cv=n_cv,
                                                                         n_era=n_era,
                                                                         seed=cv_seed,
                                                                         era_list=era_list):
            count += 1

            print('======================================================')
            print('Training on the Cross Validation Set: {}/{}'.format(count, n_cv))
            print('Validation Set Era: ', valid_era)
            print('------------------------------------------------------')

            clf = self.get_clf(parameters)

            idx_category = [x_train.shape[1] - 1]
            print('Index of categorical feature: {}'.format(idx_category))

            clf.fit(x_train, y_train, sample_weight=w_train,
                    categorical_feature=idx_category,
                    eval_set=[(x_train, y_train), (x_valid, y_valid)],
                    eval_names=['train', 'eval'],
                    early_stopping_rounds=100,
                    eval_sample_weight=[w_train, w_valid],
                    eval_metric='logloss', verbose=True)

            # Feature Importance
            if show_importance:
                self.get_importance(clf)

            # Prediction
            prob_test = self.predict(clf, self.x_test,
                                     pred_path=pred_path + 'cv_results/lgb_sk_cv_{}_'.format(count))

            # Save Train Probabilities to CSV File
            prob_train = self.get_prob_train(clf, self.x_train,
                                             pred_path=pred_path + 'cv_prob_train/lgb_sk_cv_{}_'.format(count))

            # Get Probabilities of Validation Set
            prob_valid = self.predict(clf, x_valid)

            # Print LogLoss
            print('------------------------------------------------------')
            print('Validation Set Era: ', valid_era)
            loss_train, loss_valid, loss_train_w, loss_valid_w = utils.print_loss_proba(clf, x_train, y_train, w_train,
                                                                                        x_valid, y_valid, w_valid)

            # Print and Get Accuracies of CV
            acc_train_cv, acc_valid_cv, acc_train_cv_era, acc_valid_cv_era = \
                utils.print_and_get_accuracy(prob_train, y_train, e_train, prob_valid, y_valid, e_valid)

            # Save Losses to File
            utils.save_loss_log(loss_log_path + 'lgb_sk_', count, parameters, n_valid, n_cv, valid_era,
                                loss_train, loss_valid, loss_train_w, loss_valid_w, acc_train_cv, acc_valid_cv,
                                acc_train_cv_era, acc_valid_cv_era)

            prob_test_total.append(list(prob_test))
            prob_train_total.append(list(prob_train))
            loss_train_total.append(loss_train)
            loss_valid_total.append(loss_valid)
            loss_train_w_total.append(loss_train_w)
            loss_valid_w_total.append(loss_valid_w)

        print('======================================================')
        print('Calculating Final Result...')

        prob_test_mean = np.mean(np.array(prob_test_total), axis=0)
        prob_train_mean = np.mean(np.array(prob_train_total), axis=0)
        loss_train_mean = np.mean(np.array(loss_train_total), axis=0)
        loss_valid_mean = np.mean(np.array(loss_valid_total), axis=0)
        loss_train_w_mean = np.mean(np.array(loss_train_w_total), axis=0)
        loss_valid_w_mean = np.mean(np.array(loss_valid_w_total), axis=0)

        # Print Total Losses
        utils.print_total_loss(loss_train_mean, loss_valid_mean, loss_train_w_mean, loss_valid_w_mean)

        # Print and Get Accuracies of CV of All Train Set
        acc_train, acc_train_era = utils.print_and_get_train_accuracy(prob_train_mean, self.y_train, self.e_train)

        # Save Final Losses to File
        utils.save_final_loss_log(loss_log_path + 'lgb_sk_', parameters, n_valid, n_cv, loss_train_mean,
                                  loss_valid_mean, loss_train_w_mean, loss_valid_w_mean, acc_train, acc_train_era)

        # Save Final Result
        utils.save_pred_to_csv(pred_path + 'final_results/lgb_sk_', self.id_test, prob_test_mean)
        utils.save_prob_train_to_csv(pred_path + 'final_prob_train/lgb_sk_', prob_train_mean, self.y_train)

    def stack_train(self, x_train, y_train, w_train, x_g_train, x_valid, y_valid,
                    w_valid, x_g_valid, x_test, x_g_test, parameters, show_importance=False):

        print('------------------------------------------------------')
        print('Training LightGBM...')
        print('------------------------------------------------------')

        idx_category = [x_g_test.shape[1] - 1]
        print('Index of categorical feature: {}'.format(idx_category))

        clf = self.get_clf(parameters)

        clf.fit(x_g_train, y_train, sample_weight=w_train,
                categorical_feature=idx_category,
                eval_set=[(x_g_train, y_train), (x_g_valid, y_valid)],
                eval_names=['train', 'eval'],
                early_stopping_rounds=100,
                eval_sample_weight=[w_train, w_valid],
                eval_metric='logloss', verbose=True)

        # Feature Importance
        if show_importance:
            self.get_importance(clf)

        # Print LogLoss
        loss_train, loss_valid, loss_train_w, loss_valid_w = utils.print_loss_proba(clf, x_g_train, y_train, w_train,
                                                                                    x_g_valid, y_valid, w_valid)

        losses = [loss_train, loss_valid, loss_train_w, loss_valid_w]

        # Prediction
        prob_valid = self.predict(clf, x_g_valid)
        prob_test = self.predict(clf, x_g_test)

        return prob_valid, prob_test, losses


# CatBoost
class CatBoost:

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te):

        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te
        self.importance = np.array([])
        self.indices = np.array([])

    @staticmethod
    def get_clf(parameters=None):

        print('Initialize Model...')

        clf = CatBoostClassifier(**parameters)

        return clf

    def get_importance(self, clf):

        print('------------------------------------------------------')
        print('Feature Importance')

        self.importance = clf.feature_importances_
        self.indices = np.argsort(self.importance)[::-1]

        feature_num = len(self.importance)

        for f in range(feature_num):
            print("%d | feature %d | %d" % (f + 1, self.indices[f], self.importance[self.indices[f]]))

    def predict(self, clf, x_g_test, pred_path=None):

        print('Predicting...')

        prob_test = np.array(clf.predict_proba(x_g_test))[:, 1]

        if pred_path is not None:
            utils.save_pred_to_csv(pred_path, self.id_test, prob_test)

        return prob_test

    def get_prob_train(self, clf, x_train, pred_path=None):

        print('Predicting...')

        prob_train = np.array(clf.predict_proba(x_train))[:, 1]

        if pred_path is not None:
            utils.save_prob_train_to_csv(pred_path, prob_train, self.y_train)

        return prob_train

    # Using sk-learn API
    def train(self, pred_path, loss_log_path, n_valid, n_cv, n_era, cv_seed,
              era_list=None, parameters=None, show_importance=False):

        # Check if directories exit or not
        utils.check_dir_model(pred_path, loss_log_path)

        print('------------------------------------------------------')
        print('Training CatBoost...')
        print('------------------------------------------------------')

        count = 0
        prob_test_total = []
        prob_train_total = []
        loss_train_total = []
        loss_valid_total = []
        loss_train_w_total = []
        loss_valid_w_total = []

        # Use Category
        for x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, \
            e_valid, valid_era in CrossValidation.era_k_fold_with_weight(x=self.x_train,
                                                                         y=self.y_train,
                                                                         w=self.w_train,
                                                                         e=self.e_train,
                                                                         n_valid=n_valid,
                                                                         n_cv=n_cv,
                                                                         n_era=n_era,
                                                                         seed=cv_seed,
                                                                         era_list=era_list):
            count += 1

            print('======================================================')
            print('Training on the Cross Validation Set: {}/{}'.format(count, n_cv))
            print('Validation Set Era: ', valid_era)
            print('------------------------------------------------------')

            clf = self.get_clf(parameters)

            idx_category = [x_train.shape[1] - 1]
            print('Index of categorical feature: {}'.format(idx_category))

            w_train = [0.001 if w == 0 else w for w in w_train]

            clf.fit(X=x_train, y=y_train, cat_features=idx_category, sample_weight=w_train,
                    baseline=None, use_best_model=None, eval_set=(x_valid, y_valid),
                    verbose=True, plot=False)

            # Feature Importance
            if show_importance:
                self.get_importance(clf)

            # Prediction
            prob_test = self.predict(clf, self.x_test,
                                     pred_path=pred_path + 'cv_results/cb_cv_{}_'.format(count))

            # Save Train Probabilities to CSV File
            prob_train = self.get_prob_train(clf, self.x_train,
                                             pred_path=pred_path + 'cv_prob_train/cb_cv_{}_'.format(count))

            # Get Probabilities of Validation Set
            prob_valid = self.predict(clf, x_valid)

            # Print LogLoss
            print('------------------------------------------------------')
            print('Validation Set Era: ', valid_era)
            loss_train, loss_valid, loss_train_w, loss_valid_w = utils.print_loss_proba(clf, x_train, y_train, w_train,
                                                                                        x_valid, y_valid, w_valid)

            # Print and Get Accuracies of CV
            acc_train_cv, acc_valid_cv, acc_train_cv_era, acc_valid_cv_era = \
                utils.print_and_get_accuracy(prob_train, y_train, e_train, prob_valid, y_valid, e_valid)

            # Save Losses to File
            utils.save_loss_log(loss_log_path + 'cb_', count, parameters, n_valid, n_cv, valid_era,
                                loss_train, loss_valid, loss_train_w, loss_valid_w, acc_train_cv, acc_valid_cv,
                                acc_train_cv_era, acc_valid_cv_era)

            prob_test_total.append(list(prob_test))
            prob_train_total.append(list(prob_train))
            loss_train_total.append(loss_train)
            loss_valid_total.append(loss_valid)
            loss_train_w_total.append(loss_train_w)
            loss_valid_w_total.append(loss_valid_w)

        print('======================================================')
        print('Calculating Final Result...')

        prob_test_mean = np.mean(np.array(prob_test_total), axis=0)
        prob_train_mean = np.mean(np.array(prob_train_total), axis=0)
        loss_train_mean = np.mean(np.array(loss_train_total), axis=0)
        loss_valid_mean = np.mean(np.array(loss_valid_total), axis=0)
        loss_train_w_mean = np.mean(np.array(loss_train_w_total), axis=0)
        loss_valid_w_mean = np.mean(np.array(loss_valid_w_total), axis=0)

        # Print Total Losses
        utils.print_total_loss(loss_train_mean, loss_valid_mean, loss_train_w_mean, loss_valid_w_mean)

        # Print and Get Accuracies of CV of All Train Set
        acc_train, acc_train_era = utils.print_and_get_train_accuracy(prob_train_mean, self.y_train, self.e_train)

        # Save Final Losses to File
        utils.save_final_loss_log(loss_log_path + 'cb_', parameters, n_valid, n_cv, loss_train_mean,
                                  loss_valid_mean, loss_train_w_mean, loss_valid_w_mean, acc_train, acc_train_era)

        # Save Final Result
        utils.save_pred_to_csv(pred_path + 'final_results/cb_', self.id_test, prob_test_mean)
        utils.save_prob_train_to_csv(pred_path + 'final_prob_train/cb_', prob_train_mean, self.y_train)


# Deep Neural Networks
class DeepNeuralNetworks:

    def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te, parameters):

        # Inputs
        self.x_train = x_tr
        self.y_train = y_tr
        self.w_train = w_tr
        self.e_train = e_tr
        self.x_test = x_te
        self.id_test = id_te

        # Hyperparameters
        self.parameters = parameters
        self.version = parameters['version']
        self.epochs = parameters['epochs']
        self.unit_number = parameters['unit_number']
        self.learning_rate = parameters['learning_rate']
        self.keep_probability = parameters['keep_probability']
        self.batch_size = parameters['batch_size']
        self.dnn_seed = parameters['seed']
        self.display_step = parameters['display_step']
        self.save_path = parameters['save_path']
        self.log_path = parameters['log_path']

    # Input Tensors
    @staticmethod
    def input_tensor(n_feature):

        inputs_ = tf.placeholder(tf.float64, [None, n_feature], name='inputs')
        labels_ = tf.placeholder(tf.float64, None, name='labels')
        loss_weights_ = tf.placeholder(tf.float64, None, name='loss_weights')
        learning_rate_ = tf.placeholder(tf.float64, name='learning_rate')
        keep_prob_ = tf.placeholder(tf.float64, name='keep_prob')
        is_train_ = tf.placeholder(tf.bool, name='is_train')

        return inputs_, labels_, loss_weights_, learning_rate_, keep_prob_, is_train_

    # Full Connected Layer
    def fc_layer(self, x_tensor, layer_name, num_outputs, keep_prob, training):

        with tf.name_scope(layer_name):
            # x_shape = x_tensor.get_shape().as_list()
            #
            # weights = tf.Variable(tf.truncated_normal([x_shape[1], num_outputs], dtype=tf.float64,
            #                                           stddev=2.0 / np.sqrt(x_shape[1])))
            #
            # biases = tf.Variable(tf.zeros([num_outputs], dtype=tf.float64))
            #
            # fc_layer = tf.add(tf.matmul(x_tensor, weights), biases)
            #
            # # Batch Normalization
            # fc_layer = tf.layers.batch_normalization(fc_layer, training=training)
            #
            # # Activate function
            # fc = tf.sigmoid(fc_layer)

            fc = tf.contrib.layers.fully_connected(x_tensor,
                                                   num_outputs,
                                                   activation_fn=tf.nn.sigmoid,
                                                   # weights_initializer=tf.truncated_normal_initializer(
                                                   # stddev=2.0 / math.sqrt(x_shape[1])),
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(
                                                       dtype=tf.float64,
                                                       seed=self.dnn_seed),
                                                   # normalizer_fn=tf.contrib.layers.batch_norm,
                                                   biases_initializer=tf.zeros_initializer(dtype=tf.float64)
                                                   )

            tf.summary.histogram('fc_layer', fc)

            fc = tf.nn.dropout(fc, keep_prob)

        return fc

    # Output Layer
    def output_layer(self, x_tensor, layer_name, num_outputs):

        with tf.name_scope(layer_name):
            #  x_shape = x_tensor.get_shape().as_list()
            #
            #  weights = tf.Variable(tf.truncated_normal([x_shape[1], num_outputs], stddev=2.0 / math.sqrt(x_shape[1])))
            #
            #  biases = tf.Variable(tf.zeros([num_outputs]))
            #
            #  with tf.name_scope('Wx_plus_b'):
            #      output_layer = tf.add(tf.matmul(x_tensor, weights), biases)
            #  tf.summary.histogram('output', output_layer)

            out = tf.contrib.layers.fully_connected(x_tensor,
                                                    num_outputs,
                                                    activation_fn=None,
                                                    weights_initializer=tf.contrib.layers.xavier_initializer(
                                                        dtype=tf.float64,
                                                        seed=self.dnn_seed),
                                                    biases_initializer=tf.zeros_initializer())

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
            ones = tf.ones_like(y, dtype=tf.float64)
            loss = - tf.reduce_sum(w * (y * tf.log(prob) + (ones - y) * tf.log(ones - prob)))
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
    def get_prob(self, sess, logits, x, batch_num, inputs, keep_prob, is_train):

        logits_pred = np.array([])

        for x_batch in self.get_batches_for_predict(x, batch_num):
            logits_pred_batch = sess.run(logits, {inputs: x_batch, keep_prob: 1.0, is_train: False})
            logits_pred_batch = logits_pred_batch.flatten()
            logits_pred = np.concatenate((logits_pred, logits_pred_batch))

        prob = 1.0 / (1.0 + np.exp(-logits_pred))

        return prob

    # Training
    def train(self, pred_path, loss_log_path, n_valid, n_cv, n_era, cv_seed,
              era_list=None, parameters=None, show_importance=False):

        # Check if directories exit or not
        utils.check_dir_model(pred_path, loss_log_path)

        # Build Network
        tf.reset_default_graph()
        train_graph = tf.Graph()

        with train_graph.as_default():

            # Inputs
            feature_num = self.x_train.shape[1]
            inputs, labels, weights, lr, keep_prob, is_train = self.input_tensor(feature_num)

            # Logits
            logits = self.model(inputs, self.unit_number, keep_prob, is_train)
            logits = tf.identity(logits, name='logits')

            # Loss
            with tf.name_scope('Loss'):
                # cost_ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
                cost_ = self.log_loss(logits, weights, labels)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr).minimize(cost_)

        # Training
        print('Training DNN...')

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

            for x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, \
                e_valid, valid_era in CrossValidation.era_k_fold_with_weight(self.x_train,
                                                                             self.y_train,
                                                                             self.w_train,
                                                                             self.e_train,
                                                                             n_valid=n_valid,
                                                                             n_cv=n_cv,
                                                                             n_era=n_era,
                                                                             seed=cv_seed,
                                                                             era_list=era_list):

                cv_counter += 1

                print('======================================================')
                print('Training on the Cross Validation Set: {}'.format(cv_counter))
                print('Validation Set Era: ', valid_era)
                print('------------------------------------------------------')

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
                                            is_train: True})

                        if str(cost) == 'nan':
                            raise ValueError('NaN BUG!!! Try Another Seed!!!')

                        if batch_counter % self.display_step == 0 and batch_i > 0:

                            summary_train, cost_train = sess.run([merged, cost_],
                                                                 {inputs: batch_x,
                                                                  labels: batch_y,
                                                                  weights: batch_w,
                                                                  keep_prob: 1.0,
                                                                  is_train: False})
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
                                                                          is_train: False})

                                valid_writer.add_summary(summary_valid_i, batch_counter)

                                cost_valid_all.append(cost_valid_i)

                            cost_valid = sum(cost_valid_all) / len(cost_valid_all)

                            total_time = time.time() - start_time

                            print('CV: {} |'.format(cv_counter),
                                  'Epoch: {}/{} |'.format(epoch_i + 1, self.epochs),
                                  'Batch: {} |'.format(batch_counter),
                                  'Time: {:>3.2f}s |'.format(total_time),
                                  'Train_Loss: {:>.8f} |'.format(cost_train),
                                  'Valid_Loss: {:>.8f}'.format(cost_valid))

                # Save Model
                # print('Saving model...')
                # saver = tf.train.Saver()
                # saver.save(sess, self.save_path + 'model.' + self.version + '.ckpt')

                # Prediction
                print('------------------------------------------------------')
                print('Predicting...')
                prob_train = self.get_prob(sess, logits, x_train, self.batch_size, inputs, keep_prob, is_train)
                prob_valid = self.get_prob(sess, logits, x_valid, self.batch_size, inputs, keep_prob, is_train)
                prob_test = self.get_prob(sess, logits, self.x_test, self.batch_size, inputs, keep_prob, is_train)

                loss_train, loss_valid, \
                loss_train_w, loss_valid_w = utils.print_loss_dnn(prob_train, prob_valid,
                                                                  y_train, w_train, y_valid, w_valid)
                prob_test_total.append(prob_test)
                prob_train_total.append(prob_train)
                loss_train_total.append(loss_train)
                loss_valid_total.append(loss_valid)
                loss_train_w_total.append(loss_train_w)
                loss_valid_w_total.append(loss_valid_w)

                # Print and Get Accuracies of CV
                acc_train_cv, acc_valid_cv, acc_train_cv_era, acc_valid_cv_era = \
                    utils.print_and_get_accuracy(prob_train, y_train, e_train, prob_valid, y_valid, e_valid)

                utils.save_loss_log(loss_log_path + 'dnn_', cv_counter, self.parameters, n_valid, n_cv,
                                    valid_era, loss_train, loss_valid, loss_train_w, loss_valid_w,
                                    acc_train_cv, acc_valid_cv, acc_train_cv_era, acc_valid_cv_era)

                utils.save_pred_to_csv(pred_path + 'cv_results/dnn_cv_{}_'.format(cv_counter),
                                       self.id_test, prob_test)

            # Final Result
            print('======================================================')
            print('Calculating Final Result...')

            prob_test_mean = np.mean(np.array(prob_test_total), axis=0)
            prob_train_mean = np.mean(np.array(prob_train_total), axis=0)
            loss_train_mean = np.mean(np.array(loss_train_total), axis=0)
            loss_valid_mean = np.mean(np.array(loss_valid_total), axis=0)
            loss_train_w_mean = np.mean(np.array(loss_train_w_total), axis=0)
            loss_valid_w_mean = np.mean(np.array(loss_valid_w_total), axis=0)

            # Print Total Losses
            utils.print_total_loss(loss_train_mean, loss_valid_mean, loss_train_w_mean, loss_valid_w_mean)

            # Print and Get Accuracies of CV of All Train Set
            acc_train, acc_train_era = utils.print_and_get_train_accuracy(prob_train_mean, self.y_train, self.e_train)

            # Save Final Losses to File
            utils.save_final_loss_log(loss_log_path + 'dnn_', self.parameters, n_valid, n_cv, loss_train_mean,
                                      loss_valid_mean, loss_train_w_mean, loss_valid_w_mean, acc_train, acc_train_era)

            utils.save_pred_to_csv(pred_path + 'final_results/dnn_', self.id_test, prob_test_mean)
            utils.save_prob_train_to_csv(pred_path + 'final_prob_train/dnn_', prob_train_mean, self.y_train)

    def stack_train(self, x_train, y_train, w_train, x_g_train, x_valid, y_valid,
                    w_valid, x_g_valid, x_test, x_g_test, parameters=None, show_importance=False):

        print('------------------------------------------------------')
        print('Training Deep Neural Network...')
        print('------------------------------------------------------')

        # Build Network
        tf.reset_default_graph()
        train_graph = tf.Graph()

        with train_graph.as_default():

            # Inputs
            feature_num = x_train.shape[1]
            inputs, labels, weights, lr, keep_prob, is_train = self.input_tensor(feature_num)

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
            print('Predicting...')

            logits_pred_train = sess.run(logits, {inputs: x_train, keep_prob: 1.0, is_train: False})
            logits_pred_valid = sess.run(logits, {inputs: x_valid, keep_prob: 1.0, is_train: False})
            logits_pred_test = sess.run(logits, {inputs: x_test, keep_prob: 1.0, is_train: False})

            logits_pred_train = logits_pred_train.flatten()
            logits_pred_valid = logits_pred_valid.flatten()
            logits_pred_test = logits_pred_test.flatten()

            prob_train = 1.0 / (1.0 + np.exp(-logits_pred_train))
            prob_valid = 1.0 / (1.0 + np.exp(-logits_pred_valid))
            prob_test = 1.0 / (1.0 + np.exp(-logits_pred_test))

            loss_train, loss_valid, \
            loss_train_w, loss_valid_w = utils.print_loss_dnn(prob_train, prob_valid,
                                                              y_train, w_train, y_valid, w_valid)

            losses = [loss_train, loss_valid, loss_train_w, loss_valid_w]

            return prob_valid, prob_test, losses


# # DNN using Keras
# class KerasDeepNeuralNetworks:
#
#     def __init__(self, x_tr, y_tr, w_tr, e_tr, x_te, id_te, parameters):
#
#         # Inputs
#         self.x_train = x_tr
#         self.y_train = y_tr
#         self.w_train = w_tr
#         self.e_train = e_tr
#         self.x_test = x_te
#         self.id_test = id_te
#
#         # Hyperparameters
#         self.batch_size = parameters['batch_size']
#         self.epochs = parameters['epochs']
#         self.learning_rate = parameters['learning_rate']
#         self.unit_num = parameters['unit_number']
#         self.keep_prob = parameters['keep_probability']
#
#     def train(self, pred_path, n_valid, n_cv, n_era):
#
#         model = Sequential()
#
#         feature_num = list(self.x_train.shape)[1]
#
#         model.add(Dense(self.unit_num[0],
#                         kernel_initializer=initializers.TruncatedNormal(stddev=0.05),
#                         bias_initializer='zeros',
#                         activation='sigmoid',
#                         input_dim=feature_num))
#         model.add(Dropout(self.keep_prob))
#
#         for i in range(len(self.unit_num)-1):
#             model.add(Dense(self.unit_num[i+1],
#                             kernel_initializer=initializers.TruncatedNormal(stddev=0.05),
#                             bias_initializer='zeros',
#                             activation='sigmoid'))
#             model.add(Dropout(self.keep_prob))
#
#         model.compile(loss='binary_crossentropy',
#                       optimizer=optimizers.Adam(self.learning_rate),
#                       metrics=['accuracy'])
#
#         start_time = time.time()
#
#         cv_counter = 0
#
#         prob_total = []
#
#         for x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, \
#             e_valid, valid_era in CrossValidation.era_k_fold_with_weight(self.x_train,
#                                                                          self.y_train,
#                                                                          self.w_train,
#                                                                          self.e_train,
#                                                                          n_valid,
#                                                                          n_era=n_era,
#                                                                          n_cv,
#                                                                          era_list=era_list):
#
#             cv_counter += 1
#
#             print('======================================================')
#             print('Training on the Cross Validation Set: {}'.format(cv_counter))
#
#             model.fit(x_train,
#                       y_train,
#                       epochs=self.epochs,
#                       batch_size=self.batch_size,
#                       verbose=1)
#
#             cost_train = model.evaluate(x_train, y_train, verbose=1)
#             cost_valid = model.evaluate(x_valid, y_valid, verbose=1)
#
#             total_time = time.time() - start_time
#
#             print('CV: {} |'.format(cv_counter),
#                   'Time: {:>3.2f}s |'.format(total_time),
#                   'Train_Loss: {:>.8f} |'.format(cost_train),
#                   'Valid_Loss: {:>.8f}'.format(cost_valid))
#
#             # Prediction
#             print('Predicting...')
#
#             prob_test = model.predict(self.x_test)
#
#             prob_total.append(list(prob_test))
#
#             utils.save_pred_to_csv(pred_path + 'cv_results/dnn_keras_cv_{}_'.format(cv_counter),
#                                    self.id_test, prob_test)
#
#         # Final Result
#         print('======================================================')
#         print('Calculating Final Result...')
#
#         prob_mean = np.mean(np.array(prob_total), axis=0)
#
#         utils.save_pred_to_csv(pred_path + 'final_results/dnn_keras_', self.id_test, prob_mean)


# Cross Validation
class CrossValidation:

    trained_cv = []

    @staticmethod
    def sk_k_fold_with_weight(x, y, w, n_splits, n_cv, seed=None):

        if seed is not None:
            np.random.seed(seed)

        if n_cv % n_splits != 0:
            raise ValueError('n_cv must be an integer multiple of n_splits!')

        n_repeats = int(n_cv / n_splits)

        era_k_fold = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

        for train_index, valid_index in era_k_fold.split(x, y):
            np.random.shuffle(train_index)
            np.random.shuffle(valid_index)

            # Training data
            x_train = x[train_index]
            y_train = y[train_index]
            w_train = w[train_index]

            # Validation data
            x_valid = x[valid_index]
            y_valid = y[valid_index]
            w_valid = w[valid_index]

            yield x_train, y_train, w_train, x_valid, y_valid, w_valid

    @staticmethod
    def sk_group_k_fold(x, y, e, n_cv):

        era_k_fold = GroupKFold(n_splits=n_cv)

        for train_index, valid_index in era_k_fold.split(x, y, e):
            # Training data
            x_train = x[train_index]
            y_train = y[train_index]

            # Validation data
            x_valid = x[valid_index]
            y_valid = y[valid_index]

            yield x_train, y_train, x_valid, y_valid

    @staticmethod
    def sk_group_k_fold_with_weight(x, y, w, e, n_cv):

        era_k_fold = GroupKFold(n_splits=n_cv)

        for train_index, valid_index in era_k_fold.split(x, y, e):
            # Training data
            x_train = x[train_index]
            y_train = y[train_index]
            w_train = w[train_index]

            # Validation data
            x_valid = x[valid_index]
            y_valid = y[valid_index]
            w_valid = w[valid_index]

            yield x_train, y_train, w_train, x_valid, y_valid, w_valid

    @staticmethod
    def era_k_fold_split_all_random(e, n_valid, n_cv, n_era, seed=None, era_list=None):

        if seed is not None:
            np.random.seed(seed)

        for i in range(n_cv):

            if era_list is None:
                era_list = range(1, n_era + 1)

            era_idx = [era_list]
            valid_group = np.random.choice(era_idx, n_valid, replace=False)

            train_index = []
            valid_index = []

            for ii, ele in enumerate(e):

                if ele in valid_group:
                    valid_index.append(ii)
                else:
                    train_index.append(ii)

            np.random.shuffle(train_index)
            np.random.shuffle(valid_index)

            yield train_index, valid_index

    @staticmethod
    def era_k_fold_with_weight_all_random(x, y, w, e, n_valid, n_cv, n_era, seed=None, era_list=None):

        if seed is not None:
            np.random.seed(seed)

        for i in range(n_cv):

            if era_list is None:
                era_list = range(1, n_era + 1)

            era_idx = [era_list]
            valid_group = np.random.choice(era_idx, n_valid, replace=False)

            train_index = []
            valid_index = []

            for ii, ele in enumerate(e):

                if ele in valid_group:
                    valid_index.append(ii)
                else:
                    train_index.append(ii)

            np.random.shuffle(train_index)
            np.random.shuffle(valid_index)

            # Training data
            x_train = x[train_index]
            y_train = y[train_index]
            w_train = w[train_index]

            # Validation data
            x_valid = x[valid_index]
            y_valid = y[valid_index]
            w_valid = w[valid_index]

            yield x_train, y_train, w_train, x_valid, y_valid, w_valid

    @staticmethod
    def era_k_fold_split(e, n_valid, n_cv, n_era, seed=None, era_list=None):

        if seed is not None:
            np.random.seed(seed)

        n_traverse = n_era // n_valid
        n_rest = n_era % n_valid

        if n_rest != 0:
            n_traverse += 1

        if n_cv % n_traverse != 0:
            raise ValueError

        n_epoch = n_cv // n_traverse
        trained_cv = []

        for epoch in range(n_epoch):

            if era_list is None:
                era_list = range(1, n_era + 1)

            era_idx = [era_list]

            if n_rest == 0:

                for i in range(n_traverse):

                    # Choose eras that have not used
                    if trained_cv:
                        valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                        while any(set(valid_era) == i_cv for i_cv in trained_cv):
                            print('This CV split has been chosen, choosing another one...')
                            if set(valid_era) != set(era_idx[i]):
                                valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                            else:
                                valid_era = np.random.choice(range(1, n_era + 1), n_valid, replace=False)
                    else:
                        valid_era = np.random.choice(era_idx[i], n_valid, replace=False)

                    # Generate era set for next choosing
                    if i != n_traverse - 1:
                        era_next = [rest for rest in era_idx[i] if rest not in valid_era]
                        era_idx.append(era_next)

                    train_index = []
                    valid_index = []

                    # Generate train-validation split index
                    for ii, ele in enumerate(e):

                        if ele in valid_era:
                            valid_index.append(ii)
                        else:
                            train_index.append(ii)

                    np.random.shuffle(train_index)
                    np.random.shuffle(valid_index)

                    yield train_index, valid_index

            # n_cv is not an integer multiple of n_valid
            else:

                for i in range(n_traverse):

                    if i != n_traverse - 1:

                        if trained_cv:
                            valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                            while any(set(valid_era) == i_cv for i_cv in trained_cv):
                                print('This CV split has been chosen, choosing another one...')
                                valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                        else:
                            valid_era = np.random.choice(era_idx[i], n_valid, replace=False)

                        era_next = [rest for rest in era_idx[i] if rest not in valid_era]
                        era_idx.append(era_next)

                        train_index = []
                        valid_index = []

                        for ii, ele in enumerate(e):

                            if ele in valid_era:
                                valid_index.append(ii)
                            else:
                                train_index.append(ii)

                        np.random.shuffle(train_index)
                        np.random.shuffle(valid_index)

                        yield train_index, valid_index

                    else:

                        era_idx_else = [t for t in list(range(1, n_era + 1)) if t not in era_idx[i]]

                        valid_era = era_idx[i] + list(np.random.choice(era_idx_else, n_valid - n_rest, replace=False))
                        while any(set(valid_era) == i_cv for i_cv in trained_cv):
                            print('This CV split has been chosen, choosing another one...')
                            valid_era = era_idx[i] + list(
                                np.random.choice(era_idx_else, n_valid - n_rest, replace=False))

                        train_index = []
                        valid_index = []

                        for ii, ele in enumerate(e):

                            if ele in valid_era:
                                valid_index.append(ii)
                            else:
                                train_index.append(ii)

                        np.random.shuffle(train_index)
                        np.random.shuffle(valid_index)

                        yield train_index, valid_index

    @staticmethod
    def era_k_fold_with_weight(x, y, w, e, n_valid, n_cv, n_era, seed=None, era_list=None):

        if seed is not None:
            np.random.seed(seed)

        n_traverse = n_era // n_valid
        n_rest = n_era % n_valid

        if n_rest != 0:
            n_traverse += 1

        if n_cv % n_traverse != 0:
            raise ValueError

        n_epoch = n_cv // n_traverse
        trained_cv = []

        for epoch in range(n_epoch):

            if era_list is None:
                era_list = range(1, n_era + 1)

            era_idx = [era_list]

            if n_rest == 0:

                for i in range(n_traverse):

                    # Choose eras that have not used
                    if trained_cv:
                        valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                        while any(set(valid_era) == i_cv for i_cv in trained_cv):
                            print('This CV split has been chosen, choosing another one...')
                            if set(valid_era) != set(era_idx[i]):
                                valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                            else:
                                valid_era = np.random.choice(era_list, n_valid, replace=False)
                    else:
                        valid_era = np.random.choice(era_idx[i], n_valid, replace=False)

                    # Generate era set for next choosing
                    if i != n_traverse - 1:
                        era_next = [rest for rest in era_idx[i] if rest not in valid_era]
                        era_idx.append(era_next)

                    train_index = []
                    valid_index = []

                    # Generate train-validation split index
                    for ii, ele in enumerate(e):

                        if ele in valid_era:
                            valid_index.append(ii)
                        else:
                            train_index.append(ii)

                    np.random.shuffle(train_index)
                    np.random.shuffle(valid_index)

                    # Training data
                    x_train = x[train_index]
                    y_train = y[train_index]
                    w_train = w[train_index]
                    e_train = e[train_index]

                    # Validation data
                    x_valid = x[valid_index]
                    y_valid = y[valid_index]
                    w_valid = w[valid_index]
                    e_valid = e[valid_index]

                    trained_cv.append(set(valid_era))

                    yield x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, e_valid, valid_era

            # n_cv is not an integer multiple of n_valid
            else:

                for i in range(n_traverse):

                    if i != n_traverse - 1:

                        if trained_cv:
                            valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                            while any(set(valid_era) == i_cv for i_cv in trained_cv):
                                print('This CV split has been chosen, choosing another one...')
                                valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                        else:
                            valid_era = np.random.choice(era_idx[i], n_valid, replace=False)

                        era_next = [rest for rest in era_idx[i] if rest not in valid_era]
                        era_idx.append(era_next)

                        train_index = []
                        valid_index = []

                        for ii, ele in enumerate(e):

                            if ele in valid_era:
                                valid_index.append(ii)
                            else:
                                train_index.append(ii)

                        np.random.shuffle(train_index)
                        np.random.shuffle(valid_index)

                        # Training data
                        x_train = x[train_index]
                        y_train = y[train_index]
                        w_train = w[train_index]
                        e_train = e[train_index]

                        # Validation data
                        x_valid = x[valid_index]
                        y_valid = y[valid_index]
                        w_valid = w[valid_index]
                        e_valid = e[valid_index]

                        trained_cv.append(set(valid_era))

                        yield x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, e_valid, valid_era

                    else:

                        era_idx_else = [t for t in list(era_list) if t not in era_idx[i]]

                        valid_era = era_idx[i] + list(np.random.choice(era_idx_else, n_valid - n_rest, replace=False))
                        while any(set(valid_era) == i_cv for i_cv in trained_cv):
                            print('This CV split has been chosen, choosing another one...')
                            valid_era = era_idx[i] + list(
                                np.random.choice(era_idx_else, n_valid - n_rest, replace=False))

                        train_index = []
                        valid_index = []

                        for ii, ele in enumerate(e):

                            if ele in valid_era:
                                valid_index.append(ii)
                            else:
                                train_index.append(ii)

                        np.random.shuffle(train_index)
                        np.random.shuffle(valid_index)

                        # Training data
                        x_train = x[train_index]
                        y_train = y[train_index]
                        w_train = w[train_index]
                        e_train = e[train_index]

                        # Validation data
                        x_valid = x[valid_index]
                        y_valid = y[valid_index]
                        w_valid = w[valid_index]
                        e_valid = e[valid_index]

                        trained_cv.append(set(valid_era))

                        yield x_train, y_train, w_train, e_train, x_valid, y_valid, w_valid, e_valid, valid_era

    def era_k_fold_for_stack(self, x, y, w, e, x_g, n_valid, n_cv, n_era, seed=None):

        if seed is not None:
            np.random.seed(seed)

        n_traverse = n_era // n_valid
        n_rest = n_era % n_valid

        if n_rest != 0:
            n_traverse += 1

        if n_cv % n_traverse != 0:
            raise ValueError

        n_epoch = n_cv // n_traverse

        for epoch in range(n_epoch):

            era_idx = [list(range(1, n_era + 1))]

            if n_rest == 0:

                for i in range(n_traverse):

                    # Choose eras that have not used
                    if self.trained_cv:
                        valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                        while any(set(valid_era) == i_cv for i_cv in self.trained_cv):
                            print('This CV split has been chosen, choosing another one...')
                            if set(valid_era) != set(era_idx[i]):
                                valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                            else:
                                valid_era = np.random.choice(range(1, n_era + 1), n_valid, replace=False)
                    else:
                        valid_era = np.random.choice(era_idx[i], n_valid, replace=False)

                    # Generate era set for next choosing
                    if i != n_traverse - 1:
                        era_next = [rest for rest in era_idx[i] if rest not in valid_era]
                        era_idx.append(era_next)

                    train_index = []
                    valid_index = []

                    # Generate train-validation split index
                    for ii, ele in enumerate(e):

                        if ele in valid_era:
                            valid_index.append(ii)
                        else:
                            train_index.append(ii)

                    np.random.shuffle(train_index)
                    np.random.shuffle(valid_index)

                    # Training data
                    x_train = x[train_index]
                    y_train = y[train_index]
                    w_train = w[train_index]
                    x_g_train = x_g[train_index]

                    # Validation data
                    x_valid = x[valid_index]
                    y_valid = y[valid_index]
                    w_valid = w[valid_index]
                    x_g_valid = x_g[valid_index]

                    self.trained_cv.append(set(valid_era))

                    yield x_train, y_train, w_train, x_g_train, x_valid, \
                          y_valid, w_valid, x_g_valid, valid_index, valid_era

            # n_cv is not an integer multiple of n_valid
            else:

                for i in range(n_traverse):

                    if i != n_traverse - 1:

                        if self.trained_cv:
                            valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                            while any(set(valid_era) == i_cv for i_cv in self.trained_cv):
                                print('This CV split has been chosen, choosing another one...')
                                valid_era = np.random.choice(era_idx[i], n_valid, replace=False)
                        else:
                            valid_era = np.random.choice(era_idx[i], n_valid, replace=False)

                        era_next = [rest for rest in era_idx[i] if rest not in valid_era]
                        era_idx.append(era_next)

                        train_index = []
                        valid_index = []

                        for ii, ele in enumerate(e):

                            if ele in valid_era:
                                valid_index.append(ii)
                            else:
                                train_index.append(ii)

                        np.random.shuffle(train_index)
                        np.random.shuffle(valid_index)

                        # Training data
                        x_train = x[train_index]
                        y_train = y[train_index]
                        w_train = w[train_index]
                        x_g_train = x_g[train_index]

                        # Validation data
                        x_valid = x[valid_index]
                        y_valid = y[valid_index]
                        w_valid = w[valid_index]
                        x_g_valid = x_g[valid_index]

                        self.trained_cv.append(set(valid_era))

                        yield x_train, y_train, w_train, x_g_train, x_valid, \
                              y_valid, w_valid, x_g_valid, valid_index, valid_era

                    else:

                        era_idx_else = [t for t in list(range(1, n_era + 1)) if t not in era_idx[i]]

                        valid_era = era_idx[i] + list(
                            np.random.choice(era_idx_else, n_valid - n_rest, replace=False))
                        while any(set(valid_era) == i_cv for i_cv in self.trained_cv):
                            print('This CV split has been chosen, choosing another one...')
                            valid_era = era_idx[i] + list(
                                np.random.choice(era_idx_else, n_valid - n_rest, replace=False))

                        train_index = []
                        valid_index = []

                        for ii, ele in enumerate(e):

                            if ele in valid_era:
                                valid_index.append(ii)
                            else:
                                train_index.append(ii)

                        np.random.shuffle(train_index)
                        np.random.shuffle(valid_index)

                        # Training data
                        x_train = x[train_index]
                        y_train = y[train_index]
                        w_train = w[train_index]
                        x_g_train = x_g[train_index]

                        # Validation data
                        x_valid = x[valid_index]
                        y_valid = y[valid_index]
                        w_valid = w[valid_index]
                        x_g_valid = x_g[valid_index]

                        self.trained_cv.append(set(valid_era))

                        yield x_train, y_train, w_train, x_g_train, x_valid, \
                              y_valid, w_valid, x_g_valid, valid_index, valid_era


# Grid Search
def grid_search(log_path, tr_x, tr_y, tr_e, clf, n_valid, n_cv, n_era, cv_seed, params, params_grid):
    start_time = time.time()

    grid_search_model = GridSearchCV(estimator=clf,
                                     param_grid=params_grid,
                                     scoring='neg_log_loss',
                                     verbose=1,
                                     n_jobs=-1,
                                     cv=CrossValidation.era_k_fold_split(e=tr_e, n_valid=n_valid,
                                                                         n_cv=n_cv, n_era=n_era, seed=cv_seed),
                                     # cv=5
                                     )

    # Start Grid Search
    print('Grid Searching...')

    grid_search_model.fit(tr_x, tr_y, tr_e)

    best_parameters = grid_search_model.best_estimator_.get_params()
    best_score = grid_search_model.best_score_

    print('Best score: %0.6f' % best_score)
    print('Best parameters set:')

    for param_name in sorted(params_grid.keys()):
        print('\t%s: %r' % (param_name, best_parameters[param_name]))

    total_time = time.time() - start_time

    utils.save_grid_search_log(log_path, params, params_grid, best_score, best_parameters, total_time)


if __name__ == '__main__':

    pass

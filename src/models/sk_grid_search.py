import time
from models import utils
from sklearn.model_selection import GridSearchCV
from models.cross_validation import CrossValidation


class SKLearnGridSearch(object):

    def __init__(self):
        pass

    @staticmethod
    def grid_search(log_path, tr_x, tr_y, tr_e, clf, n_valid, n_cv, n_era, cv_seed, params, params_grid):
        """
             Grid Search
        """
        start_time = time.time()

        utils.check_dir([log_path])

        grid_search_model = GridSearchCV(estimator=clf,
                                         param_grid=params_grid,
                                         scoring='neg_log_loss',
                                         verbose=1,
                                         n_jobs=-1,
                                         cv=CrossValidation.era_k_fold_split(e=tr_e, n_valid=n_valid,
                                                                             n_cv=n_cv, n_era=n_era, seed=cv_seed))

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

from models import utils
from config import cfg
import numpy as np
import pandas as pd
from models.strategy import Strategy


def calculate_profit(strategy_args_, test_path_, profit_path_):

    test_f = pd.read_csv(test_path_, header=0, dtype=np.float64)

    prob = test_f['proba']

    pct_test = utils.load_preprocessed_data(cfg.preprocessed_data_path)[5]

    f_strategy = strategy_args_['f_strategy']
    strategy_args_.pop('f_strategy')
    pred_s = pct_test.drop(['index'], axis=1)
    pred_s['prob'] = prob

    profit = f_strategy(pred_s, **strategy_args_, save_path=profit_path_)

    print('Profit: {}'.format(profit))


if __name__ == '__main__':

    test_path = '../results/xgb_idx-1_t-999_c-95_result.csv'
    profit_path = '../results/profit/xgb_idx-1_t-999_c-95_'

    strategy_args = {'f_strategy': Strategy.calc_profit,
                     'buy_count': 15,
                     'fee': 0.0034,
                     'ascending': False,
                     'save_result': True,
                     'save_image': True}

    calculate_profit(strategy_args, test_path, profit_path)

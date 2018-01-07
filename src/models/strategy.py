import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


class Strategy(object):

    def __init__(self):
        pass

    @staticmethod
    def calc_profit(pred, buy_count=15, fee=0.0034, ascending=False,
                    save_result=False, save_image=False, save_path=None):

        prediction = pred.copy()
        day_profit = prediction.groupby('date').apply(
            lambda x: x.sort_values('prob', ascending=ascending)[:buy_count]['pct'].mean())
        day_profit -= fee
        day_profit_array = day_profit.as_matrix()

        result = np.zeros(len(day_profit_array)+1)
        result[0] = 1
        for i in range(len(day_profit_array)):
            result[i+1] = result[i] * (1 + day_profit_array[i])

        final_profit = result[-1]-1

        day_profit_array = np.concatenate((day_profit_array.tolist(), [0.]))

        if save_result:
            df = pd.DataFrame({'reserve': result, 'day_profit': day_profit_array})
            df.to_csv(save_path + 'day_profit.csv', sep=',', index=True)

        if save_image:
            plt.figure(figsize=(12, 6))
            plt.plot(result, color='red')
            plt.savefig(save_path + 'reserve.jpg')
            plt.close('all')
            plt.figure(figsize=(12, 6))
            plt.plot(day_profit_array, color='blue')
            plt.savefig(save_path + 'day_profit.jpg')

        return final_profit

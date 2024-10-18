import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

def get_baseline(y_train):
    '''Calculates and returns the baseline.'''
    baselines = pd.DataFrame(y_train)
    baselines['mean'] = baselines['calories_burned'].mean()
    baseline_rmse = np.sqrt(mean_squared_error(baselines['calories_burned'], baselines['mean']))
    print('Baseline calculated.')
    print(f'Baseline: {round(baseline_rmse, 2)}')
    return baseline_rmse

def make_preds(X_train, y_train, X_val, y_val, X_test, y_test):
    '''Fits the model to the train dataset and makes predictions.'''
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    train_preds = lr.predict(X_train)
    train_rmse = round(np.sqrt(mean_squared_error(y_train, train_preds)), 2)
    val_preds = lr.predict(X_val)
    val_rmse = round(np.sqrt(mean_squared_error(y_val, val_preds)), 2)
    test_preds = lr.predict(X_test)
    test_rmse = round(np.sqrt(mean_squared_error(y_test, test_preds)), 2)
    print('Predictions evaluated.')
    print(f'Train RMSE: {train_rmse}.')
    print(f'Validation RMSE: {val_rmse}.')
    print(f'Test RMSE: {test_rmse}.')
    return test_rmse

def visualize(baseline, test_rmse):
    '''Shows the staggering difference between baseline and result.'''
    sns.barplot(x=['Baseline', 'Model'], y=[baseline, test_rmse], palette=['gainsboro', 'lightgreen'], edgecolor='black')
    sns.despine()
    plt.ylabel('RMSE')
    plt.title('Model Outperforms Baseline')
    plt.show()
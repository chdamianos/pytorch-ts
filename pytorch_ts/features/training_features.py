import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pandas.core.frame import DataFrame
from tqdm import tqdm


class TrainingFeatures:
    def __init__(self, target_column_name="sales",
                 train_sequence_length=int(365 + np.floor(365/2)),
                 predict_sequence_length=90,
                 min_step=5, max_step=15,
                 numerical_features=None,
                 categorical_features=None,
                 group_by_columns=None,
                 n_jobs=6) -> None:
        if numerical_features is None:
            numerical_features = ['sales', 'cos_dayofweek', 'sin_dayofweek',
                                  'cos_month', 'sin_month', 'cos_day', 'sin_day']
        if categorical_features is None:
            categorical_features = ['store', 'item']
        if group_by_columns is None:
            group_by_columns = ['store', 'item']
        self.target_column_name = target_column_name
        self.train_sequence_length = train_sequence_length
        self.predict_sequence_length = predict_sequence_length
        self.min_step = min_step
        self.max_step = max_step
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.n_jobs = n_jobs
        self.group_by_columns = group_by_columns

    def create_training_feautures(self, train_pdf: DataFrame) -> DataFrame:
        create_sequence_func_pdf = self.create_sequence_wrapper_pdf()
        return self.applyParallel(train_pdf.groupby(self.group_by_columns), create_sequence_func_pdf, self.njobs)

    def create_sequence_wrapper_pdf(self, train_sequence_length, predict_sequence_length, min_step,
                                    max_step, numerical_features):
        def create_sequence_pdf(pdf):
            item = pdf.iloc[0]['item']
            store = pdf.iloc[0]['store']
            pdf = self.add_time_features(pdf)
            pdf = self.add_cyclical_features(pdf)
            idx = 0
            train_seq = []
            target_seq = []
            store_list = []
            item_list = []
            step_list = []
            step = np.random.randint(min_step, max_step+1)
            while True:
                train_part_pdf = pdf.iloc[0+idx:train_sequence_length+idx]
                target_part_pdf = pdf.iloc[train_sequence_length +
                                           idx:train_sequence_length+predict_sequence_length+idx]
                if train_part_pdf.shape[0] != train_sequence_length or target_part_pdf.shape[0] != predict_sequence_length:
                    break
                mu_ = train_part_pdf['sales'].mean()
                std_ = train_part_pdf['sales'].std()
                train_part_pdf['sales'] = (train_part_pdf['sales']-mu_)/std_
                target_part_pdf['sales'] = (target_part_pdf['sales']-mu_)/std_
                train_seq.append(train_part_pdf[numerical_features].values)
                target_seq.append(target_part_pdf[numerical_features].values)
                store_list.append(store)
                item_list.append(item)
                step_list.append(step)
                idx += step
            return pd.DataFrame({'train_seq': train_seq, 'target_seq': target_seq, 'store': store_list, 'item': item_list, 'step': step_list})
        return create_sequence_pdf

    def applyParallel(dfGrouped, func, n_jobs=multiprocessing.cpu_count()):
        retLst = Parallel(n_jobs=n_jobs)(delayed(func)(group)
                                         for name, group in tqdm(dfGrouped))
        return pd.concat(retLst)

    def sin_transform(values, n_values_in_cycle):
        return np.sin((2 * np.pi * values) / (n_values_in_cycle))

    def cos_transform(values, n_values_in_cycle):
        return np.cos(2 * np.pi * values / (n_values_in_cycle))

    def add_time_features(pdf, date_column='date'):
        pdf = pdf.sort_values(by=[date_column], ascending=True)
        pdf['dayofweek'] = pdf[date_column].dt.dayofweek
        pdf['month'] = pdf[date_column].dt.month
        pdf['day'] = pdf[date_column].dt.day
        return pdf

    def add_cyclical_features(self, pdf):
        assert 'dayofweek' in pdf.columns
        assert 'month' in pdf.columns
        assert 'day' in pdf.columns
        for k, v in {'dayofweek': 7, 'month': 12, 'day': 31}.items():
            pdf["cos_"+k] = self.cos_transform(pdf[k].values, v)
            pdf["sin_"+k] = self.sin_transform(pdf[k].values, v)
        return pdf

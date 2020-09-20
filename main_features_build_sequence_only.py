print("here")
import warnings

import pandas as pd
from tqdm.notebook import tqdm
import ts_utils

warnings.filterwarnings('ignore')
tqdm.pandas()

filter_ids = ['10_1', '10_10', '10_11', '10_12', '10_13', '10_14', '10_15',
              '10_16', '10_17', '10_18', '10_19', '10_2']
scaled_data = pd.read_pickle('./data/processed_data_test_stdscaler.pkl')
scaled_data_filtered = scaled_data[scaled_data['store_item_id'].isin(filter_ids)]
scaled_data_filtered_10_1 = scaled_data_filtered.query("store_item_id=='10_1'")
output = ts_utils.sequence_builder.split_sequences(scaled_data_filtered_10_1, n_steps_in=180, n_steps_out=90,
                                                   x_cols=['sales', 'dayofweek_sin', 'dayofweek_cos', 'month_sin',
                                                           'month_cos', 'year_mod', 'day_sin', 'day_cos'],
                                                   y_cols=['sales', 'dayofweek_sin', 'dayofweek_cos', 'month_sin',
                                                           'month_cos', 'year_mod', 'day_sin', 'day_cos'],
                                                   additional_columns=['item', 'store', 'date', 'yearly_corr'], step=1,
                                                   lag_fns=[ts_utils.sequence_builder.last_year_lag])
print("*****************************************************************************")
print(scaled_data_filtered_10_1['store_item_id'].unique())
print(scaled_data_filtered_10_1.head())
print(scaled_data_filtered_10_1.shape)
print(output[0].shape)
print(output[1].shape)
print(output[2])
print(len(output[2]))

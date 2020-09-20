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
sequence_data = ts_utils.sequence_builder.sequence_builder(scaled_data_filtered, 180, 90,
                                                           'store_item_id',
                                                           ['sales', 'dayofweek_sin', 'dayofweek_cos', 'month_sin',
                                                            'month_cos', 'year_mod', 'day_sin', 'day_cos'],
                                                           'sales',
                                                           ['sales', 'dayofweek_sin', 'dayofweek_cos', 'month_sin',
                                                            'month_cos', 'year_mod', 'day_sin', 'day_cos'],
                                                           ['item', 'store', 'date', 'yearly_corr'],
                                                           lag_fns=[ts_utils.sequence_builder.last_year_lag]
                                                           )
# sequence_data.to_pickle('./sequence_data/sequence_data_stdscaler_test.pkl')

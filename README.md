# Data 
https://www.kaggle.com/c/demand-forecasting-kernels-only/data 
# Feature engineering
## train data
```
         date  store  item  sales
0  2013-01-01      1     1     13
1  2013-01-02      1     1     11
2  2013-01-03      1     1     14
3  2013-01-04      1     1     13
4  2013-01-05      1     1     10
```
## test data
```
   id        date  store  item
0   0  2018-01-01      1     1
1   1  2018-01-02      1     1
2   2  2018-01-03      1     1
3   3  2018-01-04      1     1
4   4  2018-01-05      1     1
```
## target
we want to predict store/item sales from historical data
## steps
### add `sales` to test data
```python
test['sales'] = np.nan
```
### concatenate test/train data
```python
data = pd.concat([train, test], ignore_index=True)
```
### concatenate store/item
```python
data['store_item_id'] = data['store'].astype(str) + '_' + data['item'].astype(str)
```
```
        date  store  item  sales  id store_item_id
0 2013-01-01      1     1   13.0 NaN           1_1
1 2013-01-02      1     1   11.0 NaN           1_1
2 2013-01-03      1     1   14.0 NaN           1_1
3 2013-01-04      1     1   13.0 NaN           1_1
4 2013-01-05      1     1   10.0 NaN           1_1
```
### add day of week, day of month, year, month, normalized year (`year_mod`)
```python
data['dayofweek'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year
data['day'] = data['date'].dt.day
data['year_mod'] = (data['year'] - data['year'].min()) / (data['year'].max() - data['year'].min())
```
```
        date  store  item  sales  id store_item_id  dayofweek  month  year  \
0 2013-01-01  1      1     13.0  NaN  1_1           1          1      2013   
1 2013-01-02  1      1     11.0  NaN  1_1           2          1      2013   
2 2013-01-03  1      1     14.0  NaN  1_1           3          1      2013   
3 2013-01-04  1      1     13.0  NaN  1_1           4          1      2013   
4 2013-01-05  1      1     10.0  NaN  1_1           5          1      2013   
   day  year_mod  
0  1    0.0       
1  2    0.0       
2  3    0.0       
3  4    0.0      
```
### transform cyclical features 
* see https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
```python
def sin_transform(values):
    return np.sin(2 * np.pi * values / len(set(values)))
def cos_transform(values):
    return np.cos(2 * np.pi * values / len(set(values)))
```
```python
data['dayofweek_sin'] = sin_transform(data['dayofweek'])
data['dayofweek_cos'] = cos_transform(data['dayofweek'])
data['month_sin'] = sin_transform(data['month'])
data['month_cos'] = cos_transform(data['month'])
data['day_sin'] = sin_transform(data['day'])
data['day_cos'] = cos_transform(data['day'])
```
```
        date  store  item  sales  id store_item_id  dayofweek  month  year  \
0 2013-01-01  1      1     13.0  NaN  1_1           1          1      2013   
1 2013-01-02  1      1     11.0  NaN  1_1           2          1      2013   
2 2013-01-03  1      1     14.0  NaN  1_1           3          1      2013   
3 2013-01-04  1      1     13.0  NaN  1_1           4          1      2013   
4 2013-01-05  1      1     10.0  NaN  1_1           5          1      2013   
   day  year_mod  dayofweek_sin  dayofweek_cos  month_sin  month_cos  \
0  1    0.0       0.781831       0.623490       0.5        0.866025    
1  2    0.0       0.974928      -0.222521       0.5        0.866025    
2  3    0.0       0.433884      -0.900969       0.5        0.866025    
3  4    0.0      -0.433884      -0.900969       0.5        0.866025    
4  5    0.0      -0.974928      -0.222521       0.5        0.866025    
    day_sin   day_cos  
0  0.201299  0.979530  
1  0.394356  0.918958  
2  0.571268  0.820763  
3  0.724793  0.688967  
4  0.848644  0.528964  
```
### drop `id` and sort by store/item date
```python
data.drop('id', axis=1, inplace=True)
data = data.sort_values(['store_item_id', 'date'])
```
```
            date  store  item  sales store_item_id  dayofweek  month  year  \
16434 2013-01-01  10     1     14.0   10_1          1          1      2013   
16435 2013-01-02  10     1     14.0   10_1          2          1      2013   
16436 2013-01-03  10     1     16.0   10_1          3          1      2013   
16437 2013-01-04  10     1     17.0   10_1          4          1      2013   
16438 2013-01-05  10     1     12.0   10_1          5          1      2013   
       day  year_mod  dayofweek_sin  dayofweek_cos  month_sin  month_cos  \
16434  1    0.0       0.781831       0.623490       0.5        0.866025    
16435  2    0.0       0.974928      -0.222521       0.5        0.866025    
16436  3    0.0       0.433884      -0.900969       0.5        0.866025    
16437  4    0.0      -0.433884      -0.900969       0.5        0.866025    
16438  5    0.0      -0.974928      -0.222521       0.5        0.866025    
        day_sin   day_cos  
16434  0.201299  0.979530  
16435  0.394356  0.918958  
16436  0.571268  0.820763  
16437  0.724793  0.688967  
16438  0.848644  0.528964  
```
### ACF
* The ACF gives the correlation of a value in a timeseries to values from zero 
 up to `k` lags before that value. 
* Below is the acf of the sales of item 1 in store 1 for `k` 366
![acf](./images/acf.png)
* from the plot you can see that there is strong correlation between weekly sales 
acf(~0.6), i.e. sales that happen at `sales[i]` and `sales[i+7]` are related. 
Also the ACF is increased (~0.4) at a yearly lag (`k`=365) , i.e. sales that 
happen at `sales[i]` and `sales[i+365]` are related. 
* for more on ACF
https://www.philippe-fournier-viger.com/spmf/TimeSeriesAutocorellation.php
* the ACF is used as a feature, specifically the yearly ACF is calculated by
    ```python
    from statsmodels.tsa.stattools import acf
    def get_yearly_autocorr(data_):
        ac = acf(data_, nlags=366)
        return (0.5 * ac[365]) + (0.25 * ac[364]) + (0.25 * ac[366])
    ```
    * note that the calculation is a weighted one around lag 365 
### scale data and add yearly ACF for each store/item
* scale based on train data only
    ```python
    mode = 'test'
    if mode == 'valid':
        scale_data = train[train['date'] < '2017-01-01']
    else:
        scale_data = train[train['date'] >= '2014-01-01']
    ```
* the target, `sales`, will be scaled based on mean and standard deviation
in the **train data**
    ```python
    scale_map = {}
    scaled_data = pd.DataFrame()
    # go through each store/item
    for store_item_id, item_data in data.groupby('store_item_id', as_index=False):
        # get sales data for specific store/item from TRAIN DATA
        sidata = scale_data.loc[scale_data['store_item_id'] == store_item_id, 'sales']
        mu = sidata.mean()
        sigma = sidata.std()
        # get yearly ACF of specific store/item
        yearly_autocorr = get_yearly_autocorr(sidata)
        # scale sales based on mu/std from TRAIN DATA
        item_data.loc[:, 'sales'] = (item_data['sales'] - mu) / sigma
        # save scaling record in dict
        scale_map[store_item_id] = {'mu': mu, 'sigma': sigma}
        # add average sales and yearly autocorellation as features
        item_data['mean_sales'] = mu
        item_data['yearly_corr'] = yearly_autocorr
        # append data to main features dataframe
        scaled_data = pd.concat([scaled_data, item_data], ignore_index=True)
    ```
* scale yeary ACF and average sales
    ```python
    scaled_data['yearly_corr'] = (
            (scaled_data['yearly_corr'] - scaled_data['yearly_corr'].mean()) / scaled_data['yearly_corr'].std())
    scaled_data['mean_sales'] = (scaled_data['mean_sales'] - scaled_data['mean_sales'].mean()) / scaled_data[
        'mean_sales'].std()
    ```
   * this might be wrong since the train data are not used ??? 
        * but could be ok since `yearly_corr` and `mean_sales` are based on the train
        data?
### reduce memory of pandas dataframe
```python
scaled_data = reduce_mem_usage(scaled_data)
```
### create sequences
HERE!
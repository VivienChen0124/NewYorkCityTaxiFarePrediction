import numpy as np
import pandas as pd
import os
from google.cloud import bigquery
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV

QUERY_TRAIN = """
              SELECT *
              FROM `cloud-training-demos.taxifare_kaggle.train`
			  """
QUERY_TEST = """
             SELECT *
             FROM `cloud-training-demos.taxifare_kaggle.test_features`
             """

# training dataset
client = bigquery.Client()
query_job = client.query(QUERY_TRAIN)
train_df = query_job.to_dataframe()

#print(os.listdir('../input'))
#train_df = pd.read_csv('../input/train.csv', nrows = 20_000_000)
print(train_df.dtypes)

# add new features
def add_travel_vector_features(df):
    df['abs_diff_longitude'] = (df.dropoff_longitude - df.pickup_longitude).abs()
    df['abs_diff_latitude'] = (df.dropoff_latitude - df.pickup_latitude).abs()
    
add_travel_vector_features(train_df)

train_df['pickup_datetime'] = train_df['pickup_datetime'].str.slice(0,16)
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'], utc = True, format = '%Y-%m-%d %H:%M')
train_df.dtypes
train_df['year'] = pd.DatetimeIndex(train_df['pickup_datetime']).year
train_df.dtypes
train_df['month'] = pd.DatetimeIndex(train_df['pickup_datetime']).month
train_df.dtypes
train_df['hour'] = pd.DatetimeIndex(train_df['pickup_datetime']).hour
train_df.dtypes

# testing set
query_job = client.query(QUERY_TEST)
test_df = query_job.to_dataframe()
#test_df = pd.read_csv('../input/test.csv')
print(test_df.dtypes)

add_travel_vector_features(test_df)

test_df['pickup_datetime'] = test_df['pickup_datetime'].str.slice(0,16)
test_df['pickup_datetime'] = pd.to_datetime(test_df['pickup_datetime'], utc = True, format = '%Y-%m-%d %H:%M')
test_df.dtypes
test_df['year'] = pd.DatetimeIndex(test_df['pickup_datetime']).year
test_df.dtypes
test_df['month'] = pd.DatetimeIndex(test_df['pickup_datetime']).month
test_df.dtypes
test_df['hour'] = pd.DatetimeIndex(test_df['pickup_datetime']).hour
test_df.dtypes

# dataset preprocessing
print(train_df.isnull().sum())

print('Old size: %d' % len(train_df))
train_df = train_df.dropna(how = 'any', axis = 'rows')
print('New size: %d' % len(train_df))

#plot = train_df.iloc[:2000].plot.scatter('dropoff_longitude', 'dropoff_latitude')
#plot_diff = train_df.iloc[:2000].plot.scatter('abs_diff_longitude', 'abs_diff_latitude')
#plot_yearmonth = train_df.iloc[:2000].plot.scatter('year', 'month')
#plot_monthhour = train_df.iloc[:2000].plot.scatter('month', 'hour')


print('Old size: %d' % len(train_df))
train_df = train_df[(train_df.pickup_longitude < -60) & (train_df.dropoff_longitude < -60) & (train_df.pickup_latitude > 30) & (train_df.dropoff_latitude > 30) & (train_df.abs_diff_longitude < 10) & (train_df.abs_diff_latitude < 10)]
print('New size: %d' % len(train_df))

def get_input_matrix(df):
	return np.column_stack((df.year, df.month, df.hour, df.pickup_longitude, df.pickup_latitude, df.dropoff_longitude, df.dropoff_latitude, df.abs_diff_longitude, df.abs_diff_latitude))

# training set
train_X = get_input_matrix(train_df)	
train_y = np.array(train_df['fare_amount'])
print(train_X.shape)
print(train_y.shape)

# set the parameters by cross-validation
#tuned_parameters = {'max_depth':[17, 18, 19, 20, 21]}
#tuned_parameters = {'n_estimators': [20, 30, 40]}
#clf = GridSearchCV(RandomForestRegressor(max_depth = 20, random_state = 0, n_jobs = -1), tuned_parameters, cv = 5)
#clf.fit(train_X, train_y)
#print(clf.best_params_)
#print(clf.best_score_)
regr_1 = RandomForestRegressor(n_estimators = 30, max_depth = 20, random_state = 0, n_jobs = -1)
regr_1.fit(train_X, train_y)

test_X = get_input_matrix(test_df)
test_y = regr_1.predict(test_X)
#test_y = clf.predict(test_X)

# submission file
submission = pd.DataFrame({'key': test_df.key, 'fare_amount': test_y}, columns = ['key', 'fare_amount'])
submission.to_csv('submission.csv', index = False)

print(os.listdir('.'))
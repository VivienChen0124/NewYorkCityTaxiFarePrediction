import numpy as np
import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Dense

# dataset
#print(os.listdir('../input'))
train_df = pd.read_csv('train.csv')
train_df.dtypes

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
test_df = pd.read_csv('test.csv')
test_df.dtypes

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

# model 
seed = 7
np.random.seed(seed)

model = Sequential()

model.add(Dense(16, input_dim = 9, init = 'uniform', activation = 'relu'))
model.add(Dense(8, init = 'uniform', activation = 'relu'))
model.add(Dense(1, init = 'uniform', activation = 'sigmoid'))

model.compile(loss = 'mean_squared_error', optimizer = 'sgd', metrics = ['accuracy'])

model.fit(train_X, train_y, nb_epoch = 150, batch_size = 10)

test_X = get_input_matrix(test_df)
test_y = model.predict(test_X)

# submission file
submission = pd.DataFrame({'key': test_df.key, 'fare_amount': test_y}, columns = ['key', 'fare_amount'])
submission.to_csv('submission.csv', index = False)

print(os.listdir('.'))
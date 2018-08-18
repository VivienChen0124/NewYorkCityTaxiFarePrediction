import numpy as np
import pandas as pd
import os
from google.cloud import bigquery
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV

QUERY_TRAIN = """
              SELECT fare_amount
              FROM `cloud-training-demos.taxifare_kaggle.train`
              LIMIT 100
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
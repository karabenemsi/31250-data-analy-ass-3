from datetime import datetime
import numpy as np
import pandas as pd


from sklearn import preprocessing
import helpers

pd.set_option('display.width', 0)


def preprocess(train_file, test_file, limit=None, remove_low_variance=True, remove_outliers=True):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    if limit is None:
        limit = len(train_df)
    if 0 < limit < len(train_df):
        print('Limited Sample: ' + str(limit))
        train_df = train_df.sample(n=limit)

    train_df = helpers.parse_data(train_df)
    test_df = helpers.parse_data(test_df)

    # Feature Pre-Selection

    keepColumns = ['QuoteConversion_Flag']
    # Drop Personal_info5, it has lot of empty values
    train_df.drop(columns=['Personal_info5'], inplace=True)
    test_df.drop(columns=['Personal_info5'], inplace=True)
    # Remove Rows with empty values
    train_df.dropna(inplace=True)
    # Fill empty values in test dataset, both are YN-Values, replace with previous value
    test_df.fillna(method='ffill', inplace=True)

    train_df, keepColumns = helpers.categorical_to_many(train_df, ['Geographic_info5'], keepColumns)
    test_df, a = helpers.categorical_to_many(test_df, ['Geographic_info5'], keepColumns)

    if remove_low_variance:
        train_df, removed_columns = helpers.remove_low_variance(train_df, keepColumns)
        test_df.drop(columns=removed_columns, inplace=True)

    print('DataFrame shape after feature selection:' + str(train_df.shape))

    # Detect and Remove outliers
    if remove_outliers:
        train_df = helpers.remove_outliers(train_df)

    print('DataFrame shape after outlier removal:' + str(train_df.shape))
    # Extract dependent variable from dataset

    train_dv = np.array(train_df['QuoteConversion_Flag'])
    train_data = np.array(train_df.drop(columns=['QuoteConversion_Flag']))
    test_data = np.array(test_df)

    # Scale things
    standard_scaler = preprocessing.StandardScaler()
    train_data = standard_scaler.fit_transform(train_data)
    test_data = standard_scaler.fit_transform(test_data)

    # Normalize it to be more gaussian
    train_data = preprocessing.normalize(train_data, return_norm=False)
    test_data = preprocessing.normalize(test_data, return_norm=False)

    return train_dv, train_data, test_data, train_df, test_df

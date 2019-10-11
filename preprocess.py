from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
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

    keepColumns = ['QuoteConversion_Flag']

    # TODO: Do this for all categorical
    train_df, keepColumns = helpers.categorical_to_many(train_df, ['Geographic_info5'],
                                                        keepColumns)
    test_df, a = helpers.categorical_to_many(test_df, ['Geographic_info5'],
                                             keepColumns)

    # Fill up train and test frame to have the same column length
    for key in list(set(train_df.keys()) - set(test_df.keys())):
        test_df.loc[:, key] = pd.Series(np.zeros(len(test_df['Original_Quote_Date'])), index=test_df.index)
    for key in list(set(test_df.keys()) - set(train_df.keys())):
        train_df.loc[:, key] = pd.Series(np.zeros(len(train_df['Original_Quote_Date'])), index=train_df.index)

    # Feature Selection

    # Drop Personal_info5, it has lot of empty values
    train_df.drop(columns=['Personal_info5'], inplace=True)
    test_df.drop(columns=['Personal_info5'], inplace=True)
    # Remove Rows with empty values
    train_df.dropna(inplace=True)
    # Fill empty values in test dataset, both are YN-Values, replace with previous value
    test_df.fillna(method='ffill', inplace=True)

    if remove_low_variance:
        train_df, removed_columns = helpers.remove_low_variance(train_df, keepColumns)
        test_df.drop(columns=removed_columns, inplace=True)

    print('DataFrame shape after feature selection:' + str(train_df.shape))

    # Detect and Remove outliers
    if remove_outliers:
        train_df = helpers.remove_outliers(train_df)

    print('DataFrame shape after outlier removal:' + str(train_df.shape))
    # Extract dependent variable from dataset

    # TODO: Binning

    # Convert to numpy-arrays
    train_dv = np.array(train_df['QuoteConversion_Flag'])
    train_data = np.array(train_df.drop(columns=['QuoteConversion_Flag', 'Quote_ID']))
    test_data = np.array(test_df.drop(columns=['QuoteConversion_Flag', 'Quote_ID']))

    # Extract numeric values to scale them

    # Scale things
    # Only Scale non-boolean
    #  Original_Quote_Date  Field_info1  Field_info3  Coverage_info1  Coverage_info2  Coverage_info3  XSales_info1  Sales_info2  Sales_info3   Sales_info4   Sales_info5  Personal_info2  Personal_info3  Property_info3  XProperty_info4  Property_info5  Geographic_info1  Geographic_info2  Geographic_info3
    #  Original_Quote_Date  Field_info1  Field_info3  Coverage_info1  Coverage_info2  Coverage_info3  XSales_info1  Sales_info2  Sales_info3                 Sales_info5  Personal_info2  Property_info3                  XProperty_info4  Property_info5  Geographic_info1  Geographic_info2  Geographic_info3
    scale_indices = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 12, 15, 16, 17, 18]
    # scale_indices = range(0, len(train_data[0]))

    train_scale_array = train_data[:, scale_indices]
    test_scale_array = test_data[:,scale_indices]

    print(f'Train: {train_scale_array.shape}')
    print(f'Test : {test_scale_array.shape}')
    standard_scaler = StandardScaler()
    standard_scaler.fit(train_scale_array)
    train_scale_array = standard_scaler.transform(train_scale_array)
    test_scale_array = standard_scaler.transform(test_scale_array)

    # Normalize it to be more gaussian
    train_scale_array = normalize(train_scale_array, return_norm=False, axis=0)
    test_scale_array = normalize(test_scale_array, return_norm=False, axis=0)

    for j in range(0, len(train_data)):
        for index, i in enumerate(scale_indices):
            train_data[j,i] = train_scale_array[j,index]

    for j in range(0, len(test_data)):
        for index, i in enumerate(scale_indices):
            test_data[j,i] = test_scale_array[j,index]


    return train_dv, train_data, test_data, train_df, test_df

from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats


def string_to_bool(string):
    if string == 'Y':
        return 1.0
    if string == 'N':
        return 0.0
    # If its anything else return none
    return None


def string_to_value(string):
    if string == '':
        return None
    if len(string) == 1:
        # return ord(string) - 64
        return ord(string)
    value = 0
    for index, char in enumerate(string):
        value += ord(char)
        # value += (ord(char) - 64) * pow(10, index)
    return value


def format_amount(string):
    return int(string.replace(',', ''))


def str_to_timestamp(date_string):
    return int(datetime.strptime(date_string.zfill(10), '%d/%m/%Y').timestamp())


def parse_data(df):
    # Replace -1 with None as it represents an empty value
    # data.replace(-1, inplace=True)
    # print(data.describe())

    # Convert Date
    df['Original_Quote_Date'] = df['Original_Quote_Date'].apply(str_to_timestamp)

    # Convert bool-values to int of 1 and 0
    df['Field_info4'] = df['Field_info4'].apply(string_to_bool)
    df['Personal_info1'] = df['Personal_info1'].apply(string_to_bool)
    df['Property_info1'] = df['Property_info1'].apply(string_to_bool)
    df['Geographic_info4'] = df['Geographic_info4'].apply(string_to_bool)

    # Convert string to int values
    df['Field_info1'] = df['Field_info1'].apply(string_to_value)
    df['Coverage_info3'] = df['Coverage_info3'].apply(string_to_value)
    df['Sales_info4'] = df['Sales_info4'].apply(string_to_value)
    df['Personal_info3'] = df['Personal_info3'].apply(string_to_value)
    df['Property_info3'] = df['Property_info3'].apply(string_to_value)

    # Convert special amount to int
    df['Field_info3'] = df['Field_info3'].apply(format_amount)
    return df


def categorical_to_many(df, columns, keep_columns=None):
    # Change Categorical
    if keep_columns is None:
        keep_columns = []
    dummies = dict()
    for col in columns:
        dummies[col] = pd.get_dummies(df[col])
    for dum in dummies:
        # Keep generated columns as they might include lots of empty(same) values
        keep_columns = keep_columns + list(dummies[dum].keys())
        df.drop(columns=[dum], inplace=True)
        df = pd.concat([df, dummies[dum]], axis=1)
    return df, keep_columns


def remove_low_variance(df, keep_columns=None):
    # Remove features with low variance
    if keep_columns is None:
        keep_columns = []
    remove = []
    for col in df:
        if col not in keep_columns:
            var = df.loc[:, col].var()
            # If variance is really low remember for removal
            if var < (.8 * (1 - .8)):
                remove.append(col)
                print('Remove ' + col + ' with variance of ' + str(var))

    # Drop all features with low variance
    return df.drop(columns=remove), remove


def remove_outliers(df):
    # Calculate z-score and store in numpy-array
    z = np.abs(stats.zscore(df))
    # Remove axis where z-score is lower than 3 (p=0.0013 of being in normal distribution)
    return df[(z < 3).all(axis=1)]


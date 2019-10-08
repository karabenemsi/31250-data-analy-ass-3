from datetime import datetime
import numpy as np
import pandas as pd
from sklearn import preprocessing

pd.set_option('display.width', 0)


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
        return ord(string) - 65
    value = 0
    for index, char in enumerate(string):
        value += (ord(char)-65) * pow(10, index)
    return value

def format_amount(string):
    return int(''.join(string.split(',')))



def str_to_timestamp(x):
    return int(datetime.strptime(x.zfill(10), '%d/%m/%Y').timestamp())


data = pd.read_csv('TrainingSet.csv')
print(data.shape)


# Convert Date
data['Original_Quote_Date'] = data['Original_Quote_Date'].apply(str_to_timestamp)

# Convert bool-values to int of 1 and 0
data['Field_info4'] = data['Field_info4'].apply(string_to_bool)
data['Personal_info1'] = data['Personal_info1'].apply(string_to_bool)
data['Property_info1'] = data['Property_info1'].apply(string_to_bool)
data['Geographic_info4'] = data['Geographic_info4'].apply(string_to_bool)

# Convert string to int values
data['Field_info1'] = data['Field_info1'].apply(string_to_value)
data['Coverage_info3'] = data['Coverage_info3'].apply(string_to_value)
data['Sales_info4'] = data['Sales_info4'].apply(string_to_value)
data['Personal_info3'] = data['Personal_info3'].apply(string_to_value)
data['Property_info3'] = data['Property_info3'].apply(string_to_value)

#Convert special amount to int
data['Field_info3'] = data['Field_info3'].apply(format_amount)

# Find empty values
print(data.isnull().sum())

# Feature Pre-Selection

keepColumns = ['QuoteConversion_Flag']
# Drop Personal_Info5 and Geographic_info3 as most of it is empty anyway
data.drop(columns=['Personal_info5', 'Geographic_info3'], inplace=True)
# Remove Rows with empty values
data.dropna(inplace=True)

# Change Categorical to num
dummies = dict()
dummies['Geographic_info5'] = pd.get_dummies(data['Geographic_info5'])
for dum in dummies:
    # Keep generated columns as they might include lots of empty(same) values
    keepColumns = keepColumns + list(dummies[dum].keys())
    data.drop(columns=[dum], inplace=True)
    data = pd.concat([data, dummies[dum]], axis=1)

print(data.describe())

# Remove rows with low variance
remove = []
print(keepColumns)
for col in data:
    if col not in keepColumns:
        print(col)
        var = data.loc[:, col].var()
        # If variance is really low remember for removal
        if var < (.8 * (1 - .8)):
            remove.append(col)
            print('Remove ' + col + ' with variance of ' + str(var))

# Drop all rows with low variance
# data.drop(columns=remove, inplace=True)

print(data.shape)



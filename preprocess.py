import csv
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold

pd.set_option('display.width', 0)


def stringtobool(string):
    if string == 'Y':
        return 1
    if string == 'N':
        return 0
    if string == '1':
        return 1
    if string == '0':
        return 0
    # If its anything else return none
    return None


def stringValue(string):
    if string == '':
        return None

    if len(string) == 1:
        return ord(string) - 65

    value = 0
    for index, char in enumerate(string):
        value += (ord(char)-65) * pow(10, index)
    return value


rawData = []

with open('TrainingSet.csv', 'r') as csvfile:
    trainingSet = csv.DictReader(csvfile)
    i = 0
    for row in trainingSet:
        # rawData.append(row)

        if i < 20:
            rawData.append(row)
            i += 1

print(len(rawData))
# Convert Dates into python datetime and other basic parsing
for row in rawData:
    row['Quote_ID'] = int(row['Quote_ID'])
    row['Original_Quote_Date'] = int(datetime.strptime(row['Original_Quote_Date'].zfill(10), '%d/%m/%Y').timestamp())
    row['QuoteConversion_Flag'] = stringtobool(row['QuoteConversion_Flag'])
    row['Field_info1'] = stringValue(row['Field_info1'])
    row['Field_info2'] = float(row['Field_info2']) if row['Field_info2'] != '' else None
    # Remove thousand indicator
    row['Field_info3'] = row['Field_info3'].replace(',', '')
    row['Field_info3'] = int(row['Field_info3']) if row['Field_info3'] != '' else None
    row['Field_info4'] = stringtobool(row['Field_info4'])

    row['Coverage_info1'] = int(row['Coverage_info1']) if row['Coverage_info1'] != '' else None
    row['Coverage_info2'] = int(row['Coverage_info2']) if row['Coverage_info2'] != '' else None
    row['Coverage_info3'] = stringValue(row['Coverage_info3'])

    row['Sales_info1'] = stringtobool(row['Sales_info1'])
    row['Sales_info2'] = int(row['Sales_info2']) if row['Sales_info2'] != '' else None
    row['Sales_info3'] = int(row['Sales_info2']) if row['Sales_info2'] != '' else None
    row['Sales_info4'] = stringValue(row['Sales_info4'])
    row['Sales_info5'] = int(row['Sales_info5']) if row['Sales_info5'] != '' else None

    row['Personal_info1'] = stringtobool(row['Personal_info1'])
    row['Personal_info2'] = int(row['Personal_info2']) if row['Personal_info2'] != '' else None
    row['Personal_info3'] = stringValue(row['Personal_info3'])
    row['Personal_info4'] = stringtobool(row['Personal_info4'])
    row['Personal_info5'] = int(row['Personal_info5']) if row['Personal_info5'] != '' else None

    row['Property_info1'] = stringtobool(row['Property_info1'])
    row['Property_info2'] = stringtobool(row['Property_info2'])
    row['Property_info3'] = stringValue(row['Property_info3'])
    row['Property_info4'] = stringtobool(row['Property_info4'])
    row['Property_info5'] = int(row['Property_info5']) if row['Property_info5'] != '' else None

    row['Geographic_info1'] = int(row['Geographic_info1']) if row['Geographic_info1'] != '' else None
    row['Geographic_info2'] = int(row['Geographic_info2']) if row['Geographic_info2'] != '' else None
    row['Geographic_info3'] = int(row['Geographic_info3']) if row['Geographic_info3'] != '' else None
    row['Geographic_info4'] = stringtobool(row['Geographic_info4'])
    # row['Geographic_info5'] = stringValue(row['Geographic_info5'])

# Find empty values
count = dict()
countNone = dict()
for row in rawData:
    if -1 in row.values():
        for item in row.items():
            if item[1] is -1:
                count[item[0]] = count[item[0]] + 1 if item[0] in count else 1
    if None in row.values():
        for item in row.items():
            if item[1] is None:
                countNone[item[0]] = countNone[item[0]] + 1 if item[0] in countNone else 1

print('-1 in: ' + str(count))
print('None in: ' + str(countNone))

data = pd.DataFrame(rawData)
# print(data.isnull().sum())

# Feature Selection

# Drop Personal_Info5 and Geographic_info3 as most of it is empty anyway
data.drop(columns=['Personal_info5', 'Geographic_info3'], inplace=True)
# Remove Rows with empty values
data.dropna(inplace=True)

# Remove rows with low variance
remove = []
for col in data:
    var = data.loc[:, col].var()
    # If variance is really low remember for removal it (except the Quote flag)
    if var < (.8 * (1 - .8))and col != 'QuoteConversion_Flag':
        remove.append(col)
        print('Remove ' + col + ' with variance of ' + str(var))

# Drop all rows with low variance
data.drop(columns=remove, inplace=True)






print(data.shape)



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
        return ord(string)

    value = 0
    for index, char in enumerate(string):
        value += ord(char) * pow(10, index)
    return value


rawData = []

with open('TrainingSet.csv', 'r') as csvfile:
    trainingSet = csv.DictReader(csvfile)
    i = 0
    for row in trainingSet:
        rawData.append(row)

        # if i < 50:
        #     rawData.append(row)
        #     i += 1

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
    row['Geographic_info5'] = stringValue(row['Geographic_info5'])

# Find empty values
count = dict()
for row in rawData:
    if -1 in row.values():
        for item in row.items():
            if item[1] is -1:
                count[item[0]] = count[item[0]] + 1 if item[0] in count else 1

print('-1 in: ' + str(count))

data = pd.DataFrame(rawData)

# Drop Personal_Info5 as most of it is empty anyway
data = data.drop(columns=['Personal_info5', 'Geographic_info3'])
# Remove Rows with empty values
data = data.dropna()

# Feature Selection
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
data = sel.fit_transform(data)


print(data.shape)

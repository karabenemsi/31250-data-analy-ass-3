import preprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score


def smote_train_model(classifier_model, x, y):
    # Use SMOTE to oversample the dataset for better training accuracy
    sm = SMOTE()
    x_train_oversampled, y_train_oversampled = sm.fit_sample(x, y)

    # Fit and predict
    classifier_model.fit(x_train_oversampled, y_train_oversampled)
    return classifier_model


def decide_for_unsure(df, name):
    df['Sum'] = df.sum(axis=1)
    final = []
    width = len(df.keys()) - 1
    print(df.keys())
    count = []
    # init counter
    for i in range(0, width + 1):
        count.append(0)
    for row in df['Sum']:
        if 0 < row < width:
            # Do +1 as most of the data points tend to be 0 rather than 1
            final.append(1 if row > ((width / 2) + 1) else 0)
            count[row] += 1
        else:
            count[row] += 1
            final.append(0 if row == 0 else 1)
    df['Final'] = final
    for i in range(0, width + 1):
        print(str(count[i]) + ' times ' + str(i) + ' on ' + name)
    print(str(sum(count[1:-1]) / len(df) * 100) + '% Unsure on ' + name)
    return df


def save_to_file(df, prediction, suffix):
    df['QuoteConversion_Flag'] = pd.Series(prediction, index=df.index)

    toDrop = []
    for col in df.columns:
        if col not in ['Quote_ID', 'QuoteConversion_Flag']:
            toDrop.append(col)
    df.drop(columns=toDrop, inplace=True)
    df.to_csv(f'./results/Kaggle_Submission{suffix}.csv', index=False)
    print('Written to file')


# Get Preprocessed Data
train_target, train_data, test_data, train_df, test_df = preprocess.preprocess(
    "TrainingSet.csv", 'TestSet.csv', limit=None, remove_low_variance=True, remove_outliers=True)
X_g_train, X_g_test, y_g_train, y_g_test = train_test_split(train_data, train_target, test_size=0.30)
print(f'TrainSet has {train_target.sum()} times 1')

# Init some variables for later use
result_predict = dict()
test_predict = dict()

# Random Forest
for i in range(0,100):
    print('Start Random Forest No.' + str(i))
    model = RandomForestClassifier(n_estimators=185, criterion='entropy', n_jobs=-1)
    model = smote_train_model(model, X_g_train, y_g_train)
    y_predict = model.predict(X_g_test)
    print(roc_auc_score(y_g_test, y_predict))

    result_predict['RandomForest_' + str(i)] = np.array(model.predict(test_data))
    test_predict['RandomForest_' + str(i)] = np.array(model.predict(X_g_test))

result_df = pd.DataFrame(result_predict)
t_df = pd.DataFrame(test_predict)

rf_df = decide_for_unsure(result_df, 'TestSet')
rf2_df = decide_for_unsure(t_df, 'TrainSet')

print(roc_auc_score(y_g_test, rf2_df['Final']))
save_to_file(test_df, rf_df['Final'], '_rf')

# SVM
print('SVM')
model = SVC(gamma='auto', kernel='rbf')
model = smote_train_model(model, X_g_train, y_g_train)
y_predict = model.predict(X_g_test)
print(roc_auc_score(y_g_test, y_predict))

result_predict['SVM'] = np.array(model.predict(test_data))
test_predict['SVM'] = np.array(model.predict(X_g_test))
save_to_file(test_df, result_predict['SVM'], '_svm')

# Neural Network
print('Start MLPClassifier')
model = MLPClassifier(solver='adam', alpha=0.0001, learning_rate_init=0.001,
                      hidden_layer_sizes=(17, 11), max_iter=1000, warm_start=True)
model = smote_train_model(model, X_g_train, y_g_train)
y_predict = model.predict(X_g_test)
print(roc_auc_score(y_g_test, y_predict))

result_predict['mlpNetwork'] = np.array(model.predict(test_data))
test_predict['mlpNetwork'] = np.array(model.predict(X_g_test))
save_to_file(test_df, result_predict['mlpNetwork'], '_mlp')

# Do stuff with unsure rows

result_df = pd.DataFrame(result_predict)
result_df = decide_for_unsure(result_df, 'TestSet')

print('\n Test:')
# Do it for test to
t_df = pd.DataFrame(test_predict)
t_df = decide_for_unsure(t_df, 'TrainingSet')
print(roc_auc_score(y_g_test, t_df['Final']))

# Save Result to file
save_to_file(test_df, result_df['Final'], '_all')

import preprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score


def k_fold_model(classifier_model, x, y):
    cv = StratifiedKFold(n_splits=5)
    for train_idx, test_idx, in cv.split(x, y):
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]

        # Use SMOTE to oversample the dataset for better training accuracy
        sm = SMOTE()
        x_train_oversampled, y_train_oversampled = sm.fit_sample(x_train, y_train)

        # Fit and predict
        classifier_model.fit(x_train_oversampled, y_train_oversampled)
        y_pred = classifier_model.predict(x_test)

        print(f'auc: {roc_auc_score(y_test, y_pred)}')
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


# Get Preprocessed Data
train_target, train_data, test_data, train_df, test_df = preprocess.preprocess(
    "TrainingSet.csv", 'TestSet.csv', limit=None, remove_low_variance=True, remove_outliers=True)
X_g_train, X_g_test, y_g_train, y_g_test = train_test_split(train_data, train_target, test_size=0.30)
print(f'TrainSet has {train_target.sum()} times 1')

# Init some variables for later use
result_predict = dict()
test_predict = dict()

# Random Forest
print('Start Random Forest')
model = RandomForestClassifier(n_estimators=98, criterion='entropy', n_jobs=-1, warm_start=True)
#model = k_fold_model(model, train_data, train_target)

cv = StratifiedKFold(n_splits=5)
for train_idx, test_idx, in cv.split(X_g_train, y_g_train):
    x_train, y_train = X_g_train[train_idx], y_g_train[train_idx]
    x_test, y_test = X_g_train[test_idx], y_g_train[test_idx]

    # Use SMOTE to oversample the dataset for better training accuracy
    sm = SMOTE()
    x_train_oversampled, y_train_oversampled = sm.fit_sample(x_train, y_train)

    # Fit and predict
    model.fit(x_train_oversampled, y_train_oversampled)
    y_pred = model.predict(x_test)
    esti = model.get_params()['n_estimators']
    model.set_params(**{'n_estimators':esti + 98})

    print(f'auc: {roc_auc_score(y_test, y_pred)}')


y_predict = model.predict(X_g_test)
print(roc_auc_score(y_g_test, y_predict))

result_predict['RandomForest'] = np.array(model.predict(test_data))
test_predict['RandomForest'] = np.array(model.predict(X_g_test))

# SVM
print('SVM')
model = SVC(gamma='auto', kernel='rbf')
model = k_fold_model(model, X_g_train, y_g_train)
y_predict = model.predict(X_g_test)
print(roc_auc_score(y_g_test, y_predict))

result_predict['SVM'] = np.array(model.predict(test_data))
test_predict['SVM'] = np.array(model.predict(X_g_test))

# Neural Network
print('Start MLPClassifier')
model = MLPClassifier(solver='adam', alpha=0.0001, learning_rate_init=0.001,
                      hidden_layer_sizes=(100), max_iter=1000, warm_start=True)
model = k_fold_model(model, X_g_train, y_g_train)
y_predict = model.predict(X_g_test)
print(roc_auc_score(y_g_test, y_predict))

result_predict['mlpNetwork'] = np.array(model.predict(test_data))
test_predict['mlpNetwork'] = np.array(model.predict(X_g_test))

# Do stuff with unsure rows

result_df = pd.DataFrame(result_predict)
result_df = decide_for_unsure(result_df, 'TestSet')

print('\n Test:')
# Do it for test to
t_df = pd.DataFrame(test_predict)
t_df = decide_for_unsure(t_df, 'TrainingSet')
print(roc_auc_score(y_g_test, t_df['Final']))

# Save Result to filen
test_df['QuoteConversion_Flag'] = pd.Series(result_df['Final'], index=test_df.index)

toDrop = []
for col in test_df.columns:
    if col not in ['Quote_ID', 'QuoteConversion_Flag']:
        toDrop.append(col)
test_df.drop(columns=toDrop, inplace=True)
test_df.to_csv('Kaggle_Submission.csv', index=False)
print('Written to file')

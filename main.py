import preprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV


def kFoldModel(model, X, y):
    cv = StratifiedKFold(n_splits=5)
    for train_idx, test_idx, in cv.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Use SMOTE to oversample the dataset for better training accuracy
        sm = SMOTE()
        X_train_oversampled, y_train_oversampled = sm.fit_sample(X_train, y_train)

        # Fit and predict
        model.fit(X_train_oversampled, y_train_oversampled)
        y_pred = model.predict(X_test)

        print(f'auc: {roc_auc_score(y_test, y_pred)}')
    return model



## Get Preprocessed Data
train_target, train_data, test_data, train_df, test_df = preprocess.preprocess(
    "TrainingSet.csv", 'TestSet.csv', limit=6000, remove_low_variance=True, remove_outliers=True)
X_g_train, X_g_test, y_g_train, y_g_test = train_test_split(train_data, train_target, test_size=0.30)
print(f'Trainset has {train_target.sum()} times 1')

## Init some variables for later use
result_predict = dict()
test_predict = dict()


## Random Forest
print('Start Random Forest')
model = RandomForestClassifier(n_estimators=128, criterion='entropy', n_jobs=-1)
model = kFoldModel(model, train_data, train_target)
y_predict = model.predict(X_g_test)
print(roc_auc_score(y_g_test, y_predict))

result_predict['RandomForest'] = np.array(model.predict(test_data))
test_predict['RandomForest'] = np.array(model.predict(X_g_test))

## k-nearest neighbour
print('Start k-nearest neighbour')
model = KNeighborsClassifier(n_neighbors=20, weights='uniform', n_jobs=-1)
model = kFoldModel(model, train_data, train_target)
y_predict = model.predict(X_g_test)
print(roc_auc_score(y_g_test, y_predict))

result_predict['KNeighbors'] = np.array(model.predict(test_data))
test_predict['KNeighbors'] = np.array(model.predict(X_g_test))

## SVM
print('SVM')
model = SVC(gamma='auto', kernel='rbf')
model = kFoldModel(model, train_data, train_target)
y_predict = model.predict(X_g_test)
print(roc_auc_score(y_g_test, y_predict))

result_predict['SVM'] = np.array(model.predict(test_data))
test_predict['SVM'] = np.array(model.predict(X_g_test))



## Neural Network
print('Start MLPClassifier')
model = MLPClassifier(solver='adam', alpha=0.001, learning_rate_init=0.001,
                      hidden_layer_sizes=(7, 11), max_iter=1000)
model = kFoldModel(model, train_data, train_target)
y_predict = model.predict(X_g_test)
print(roc_auc_score(y_g_test, y_predict))

result_predict['mlpNetwork'] = np.array(model.predict(test_data))
test_predict['mlpNetwork'] = np.array(model.predict(X_g_test))

## Do stuff with unsure rows

result_df = pd.DataFrame(result_predict)
result_df['Sum'] = result_df.sum(axis=1)

count = 0
final = []
width = len(result_df.keys()) - 1
for row in result_df['Sum']:
    if 0 < row < width:
        final.append(1 if row > (width / 2) else 0)
        count += 1
    else:
        final.append(0 if row == 0 else 1)
result_df['Final'] = final
print(str(count / len(result_df) * 100) + '% Unsure on TestSet')

# Do it for test to
t_df = pd.DataFrame(test_predict)
t_df['Sum'] = t_df.sum(axis=1)

count = 0
final = []
width = len(t_df.keys()) - 1
for row in t_df['Sum']:
    if 0 < row < width:
        final.append(1 if row > (width / 2) else 0)
        count += 1
    else:
        final.append(0 if row == 0 else 1)
t_df['Final'] = final
print(str(count / len(t_df) * 100) + '% Unsure on TrainTest')

print(f'AUC: {roc_auc_score(y_g_test, list(t_df["Final"]))}')

### Save Result to file
test_df['QuoteConversion_Flag'] = pd.Series(result_df['Final'], index=test_df.index)

todrop = []
for col in test_df.columns:
    if col not in ['Quote_ID', 'QuoteConversion_Flag']:
        todrop.append(col)
test_df.drop(columns=todrop, inplace=True)
test_df.to_csv('Kaggle_Submission.csv', index=False)
print('Written to file')

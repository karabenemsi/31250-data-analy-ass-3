import preprocess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


train_target, train_data, test_data, column_names = preprocess.preprocess("TrainingSet.csv", 'TestSet.csv', limit=80)

X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size=0.30)

# Random Forest
model = RandomForestClassifier(n_estimators=100, criterion='entropy', n_jobs=-1, warm_start=True)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

print(accuracy_score(y_test, y_predict))
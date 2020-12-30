from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split,KFold, StratifiedKFold
import numpy as np

digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.3)

# logreg = LogisticRegression()
# logreg.fit(X_train, y_train)
# lr_score = logreg.score(X_test, y_test)
# print("LR",lr_score)

# svm = SVC()
# svm.fit(X_train, y_train)
# svm_score = svm.score(X_test, y_test)
# print("SVM",svm_score)

# rf = RandomForestClassifier(n_estimators=40)
# rf.fit(X_train,y_train)
# rf_score = rf.score(X_test, y_test)
# print("RF",rf_score)

kf= KFold(n_splits=3)
for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9,10]):
    #print(train_index)
    #print(test_index)
    pass

def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

# generic_scorer= get_score(LogisticRegression(), X_train, X_test, y_train, y_test)
# print(generic_scorer)

skf = StratifiedKFold(n_splits=3)

scores_l = []
scores_svm = []
scores_rf = []

# for train_index, test_index in kf.split(digits.data):
#     X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index], digits.target[train_index], digits.target[test_index]
#     scores_l.append(get_score(LogisticRegression(), X_train, X_test, y_train, y_test))
#     scores_svm.append(get_score(SVC(), X_train, X_test, y_train, y_test))
#     scores_rf.append(get_score(RandomForestClassifier(), X_train, X_test, y_train, y_test))

for train_index, test_index in skf.split(digits.data, digits.target):
    X_train, X_test, y_train, y_test = digits.data[train_index], digits.data[test_index], digits.target[train_index], digits.target[test_index]
    scores_l.append(get_score(LogisticRegression(max_iter=10000), X_train, X_test, y_train, y_test))
    scores_svm.append(get_score(SVC(), X_train, X_test, y_train, y_test))
    scores_rf.append(get_score(RandomForestClassifier(n_estimators=40), X_train, X_test, y_train, y_test))


print(scores_l)
print(scores_svm)
print(scores_rf)
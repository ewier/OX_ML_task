from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     score = clf.score(X_test, y_test)
#     return score, y_pred

def get_cross_val_score(clf, X_t, y_t, cv=5):
    scores = cross_val_score(clf, X_t, y_t, cv=cv)
    return scores.mean(), scores.std()

def dummy(X_train, X_test, y_train, y_test):
    dummy_clf = DummyClassifier(strategy="most_frequent")
    # score, y_pred = train_and_evaluate(dummy_clf, X_train, X_test, y_train, y_test)
    mean, std = get_cross_val_score(dummy_clf, X_train, y_train)
    print(f'Dummy Classifier cross-validation score:\nMean                  : {mean}\nStandard Deviation    : {std}\n')

def logistic_regression(X_train, X_test, y_train, y_test):
    lr_clf = LogisticRegression(random_state=0)
    # score, y_pred = train_and_evaluate(lr_clf, X_train, X_test, y_train, y_test)
    mean, std = get_cross_val_score(lr_clf, X_train, y_train)
    print(f'Logistic regression cross-validation score:\nMean               : {mean}\nStandard Deviation : {std}\n')

def run_baselines(X_train, X_test, y_train, y_test):
    dummy(X_train, X_test, y_train, y_test)
    logistic_regression(X_train, X_test, y_train, y_test)
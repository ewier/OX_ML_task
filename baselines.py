from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def get_cross_val_score(clf, X_t, y_t, cv=5):
    scores = cross_val_score(clf, X_t, y_t, cv=cv)
    return scores.mean(), scores.std()

def dummy(X_train, y_train):
    '''
    The first baseline model - dummy classifier.
    
    '''
    dummy_clf = DummyClassifier(strategy="most_frequent")
    mean, std = get_cross_val_score(dummy_clf, X_train, y_train)
    print(f'Dummy Classifier cross-validation score:\nMean                  : {mean}\nStandard Deviation    : {std}\n')

def logistic_regression(X_train, y_train):
    '''
    The second baseline model - logistic regression.

    Note:
    As mentioned in the readme file, the algorithm doesn't converge. 
    This can be corrected by increasing the maximum number of iterations or changing the solver function.

    '''
    lr_clf = LogisticRegression(random_state=0)
    mean, std = get_cross_val_score(lr_clf, X_train, y_train)
    print(f'Logistic regression cross-validation score:\nMean               : {mean}\nStandard Deviation : {std}\n')

def run_baselines(X_train, y_train):
    dummy(X_train, y_train)
    logistic_regression(X_train, y_train)
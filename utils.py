from pandas import read_csv
from numpy import array as np_array
from sklearn.model_selection import train_test_split

def translate_y(y):
    y = [x-1 for x in y]
    return y

def get_data_from_file():
    covtype_df = read_csv(r'./covtype.data', header=None)
    y = covtype_df[:][54]
    X = covtype_df.drop(columns=[54])
    y = translate_y(y)
    return np_array(X), np_array(y)

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

def trim_data(X_train, X_test, y_train, y_test, train_size=5000, test_size=1000):
    X_train, X_test, y_train, y_test = X_train[:train_size], X_test[:test_size], y_train[:train_size], y_test[:test_size]
    return X_train, X_test, y_train, y_test

def get_data():
    X, y = get_data_from_file()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train, X_test, y_train, y_test = trim_data(X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test


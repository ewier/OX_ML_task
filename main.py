from knn import run_knn
from utils import get_data
from baselines import run_baselines
from neural_network import run_example


def main(train_size=None, test_size=None):
    '''
    Run the solution. The result will be printed in terminal.

    To use the whole dataset, set:
    train_size, test_size = None, None
    
    '''

    print(" ------------- DATA ------------ ")
    X_train, X_test, y_train, y_test = get_data(train_size, test_size)

    print(" ------------- KNN ------------- ")
    run_knn(X_train, X_test, y_train, y_test)

    print(" ---------- BASELINES ---------- ")
    run_baselines(X_train, y_train)

    print(" ------- NEURAL  NETWORK ------- ")
    run_example(X_train, X_test, y_train, y_test)

# train_size, test_size = None, None
train_size, test_size = 5000, 1000
main(train_size, test_size)
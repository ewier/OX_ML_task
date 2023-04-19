from knn import run_knn
from utils import get_data
from baselines import run_baselines
from neural_network import run_example


def main():
    print(" ------------- DATA ------------ ")
    X_train, X_test, y_train, y_test = get_data()

    print(" ------------- KNN ------------- ")
    run_knn(X_train, X_test, y_train, y_test)

    print(" ---------- BASELINES ---------- ")
    run_baselines(X_train, X_test, y_train, y_test)

    print(" ------- NEURAL  NETWORK ------- ")
    run_example(X_train, X_test, y_train, y_test)

main()
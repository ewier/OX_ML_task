from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


def get_model():
    shape = (54,)
    model = Sequential([
        Dense(54, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(7, activation='softmax')
        
    ])

    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

def run_example(X_train, X_test, y_train, y_test):
    '''
    Neural network example model - no hyperparameters were adjusted

    '''
    model = get_model()
    model.fit(X_train, y_train, epochs=120, verbose=0)
    baseline_eval = model.evaluate(X_test, y_test, return_dict=True, verbose=0)
    loss, accuracy = baseline_eval['loss'], baseline_eval['accuracy']
    print(f'Neural network evaluation:\nLoss     : {loss}\nAccuracy : {accuracy}\n')

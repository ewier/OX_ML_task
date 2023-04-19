from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


def get_model(loss_fun='sparse_categorical_crossentropy', metrics=['accuracy'], layer_num=3, layer_activations=['relu','relu','softmax']):
    shape = (54,)
    model = Sequential([
        Dense(54, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(7, activation='softmax')
        
    ])

    model.compile(loss = loss_fun, optimizer = 'adam', metrics = metrics)
    return model

def run_example(X_train, X_test, y_train, y_test):
    model = get_model()
    model.fit(X_train, y_train, epochs=120, verbose=0)
    baseline_eval = model.evaluate(X_test, y_test, return_dict=True, verbose=0)
    loss, accuracy = baseline_eval['loss'], baseline_eval['accuracy']
    print(f'Neural network evaluation:\nLoss     : {loss}\nAccuracy : {accuracy}\n')

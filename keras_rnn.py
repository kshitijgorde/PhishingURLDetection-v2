import matplotlib.pyplot as plt
import numpy as np
import time
import csv
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential



def build_model():
    model = Sequential()
    layers = [1, 50, 100, 1]

    model.add(LSTM(
        layers[1],
        input_shape=(None, layers[0]),
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        layers[3], activation="sigmoid"))
    # model.add(Activation("sigmoid"))

    start = time.time()
    # model.compile(loss="mse", optimizer="rmsprop")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print( "Compilation Time : ", time.time() - start)
    return model


def run_network(model=None, data=None):

    global_start_time = time.time()
    epochs = 1
    ratio = 0.5
    sequence_length = 12
    #path_to_dataset = 'Features_T10_Outresponse-Bofa.csv'
    dataset = csv.reader(open('Features_T10_Outresponse-Bofa.csv'), delimiter=",")
    next(dataset)
    dataset = list(dataset)
    # split into input and output variables
    dataset = np.array(dataset)
    X = dataset[:,1:12].astype(float)
    Y = dataset[:,12].astype(float)
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    # labelencoder_y_1 = LabelEncoder()
    # y = labelencoder_y_1.fit_transform(Y)
    from keras.utils.np_utils import to_categorical
    y = to_categorical(Y)
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size=0.20)
    X_Train = np.reshape(X_Train, (X_Train.shape[0], X_Train.shape[1], 1))
    X_Test = np.reshape(X_Test, (X_Test.shape[0], X_Test.shape[1], 1))
    print( '\nData Loaded. Compiling...\n')

    if model is None:
        model = build_model()

    try:
        model.fit(
            X_Train, Y_Train,
            batch_size=100, nb_epoch=epochs, validation_split=0.05)
        predicted = model.predict(X_Test)
        predictions = [float(round(x)) for x in predicted]
        import collections
        print(collections.Counter(predictions))
        print(predicted)


    except KeyboardInterrupt:
        print( 'Training duration (s) : ', time.time() - global_start_time)
        return model, Y_Test, 0

    try:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(Y_Test[:100])
        plt.plot(predicted[:100])
        #plt.show()
    except Exception as e:
        print( str(e))
    print( 'Training duration (s) : ', time.time() - global_start_time)
    from sklearn.metrics import f1_score
    f1 = f1_score(Y_Test,predicted,pos_label='1')
    print(f1)
    return model, Y_Test, predicted


if __name__ == '__main__':
    run_network()


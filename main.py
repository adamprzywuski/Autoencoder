from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import numpy as np

database=load_digits()

def zad1():
    plt.gray()
    # print(database)
    print("The database has " + str(len(database.images)))
    print(database["target"])

    x = random.randint(0, 1000)
    print(x)
    for a in range(0, 10):

        b = 0
        while (database.target[x + a + b] != a):
            b = b + 1
        plt.matshow(database.images[x + a])
        print(database.target[x + a + b])

    # plt.show()
    print("The pictures are 8x8 pixels in the square model")
    print("Printing histogram")
    array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for a in range(0, 1797):
        array[database.target[a]] = array[database.target[a]] + 1
    plt.hist(database.target)
    plt.show()

def zad2():



    x_train,x_test,y_train,y_test=train_test_split(database.data,database.target,test_size=0.2)

    y_train=to_categorical(y_train)
    y_test=to_categorical(y_test)


    input_layer=Input(shape=(64))

    dense_layer=Dense(10,activation='sigmoid')(input_layer)
    net_model=Model(inputs=input_layer,outputs=dense_layer)
    net_model.compile(loss='mse',optimizer=Adam())



    net_model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100)


    print("TESTING")
    loss= net_model.evaluate(x_test, y_test)
    print(loss)



if __name__ == '__main__':
    zad2()

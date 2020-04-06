from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
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
    #print(database["target"])
    print(database.images[1].shape)
    print(database.images[1].size)


    for a in range(0, 10):
        x = random.randint(0, 1000)
        b = 0
        while (database.target[x + a + b] != a):
            b = b + 1
        #plt.matshow(database.images[x + a+b])
        print(database.target[x + a + b])

    # plt.show()
    print("The pictures are 8x8 pixels in the square model")
    print("Printing histogram")
    array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for a in range(0, 1797):
        array[database.target[a]] = array[database.target[a]] + 1
    plt.xticks(range(10))
    plt.hist(database.target)

    plt.show()

def zad2():
    #creating data to input,output,training,testing
    x_train,x_test,y_train,y_test=train_test_split(database.data,database.target,test_size=0.2)
    #to get output to the categorical
    y_train=to_categorical(y_train)
    y_test=to_categorical(y_test)

    #creating input layer
    input_layer=Input(shape=(64))
    #creating output layer
    dense_layer=Dense(10,activation='sigmoid')(input_layer)

    #creating model
    net_model=Model(inputs=input_layer,outputs=dense_layer)
    net_model.compile(loss='mse',optimizer=Adam())


    #training the data
    net_model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=100)

    #Testing data
    print("TESTING")
    loss= net_model.evaluate(x_test, y_test)
    print(loss)

    x=random.randint(0,100)
    prediction=net_model.predict(x_test)
    print(prediction[x])




def displaying_value(array):
    plt.gray()
    buffer=[]
    i=0

    for b in range(0,8):
        arrbuff = []
        for c in range(0,8):


            arrbuff.append(array[i])
            i=i+1
        buffer.append(arrbuff)
    print(buffer)
   # plt.imshow(np.reshape(array (8, 8)), cmap=plt.cm.gray_r)
    plt.matshow(buffer)


def zad3():
    #Preparing data
    x_train,x_test,y_train,y_test=train_test_split(database.data,database.target,test_size=0.2)
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    #Creating input data
    input_layer=Input(shape=(64))
    #Creating others layers
    encoded=Dense(5,activation='relu')(input_layer)
    decoded=Dense(64,activation='sigmoid')(encoded)
    #Creating model
    autoencoder=Model(input_layer,decoded)
    autoencoder.compile(optimizer=Adam(),loss="mse")
    #Training model
    autoencoder.fit(x_train,x_train,validation_data=(x_test,x_test),epochs=100)
    #Testing
    (loss) = autoencoder.evaluate(x_train, x_train, verbose=0)
    print(loss)


    #Printing values of layers
    weights=autoencoder.get_weights()
    for a in weights:
        print(a.shape)



    #Creating encoder
    encoder=Model(inputs=input_layer,outputs=encoded)
    encoder.compile(optimizer=Adam(), loss="mse")


    #Creating embbaings value
    encoded_train = encoder.predict(x_train)
    encoded_test = encoder.predict(x_test)
    print(encoded_train)

    #Creating decoder
    input_layer2=Input(5)
    encoded_layer=Dense(64,activation='relu')(input_layer2)
    decoder=Model(input_layer2,encoded_layer)
    decoder.compile(optimizer=Adam(), loss="mse")
    decoder.fit(encoded_train,x_train,epochs=100)

    predictions_values=decoder.predict(encoded_train)
    x=random.randint(0,100)
    displaying_value(predictions_values[x])
    displaying_value(x_test[x])
    plt.show()
















if __name__ == '__main__':
    print(database.target[100])
    zad3()

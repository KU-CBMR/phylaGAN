from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
#from keras.datasets import mnist
import numpy as np
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

#(x_train, _), (x_test, _) = mnist.load_data()

#x_train = x_train.astype('float32') / 255.
#x_test = x_test.astype('float32') / 255.
#x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
#x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

#x_trainCar =np.load('/home/b15ee015/Documents/MatData/MatCartrain128.npy')
#x_trainReal=np.load('/home/b15ee015/Documents/MatData/MatRealtrain128.npy')
x_train =np.load('/home/b15ee015/Documents/MatData/MatFloorplantrain128.npy')
x_test =np.load('/home/b15ee015/Documents/MatData/MatFloorplanVal128.npy')

input_img = Input(shape=(128, 128, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)    #64*128*128
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)    #32*128*128
x = MaxPooling2D((2, 2), padding='same')(x)                             #32*64*64
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)            #16*64*64
x = MaxPooling2D((2, 2), padding='same')(x)                             #16*32*32
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)             #8*32*32
encoded = MaxPooling2D((2, 2), padding='same')(x)                       #8*16*16

# at this point the representation is (8, 16, 16) i.e. 1024-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)       #8*16*16
x = UpSampling2D((2, 2))(x)                                             #8*32*32
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)             #8*32*32
x = UpSampling2D((2, 2))(x)                                             #16*64*64
x = Conv2D(16, (3, 3), activation='relu',padding='same')(x)             #16*64*64
x = UpSampling2D((2, 2))(x)                                             #16*128*128
x = Conv2D(32, (3, 3), activation='relu',padding='same')(x)             #32*128*128
x = Conv2D(64, (3, 3), activation='relu',padding='same')(x)             #64*128*128
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)    #1*128*128

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

print(autoencoder.summary())

autoencoder.fit(x_train, x_train,
                epochs=2000,
                batch_size=20,
                shuffle=True,
                validation_data=(x_train, x_train),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
                
autoencoder.save_weights('autoencoder.h5')
#autoencoder.load_weights('autoencoder.h5')

decoded_imgs = autoencoder.predict(x_train)

n = 10
plt.figure(figsize=(128, 128))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_train[i].reshape(128, 128))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n+1)
    plt.imshow(decoded_imgs[i].reshape(128, 128))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

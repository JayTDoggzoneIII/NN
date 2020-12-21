# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import imageio

def ans(model, filename, display = True):
    img = imageio.imread(filename)
    img = np.mean(img, 2, dtype = float)
    img = img/255
    if (display):
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img, cmap = plt.cm.binary)
        plt.xlabel(filename)
        img = np.expand_dims(img, 0)
        img = np.expand_dims(img, -1)
        plt.title(f"Number: {np.argmax(model.predict(img))}")
        plt.show()    
    return 

MNIST = tf.keras.datasets.mnist
[x_train, y_train],[x_test, y_test] = MNIST.load_data()


print(x_test.shape, y_test.shape, sep = '\n')

x_train =  x_train/255
x_test = x_test/255

plt.figure(figsize = (10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap = plt.cm.binary)
    plt.xlabel(y_train[i])
plt.show()

model =  tf.keras.models.Sequential(
    
    [#Коволюционный слой определяет прямые
    tf.keras.layers.Conv2D(input_shape=(28, 28, 1), filters=32, kernel_size=(5,5),activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    #А тут коволюционный слой определяет углы между прямыми
    tf.keras.layers.Conv2D(filters=64, kernel_size=(5,5),activation='relu'),
    tf.keras.layers.MaxPool2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)]
)

print(model.summary()) #вывод структуры Нейронной Сети в консоль

model.compile(
    
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

Model = model.fit(x_train.reshape(-1, 28, 28, 1), y_train, batch_size = 32, epochs = 10) #размер батча (32 картинки), после которых будет выполняться корректировка весов

print(model.evaluate(x_test.reshape(-1, 28, 28, 1), y_test))

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5, i+1)
    plt.xticks([])
    plt.yticks([])
    x = np.expand_dims(x_test[i], axis = 0)
    res = model.predict(x.reshape(-1, 28, 28, 1))

    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.title(f"Number: {np.argmax(res)}")
plt.show()

accu_values = Model.history['accuracy']
epochs = range(1,len(accu_values) + 1)
plt.plot(epochs,accu_values,label = 'Метрика качества')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()


accu_values = Model.history['loss']
epochs = range(1,len(accu_values) + 1)
plt.plot(epochs,accu_values,label = 'Функция потерь')
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


for i in range(5):
    filename = f'{i}.png'
    ans(model, filename)

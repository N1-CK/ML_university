from keras import layers, callbacks
from keras.models import Model, load_model
import cv2
import numpy as np
import string
import os
import pickle
import matplotlib.pyplot as plt


symbols = string.ascii_lowercase + "0123456789"
num_symbols = len(symbols)
img_shape = (50, 200, 1)


def create_model():
    img = layers.Input(shape=img_shape)
    conv1 = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(img)
    mp1 = layers.MaxPooling2D(padding='same')(conv1)  # 100x25
    conv2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp1)
    mp2 = layers.MaxPooling2D(padding='same')(conv2)  # 50x13
    conv3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp2)
    bn = layers.BatchNormalization()(conv3)
    mp3 = layers.MaxPooling2D(padding='same')(bn)  # 25x7

    flat = layers.Flatten()(mp3)
    outs = []
    for _ in range(5):
        dens1 = layers.Dense(64, activation='relu')(flat)
        drop = layers.Dropout(0.5)(dens1)
        res = layers.Dense(num_symbols, activation='sigmoid')(drop)

        outs.append(res)

    model = Model(img, outs)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model


def preprocess_data():
    n_samples = len(os.listdir('input/samples/samples'))
    X = np.zeros((n_samples, 50, 200, 1))
    y = np.zeros((5, n_samples, num_symbols))

    for i, pic in enumerate(os.listdir('input/samples/samples')):
        img = cv2.imread(os.path.join('input/samples/samples', pic), cv2.IMREAD_GRAYSCALE)
        pic_target = pic[:-4]
        if len(pic_target) < 6:
            img = img / 255.0
            img = np.reshape(img, (50, 200, 1))
            targs = np.zeros((5, num_symbols))
            for j, l in enumerate(pic_target):
                ind = symbols.find(l)
                targs[j, ind] = 1
            X[i] = img
            y[:, i] = targs

    return X, y


# X, y = preprocess_data()
# X_train, y_train = X[:970], y[:, :970]
# X_test, y_test = X[970:], y[:, 970:]
# model.fit(X_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]], batch_size=32, epochs=30, verbose=1, validation_split=0.2)

def predict(filepath):
    # Проверяем, что файл изображения существует.
    if not os.path.exists(filepath):
        print("Файл изображения не существует.")
        return None

    # Проверяем, что изображение является изображением в градациях серого
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Изображение не является изображением в градациях серого.")
        return None

    # Загружаем модель
    with open('input/model.pkl', 'rb') as f:
        model1 = pickle.load(f)
        f.close

    if img.shape[1] > 200:
        resized_img = cv2.resize(img, (200, 50))
    else:
        resized_img = img

    normalized_img = resized_img / 255.0

    # Делаем предсказание
    res = np.array(model1.predict(normalized_img[np.newaxis, :, :, np.newaxis]))
    ans = np.reshape(res, (5, 36))
    l_ind = []
    probs = []
    for a in ans:
        l_ind.append(np.argmax(a))
    capt = ''
    for l in l_ind:
        capt += symbols[l]

    return capt

print(predict('input/predict/2cegf.png'))
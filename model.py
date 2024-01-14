from keras import layers
from keras.models import Model
import cv2
import numpy as np
import string
import os
import pickle
import mlflow.keras


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


def preprocess_data(filepath):
    n_samples = len(os.listdir(filepath))
    X = np.zeros((n_samples, 50, 200, 1))
    y = np.zeros((5, n_samples, num_symbols))

    for i, pic in enumerate(os.listdir(filepath)):
        img = cv2.imread(os.path.join(filepath, pic), cv2.IMREAD_GRAYSCALE)
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


def train_model(model1, batch_size1, epochs1, verbose1, validation_split1, filepath_captchas1):
    X, y = preprocess_data(filepath_captchas1)
    X_train, y_train = X[:970], y[:, :970]
    X_test, y_test = X[970:], y[:, 970:]
    model1.fit(X_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4]], batch_size=batch_size1,
               epochs=epochs1, verbose=verbose1, validation_split=validation_split1)
    score = model1.evaluate(X_test, [y_test[0], y_test[1], y_test[2], y_test[3], y_test[4]], verbose=verbose1)

    return model1, score


# def predict(filepath):
#     if not os.path.exists(filepath):
#         print("Файл изображения не существует.")
#         return None
#
#     img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         print("Изображение не является изображением в градациях серого.")
#         return None
#
#     with open('input/model.pkl', 'rb') as f:
#         model1 = pickle.load(f)
#         f.close
#
#     if img.shape[1] > 200:
#         resized_img = cv2.resize(img, (200, 50))
#     else:
#         resized_img = img
#
#     normalized_img = resized_img / 255.0
#
#     res = np.array(model1.predict(normalized_img[np.newaxis, :, :, np.newaxis]))
#     ans = np.reshape(res, (5, 36))
#     l_ind = []
#     for a in ans:
#         l_ind.append(np.argmax(a))
#     capt = ''
#     for l in l_ind:
#         capt += symbols[l]
#
#     return capt


if __name__ == "__main__":
    with mlflow.start_run():
        batch_size = 32
        epochs = 30
        verbose = 0.1
        validation_split = 0.2
        filepath_captchas = 'input/samples/samples1'

        model = create_model()
        model, score = train_model(model, batch_size, epochs, verbose, validation_split, filepath_captchas)

        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("verbose", verbose)
        mlflow.log_param("validation_split", validation_split)

        # mlflow.log_metrics("score", score)

        mlflow.keras.log_model(model, "models")
        mlflow.log_artifact('input/model.pkl')
        mlflow.models.signature.set_signature("")
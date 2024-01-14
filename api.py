import base64
import pickle

import cv2
from flask import Flask, request, jsonify
import numpy as np
import string

app1 = Flask(__name__)

symbols = string.ascii_lowercase + "0123456789"
num_symbols = len(symbols)
img_shape = (50, 200, 1)


@app1.route('/api/test_hello', methods=['GET'])
def api_predict2():
    pred = 'hello'
    return jsonify({'prediction': pred})


@app1.route('/api/predict', methods=['POST'])
def predict4():
    try:
        image_data = request.files['image'].read()
        decoded_image = base64.b64decode(image_data)

        img = cv2.imdecode(np.frombuffer(decoded_image, np.uint8), cv2.IMREAD_GRAYSCALE)

        if img is None:
            return jsonify({"error": "Изображение не является изображением в градациях серого."}), 400

        with open('./mlruns/0/2f00cf67c4384f53a158d435cf280fbb/artifacts/model.pkl', 'rb') as f:
            model1 = pickle.load(f)

        if img.shape[1] > 200:
            resized_img = cv2.resize(img, (200, 50))
        else:
            resized_img = img

        normalized_img = resized_img / 255.0

        res = np.array(model1.predict(normalized_img[np.newaxis, :, :, np.newaxis]))
        ans = np.reshape(res, (5, 36))
        l_ind = []
        for a in ans:
            l_ind.append(np.argmax(a))
        capt = ''
        for l in l_ind:
            capt += symbols[l]

        return jsonify({"prediction": capt})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app1.run(port=5001)

import base64
import pickle

import requests
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)


@app.route("/")
def main():
    return render_template("main.html", current_page='main')


@app.route("/about")
def about():
    return render_template("about.html", current_page='about')

# Predict without API
# @app.route("/result2", methods=["POST"])
# def result2():
#     if request.method == "POST":
#         image = request.files["image"]
#
#         image_filename = secure_filename(image.filename)
#         if image_filename != '':
#             print(os.path.join('static', image_filename))
#             image.save(os.path.join('static', image_filename))
#
#             pred = predict(os.path.join('static', image_filename))
#             print(pred)
#
#             # Get the image src
#             image_src = f"/static/{image_filename}"
#
#             return render_template("result.html", pred=pred, image_src=image_src)
#         else:
#             return render_template("main.html")
#     else:
#         return render_template("main.html")


@app.route("/result", methods=["POST"])
def result():
    if request.method == "POST":
        image = request.files["image"]

        image_filename = secure_filename(image.filename)
        try:
            if image_filename != '':
                print(os.path.join('static', image_filename))
                image.save(os.path.join('static', image_filename))

                with open(os.path.join('static', image_filename), "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read())

                files = {"image": encoded_image}
                response = requests.post(
                    "http://localhost:5001/api/predict",
                    files=files
                )

                if response.status_code == 200:
                    pred_out = response.json()['prediction']
                else:
                    pred_out = ('Error:', response.status_code, response.text)

                image_src = f"/static/{image_filename}"

                return render_template("main.html", pred=pred_out, image_src=image_src, result_div='on', current_page='main')
        except Exception as ex:
            print(ex)
            fault = 'Подключение не установлено. Попробуйте позднее'
            return render_template("main.html", pred=fault, image_src='false', result_div='on', current_page='main')
    else:
        return render_template("main.html")


if __name__ == "__main__":
    app.run(port=8100)

import pickle

from flask import Flask, request, render_template
from joblib import load
from werkzeug.utils import secure_filename
import os

from model import predict

app = Flask(__name__)

@app.route("/")
def main():
    return render_template("main.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/result", methods=["POST"])
def result():
    if request.method == "POST":
        image = request.files["image"]
        image_filename = secure_filename(image.filename)
        if image_filename != '':
            print(os.path.join('static', image_filename))
            image.save(os.path.join('static', image_filename))

            pred = predict(os.path.join('static', image_filename))
            print(pred)

            # Get the image src
            image_src = f"/static/{image_filename}"

            return render_template("result.html", pred=pred, image_src=image_src)
        else:
            return render_template("main.html")
    else:
        return render_template("main.html")



if __name__ == "__main__":
    app.run()

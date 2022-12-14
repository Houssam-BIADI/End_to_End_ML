import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.json["data"]
    # array_data = np.array(list(data.values())).reshape(1, -1)
    array_data = pd.DataFrame(data, index=[0])
    output = round(model.predict_proba(array_data)[0][1], 3)
    return jsonify(output)


@app.route("/predict", methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    array_data = np.array(data).reshape(1, -1)
    output = round(model.predict_proba(array_data)[0][1], 3)
    return render_template(
        "home.html",
        prediction_text=f"your probabilty of having a heart desease is {output}",
    )


if __name__ == "__main__":
    app.run(debug=True)

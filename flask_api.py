#%%
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import json

WEIGHTS_DIR = "weights/"
FLASK_API = Flask(__name__)

#%%


def get_iris_model():
    loaded_clf = joblib.load(WEIGHTS_DIR + "clf_iris.joblib")
    return loaded_clf


def str_to_float_list(arg):
    arg = arg.split(",")
    arg = [float(x) for x in arg]
    return arg


loaded_clf = get_iris_model()
#%%Postman
def get_params_postman(request):
    sep_length = str_to_float_list(request.args.get("sepLen"))
    sep_width = str_to_float_list(request.args.get("sepWid"))
    pet_length = str_to_float_list(request.args.get("petLen"))
    pet_width = str_to_float_list(request.args.get("petWid"))

    return (sep_length, sep_width, pet_length, pet_width)


@FLASK_API.route("/predict_class_postman", methods=["GET", "POST"])
def predict_class_postman():
    (sep_length, sep_width, pet_length, pet_width) = get_params_postman(request)
    new_row = pd.DataFrame(
        {
            "sepal length (cm)": [float(x) for x in sep_length],
            "sepal width (cm)": [float(x) for x in sep_width],
            "petal length (cm)": [float(x) for x in pet_length],
            "petal width (cm)": [float(x) for x in pet_width],
        }
    )
    y_pred = list(loaded_clf.predict(new_row))
    y_pred = [str(x) for x in y_pred]

    response = {"y_pred": ",".join(y_pred)}
    return jsonify(response)


#%%CURL
def get_params_curl(request):
    request_input = request.form.get("input")
    request_input = json.loads(request_input)

    sep_length = str_to_float_list(request_input["sepLen"])
    sep_width = str_to_float_list(request_input["sepWid"])
    pet_length = str_to_float_list(request_input["petLen"])
    pet_width = str_to_float_list(request_input["petWid"])

    return (sep_length, sep_width, pet_length, pet_width)


@FLASK_API.route("/predict_class_curl", methods=["GET", "POST"])
def predict_class_curl():
    (sep_length, sep_width, pet_length, pet_width) = get_params_curl(request)
    new_row = pd.DataFrame(
        {
            "sepal length (cm)": [float(x) for x in sep_length],
            "sepal width (cm)": [float(x) for x in sep_width],
            "petal length (cm)": [float(x) for x in pet_length],
            "petal width (cm)": [float(x) for x in pet_width],
        }
    )
    y_pred = list(loaded_clf.predict(new_row))
    y_pred = [str(x) for x in y_pred]

    response = {"y_pred": ",".join(y_pred)}
    return jsonify(response)


#%%
if __name__ == "__main__":
    FLASK_API.debug = True
    FLASK_API.run(host="0.0.0.0", port="8080")
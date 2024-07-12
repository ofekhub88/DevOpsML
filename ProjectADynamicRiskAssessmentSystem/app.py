from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import diagnostics as diag
import json
import os


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = "1652d576-484a-49fd-913a-6879acfa6ba4"

with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=["POST", "OPTIONS"])
def predict():
    # call the prediction function you created in Step 3
    # read the body of the request csv_location
    # return the prediction as a json object
    csv_location = request.json["csv_location"]
    results = diag.model_predictions(csv_location)
    with open("apireturns.txt", "a") as f:
       f.write("prediction: " + str(results) + "\n")
    return jsonify({"prediction": {"input": csv_location, "output": str(results)}})


#######################Scoring Endpoint
@app.route("/scoring", methods=["GET", "OPTIONS"])
def statsscoring():
    with open(config["output_model_path"] + "/latestscore.txt", "rb") as f:
        score = f.read()
    with open("apireturns.txt", "a") as f:
        f.write("score: " + str(score) + "\n")
    return score


#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=["GET", "OPTIONS"])
def statssummarystats():
    stat_results = diag.dataframe_summary()
    with open("apireturns.txt", "a") as f:
        f.write("summary statistics: " + str(stat_results) + "\n")
    return stat_results


#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=["GET", "OPTIONS"])
def statsdiagnostics():
    stat_timing = diag.execution_time()
    na_status = diag.missing_data()
    results = {"timing": stat_timing, "missing_data": na_status}
    with open("apireturns.txt", "a") as f:
        f.write("diagnostics: " + str(results) + "\n")
    return results


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)

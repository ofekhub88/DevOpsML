import training
import scoring
import deployment
import diagnostics as diag
import ingestion as inges
import reporting
import json, os
import config as cfg

LOGGER = cfg.get_logger()

with open("config.json") as f:
    config = json.load(f)

with open("version") as f:
    version = json.load(f)

output_folder_path = config["output_folder_path"] + "/"
input_folder_path = config["input_folder_path"] + "/"
prod_deployment_path = config["prod_deployment_path"] + "/"


def validate_new_data():
    # Checking and Reading New Data
    LOGGER.info("Checking and Reading New Data")
    ingested_files = []
    with open(prod_deployment_path + "ingestedfiles.txt", "r") as f:
        ingested_files = f.read().split(",")[-1]

    new_files = []
    for file in os.listdir(input_folder_path):
        if file.endswith(".csv") and file not in ingested_files:
            # validate is a csv file
            new_files.append(file)

    # second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    if len(new_files) == 0:
        LOGGER.info("No new data found. Exiting.")
        return False
    else:
        inges.merge_multiple_dataframe()
        return True


##################Checking for model drift
# check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
def model_drift():
    with open(prod_deployment_path + "latestscore.txt", "r") as f:
        latest_score = float(f.read().split(",")[-1])

    new_score = scoring.score_model(output_folder_path + "finaldata.csv")

    ##################Deciding whether to proceed, part 2
    # if you found model drift, you should proceed. otherwise, do end the process here
    if new_score < latest_score:
        LOGGER.info(f"""No model drift detected. Exiting.
                    The latest score: {latest_score}
                    The new Score   : {new_score}""")
        return False
    else:
        LOGGER.info(
            f"""Model drift detected. Proceeding 
                    with re-training and re-deployment.
                    The latest score: {latest_score}
                    The new Score   : {new_score} """
        )
        return True


if __name__ == "__main__":
    if not validate_new_data():
        exit(0)
    if not model_drift():
        exit(0)
    LOGGER.info("Training model started")
    training.train_model()
    LOGGER.info("Training model completed")
    deployment.store_model_into_pickle("trainedmodel.pkl")
    LOGGER.info("Model deployment completed")
    version["Latest"] += 1
    with open("version", "w") as f:
        json.dump(version, f)

    ##################Diagnostics and reporting
    # run diagnostics.py and reporting.py for the re-deployed model
    diag.diagnostics(output_folder_path + "finaldata.csv")
    LOGGER.info("Diagnostics completed")
    reporting.score_model()

import requests
import json, os
import config as cfg

os.environ["http_proxy"] = ""
LOGGER = cfg.get_logger()

# Specify a URL that resolves to your workspace
URL = "http://localhost:8000/"

with open("version", "r") as f:
    version = json.load(f)["Latest"]
version = str(version)

# Call each API endpoint and store the responses
body = {"csv_location": "testdata/testdata.csv"}
LOGGER.info("Calling the API endpoints scoring")
response2 = requests.get(URL + "scoring")
LOGGER.info("Calling the API endpoints prediction")
response1 = requests.post(URL + "prediction", json=body)
LOGGER.info("Calling the API endpoints summarystats")
response3 = requests.get(URL + "summarystats")
LOGGER.info("Calling the API endpoints diagnostics")
response4 = requests.get(URL + "diagnostics")
# combine all API responses

responses = {
    "prediction": response1.text,
    "scoring": response2.text,
    "summarystats": str(response3.json()),
    "diagnostics": str(response4.json()),
}

# write the responses to your workspace
LOGGER.info
with open(f"apireturns{version}.txt", "a") as f:
    f.write(str(responses) + "\n")

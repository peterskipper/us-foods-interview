# US Foods Interview
Thanks for checking this out! This was a fun one to implement.

## Installation and Setup
This app was built using poetry. Install instructions [here](https://python-poetry.org/docs/)
You can install the dependencies from root via 
```shell
poetry install
```
Activate the virtualenv by running
```shell
poetry shell
```
You will also need training data to build a model. Go to [this page on kaggle](https://www.kaggle.com/c/demand-forecasting-kernels-only/data)
and download the `train.csv` file. Move it to the path `app/model/data/train.csv`

## Updating Dependencies
I don't know why you're changing the deps for my mini project? 
But if you need to, you should also update the `requirements.txt` via the following command:
```shell
poetry export --without-hashes --format=requirements.txt > requirements.txt
```
Don't forget to rebuild the Docker container as well!

## Training
The model training script is at `app/train_script.py`. Run
```shell
python app/train_script.py
```
to build a new model. This takes ~2 minutes locally.

## Tests
We have some basic test pipelines. To run unit tests:
```shell
pytest tests
```
Linting:
```shell
pylint app tests
```
Type Checks:
```shell
mypy app tests
```
Paint It Black:
```shell
black app tests
```
## Deployment
We use docker to deploy. **You'll need to train a model before you can deploy the API** (see command above)
```shell
docker build -t us-foods-ml-interview-app .
```
Then to run the API
```shell
docker run -d --name <fancy-name-here> -p 80:80 us-foods-ml-interview-app
```
You can now go to `http://localhost/docs` to see the API documentation.

To shut it down after:
```shell
docker stop <container id>
```
## Querying the API
The API was tested using python's `requests` library
```python
import requests

# Check status endpoint to see if API is running
resp = requests.get("http://127.0.0.1:80/status").json()
print(resp)
# {'status': 'OK'}

# Get a sales prediction from the model
payload = {"date": "2015-06-08", "store": 39, "item": 2}
resp = requests.post("http://localhost:80/predict", json=payload).json()
print(resp)
# {'sales': 65.63952561327561}
```

## Extensions
This is (hopefully) a good start to production-izing this model, but there's a lot of things I haven't had time to implement yet:
- A CI pipeline. We have tests, but I haven't hooked them up to run automatically yet
- A more configurable training pipeline. I used a random forest out of the box and it works decently well. But in a real life scenario, we'd want a pipeline that allows us to try different hyperparam combinations, and to retrain automatically as more training data arrives.
- Schema checks. I do some basic type checking on the model inputs (int, str etc.). But to make this more robust we'd want to set maximums and minimums to ensure we're not predicting on e.g. items the model has never heard of.
- Monitoring the API. I've added logging to capture model inputs and outputs. But ideally we'd like to stash this somewhere on s3 and make it observable. We want alerts when the model goes down and warnings when the inputs have drifted significantly from the training data.

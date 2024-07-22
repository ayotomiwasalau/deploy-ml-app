# Deploying a machine learning model with FastAPI

To run the training script of the model, from the project root directory, run `python app/src/train_model.py` 


`main.py` contains the endpoint for the interacting with the model

```
GET / HTTP/2.0
POST /predict HTTP/2.0
```



```
   import json
   import requests

   data = {
         "age": 45,
         "workclass": "Private",
         "fnlgt": 178356,
         "education": "Bachelors",
         "education_num": 9,
         "marital_status": "Never-married",
         "occupation": "Prof-specialty",
         "relationship": "Husband",
         "race": "White",
         "sex": "Female",
         "capital_gain": 4236,
         "capital_loss": 0,
         "hours_per_week": 45,
         "native_country": "Puerto-Rico"
      }

   response = requests.post("https://deploy-ml-app.onrender.com/predict", data=json.dumps(data))

   print(response.status_code)
   print(response.json())

```

import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.src.ml.data import process_data
from app.src.ml.model import inference


path = "app/model/lb.pkl"
lb = pickle.load(open(path, 'rb'))

path = "app/model/encoder.pkl"
encoder = pickle.load(open(path, 'rb'))

path = "app/model/rf_model.pkl" 
model = pickle.load(open(path, 'rb'))

class Data(BaseModel):
    age: int = Field(None, example=45)
    workclass: str = Field(None, example='Private')
    fnlgt: int = Field(None, example=178356)
    education: str = Field(None, example='Bachelors')
    education_num: int = Field(None, example=9)
    marital_status: str = Field(None, example='Never-married')
    occupation: str = Field(None, example='Prof-specialty')
    relationship: str = Field(None, example='Husband')
    race: str = Field(None, example='White')
    sex: str = Field(None, example='Female')
    capital_gain: int = Field(None, example=4236)
    capital_loss: int = Field(None, example=0)
    hours_per_week: int = Field(None, example=45)
    native_country: str = Field(None, example='Puerto-Rico')


app = FastAPI()

@app.get('/')
async def welcome():
    return "Welcome!"


@app.post("/predict")
async def make_prediction(data: Data):
    data_dict = data.dict()
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}

    data = pd.DataFrame.from_dict(data)

    try:

        cat_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]

        data_processed, _, _, _ = process_data(
            data,
            categorical_features=cat_features,
            label=None,
            training=False,
            encoder=encoder,
            lb=lb
        )

        inference_output = inference(model=model, X=data_processed)[0]

        if inference_output == 0:
            output = '<=50K' 
        else: 
            output = '>50K'

        return {"message": output}
    except Exception as e:
         raise HTTPException(status_code=422, detail={"message": {e}})


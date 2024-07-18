# Script to train machine learning model.
import os
import pickle
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, calculate_slices

logging.basicConfig(level=logging.INFO)

# Add code to load in the data.
root_path = "app" #os.path.dirname(__file__) 
path = os.path.join(root_path, 'data/cleaned_census.csv')
data = pd.read_csv(path)

data = data.drop(['Unnamed: 0'], axis=1)

logging.info('Preprocessing data...')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, encoder=encoder, lb=lb, label="salary", training=False
)

logging.info('Training model...')

# Train model.
rf_model = train_model(X_train, y_train)

logging.info('Saving model and encoder...')

# Save the trained model
mdl_path = os.path.join(root_path, './model/rf_model.pkl')
pickle.dump(rf_model, open(mdl_path, 'wb'))

# Save the encoder
encoder_path = os.path.join(root_path, './model/encoder.pkl')
pickle.dump(encoder, open(encoder_path, 'wb'))

# Save the label binarizer
lb_path = os.path.join(root_path, './model/lb.pkl')
pickle.dump(lb, open(lb_path, 'wb'))

logging.info('Inferencing...')

#load model for inferencing
ld_model = pickle.load(open(mdl_path, 'rb'))

# inferencing
y_pred = inference(ld_model, X_test)

# classification metrics
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
logging.info('Metrics for full data set')
logging.info(f'precision - {precision}, recall - {recall}, fbeta - {fbeta}')


logging.info('Check app/log/slice_output.txt for metrics on each categroprical feature slice.')

# Compute slices and their metrics
calculate_slices(data, ld_model, cat_features, encoder, lb)

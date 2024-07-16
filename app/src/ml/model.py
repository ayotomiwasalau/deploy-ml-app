from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

try:
    from app.src.ml.data import process_data
except ImportError:
    from ml.data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    clf_model = RandomForestClassifier()

    model = clf_model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds

def calculate_slices(data, model, cat_features, encoder, lb):

    for col in cat_features:
        for slice in data[col].unique():
            slice_df = data[data[col] == slice]
            colsize, rowsize = slice_df.shape[0], slice_df.shape[1]
            X_slice, y_slice, _, _ = process_data(
                slice_df, categorical_features=cat_features, 
                training=False,label="salary", encoder=encoder, lb=lb
            )

            y_pred_slice = inference(model, X_slice)

            precision, recall, fbeta = compute_model_metrics(y_slice, y_pred_slice)

            with open("app/log/slice_output.txt", "a") as f:
                print(f"{col}: {slice}, size: {colsize:,}, {rowsize:,}", file=f)
                print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {fbeta:.4f}", file=f)
                print("", file=f)

        

# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model was created by Ayotomiwa Salau. The model is a random forest model from the `scikit-learn` library for classification task.

## Intended Use
The model is used to predict if an individual earns over or below $50,000 per annum

## Training Data
The training data is from the `UC Irvin ML repository`. Extraction was done by Barry Becker from the 1994 Census database. 

## Evaluation Data
The evaluation and training data are from the same source. Both were preprocessed and encoded using `OneHotEncoder` and `LabelBinarizer`
The train and evaluation data were split 80%-20%. 

## Metrics
The metrics used to evaluate the model are `precision`, `recall` and `fbeta` score.

The scores are as follows
`precision` - 0.7294, `recall` - 0.6219, `fbeta` - 0.6714

## Ethical Considerations
The model performance a bit pooly on slices such as native country (`Trinadad&Tobago`, `Columbia` etc)

## Caveats and Recommendations

Get more data on some native countries for very adequate predictions.

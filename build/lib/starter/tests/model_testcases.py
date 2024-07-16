import pytest
import numpy as np
from starter.ml.model import train_model, compute_model_metrics, inference, calculate_slices

def test_train_model():
    """
    # test the train model function to make sure it works
    """
    X = np.random.rand(35, 16)
    y = np.random.randint(2, size=35)
    model = train_model(X, y)

    print(type(model))


# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    # add description for the second test
    """
    # Your code here
    pass


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    # add description for the third test
    """
    # Your code here
    pass

# test
# asert inferencing returns 0 or 1
# assert metrics is between 0 and 1

test_train_model()
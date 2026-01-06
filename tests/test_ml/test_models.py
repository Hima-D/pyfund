import pytest
import numpy as np
import pandas as pd
from pyfundlib.ml.models.xgboost import XGBoostModel

def test_xgboost_model_fit_predict():
    """Test basic fit and predict for XGBoostModel."""
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 100))
    
    model = XGBoostModel(task="classification", n_estimators=10)
    model.fit(X, y)
    
    preds = model.predict(X)
    assert len(preds) == 100
    assert np.all((preds == 0) | (preds == 1))

def test_xgboost_model_feature_importance():
    """Test feature importance extraction."""
    X = pd.DataFrame(np.random.rand(100, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.random.randint(0, 2, 100))
    
    model = XGBoostModel(task="classification", n_estimators=10)
    model.fit(X, y)
    
    importance = model.feature_importance()
    assert len(importance) == 5

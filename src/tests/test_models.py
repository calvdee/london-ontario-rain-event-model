from src.app.model.models import evalute_models
from src.app.feature_data import load_feature_data
from settings import WEEKLY_DATA_PATH

def test_evaluate_models():
  X = load_feature_data(WEEKLY_DATA_PATH)
  evalute_models(X)
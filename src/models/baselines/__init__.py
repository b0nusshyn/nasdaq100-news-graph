"""Baseline ladder (README, Evaluation). Each step isolates one component."""

from src.models.baselines.arima import Arima
from src.models.baselines.buy_and_hold import BuyAndHold
from src.models.baselines.logistic import Logistic
from src.models.baselines.lstm_news import LSTMNews
from src.models.baselines.lstm_price import LSTMPrice
from src.models.baselines.xgb import XGBoost

LADDER = (BuyAndHold, Arima, Logistic, XGBoost, LSTMPrice, LSTMNews)
BASELINES = {cls.name: cls for cls in LADDER}

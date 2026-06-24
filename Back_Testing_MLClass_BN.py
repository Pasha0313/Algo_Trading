# strategies_ml.py
import pandas as pd
from backtesting import Strategy

class SimpleClassificationUD(Strategy):
    def init(self):
        # These must be injected dynamically after class creation
        self.model = None
        self.features = None
        self.already_bought = False

    def next(self):
        row = pd.DataFrame({feat: [getattr(self.data, feat)[-1]] for feat in self.features})
        prediction = self.model.predict(row)[0]

        if prediction == 1 and self.already_bought == False:
            self.buy()
            self.already_bought = True
        elif prediction == 0 and self.already_bought == True:
            self.sell()
            self.already_bought = False
import pandas as pd
import numpy as np

class CustomBacktester:
    def __init__(self, df, initial_balance=10000):
        self.df = df
        self.balance = initial_balance
        self.position = 0
        self.trades = []
    
    def run(self, strategy_func):
        for i in range(60, len(self.df)):
            signal = strategy_func(self.df.iloc[:i])
            # Simplified execution
            price = self.df['close'].iloc[i]
            if signal == 1 and self.position <= 0:
                self.position = 1
                self.trades.append(('buy', price))
            elif signal == -1 and self.position >= 0:
                self.position = -1
                self.trades.append(('sell', price))
        print(f'Backtest complete. {len(self.trades)} trades. Final balance: {self.balance}')
        return self.balance
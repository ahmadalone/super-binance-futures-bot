import os
import time
import hmac
import hashlib
import json
import asyncio
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv
import logging
import websockets
from datetime import datetime

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ====================== CONFIG ======================
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_API_SECRET')
BASE_URL = "https://testnet.binancefuture.com"  # Change to production when ready
SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVAL = "5m"
LEVERAGE = 5
POSITION_SIZE_PCT = 0.08
STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.05
RETRAIN_HOURS = 1

# ====================== MODELS ======================
class PriceLSTM(nn.Module):
    def __init__(self, input_size=12, hidden_size=64, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.25)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class ActorCritic(nn.Module):
    def __init__(self, state_dim=16, action_dim=3):  # hold/long/short
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
        self.actor = nn.Sequential(nn.Linear(128, action_dim), nn.Softmax(dim=-1))
        self.critic = nn.Linear(128, 1)
    
    def forward(self, state):
        shared = self.shared(state)
        return self.actor(shared), self.critic(shared)

# ====================== HELPERS ======================
def signed_request(method, endpoint, params=None):
    if params is None:
        params = {}
    params['timestamp'] = int(time.time() * 1000)
    query = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
    signature = hmac.new(API_SECRET.encode('utf-8'), query.encode('utf-8'), hashlib.sha256).hexdigest()
    params['signature'] = signature
    url = f"{BASE_URL}{endpoint}"
    headers = {'X-MBX-APIKEY': API_KEY}
    if method.upper() == 'GET':
        return requests.get(url, params=params, headers=headers).json()
    return requests.post(url, params=params, headers=headers).json()

def get_klines(symbol, limit=1000):
    params = {'symbol': symbol, 'interval': INTERVAL, 'limit': limit}
    data = requests.get(f"{BASE_URL}/fapi/v1/klines", params=params).json()
    df = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'taker_bv', 'taker_qv', 'ignore'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col])
    df['return'] = df['close'].pct_change()
    return df

def add_features(df):
    df['ema_fast'] = df['close'].ewm(span=12).mean()
    df['ema_slow'] = df['close'].ewm(span=26).mean()
    df['macd'] = df['ema_fast'] - df['ema_slow']
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_mid'] - 2 * df['bb_std']
    df['volatility'] = df['return'].rolling(20).std()
    return df.dropna()

class TradingDataset(Dataset):
    def __init__(self, data, seq_length=60):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_length = seq_length
    def __len__(self):
        return len(self.data) - self.seq_length
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx + self.seq_length, 3]  # close price
        return x, y

# ====================== SENTIMENT ======================
def get_market_sentiment(symbol):
    try:
        query = symbol.replace('USDT', '').lower()
        resp = requests.get(f"https://api.coingecko.com/api/v3/coins/{query}/market_chart?vs_currency=usd&days=1").json()
        prices = resp.get('prices', [])
        if len(prices) > 10:
            recent_change = (prices[-1][1] - prices[-10][1]) / prices[-10][1]
            return np.clip(recent_change * 5, -1, 1)
    except:
        pass
    return 0.0

# ====================== CORE TRADER ======================
class SuperTrader:
    def __init__(self):
        self.lstm_models = {sym: PriceLSTM() for sym in SYMBOLS}
        self.ppo = ActorCritic()
        self.ppo_optimizer = optim.Adam(self.ppo.parameters(), lr=3e-4)
        self.live_data = {sym: pd.DataFrame() for sym in SYMBOLS}
        self.positions = {sym: 0 for sym in SYMBOLS}
        self.last_retrain = time.time()
    
    def prepare_data(self, df):
        features = ['open','high','low','close','volume','return','macd','rsi','bb_upper','bb_lower','volatility']
        data = df[features].values
        mean, std = np.mean(data, axis=0), np.std(data, axis=0) + 1e-8
        return (data - mean) / std, mean, std
    
    def train_lstm(self, df):
        norm_data, _, _ = self.prepare_data(df)
        dataset = TradingDataset(norm_data)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        # Simplified training (use specific model in full impl)
        optimizer = optim.Adam(list(self.lstm_models.values())[0].parameters(), lr=0.001)
        criterion = nn.MSELoss()
        for epoch in range(5):
            for x, y in loader:
                optimizer.zero_grad()
                pred = list(self.lstm_models.values())[0](x.unsqueeze(0) if x.dim()==2 else x)
                loss = criterion(pred, y.unsqueeze(1))
                loss.backward()
                optimizer.step()
        logging.info("LSTM retrained")
    
    def get_ppo_action(self, state):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        probs, _ = self.ppo(state_t)
        action = torch.argmax(probs).item()
        return action
    
    def execute_trade(self, symbol, action):
        if action == 0: return
        side = 'BUY' if action == 1 else 'SELL'
        try:
            account = signed_request('GET', '/fapi/v2/account')
            balance = float(account.get('availableBalance', 1000))
            price = self.live_data[symbol]['close'].iloc[-1]
            qty = round((balance * POSITION_SIZE_PCT * LEVERAGE) / price * 0.95, 3)
            params = {'symbol': symbol, 'side': side, 'type': 'MARKET', 'quantity': qty}
            resp = signed_request('POST', '/fapi/v1/order', params)
            logging.info(f"Executed {side} on {symbol}: {resp}")
            self.positions[symbol] = 1 if action == 1 else -1
        except Exception as e:
            logging.error(f"Trade error: {e}")
    
    async def ws_handler(self):
        uri = "wss://fstream.binance.com/ws"
        async with websockets.connect(uri) as websocket:
            subscribe_msg = {
                "method": "SUBSCRIBE",
                "params": [f"{s.lower()}@kline_{INTERVAL}" for s in SYMBOLS],
                "id": 1
            }
            await websocket.send(json.dumps(subscribe_msg))
            async for message in websocket:
                data = json.loads(message)
                if 'k' in data.get('data', {}):
                    k = data['data']['k']
                    sym = data['data']['s']
                    if sym in self.live_data:
                        new_row = pd.DataFrame([{
                            'open': float(k['o']), 'high': float(k['h']), 'low': float(k['l']),
                            'close': float(k['c']), 'volume': float(k['v']), 'open_time': k['t']
                        }])
                        self.live_data[sym] = pd.concat([self.live_data[sym], new_row]).tail(500)
    
    def run_main_loop(self):
        while True:
            try:
                for sym in SYMBOLS:
                    if len(self.live_data[sym]) < 100:
                        df = get_klines(sym)
                        self.live_data[sym] = add_features(df)
                    
                    sentiment = get_market_sentiment(sym)
                    df = self.live_data[sym]
                    norm, m, s = self.prepare_data(df)
                    state = np.append(norm[-1], [sentiment, self.positions[sym]])
                    
                    action = self.get_ppo_action(state)
                    self.execute_trade(sym, action)
                
                if time.time() - self.last_retrain > RETRAIN_HOURS * 3600:
                    for sym in SYMBOLS:
                        self.train_lstm(self.live_data[sym])
                    self.last_retrain = time.time()
                
                time.sleep(30)
            except Exception as e:
                logging.error(f"Loop error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    trader = SuperTrader()
    trader.run_main_loop()
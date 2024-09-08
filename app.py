from flask import Flask, Response
import requests
import json
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from chronos import ChronosPipeline

# create our Flask app
app = Flask(__name__)

# define the Hugging Face model we will use
model_name = "amazon/chronos-t5-tiny"

# Create a scaler for normalizing data
scaler = MinMaxScaler()

def get_binance_url(token):
    """Build the Binance Klines API URL for the specified token."""
    base_url = "https://api.binance.com/api/v3/klines"
    token_map = {
        'ETH': 'ETHUSDT',
        'SOL': 'SOLUSDT',
        'BTC': 'BTCUSDT',
        'BNB': 'BNBUSDT',
        'ARB': 'ARBUSDT'
    }
    
    token = token.upper()
    if token in token_map:
        symbol = token_map[token]
        url = f"{base_url}?symbol={symbol}&interval=1d&limit=30"
        return url
    else:
        raise ValueError("Unsupported token")

# define our endpoint
@app.route("/inference/<string:token>")
def get_inference(token):
    """Generate inference for the given token."""
    try:
        # use a pipeline as a high-level helper
        pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    except Exception as e:
        return Response(json.dumps({"pipeline error": str(e)}), status=500, mimetype='application/json')

    try:
        # Get the data from Binance Kline API
        url = get_binance_url(token)
    except ValueError as e:
        return Response(json.dumps({"error": str(e)}), status=400, mimetype='application/json')

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()

        # Parsing the Binance API response
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", 
                                         "close_time", "quote_asset_volume", "number_of_trades", 
                                         "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df["close"] = pd.to_numeric(df["close"])
        
        # Using close prices for normalization and predictions
        df = df[:-1]  # removing today's price
        df["normalized_price"] = scaler.fit_transform(df["close"].values.reshape(-1, 1))
        
        print(df.tail(5))
    else:
        return Response(json.dumps({"Failed to retrieve data from the API": str(response.text)}), 
                        status=response.status_code, 
                        mimetype='application/json')

    # Use the normalized close price data for prediction
    context = torch.tensor(df["normalized_price"].values)
    prediction_length = 5  # Predict for more days

    try:
        forecast = pipeline.predict(context, prediction_length)  # shape [num_series, num_samples, prediction_length]
        
        # Inverse transform to get the actual price back from normalized predictions
        forecast_values = scaler.inverse_transform(forecast[0].mean(dim=0).unsqueeze(0).numpy())
        
        # Prepare response data
        prediction_data = {
            "forecasted_prices": forecast_values.tolist()
        }
        return Response(json.dumps(prediction_data), status=200, mimetype='application/json')
    except Exception as e:
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

# run our Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)

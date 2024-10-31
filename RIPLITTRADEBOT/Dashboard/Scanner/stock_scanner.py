import alpaca_trade_api as tradeapi
import pandas as pd
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import yfinance as yf
import webview
import threading
import openai

# Alpaca API configuration
ALPACA_API_KEY = "PKKTH5ONSE1UBXDXYIOX"
ALPACA_SECRET_KEY = "gedAkYqXjePoKAZWfCc5p576cGS38RzYCMOqzXXi"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# OpenAI API configuration
openai.api_key = "sk-proj-UzqjbkLRu7qQHEEbjMqmT3BlbkFJRouOeWZctZX1XJyaFXLa"

# Initialize Alpaca API
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

# Define the Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.H1("Top Gainers Stock Scanner"),
    dcc.Interval(id='interval-component', interval=60*1000, n_intervals=0),  # Update every minute
    dash_table.DataTable(
        id='live-update-table',
        columns=[
            {'name': 'Symbol/News', 'id': 'symbol'},
            {'name': 'Price', 'id': 'price'},
            {'name': 'Volume', 'id': 'volume'},
            {'name': 'Float', 'id': 'float'},
            {'name': 'Relative Volume', 'id': 'relative_volume'},
            {'name': 'Change From Close(%)', 'id': 'percent_change'}
        ],
        style_cell={'textAlign': 'center'},
        style_header={
            'backgroundColor': 'rgb(30, 30, 30)',
            'color': 'white'
        },
        style_data_conditional=[
            {
                'if': {'column_id': 'percent_change'},
                'backgroundColor': 'rgb(50, 50, 50)',
                'color': 'white'
            }
        ],
        style_table={'overflowX': 'auto'}
    ),
    html.Div(id='summary-output'),
    html.Div(id='news-log', style={'whiteSpace': 'pre-line', 'height': '200px', 'overflowY': 'scroll', 'backgroundColor': '#f0f0f0', 'padding': '10px', 'marginTop': '20px'})
])

def get_trending_news(symbol):
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        if news:
            return True, news[0]['title']  # Return True and the title of the latest news
        return False, ''
    except Exception as e:
        print(f"Error fetching news for {symbol}: {e}")
        return False, ''

def batch_symbols(symbols, batch_size=100):
    for i in range(0, len(symbols), batch_size):
        yield symbols[i:i + batch_size]

def get_top_gainers():
    assets = api.list_assets(status='active')
    symbols = [asset.symbol for asset in assets if asset.tradable and asset.fractionable is False]  # Ensure only valid stock symbols
    df_list = []

    for symbol_batch in batch_symbols(symbols):
        bars = api.get_bars(symbol_batch, "1Min", limit=1).df.reset_index()
        df_list.append(bars)

    df = pd.concat(df_list)
    df['percent_change'] = (df['close'] - df['open']) / df['open'] * 100
    df = df[df['percent_change'] > 10]  # At least 10% increase

    # Fetch additional data like float, volume, etc.
    assets_info = {asset.symbol: asset for asset in assets if asset.symbol in df['symbol'].unique()}

    result = []
    news_log = []

    for symbol, row in df.iterrows():
        asset = assets_info.get(symbol)
        if asset:
            float_data = 20  # Fetch float data from another source if available
            relative_volume = row['volume'] / float_data if float_data > 0 else 0
            has_news, news_title = get_trending_news(row['symbol'])
            if (row['close'] < 20 and float_data < 20 and relative_volume > 10) or row['percent_change'] > 20:
                result.append({
                    'symbol': row['symbol'],
                    'price': row['close'],
                    'volume': row['volume'],
                    'float': float_data,
                    'relative_volume': relative_volume,
                    'percent_change': row['percent_change']
                })
                if has_news:
                    news_log.append(f"{row['symbol']}: {news_title}")

    return pd.DataFrame(result), '\n'.join(news_log)

def analyze_with_openai(df):
    stocks_info = df.to_dict('records')
    prompt = (
        "Analyze the following stock data, identify trends, and summarize the biggest gainers:\n\n"
        f"{stocks_info}\n\n"
        "Provide a summary of the key trends and the most significant gainers."
    )
    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=prompt,
        max_tokens=200,
        temperature=0.5
    )
    return response.choices[0].text.strip()

@app.callback(
    [Output('live-update-table', 'data'), Output('summary-output', 'children'), Output('news-log', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_table_live(n):
    df, news_log = get_top_gainers()
    summary = analyze_with_openai(df)
    return df.to_dict('records'), summary, news_log

def run_dash():
    app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter

if __name__ == '__main__':
    dash_thread = threading.Thread(target=run_dash)
    dash_thread.start()
    webview.create_window("Top Gainers Stock Scanner", "http://127.0.0.1:8050/")
    webview.start()
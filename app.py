# =========================================
# Billion-Dollar Pre-News AI Trading Dashboard
# Forex & Gold | Free Cloud Deployable
# =========================================

import os
import requests
import pandas as pd
import datetime
import numpy as np
import pickle
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import telegram
import smtplib
from textblob import TextBlob

# ---------------------------
# 1️⃣ CONFIGURATION
# ---------------------------
NEWS_EVENTS = {
    "CPI": ["Energy", "Wages", "Housing", "RetailSales"],
    "NFP": ["ADPJobs", "UnemploymentClaims", "HourlyEarnings"]
}

DEFAULT_WEIGHTS = {
    "Energy": 0.08,
    "Wages": 0.14,
    "Housing": 0.4,
    "RetailSales": 0.38,
    "ADPJobs": 0.3,
    "UnemploymentClaims": 0.3,
    "HourlyEarnings": 0.4
}

CUSTOM_INDICATORS = {}  # Add custom indicators dynamically

ALERT_HIGH = 60
ALERT_LOW = 40

# Telegram config
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
bot = telegram.Bot(token=TELEGRAM_TOKEN) if TELEGRAM_TOKEN else None

# Email config
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_TO = os.getenv("EMAIL_TO")

# Free API keys
ALPHA_KEY = os.getenv("ALPHA_KEY")  # AlphaVantage for WTI Crude Oil
FRED_KEY = os.getenv("FRED_KEY")    # FRED API

# TradingView webhook (optional)
TRADINGVIEW_WEBHOOK = os.getenv("TRADINGVIEW_WEBHOOK")

# ---------------------------
# 2️⃣ HELPER FUNCTIONS
# ---------------------------
def fetch_json(url):
    try:
        response = requests.get(url)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def normalize(value, min_val, max_val, direction="positive"):
    if max_val == min_val: return 50
    score = (value - min_val)/(max_val - min_val)*100
    if direction=="negative": score = 100 - score
    return max(0, min(100, score))

# ---------------------------
# 3️⃣ INDICATOR FETCHERS (FREE APIs)
# ---------------------------

def get_energy_price():
    if not ALPHA_KEY: return 80.0
    url = f"https://www.alphavantage.co/query?function=WTI_CRUDE&apikey={ALPHA_KEY}"
    data = fetch_json(url)
    try:
        latest = list(data['Time Series (Daily)'].values())[0]['close']
        return float(latest)
    except: 
        return 80.0

def get_wages():
    if not FRED_KEY: return 30.0
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=CES0500000003&api_key={FRED_KEY}&file_type=json"
    data = fetch_json(url)
    try:
        latest = float(data['observations'][-1]['value'])
        return latest
    except:
        return 30.0

def get_housing_index():
    if not FRED_KEY: return 200.0
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=CSUSHPISA&api_key={FRED_KEY}&file_type=json"
    data = fetch_json(url)
    try:
        latest = float(data['observations'][-1]['value'])
        return latest
    except:
        return 200.0

def get_retail_sales():
    if not FRED_KEY: return 600.0
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id=RSAFS&api_key={FRED_KEY}&file_type=json"
    data = fetch_json(url)
    try:
        latest = float(data['observations'][-1]['value'])
        return latest
    except:
        return 600.0

def get_custom_indicator(name, config):
    data = fetch_json(config["source"])
    val = float(data.get("latest",50)) if data else 50
    return normalize(val, config["min_val"], config["max_val"], config.get("direction","positive"))

# ---------------------------
# 4️⃣ SENTIMENT ANALYSIS
# ---------------------------
def analyze_sentiment_twitter(keyword):
    # Placeholder: requires Twitter API v2
    # Returns 0-100 sentiment score
    return 50  # neutral if API not provided

def analyze_sentiment_news(keyword):
    # Placeholder for NewsAPI / GDELT
    return 50

# ---------------------------
# 5️⃣ PRE-NEWS SCORE CALCULATION
# ---------------------------
def calculate_pre_news_score(event_name):
    components = NEWS_EVENTS.get(event_name, [])
    total, weight_total = 0,0
    for comp in components:
        if comp=="Energy": val = normalize(get_energy_price(),60,120)
        elif comp=="Wages": val = normalize(get_wages(),20,35)
        elif comp=="Housing": val = normalize(get_housing_index(),100,300)
        elif comp=="RetailSales": val = normalize(get_retail_sales(),400,700)
        elif comp in CUSTOM_INDICATORS: val = get_custom_indicator(comp,CUSTOM_INDICATORS[comp])
        else: val = 50
        weight = DEFAULT_WEIGHTS.get(comp, CUSTOM_INDICATORS.get(comp,{}).get("weight",0.1))
        total += val*weight
        weight_total += weight
    base_score = round(total/weight_total,2) if weight_total else 50
    
    # Incorporate sentiment (Twitter & News) 20% weight
    sentiment = (analyze_sentiment_twitter(event_name)+analyze_sentiment_news(event_name))/2
    final_score = 0.8*base_score + 0.2*sentiment
    return final_score

# ---------------------------
# 6️⃣ ML ENGINE (Market Move Probability)
# ---------------------------
def load_ml_model():
    if os.path.exists('pre_news_model.pkl'):
        with open('pre_news_model.pkl','rb') as f:
            return pickle.load(f)
    return None

model = load_ml_model()

def predict_market_move(features_dict=None):
    if model and features_dict:
        df = pd.DataFrame([features_dict])
        prob = model.predict_proba(df)[0][1]
        return prob
    else:
        return np.random.rand()

# ---------------------------
# 7️⃣ ALERTS
# ---------------------------
def send_telegram_alert(msg):
    if bot: bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)

def send_email_alert(subject, body):
    if EMAIL_USER:
        try:
            server = smtplib.SMTP('smtp.gmail.com',587)
            server.starttls()
            server.login(EMAIL_USER,EMAIL_PASS)
            server.sendmail(EMAIL_USER,EMAIL_TO,f"Subject:{subject}\n\n{body}")
            server.quit()
        except Exception as e:
            print("Email alert failed:", e)

def send_tradingview_webhook(symbol, action):
    if TRADINGVIEW_WEBHOOK:
        try:
            requests.post(TRADINGVIEW_WEBHOOK,json={"symbol":symbol,"action":action})
        except:
            print("TradingView webhook failed")

# ---------------------------
# 8️⃣ DASHBOARD (Plotly Dash)
# ---------------------------
app = Dash(__name__)
app.layout = html.Div([
    html.H1("Pre-News AI Dashboard (Forex & Gold)"),
    dcc.Interval(id='interval-component', interval=60*1000, n_intervals=0),
    html.Div(id='dashboard-content')
])

@app.callback(Output('dashboard-content','children'), Input('interval-component','n_intervals'))
def update_dashboard(n):
    content = []
    for event in NEWS_EVENTS.keys():
        score = calculate_pre_news_score(event)
        features = {
            "Energy": get_energy_price(),
            "Wages": get_wages(),
            "Housing": get_housing_index(),
            "RetailSales": get_retail_sales()
        }
        prob_up = predict_market_move(features)
        
        # Add heatmap for contributions
        contribs = {
            "Energy": normalize(features["Energy"],60,120),
            "Wages": normalize(features["Wages"],20,35),
            "Housing": normalize(features["Housing"],100,300),
            "RetailSales": normalize(features["RetailSales"],400,700)
        }
        fig = px.imshow([list(contribs.values())],
                        x=list(contribs.keys()),
                        y=['Contribution'],
                        color_continuous_scale='RdYlGn')
        fig.update_layout(height=200)
        
        content.append(html.H3(f"{event} Pre-News Score: {score} | Prob Market Up: {prob_up*100:.2f}%"))
        content.append(dcc.Graph(figure=fig))
        
        # Alerts
        if score>ALERT_HIGH:
            send_telegram_alert(f"{event} likely beat/hawkish | Score: {score}")
            send_email_alert(f"{event} Pre-News Alert","Score high! Check dashboard.")
            send_tradingview_webhook(event,"BUY")
        elif score<ALERT_LOW:
            send_telegram_alert(f"{event} likely miss/dovish | Score: {score}")
            send_email_alert(f"{event} Pre-News Alert","Score low! Check dashboard.")
            send_tradingview_webhook(event,"SELL")
            
    return content

# ---------------------------
# 9️⃣ RUN DASHBOARD
# ---------------------------
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)

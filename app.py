import dash
from dash import html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import logging

logging.basicConfig(level=logging.WARNING)

# ——————————————————————————————————————————
# CONFIG
# ——————————————————————————————————————————
TICKERS = ["SPY","AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","QQQ","AMD","JPM","NFLX","BA","DIS","V"]

CL = {"bg":"#0b1121","card":"#0f172a","bdr":"rgba(255,255,255,0.06)",
      "txt":"#e2e8f0","mut":"#64748b","grid":"rgba(255,255,255,0.04)",
      "g":"#34d399","r":"#f87171","b":"#6366f1","o":"#f97316",
      "p":"#a78bfa","y":"#fbbf24","c":"#22d3ee","i":"#818cf8"}

LO = dict(template="plotly_dark", paper_bgcolor=CL["bg"], plot_bgcolor=CL["card"],
          font=dict(color="#94a3b8",size=11,family="Inter,system-ui,sans-serif"),
          margin=dict(l=55,r=20,t=40,b=35),
          xaxis=dict(gridcolor=CL["grid"]), yaxis=dict(gridcolor=CL["grid"]),
          legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,
                      bgcolor="rgba(0,0,0,0)",font=dict(size=10)))


# ——————————————————————————————————————————
# DATA PIPELINE
# ——————————————————————————————————————————
class Data:
    """Market data pipeline with caching and fallback."""
    _cache = {}

    @classmethod
    def fetch(cls, ticker, period="1y"):
        key = f"{ticker}_{period}"
        if key in cls._cache:
            return cls._cache[key].copy()
        try:
            session = requests.Session()
            session.headers["User-Agent"] = "Mozilla/5.0"
            df = yf.Ticker(ticker, session=session).history(period=period)
            if not df.empty:
                df = df[["Open","High","Low","Close","Volume"]].copy()
                df.columns = ["open","high","low","close","volume"]
                df.dropna(inplace=True)
                cls._cache[key] = df
                return df.copy()
        except Exception as e:
            logging.warning(f"yfinance error for {ticker}: {e}")

        # Fallback: simulated data
        logging.info(f"Using simulated data for {ticker}")
        np.random.seed(hash(ticker) % 2**31)
        n = {"1mo":22,"3mo":66,"6mo":126,"1y":252,"2y":504,"5y":1260}.get(period, 252)
        price = 100 + np.cumsum(np.random.randn(n) * 1.5)
        price = np.maximum(price, 10)
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
        df = pd.DataFrame({
            "open": price + np.random.randn(n)*0.3,
            "high": price + abs(np.random.randn(n))*1.0,
            "low":  price - abs(np.random.randn(n))*1.0,
            "close": price,
            "volume": np.random.randint(1_000_000, 10_000_000, n).astype(float)
        }, index=dates)
        cls._cache[key] = df
        return df.copy()


# ——————————————————————————————————————————
# TECHNICAL INDICATORS
# ——————————————————————————————————————————
class Ind:
    """Vectorized technical indicators."""
    @staticmethod
    def sma(s, p): return s.rolling(p, min_periods=p).mean()

    @staticmethod
    def rsi(s, p=14):
        d = s.diff()
        g = d.where(d > 0, 0.0).rolling(p, min_periods=p).mean()
        l = (-d.where(d < 0, 0.0)).rolling(p, min_periods=p).mean()
        return 100 - 100 / (1 + g / l)

    @staticmethod
    def bbands(s, p=20, std=2.0):
        m = s.rolling(p).mean(); sd = s.rolling(p).std()
        return m, m + std*sd, m - std*sd

    @staticmethod
    def macd(s):
        ml = s.ewm(span=12).mean() - s.ewm(span=26).mean()
        sl = ml.ewm(span=9).mean()
        return ml, sl, ml - sl

    @classmethod
    def add(cls, df, fast, slow):
        df = df.copy()
        df["sma_f"] = cls.sma(df["close"], fast)
        df["sma_s"] = cls.sma(df["close"], slow)
        df["rsi"] = cls.rsi(df["close"])
        df["bb_m"], df["bb_u"], df["bb_l"] = cls.bbands(df["close"])
        df["macd"], df["macd_sig"], df["macd_h"] = cls.macd(df["close"])
        return df


# ——————————————————————————————————————————
# STRATEGY ENGINE
# ——————————————————————————————————————————
class Strat:
    """MA Crossover strategy with optional RSI filter."""
    @staticmethod
    def signals(df, use_rsi=True):
        df = df.copy()
        df["signal"] = 0
        pf, ps = df["sma_f"].shift(1), df["sma_s"].shift(1)
        buy = (pf <= ps) & (df["sma_f"] > df["sma_s"])
        sell = (pf >= ps) & (df["sma_f"] < df["sma_s"])
        if use_rsi:
            buy = buy & (df["rsi"] < 70)
            sell = sell & (df["rsi"] > 30)
        df.loc[buy, "signal"] = 1
        df.loc[sell, "signal"] = -1
        return df


# ——————————————————————————————————————————
# BACKTESTER
# ——————————————————————————————————————————
class BT:
    """Backtesting engine with risk management."""
    @staticmethod
    def run(df, cap=100000, sl=0.02):
        pos, shares, ep, ed = 0, 0, 0, ""
        cash, peak = cap, cap
        trades, equity = [], []

        for idx, r in df.iterrows():
            sig, p = r.get("signal", 0), r["close"]
            # Stop loss
            if pos == 1 and p <= ep * (1 - sl):
                pnl = shares * (p - ep) - 2
                cash += shares * p - 1
                trades.append({"entry":ed,"ep":round(ep,2),"exit":str(idx)[:10],
                    "xp":round(p,2),"shares":shares,"pnl":round(pnl,2),
                    "ret":round((p/ep-1)*100,2)})
                pos, shares = 0, 0
            if sig == 1 and pos == 0:
                shares = int((cash * 0.95) / p)
                if shares > 0:
                    cash -= shares * p + 1
                    ep, ed, pos = p, str(idx)[:10], 1
            elif sig == -1 and pos == 1:
                pnl = shares * (p - ep) - 2
                cash += shares * p - 1
                trades.append({"entry":ed,"ep":round(ep,2),"exit":str(idx)[:10],
                    "xp":round(p,2),"shares":shares,"pnl":round(pnl,2),
                    "ret":round((p/ep-1)*100,2)})
                pos, shares = 0, 0
            eq = cash + shares * p
            peak = max(peak, eq)
            equity.append(eq)

        if pos == 1:
            fp = df.iloc[-1]["close"]
            pnl = shares * (fp - ep) - 2
            trades.append({"entry":ed,"ep":round(ep,2),"exit":str(df.index[-1])[:10],
                "xp":round(fp,2),"shares":shares,"pnl":round(pnl,2),
                "ret":round((fp/ep-1)*100,2)})

        eq = pd.Series(equity, index=df.index)
        return eq, trades, BT._m(eq, trades, cap)

    @staticmethod
    def _m(eq, trades, init):
        f = eq.iloc[-1] if len(eq) > 0 else init
        n = len(eq)
        rets = eq.pct_change().dropna()
        dd = ((eq.cummax() - eq) / eq.cummax() * 100)
        pnls = [t["pnl"] for t in trades]
        w = [p for p in pnls if p > 0]
        lo = [p for p in pnls if p <= 0]
        sharpe = round((rets.mean()/rets.std())*np.sqrt(252),2) if len(rets)>1 and rets.std()>0 else 0
        tl = abs(sum(lo))
        return {
            "ret":round((f/init-1)*100,2),
            "ann":round(((f/init)**(252/max(n,1))-1)*100,2) if n>0 else 0,
            "sharpe":sharpe,
            "mdd":round(dd.max(),2),
            "wr":round(len(w)/len(pnls)*100,1) if pnls else 0,
            "pf":round(sum(w)/tl,2) if tl>0 else 0,
            "nt":len(trades),
            "aw":round(np.mean(w),2) if w else 0,
            "al":round(np.mean(lo),2) if lo else 0,
            "final":round(f,2),
            "dd":dd,
        }


# ——————————————————————————————————————————
# CHART BUILDERS
# ——————————————————————————————————————————
def mk_price(df, fast, slow):
    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.8,0.2],vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index,open=df["open"],high=df["high"],
        low=df["low"],close=df["close"],name="Price",
        increasing_line_color=CL["g"],decreasing_line_color=CL["r"]),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["sma_f"],name=f"SMA({fast})",
        line=dict(color=CL["g"],width=1.5)),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["sma_s"],name=f"SMA({slow})",
        line=dict(color=CL["o"],width=1.5)),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["bb_u"],name="BB",
        line=dict(color="rgba(139,92,246,0.3)",width=1,dash="dot")),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df["bb_l"],name="BB Lower",showlegend=False,
        line=dict(color="rgba(139,92,246,0.3)",width=1,dash="dot"),
        fill="tonexty",fillcolor="rgba(139,92,246,0.04)"),row=1,col=1)
    buys,sells = df[df["signal"]==1], df[df["signal"]==-1]
    fig.add_trace(go.Scatter(x=buys.index,y=buys["close"],mode="markers",name="BUY",
        marker=dict(color=CL["g"],size=11,symbol="triangle-up",line=dict(width=1,color="white"))),row=1,col=1)
    fig.add_trace(go.Scatter(x=sells.index,y=sells["close"],mode="markers",name="SELL",
        marker=dict(color=CL["r"],size=11,symbol="triangle-down",line=dict(width=1,color="white"))),row=1,col=1)
    vc=[CL["g"] if df["close"].iloc[i]>=df["open"].iloc[i] else CL["r"] for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index,y=df["volume"],marker_color=vc,opacity=0.5,showlegend=False),row=2,col=1)
    fig.update_layout(**LO,height=500,xaxis_rangeslider_visible=False)
    return fig

def mk_eq(eq):
    fig=go.Figure(layout=LO)
    fig.add_trace(go.Scatter(x=eq.index,y=eq.values,fill="tozeroy",
        fillcolor="rgba(52,211,153,0.1)",line=dict(color=CL["g"],width=2),name="Equity"))
    fig.update_layout(title="Equity Curve",yaxis_tickformat="$,.0f",height=300)
    return fig

def mk_dd(dd):
    fig=go.Figure(layout=LO)
    fig.add_trace(go.Scatter(x=dd.index,y=dd.values,fill="tozeroy",
        fillcolor="rgba(248,113,113,0.15)",line=dict(color=CL["r"],width=1.5),name="Drawdown %"))
    fig.update_layout(title="Drawdown",yaxis=dict(autorange="reversed"),height=180)
    return fig

def mk_rsi(df):
    fig=go.Figure(layout=LO)
    fig.add_trace(go.Scatter(x=df.index,y=df["rsi"],line=dict(color=CL["p"],width=1.5),name="RSI"))
    fig.add_hline(y=70,line_dash="dash",line_color="rgba(248,113,113,0.5)")
    fig.add_hline(y=30,line_dash="dash",line_color="rgba(52,211,153,0.5)")
    fig.update_layout(title="RSI",yaxis=dict(range=[0,100]),height=220)
    return fig

def mk_macd(df):
    fig=go.Figure(layout=LO)
    fig.add_trace(go.Scatter(x=df.index,y=df["macd"],line=dict(color=CL["b"],width=1.5),name="MACD"))
    fig.add_trace(go.Scatter(x=df.index,y=df["macd_sig"],line=dict(color=CL["o"],width=1.5),name="Signal"))
    hc=[CL["g"] if v>=0 else CL["r"] for v in df["macd_h"]]
    fig.add_trace(go.Bar(x=df.index,y=df["macd_h"],marker_color=hc,name="Hist",opacity=0.6))
    fig.update_layout(title="MACD",height=220)
    return fig


# ——————————————————————————————————————————
# DASH APP
# ——————————————————————————————————————————
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG],
                title="Algo Trading Dashboard",
                meta_tags=[{"name":"viewport","content":"width=device-width,initial-scale=1"}])

server = app.server  # Required for gunicorn on Render

def scard(label, val, color="#e2e8f0"):
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.P(label,style={"fontSize":11,"color":CL["mut"],"margin":0,
               "textTransform":"uppercase","letterSpacing":"0.5px"}),
        html.H4(val,style={"color":color,"margin":"4px 0 0","fontWeight":700})
    ]),style={"background":CL["card"],"border":f"1px solid {CL['bdr']}",
             "borderRadius":12}),md=2,sm=4,xs=6,className="mb-2")

app.layout = dbc.Container([
    html.Div([
        html.H1("Algo Trading Dashboard",
            style={"background":"linear-gradient(135deg,#818cf8,#6366f1,#a78bfa)",
                   "WebkitBackgroundClip":"text","WebkitTextFillColor":"transparent",
                   "fontSize":26,"fontWeight":800,"margin":0}),
        html.P("MA Crossover Strategy • Real Market Data • Backtesting Engine",
               style={"color":CL["mut"],"fontSize":12,"margin":"2px 0 0"})
    ],style={"padding":"14px 0"}),

    dbc.Card(dbc.CardBody(dbc.Row([
        dbc.Col([html.Label("Ticker",style={"fontSize":11,"color":CL["mut"]}),
            dcc.Dropdown(id="tk",options=[{"label":t,"value":t} for t in TICKERS],
                value="SPY",clearable=False,style={"backgroundColor":"#1e293b"})],md=2),
        dbc.Col([html.Label("Period",style={"fontSize":11,"color":CL["mut"]}),
            dcc.Dropdown(id="pr",options=[{"label":l,"value":v} for l,v in
                [("1M","1mo"),("3M","3mo"),("6M","6mo"),("1Y","1y"),("2Y","2y"),("5Y","5y")]],
                value="1y",clearable=False,style={"backgroundColor":"#1e293b"})],md=1),
        dbc.Col([html.Label("Fast MA",style={"fontSize":11,"color":CL["mut"]}),
            dcc.Slider(id="fm",min=3,max=50,value=10,step=1,marks=None,
                tooltip={"placement":"bottom"})],md=2),
        dbc.Col([html.Label("Slow MA",style={"fontSize":11,"color":CL["mut"]}),
            dcc.Slider(id="sm",min=10,max=200,value=30,step=1,marks=None,
                tooltip={"placement":"bottom"})],md=2),
        dbc.Col([html.Label("Stop Loss %",style={"fontSize":11,"color":CL["mut"]}),
            dcc.Slider(id="sl",min=0.5,max=10,value=2,step=0.5,marks=None,
                tooltip={"placement":"bottom"})],md=2),
        dbc.Col([html.Label("RSI Filter",style={"fontSize":11,"color":CL["mut"]}),
            dbc.Checklist(id="rf",options=[{"label":" On","value":"rsi"}],
                value=["rsi"],className="small",style={"marginTop":6})],md=1),
        dbc.Col([html.Br(),
            dbc.Button("Run Backtest",id="btn",color="primary",className="w-100",
                style={"fontWeight":700,"borderRadius":8})],md=2),
    ],className="g-3 align-items-end")),
    style={"background":CL["card"],"border":f"1px solid {CL['bdr']}",
           "borderRadius":12},className="mb-3"),

    dcc.Loading(html.Div(id="stats"),type="circle"),

    dbc.Tabs([
        dbc.Tab(label="Price & Signals",children=[dcc.Loading(dcc.Graph(id="pc"))]),
        dbc.Tab(label="Equity & Risk",children=[
            dcc.Loading(dcc.Graph(id="ec")),dcc.Loading(dcc.Graph(id="dc"))]),
        dbc.Tab(label="Indicators",children=[
            dcc.Loading(dcc.Graph(id="rc")),dcc.Loading(dcc.Graph(id="mc"))]),
        dbc.Tab(label="Trades",children=[html.Div(id="tt",className="mt-3")]),
        dbc.Tab(label="Architecture",children=[html.Div(id="arch",className="mt-3")]),
    ],className="mb-3"),

    html.Div("Algo Trading System • MA Crossover Strategy • Backtester",
        style={"textAlign":"center","fontSize":10,"color":"#334155","padding":"12px 0"})
],fluid=True,style={"backgroundColor":CL["bg"],"minHeight":"100vh","color":CL["txt"]})


@app.callback(
    [Output("stats","children"),Output("pc","figure"),Output("ec","figure"),
     Output("dc","figure"),Output("rc","figure"),Output("mc","figure"),
     Output("tt","children"),Output("arch","children")],
    [Input("btn","n_clicks")],
    [State("tk","value"),State("pr","value"),State("fm","value"),
     State("sm","value"),State("sl","value"),State("rf","value")],
    prevent_initial_call=False
)
def update(n,tk,pr,fm,sm,sl,rf):
    empty = go.Figure(layout=LO)
    df = Data.fetch(tk, pr)
    if df.empty:
        msg = html.Div("No data available",style={"color":CL["r"],"padding":20})
        return msg,empty,empty,empty,empty,empty,msg,msg

    df = Ind.add(df, fm, sm)
    df = Strat.signals(df, "rsi" in (rf or []))
    eq, trades, m = BT.run(df, sl=sl/100)
    dd = m.pop("dd")

    # Stats
    stats = dbc.Row([
        scard("Return",f"{m['ret']}%",CL["g"] if m['ret']>=0 else CL["r"]),
        scard("Sharpe",str(m['sharpe']),CL["g"] if m['sharpe']>=1 else CL["y"]),
        scard("Max DD",f"{m['mdd']}%",CL["r"]),
        scard("Win Rate",f"{m['wr']}%",CL["y"]),
        scard("Profit Factor",str(m['pf']),CL["c"]),
        scard("Trades",str(m['nt'])),
    ])

    # Trade table
    if trades:
        for i,t in enumerate(trades): t["#"]=i+1
        tt = dash_table.DataTable(
            data=trades,page_size=15,sort_action="native",
            columns=[{"name":"#","id":"#"},{"name":"Entry","id":"entry"},
                {"name":"Entry$","id":"ep"},{"name":"Exit","id":"exit"},
                {"name":"Exit$","id":"xp"},{"name":"Shares","id":"shares"},
                {"name":"P&L","id":"pnl"},{"name":"Ret%","id":"ret"}],
            style_header={"backgroundColor":"#1e293b","color":"#94a3b8",
                "fontWeight":"bold","fontSize":11,"border":"1px solid #334155"},
            style_cell={"backgroundColor":CL["card"],"color":CL["txt"],
                "border":"1px solid #1e293b","fontSize":12,"padding":"6px 10px"},
            style_data_conditional=[
                {"if":{"filter_query":"{pnl}>0","column_id":"pnl"},"color":CL["g"],"fontWeight":"bold"},
                {"if":{"filter_query":"{pnl}<=0","column_id":"pnl"},"color":CL["r"],"fontWeight":"bold"},
            ])
    else:
        tt = html.P("No trades — adjust MA periods",style={"color":CL["mut"]})

    # Architecture tab
    steps = [
        ("1. Data Pipeline","Real market data via yfinance. Cached in memory.",
         "Production: FIX feeds, co-located servers, kdb+/q",CL["b"]),
        ("2. Feature Engine","SMA, RSI, Bollinger Bands, MACD — vectorized NumPy/Pandas.",
         "Production: Incremental C++ with O(1) per-tick updates",CL["p"]),
        ("3. Strategy Engine","MA Crossover with RSI filter. Stateless signal generation.",
         "Production: Event-driven C++, sub-microsecond latency",CL["g"]),
        ("4. Backtester","Event-driven sim with commission, stop-loss, drawdown halt.",
         "Production: Tick-level with order book impact modeling",CL["y"]),
        ("5. Risk Mgmt","Sharpe, drawdown, win rate, profit factor analysis.",
         "Production: Real-time VaR, position limits, circuit breakers",CL["r"]),
        ("6. Dashboard","Interactive Plotly Dash web app deployed on Render.",
         "Production: Low-latency C++ GUI or custom React frontend",CL["o"]),
    ]
    arch = html.Div([
        dbc.Card(dbc.CardBody([
            html.H5(t,style={"color":c,"margin":"0 0 6px"}),
            html.P(d,style={"color":CL["txt"],"margin":"0 0 4px","fontSize":13}),
            html.P(p,style={"color":CL["mut"],"fontSize":11,"margin":0}),
        ]),style={"background":CL["card"],"border":f"1px solid {CL['bdr']}",
                  "borderLeft":f"4px solid {c}","borderRadius":10},className="mb-2")
        for t,d,p,c in steps
    ] + [
        dbc.Card(dbc.CardBody([
            html.H5("Interview Talking Points",style={"color":CL["i"]}),
            html.Ul([
                html.Li("Latency: C++ hot path, lock-free queues, kernel bypass, FPGA acceleration"),
                html.Li("Data: kdb+/q for tick storage, Kafka for streaming, columnar analytics"),
                html.Li("Scale: Billions of events/day, partitioning, cross-asset correlation"),
                html.Li("Risk: Real-time position limits, circuit breakers, Greeks-based hedging"),
                html.Li("DevOps: CI/CD with Bazel, Docker/K8s, GCP/AWS, monitoring with Grafana"),
            ],style={"color":CL["txt"],"fontSize":13,"lineHeight":1.8})
        ]),style={"background":"rgba(99,102,241,0.08)","border":"1px solid rgba(99,102,241,0.3)",
                  "borderRadius":10},className="mt-3")
    ])

    return stats,mk_price(df,fm,sm),mk_eq(eq),mk_dd(dd),mk_rsi(df),mk_macd(df),tt,arch


# ——————————————————————————————————————————
# SERVER
# ——————————————————————————————————————————
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print(f"Starting on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)

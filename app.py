# Indian NSE + US tickers
# Display name → (Alpha Vantage symbol, yfinance symbol)
TICKER_MAP = {
    "RELIANCE": ("NSE:RELIANCE", "RELIANCE.NS"),
    "TCS":      ("NSE:TCS", "TCS.NS"),
    "INFY":     ("NSE:INFY", "INFY.NS"),
    "HDFCBANK": ("NSE:HDFCBANK", "HDFCBANK.NS"),
    "ICICIBANK":("NSE:ICICIBANK", "ICICIBANK.NS"),
    "SBIN":     ("NSE:SBIN", "SBIN.NS"),
    "ITC":      ("NSE:ITC", "ITC.NS"),
    "BHARTIARTL":("NSE:BHARTIARTL", "BHARTIARTL.NS"),
    "WIPRO":    ("NSE:WIPRO", "WIPRO.NS"),
    "LT":       ("NSE:LT", "LT.NS"),
    "TATAMOTORS":("NSE:TATAMOTORS", "TATAMOTORS.NS"),
    "HCLTECH":  ("NSE:HCLTECH", "HCLTECH.NS"),
    "MARUTI":   ("NSE:MARUTI", "MARUTI.NS"),
    "ADANIENT": ("NSE:ADANIENT", "ADANIENT.NS"),
    "TATASTEEL":("NSE:TATASTEEL", "TATASTEEL.NS"),
    "NIFTY50":  ("NSE:NIFTY50", "^NSEI"),
    "BANKNIFTY":("NSE:BANKNIFTY", "^NSEBANK"),
    "SPY":      ("SPY", "SPY"),
    "AAPL":     ("AAPL", "AAPL"),
    "NVDA":     ("NVDA", "NVDA"),
}
TICKERS = list(TICKER_MAP.keys())import dash
from dash import html, dcc, Input, Output, State, dash_table, ctx, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
import logging
import json

logging.basicConfig(level=logging.WARNING)

# ——————————————————————————————————————————
# CONFIG
# ——————————————————————————————————————————

# Indian NSE + US tickers
# Display name → (Alpha Vantage symbol, yfinance symbol)
TICKER_MAP = {
    "RELIANCE": ("NSE:RELIANCE", "RELIANCE.NS"),
    "TCS":      ("NSE:TCS", "TCS.NS"),
    "INFY":     ("NSE:INFY", "INFY.NS"),
    "HDFCBANK": ("NSE:HDFCBANK", "HDFCBANK.NS"),
    "ICICIBANK":("NSE:ICICIBANK", "ICICIBANK.NS"),
    "SBIN":     ("NSE:SBIN", "SBIN.NS"),
    "ITC":      ("NSE:ITC", "ITC.NS"),
    "BHARTIARTL":("NSE:BHARTIARTL", "BHARTIARTL.NS"),
    "WIPRO":    ("NSE:WIPRO", "WIPRO.NS"),
    "LT":       ("NSE:LT", "LT.NS"),
    "TATAMOTORS":("NSE:TATAMOTORS", "TATAMOTORS.NS"),
    "HCLTECH":  ("NSE:HCLTECH", "HCLTECH.NS"),
    "MARUTI":   ("NSE:MARUTI", "MARUTI.NS"),
    "ADANIENT": ("NSE:ADANIENT", "ADANIENT.NS"),
    "TATASTEEL":("NSE:TATASTEEL", "TATASTEEL.NS"),
    "NIFTY50":  ("NSE:NIFTY50", "^NSEI"),
    "BANKNIFTY":("NSE:BANKNIFTY", "^NSEBANK"),
    "SPY":      ("SPY", "SPY"),
    "AAPL":     ("AAPL", "AAPL"),
    "NVDA":     ("NVDA", "NVDA"),
}
TICKERS = list(TICKER_MAP.keys())

CL = {"bg":"#0b1121","card":"#0f172a","card2":"#131b2e","bdr":"rgba(255,255,255,0.06)",
      "txt":"#e2e8f0","mut":"#64748b","grid":"rgba(255,255,255,0.04)",
      "g":"#34d399","r":"#f87171","b":"#6366f1","o":"#f97316",
      "p":"#a78bfa","y":"#fbbf24","c":"#22d3ee","i":"#818cf8",
      "g2":"#10b981","r2":"#ef4444","bg2":"#1e293b"}

LO = dict(template="plotly_dark", paper_bgcolor=CL["bg"], plot_bgcolor=CL["card"],
          font=dict(color="#94a3b8",size=11,family="Inter,system-ui,sans-serif"),
          margin=dict(l=55,r=20,t=45,b=35),
          xaxis=dict(gridcolor=CL["grid"],showgrid=True),
          yaxis=dict(gridcolor=CL["grid"],showgrid=True),
          legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,
                      bgcolor="rgba(0,0,0,0)",font=dict(size=10)),
          hoverlabel=dict(bgcolor="#1e293b",bordercolor="#334155",font=dict(color="#e2e8f0",size=12)))


# ——————————————————————————————————————————
# DATA PIPELINE
# ——————————————————————————————————————————
class Data:
    _cache = {}
    _cache_time = {}
    CACHE_TTL = 300
    AV_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
    AV_BASE = "https://www.alphavantage.co/query"
    _source = {}  # Track data source per ticker

    @classmethod
    def _get_symbols(cls, ticker):
        """Get Alpha Vantage and yfinance symbols for a display ticker."""
        if ticker in TICKER_MAP:
            return TICKER_MAP[ticker]
        return (ticker, ticker)

    @classmethod
    def _fetch_alpha_vantage(cls, ticker, period="1y"):
        """Fetch from Alpha Vantage API (primary source)."""
        if not cls.AV_KEY:
            return pd.DataFrame()
        av_sym, _ = cls._get_symbols(ticker)
        try:
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": av_sym,
                "outputsize": "full" if period in ["2y","5y"] else "compact",
                "apikey": cls.AV_KEY
            }
            resp = requests.get(cls.AV_BASE, params=params, timeout=15)
            data = resp.json()

            if "Time Series (Daily)" not in data:
                logging.warning(f"Alpha Vantage: {data.get('Note', data.get('Information', 'No data'))}")
                return pd.DataFrame()

            ts = data["Time Series (Daily)"]
            rows = []
            for date_str, vals in ts.items():
                rows.append({
                    "date": pd.Timestamp(date_str),
                    "open": float(vals["1. open"]),
                    "high": float(vals["2. high"]),
                    "low": float(vals["3. low"]),
                    "close": float(vals["4. close"]),
                    "volume": float(vals["5. volume"]),
                })
            df = pd.DataFrame(rows).set_index("date").sort_index()

            # Filter to requested period
            n_days = {"1mo":30,"3mo":90,"6mo":180,"1y":365,"2y":730,"5y":1825}.get(period, 365)
            cutoff = pd.Timestamp.today() - pd.Timedelta(days=n_days)
            df = df[df.index >= cutoff]
            df = df[df["volume"] > 0]

            if not df.empty:
                logging.info(f"Alpha Vantage: {len(df)} bars for {ticker}")
                cls._source[ticker] = "Alpha Vantage"
            return df

        except Exception as e:
            logging.warning(f"Alpha Vantage error for {ticker}: {e}")
            return pd.DataFrame()

    @classmethod
    def _fetch_alpha_vantage_quote(cls, ticker):
        """Fetch real-time quote from Alpha Vantage."""
        if not cls.AV_KEY:
            return None
        av_sym, _ = cls._get_symbols(ticker)
        try:
            params = {
                "function": "GLOBAL_QUOTE",
                "symbol": av_sym,
                "apikey": cls.AV_KEY
            }
            resp = requests.get(cls.AV_BASE, params=params, timeout=10)
            data = resp.json()
            q = data.get("Global Quote", {})
            if q:
                return {
                    "price": float(q.get("05. price", 0)),
                    "change": float(q.get("10. change percent", "0").replace("%", "")),
                    "volume": int(float(q.get("06. volume", 0))),
                    "prev_close": float(q.get("08. previous close", 0)),
                }
        except Exception as e:
            logging.warning(f"AV quote error {ticker}: {e}")
        return None

    @classmethod
    def _fetch_yfinance(cls, ticker, period="1y"):
        """Backup: yfinance."""
        _, yf_sym = cls._get_symbols(ticker)
        try:
            session = requests.Session()
            session.headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            df = yf.Ticker(yf_sym, session=session).history(period=period)
            if not df.empty:
                df = df[["Open","High","Low","Close","Volume"]].copy()
                df.columns = ["open","high","low","close","volume"]
                df.dropna(inplace=True)
                df = df[df["volume"] > 0]
                if not df.empty:
                    cls._source[ticker] = "Yahoo Finance"
                return df
        except Exception as e:
            logging.warning(f"yfinance error {ticker}: {e}")
        return pd.DataFrame()

    @classmethod
    def _generate_fallback(cls, ticker, period):
        """Last resort: simulated data."""
        cls._source[ticker] = "Simulated"
        np.random.seed(hash(ticker) % 2**31)
        n = {"1mo":22,"3mo":66,"6mo":126,"1y":252,"2y":504,"5y":1260}.get(period, 252)
        price = 100 + np.cumsum(np.random.randn(n)*1.5)
        price = np.maximum(price, 10)
        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
        return pd.DataFrame({"open":price+np.random.randn(n)*0.3,
            "high":price+abs(np.random.randn(n))*1.0,
            "low":price-abs(np.random.randn(n))*1.0,
            "close":price,
            "volume":np.random.randint(1_000_000,10_000_000,n).astype(float)}, index=dates)

    @classmethod
    def fetch(cls, ticker, period="1y"):
        """Fetch data: Alpha Vantage → yfinance → simulated fallback."""
        import time as _time
        key = f"{ticker}_{period}"
        ttl = 60 if period in ["1d","5d"] else cls.CACHE_TTL
        if key in cls._cache and (_time.time() - cls._cache_time.get(key, 0)) < ttl:
            return cls._cache[key].copy()

        # Try Alpha Vantage first
        df = cls._fetch_alpha_vantage(ticker, period)

        # Backup: yfinance
        if df.empty:
            df = cls._fetch_yfinance(ticker, period)

        # Last resort: simulated
        if df.empty:
            df = cls._generate_fallback(ticker, period)

        cls._cache[key] = df
        cls._cache_time[key] = _time.time()
        return df.copy()

    @classmethod
    def get_live_prices(cls, tickers):
        """Fetch latest prices — uses AV quotes for real-time."""
        prices = {}
        for t in tickers[:8]:
            # Try Alpha Vantage real-time quote
            q = cls._fetch_alpha_vantage_quote(t)
            if q and q["price"] > 0:
                prices[t] = {"price": round(q["price"], 2), "change": round(q["change"], 2)}
                continue
            # Fallback to cached data
            try:
                d = cls.fetch(t, "1mo")
                if not d.empty and len(d) >= 2:
                    prices[t] = {"price": round(d["close"].iloc[-1], 2),
                                 "change": round((d["close"].iloc[-1]/d["close"].iloc[-2]-1)*100, 2)}
                else:
                    prices[t] = {"price": 0, "change": 0}
            except:
                prices[t] = {"price": 0, "change": 0}
        return prices

    @classmethod
    def get_source(cls, ticker):
        return cls._source.get(ticker, "Unknown")


# ——————————————————————————————————————————
# TECHNICAL INDICATORS
# ——————————————————————————————————————————
class Ind:
    @staticmethod
    def sma(s, p): return s.rolling(p, min_periods=p).mean()
    @staticmethod
    def ema(s, p): return s.ewm(span=p, adjust=False).mean()
    @staticmethod
    def rsi(s, p=14):
        d = s.diff()
        g = d.where(d>0,0.0).rolling(p,min_periods=p).mean()
        l = (-d.where(d<0,0.0)).rolling(p,min_periods=p).mean()
        return 100-100/(1+g/l)
    @staticmethod
    def bbands(s, p=20, std=2.0):
        m=s.rolling(p).mean(); sd=s.rolling(p).std()
        return m, m+std*sd, m-std*sd
    @staticmethod
    def macd(s):
        ml=s.ewm(span=12).mean()-s.ewm(span=26).mean()
        sl=ml.ewm(span=9).mean()
        return ml, sl, ml-sl
    @staticmethod
    def vwap(df):
        tp=(df["high"]+df["low"]+df["close"])/3
        return (tp*df["volume"]).cumsum()/df["volume"].cumsum()
    @staticmethod
    def atr(df, p=14):
        tr=pd.concat([df["high"]-df["low"],(df["high"]-df["close"].shift()).abs(),
            (df["low"]-df["close"].shift()).abs()],axis=1).max(axis=1)
        return tr.rolling(p).mean()

    @classmethod
    def add(cls, df, fast, slow):
        df=df.copy()
        df["sma_f"]=cls.sma(df["close"],fast)
        df["sma_s"]=cls.sma(df["close"],slow)
        df["ema_f"]=cls.ema(df["close"],fast)
        df["ema_s"]=cls.ema(df["close"],slow)
        df["rsi"]=cls.rsi(df["close"])
        df["bb_m"],df["bb_u"],df["bb_l"]=cls.bbands(df["close"])
        df["macd"],df["macd_sig"],df["macd_h"]=cls.macd(df["close"])
        df["vwap"]=cls.vwap(df)
        df["atr"]=cls.atr(df)
        return df


# ——————————————————————————————————————————
# STRATEGY ENGINE
# ——————————————————————————————————————————
class Strat:
    @staticmethod
    def signals(df, use_rsi=True, use_bb=False, ma_type="SMA"):
        df=df.copy(); df["signal"]=0
        fk = "sma_f" if ma_type=="SMA" else "ema_f"
        sk = "sma_s" if ma_type=="SMA" else "ema_s"
        pf,ps=df[fk].shift(1),df[sk].shift(1)
        buy=(pf<=ps)&(df[fk]>df[sk])
        sell=(pf>=ps)&(df[fk]<df[sk])
        if use_rsi:
            buy=buy&(df["rsi"]<70); sell=sell&(df["rsi"]>30)
        if use_bb:
            buy=buy&(df["close"]<=df["bb_l"]*1.02)
            sell=sell&(df["close"]>=df["bb_u"]*0.98)
        df.loc[buy,"signal"]=1; df.loc[sell,"signal"]=-1
        return df


# ——————————————————————————————————————————
# BACKTESTER
# ——————————————————————————————————————————
class BT:
    @staticmethod
    def run(df, cap=100000, sl=0.02):
        pos,shares,ep,ed=0,0,0,""
        cash,peak=cap,cap
        trades,equity=[],[]
        for idx,r in df.iterrows():
            sig,p=r.get("signal",0),r["close"]
            if pos==1 and p<=ep*(1-sl):
                pnl=shares*(p-ep)-2; cash+=shares*p-1
                trades.append({"entry":ed,"ep":round(ep,2),"exit":str(idx)[:10],
                    "xp":round(p,2),"shares":shares,"pnl":round(pnl,2),
                    "ret":round((p/ep-1)*100,2),"type":"Stop Loss"})
                pos,shares=0,0
            if sig==1 and pos==0:
                shares=int((cash*0.95)/p)
                if shares>0: cash-=shares*p+1; ep,ed,pos=p,str(idx)[:10],1
            elif sig==-1 and pos==1:
                pnl=shares*(p-ep)-2; cash+=shares*p-1
                trades.append({"entry":ed,"ep":round(ep,2),"exit":str(idx)[:10],
                    "xp":round(p,2),"shares":shares,"pnl":round(pnl,2),
                    "ret":round((p/ep-1)*100,2),"type":"Signal"})
                pos,shares=0,0
            equity.append(cash+shares*p)
            peak=max(peak,equity[-1])
        if pos==1:
            fp=df.iloc[-1]["close"]; pnl=shares*(fp-ep)-2
            trades.append({"entry":ed,"ep":round(ep,2),"exit":str(df.index[-1])[:10],
                "xp":round(fp,2),"shares":shares,"pnl":round(pnl,2),
                "ret":round((fp/ep-1)*100,2),"type":"Open"})
        eq=pd.Series(equity,index=df.index)
        return eq,trades,BT._m(eq,trades,cap)

    @staticmethod
    def buy_and_hold(df, cap=100000):
        """Buy & hold benchmark."""
        shares=int((cap*0.95)/df["close"].iloc[0])
        rem=cap-shares*df["close"].iloc[0]
        eq=shares*df["close"]+rem
        ret=round((eq.iloc[-1]/cap-1)*100,2)
        rets=eq.pct_change().dropna()
        sharpe=round((rets.mean()/rets.std())*np.sqrt(252),2) if len(rets)>1 and rets.std()>0 else 0
        dd=((eq.cummax()-eq)/eq.cummax()*100)
        return eq, {"ret":ret,"sharpe":sharpe,"mdd":round(dd.max(),2)}

    @staticmethod
    def optimize(df, cap=100000, sl=0.02):
        """Find best fast/slow MA combination."""
        best={"sharpe":-999}; results=[]
        for f in range(5,31,5):
            for s in range(20,101,10):
                if f>=s: continue
                d=Ind.add(df.copy(),f,s)
                d=Strat.signals(d,True,False,"SMA")
                eq,_,m=BT.run(d,cap,sl)
                results.append({"fast":f,"slow":s,"ret":m["ret"],"sharpe":m["sharpe"],
                    "mdd":m["mdd"],"wr":m["wr"],"trades":m["nt"]})
                if m["sharpe"]>best["sharpe"]: best={**m,"fast":f,"slow":s}
        return results, best

    @staticmethod
    def _m(eq,trades,init):
        f=eq.iloc[-1] if len(eq)>0 else init; n=len(eq)
        rets=eq.pct_change().dropna()
        dd=((eq.cummax()-eq)/eq.cummax()*100)
        pnls=[t["pnl"] for t in trades]
        w=[p for p in pnls if p>0]; lo=[p for p in pnls if p<=0]
        sharpe=round((rets.mean()/rets.std())*np.sqrt(252),2) if len(rets)>1 and rets.std()>0 else 0
        tl=abs(sum(lo))
        # Calmar ratio
        mdd_dec=dd.max()/100
        ann_ret=((f/init)**(252/max(n,1))-1) if n>0 else 0
        calmar=round(ann_ret/mdd_dec,2) if mdd_dec>0 else 0
        # Sortino
        neg_rets=rets[rets<0]
        sortino=round((rets.mean()/neg_rets.std())*np.sqrt(252),2) if len(neg_rets)>0 and neg_rets.std()>0 else 0
        return {
            "ret":round((f/init-1)*100,2),
            "ann":round(ann_ret*100,2),
            "sharpe":sharpe,"sortino":sortino,"calmar":calmar,
            "mdd":round(dd.max(),2),
            "wr":round(len(w)/len(pnls)*100,1) if pnls else 0,
            "pf":round(sum(w)/tl,2) if tl>0 else 0,
            "nt":len(trades),
            "aw":round(np.mean(w),2) if w else 0,
            "al":round(np.mean(lo),2) if lo else 0,
            "final":round(f,2),"dd":dd,
            "max_win":round(max(pnls),2) if pnls else 0,
            "max_loss":round(min(pnls),2) if pnls else 0,
            "expectancy":round(np.mean(pnls),2) if pnls else 0,
        }


# ——————————————————————————————————————————
# CHART BUILDERS
# ——————————————————————————————————————————
def mk_price(df, fast, slow, ma_type, show_bb, show_vwap):
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,
        row_heights=[0.65,0.15,0.20],vertical_spacing=0.02,
        subplot_titles=("","Volume","RSI"))
    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index,open=df["open"],high=df["high"],
        low=df["low"],close=df["close"],name="Price",
        increasing_line_color=CL["g"],decreasing_line_color=CL["r"]),row=1,col=1)
    # MAs
    fk="sma_f" if ma_type=="SMA" else "ema_f"
    sk="sma_s" if ma_type=="SMA" else "ema_s"
    fig.add_trace(go.Scatter(x=df.index,y=df[fk],name=f"{ma_type}({fast})",
        line=dict(color=CL["g"],width=1.5)),row=1,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=df[sk],name=f"{ma_type}({slow})",
        line=dict(color=CL["o"],width=1.5)),row=1,col=1)
    # Bollinger Bands
    if show_bb:
        fig.add_trace(go.Scatter(x=df.index,y=df["bb_u"],name="BB Upper",
            line=dict(color="rgba(139,92,246,0.3)",width=1,dash="dot"),showlegend=True),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=df["bb_l"],name="BB Lower",showlegend=False,
            line=dict(color="rgba(139,92,246,0.3)",width=1,dash="dot"),
            fill="tonexty",fillcolor="rgba(139,92,246,0.04)"),row=1,col=1)
    # VWAP
    if show_vwap:
        fig.add_trace(go.Scatter(x=df.index,y=df["vwap"],name="VWAP",
            line=dict(color=CL["c"],width=1.2,dash="dash")),row=1,col=1)
    # Signals
    buys,sells=df[df["signal"]==1],df[df["signal"]==-1]
    fig.add_trace(go.Scatter(x=buys.index,y=buys["close"],mode="markers",name="BUY",
        marker=dict(color=CL["g"],size=12,symbol="triangle-up",
            line=dict(width=2,color="white")),
        hovertemplate="<b>BUY</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>"),row=1,col=1)
    fig.add_trace(go.Scatter(x=sells.index,y=sells["close"],mode="markers",name="SELL",
        marker=dict(color=CL["r"],size=12,symbol="triangle-down",
            line=dict(width=2,color="white")),
        hovertemplate="<b>SELL</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>"),row=1,col=1)
    # Volume
    vc=[CL["g"] if df["close"].iloc[i]>=df["open"].iloc[i] else CL["r"] for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index,y=df["volume"],marker_color=vc,opacity=0.5,
        showlegend=False,hovertemplate="Vol: %{y:,.0f}<extra></extra>"),row=2,col=1)
    # RSI
    fig.add_trace(go.Scatter(x=df.index,y=df["rsi"],line=dict(color=CL["p"],width=1.5),
        name="RSI",showlegend=False,hovertemplate="RSI: %{y:.1f}<extra></extra>"),row=3,col=1)
    fig.add_hline(y=70,line_dash="dash",line_color="rgba(248,113,113,0.4)",row=3,col=1)
    fig.add_hline(y=30,line_dash="dash",line_color="rgba(52,211,153,0.4)",row=3,col=1)
    fig.add_hrect(y0=30,y1=70,fillcolor="rgba(255,255,255,0.02)",line_width=0,row=3,col=1)
    fig.update_yaxes(range=[0,100],row=3,col=1)
    fig.update_layout(**LO,height=680,xaxis_rangeslider_visible=False,
        hovermode="x unified")
    return fig

def mk_equity(eq, bh_eq, m, bh_m):
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[0.65,0.35],vertical_spacing=0.05)
    # Strategy equity
    fig.add_trace(go.Scatter(x=eq.index,y=eq.values,fill="tozeroy",
        fillcolor="rgba(52,211,153,0.08)",line=dict(color=CL["g"],width=2),
        name=f"Strategy ({m['ret']}%)",
        hovertemplate="Strategy: $%{y:,.2f}<extra></extra>"),row=1,col=1)
    # Buy & hold equity
    fig.add_trace(go.Scatter(x=bh_eq.index,y=bh_eq.values,
        line=dict(color=CL["mut"],width=1.5,dash="dash"),
        name=f"Buy & Hold ({bh_m['ret']}%)",
        hovertemplate="B&H: $%{y:,.2f}<extra></extra>"),row=1,col=1)
    # Drawdown
    dd=m["dd"]
    fig.add_trace(go.Scatter(x=dd.index,y=dd.values,fill="tozeroy",
        fillcolor="rgba(248,113,113,0.12)",line=dict(color=CL["r"],width=1.5),
        name="Drawdown %",hovertemplate="DD: %{y:.2f}%<extra></extra>"),row=2,col=1)
    fig.update_layout(**LO,height=450,hovermode="x unified")
    fig.update_yaxes(title_text="Equity ($)",tickformat="$,.0f",row=1,col=1)
    fig.update_yaxes(title_text="Drawdown %",autorange="reversed",row=2,col=1)
    return fig

def mk_macd(df):
    fig=go.Figure(layout=LO)
    fig.add_trace(go.Scatter(x=df.index,y=df["macd"],line=dict(color=CL["b"],width=1.5),name="MACD"))
    fig.add_trace(go.Scatter(x=df.index,y=df["macd_sig"],line=dict(color=CL["o"],width=1.5),name="Signal"))
    hc=[CL["g"] if v>=0 else CL["r"] for v in df["macd_h"]]
    fig.add_trace(go.Bar(x=df.index,y=df["macd_h"],marker_color=hc,name="Histogram",opacity=0.6))
    fig.update_layout(title="MACD (12, 26, 9)",height=250,hovermode="x unified")
    return fig

def mk_trade_scatter(trades):
    if not trades: return go.Figure(layout=LO).update_layout(height=300)
    pnls=[t["pnl"] for t in trades]
    rets=[t["ret"] for t in trades]
    colors=[CL["g"] if p>0 else CL["r"] for p in pnls]
    sizes=[max(8,min(25,abs(p)/50)) for p in pnls]
    fig=go.Figure(layout=LO)
    fig.add_trace(go.Scatter(x=list(range(1,len(pnls)+1)),y=pnls,mode="markers",
        marker=dict(color=colors,size=sizes,line=dict(width=1,color="white"),opacity=0.8),
        text=[f"#{i+1}: ${p:+,.2f} ({r:+.2f}%)<br>{t['entry']} → {t['exit']}<br>Type: {t.get('type','')}"
              for i,(t,p,r) in enumerate(zip(trades,pnls,rets))],
        hovertemplate="%{text}<extra></extra>",name="Trades"))
    fig.add_hline(y=0,line_color=CL["mut"],line_dash="dash",line_width=0.5)
    # Cumulative P&L line
    cum=np.cumsum(pnls)
    fig.add_trace(go.Scatter(x=list(range(1,len(cum)+1)),y=cum.tolist(),
        line=dict(color=CL["b"],width=2),name="Cumulative P&L",yaxis="y2"))
    fig.update_layout(title="Trade Analysis (bubble size = P&L magnitude)",height=350,
        yaxis2=dict(overlaying="y",side="right",gridcolor=CL["grid"],
            title="Cumulative P&L",tickformat="$,.0f"),
        yaxis=dict(title="Per-Trade P&L",tickformat="$,.0f"))
    return fig

def mk_heatmap(results):
    if not results: return go.Figure(layout=LO)
    df=pd.DataFrame(results)
    pivot=df.pivot_table(values="sharpe",index="fast",columns="slow",aggfunc="first")
    fig=go.Figure(data=go.Heatmap(
        z=pivot.values, x=[str(c) for c in pivot.columns],
        y=[str(i) for i in pivot.index],
        colorscale=[[0,"#f87171"],[0.5,"#1e293b"],[1,"#34d399"]],
        text=np.round(pivot.values,2), texttemplate="%{text}",
        textfont=dict(size=10,color="white"),
        hovertemplate="Fast: %{y}<br>Slow: %{x}<br>Sharpe: %{z:.2f}<extra></extra>",
        colorbar=dict(title="Sharpe")))
    fig.update_layout(**LO,title="Parameter Optimization — Sharpe Ratio Heatmap",
        height=350,xaxis_title="Slow MA Period",yaxis_title="Fast MA Period")
    return fig

def mk_monthly(eq):
    """Monthly returns heatmap."""
    rets=eq.resample("ME").last().pct_change().dropna()*100
    if len(rets)<2: return go.Figure(layout=LO).update_layout(height=250)
    df=pd.DataFrame({"ret":rets})
    df["year"]=df.index.year; df["month"]=df.index.month
    month_names=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot=df.pivot_table(values="ret",index="year",columns="month",aggfunc="first")
    pivot.columns=[month_names[int(c)-1] for c in pivot.columns]
    fig=go.Figure(data=go.Heatmap(
        z=pivot.values,x=pivot.columns.tolist(),y=[str(y) for y in pivot.index],
        colorscale=[[0,"#f87171"],[0.5,"#1e293b"],[1,"#34d399"]],
        text=np.round(pivot.values,1),texttemplate="%{text}%",
        textfont=dict(size=10),zmid=0,
        hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>",
        colorbar=dict(title="%")))
    fig.update_layout(**LO,title="Monthly Returns Heatmap",height=250)
    return fig


# ——————————————————————————————————————————
# DASH APP
# ——————————————————————————————————————————
app = dash.Dash(__name__,
    external_stylesheets=[dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"],
    title="Algo Trading Dashboard",
    meta_tags=[{"name":"viewport","content":"width=device-width,initial-scale=1"}])

server = app.server

def scard(label,val,color="#e2e8f0",sub=None):
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.P(label,style={"fontSize":10,"color":CL["mut"],"margin":0,
            "textTransform":"uppercase","letterSpacing":"0.8px","fontWeight":600}),
        html.H4(val,style={"color":color,"margin":"4px 0 0","fontWeight":700,"fontSize":20}),
        html.P(sub,style={"fontSize":10,"color":CL["mut"],"margin":"2px 0 0"}) if sub else None
    ]),style={"background":CL["card"],"border":f"1px solid {CL['bdr']}",
        "borderRadius":12,"transition":"transform 0.2s","cursor":"default"},
        className="stat-card"),md=True,sm=4,xs=6,className="mb-2")


# Ticker tape
def ticker_tape():
    prices = Data.get_live_prices(TICKERS[:8])
    items = []
    for t,d in prices.items():
        c = CL["g"] if d["change"]>=0 else CL["r"]
        arrow = "▲" if d["change"]>=0 else "▼"
        items.append(html.Span([
            html.Span(f"{t} ",style={"fontWeight":700,"color":CL["txt"]}),
            html.Span(f"${d['price']} ",style={"color":CL["txt"]}),
            html.Span(f"{arrow}{abs(d['change'])}%",style={"color":c,"fontWeight":600}),
        ],style={"marginRight":30,"fontSize":12}))
    return items

app.layout = dbc.Container([
    # Ticker tape
    html.Div(
        html.Div(ticker_tape(),style={"display":"flex","whiteSpace":"nowrap","padding":"8px 0",
            "animation":"scroll 30s linear infinite"}),
        style={"overflow":"hidden","background":"rgba(99,102,241,0.06)",
            "borderBottom":f"1px solid {CL['bdr']}","marginBottom":12,"borderRadius":8}),

    # Header
    html.Div([
        html.Div([
            html.H1("Algo Trading Dashboard",
                style={"background":"linear-gradient(135deg,#818cf8,#6366f1,#a78bfa)",
                    "WebkitBackgroundClip":"text","WebkitTextFillColor":"transparent",
                    "fontSize":28,"fontWeight":800,"margin":0,"letterSpacing":"-0.5px"}),
            html.P("MA Crossover Strategy • Real Market Data • Backtesting Engine • Parameter Optimization",
                style={"color":CL["mut"],"fontSize":12,"margin":"2px 0 0"})
        ],style={"flex":1}),
    ],style={"padding":"8px 0 16px","display":"flex","alignItems":"center"}),

    # Controls
    dbc.Card(dbc.CardBody([
        dbc.Row([
            dbc.Col([html.Label("Ticker",style={"fontSize":10,"color":CL["mut"],"fontWeight":600}),
                dcc.Dropdown(id="tk",options=[{"label":f"{'🇮🇳 ' if t not in ['SPY','AAPL','NVDA'] else '🇺🇸 '}{t}","value":t} for t in TICKERS],
                    value="RELIANCE",clearable=False,style={"backgroundColor":"#1e293b"})],md=2),
            dbc.Col([html.Label("Period",style={"fontSize":10,"color":CL["mut"],"fontWeight":600}),
                dcc.Dropdown(id="pr",options=[{"label":l,"value":v} for l,v in
                    [("1M","1mo"),("3M","3mo"),("6M","6mo"),("1Y","1y"),("2Y","2y"),("5Y","5y")]],
                    value="1y",clearable=False,style={"backgroundColor":"#1e293b"})],md=1),
            dbc.Col([html.Label("MA Type",style={"fontSize":10,"color":CL["mut"],"fontWeight":600}),
                dcc.Dropdown(id="mat",options=[{"label":"SMA","value":"SMA"},{"label":"EMA","value":"EMA"}],
                    value="SMA",clearable=False,style={"backgroundColor":"#1e293b"})],md=1),
            dbc.Col([html.Label(id="fm-label",style={"fontSize":10,"color":CL["mut"],"fontWeight":600}),
                dcc.Slider(id="fm",min=3,max=50,value=10,step=1,marks=None,
                    tooltip={"placement":"bottom"})],md=2),
            dbc.Col([html.Label(id="sm-label",style={"fontSize":10,"color":CL["mut"],"fontWeight":600}),
                dcc.Slider(id="sm",min=10,max=200,value=30,step=1,marks=None,
                    tooltip={"placement":"bottom"})],md=2),
            dbc.Col([html.Label(id="sl-label",style={"fontSize":10,"color":CL["mut"],"fontWeight":600}),
                dcc.Slider(id="sl",min=0.5,max=10,value=2,step=0.5,marks=None,
                    tooltip={"placement":"bottom"})],md=1),
            dbc.Col([html.Label("Overlays",style={"fontSize":10,"color":CL["mut"],"fontWeight":600}),
                dbc.Checklist(id="ov",options=[{"label":" BB","value":"bb"},
                    {"label":" VWAP","value":"vwap"},{"label":" RSI","value":"rsi"}],
                    value=["rsi"],inline=True,className="small",style={"fontSize":11})],md=2),
            dbc.Col([html.Br(),dbc.ButtonGroup([
                dbc.Button("Run",id="btn",color="primary",style={"fontWeight":700,"borderRadius":"8px 0 0 8px"}),
                dbc.Button("Optimize",id="opt-btn",color="info",outline=True,
                    style={"fontWeight":600,"borderRadius":"0 8px 8px 0","fontSize":12}),
            ],className="w-100")],md=1),
        ],className="g-2 align-items-end"),
    ]),style={"background":CL["card"],"border":f"1px solid {CL['bdr']}","borderRadius":12},className="mb-3"),

    # Stats row
    dcc.Loading(html.Div(id="stats"),type="circle",color=CL["b"]),

    # Main content
    dbc.Tabs([
        dbc.Tab(label="📈 Price & Signals",tab_id="t1",children=[
            dcc.Loading(dcc.Graph(id="pc",config={"displayModeBar":True,"modeBarButtonsToAdd":
                ["drawline","drawopenpath","eraseshape"]}))]),
        dbc.Tab(label="💰 Equity vs Benchmark",tab_id="t2",children=[
            dcc.Loading(dcc.Graph(id="ec")),
            dcc.Loading(dcc.Graph(id="monthly"))]),
        dbc.Tab(label="📊 MACD & Trades",tab_id="t3",children=[
            dcc.Loading(dcc.Graph(id="mc")),
            dcc.Loading(dcc.Graph(id="ts"))]),
        dbc.Tab(label="📋 Trade Log",tab_id="t4",children=[html.Div(id="tt",className="mt-3")]),
        dbc.Tab(label="🔬 Optimizer",tab_id="t5",children=[
            dcc.Loading(dcc.Graph(id="hm")),
            html.Div(id="opt-result",className="mt-3")]),
        dbc.Tab(label="🏗️ Architecture",tab_id="t6",children=[html.Div(id="arch",className="mt-3")]),
    ],active_tab="t1",className="mb-3"),

    # Footer
    html.Div([
        html.Span("Algo Trading System",style={"fontWeight":700,"color":CL["i"]}),
        html.Span(" • MA Crossover Strategy • Backtester • Parameter Optimizer",style={"color":CL["mut"]}),
    ],style={"textAlign":"center","fontSize":11,"padding":"16px 0","borderTop":f"1px solid {CL['bdr']}"}),

    # Auto-refresh every 5 minutes
    dcc.Interval(id="auto-refresh", interval=5*60*1000, n_intervals=0),

    # Data source indicator
    html.Div(id="data-source", style={"position":"fixed","bottom":10,"right":10,
        "fontSize":10,"padding":"4px 10px","borderRadius":6,
        "background":"rgba(15,23,42,0.9)","border":f"1px solid {CL['bdr']}","zIndex":999}),

    # Hidden stores
    dcc.Store(id="opt-data"),
],fluid=True,style={"backgroundColor":CL["bg"],"minHeight":"100vh","color":CL["txt"],
    "fontFamily":"Inter,system-ui,sans-serif"})


# --- Slider labels ---
@app.callback(Output("fm-label","children"),[Input("fm","value")])
def fl(v): return f"Fast MA: {v}"
@app.callback(Output("sm-label","children"),[Input("sm","value")])
def sl2(v): return f"Slow MA: {v}"
@app.callback(Output("sl-label","children"),[Input("sl","value")])
def sl3(v): return f"SL: {v}%"

# --- Main callback ---
@app.callback(
    [Output("stats","children"),Output("pc","figure"),Output("ec","figure"),
     Output("monthly","figure"),Output("mc","figure"),Output("ts","figure"),
     Output("tt","children"),Output("arch","children"),Output("data-source","children")],
    [Input("btn","n_clicks"),Input("auto-refresh","n_intervals")],
    [State("tk","value"),State("pr","value"),State("fm","value"),
     State("sm","value"),State("sl","value"),State("ov","value"),
     State("mat","value")],
    prevent_initial_call=False
)
def update(n,nint,tk,pr,fm,sm,sl,ov,mat):
    empty=go.Figure(layout=LO)
    ov=ov or []
    is_indian = tk not in ["SPY","AAPL","NVDA"]
    currency = "₹" if is_indian else "$"

    # Clear cache on refresh to get fresh data
    if ctx.triggered_id == "auto-refresh":
        Data._cache.clear()
        Data._cache_time.clear()

    df=Data.fetch(tk,pr)
    if df.empty:
        msg=html.Div("No data",style={"color":CL["r"],"padding":20})
        return msg,empty,empty,empty,empty,empty,msg,msg,""

    # Detect source
    source = Data.get_source(tk)
    src_color = CL["g"] if source == "Alpha Vantage" else (CL["y"] if source == "Yahoo Finance" else CL["r"])
    market = "NSE" if is_indian else "US"
    source_badge = html.Span([
        html.Span("● ", style={"color": src_color}),
        html.Span(f"{source} • {market}:{tk} • {len(df)} bars • ", style={"color": CL["txt"]}),
        html.Span(f"Updated {pd.Timestamp.now().strftime('%H:%M:%S')}", style={"color": CL["mut"]}),
    ])

    df=Ind.add(df,fm,sm)
    df=Strat.signals(df,"rsi" in ov,"bb" in ov,mat)
    eq,trades,m=BT.run(df,sl=sl/100)
    bh_eq,bh_m=BT.buy_and_hold(df)
    dd=m.pop("dd")
    m["dd"]=dd

    # Stats
    alpha=round(m["ret"]-bh_m["ret"],2)
    alpha_c=CL["g"] if alpha>=0 else CL["r"]
    stats=dbc.Row([
        scard("Total Return",f"{m['ret']}%",CL["g"] if m['ret']>=0 else CL["r"],
              f"B&H: {bh_m['ret']}%"),
        scard("Alpha",f"{alpha:+.2f}%",alpha_c,"vs Buy & Hold"),
        scard("Sharpe",str(m['sharpe']),CL["g"] if m['sharpe']>=1 else CL["y"],
              f"Sortino: {m['sortino']}"),
        scard("Max DD",f"{m['mdd']}%",CL["r"],f"Calmar: {m['calmar']}"),
        scard("Win Rate",f"{m['wr']}%",CL["y"],f"{m['nt']} trades"),
        scard("Profit Factor",str(m['pf']),CL["c"],
              f"Exp: {currency}{m['expectancy']}"),
    ])

    # Trade table
    if trades:
        for i,t in enumerate(trades): t["#"]=i+1
        tt=dash_table.DataTable(
            data=trades,page_size=15,sort_action="native",filter_action="native",
            columns=[{"name":"#","id":"#"},{"name":"Type","id":"type"},
                {"name":"Entry","id":"entry"},{"name":"Entry$","id":"ep"},
                {"name":"Exit","id":"exit"},{"name":"Exit$","id":"xp"},
                {"name":"Shares","id":"shares"},{"name":"P&L","id":"pnl"},
                {"name":"Ret%","id":"ret"}],
            style_header={"backgroundColor":"#1e293b","color":"#94a3b8",
                "fontWeight":"bold","fontSize":11,"border":"1px solid #334155"},
            style_cell={"backgroundColor":CL["card"],"color":CL["txt"],
                "border":"1px solid #1e293b","fontSize":12,"padding":"8px 12px"},
            style_data_conditional=[
                {"if":{"filter_query":"{pnl}>0","column_id":"pnl"},"color":CL["g"],"fontWeight":"bold"},
                {"if":{"filter_query":"{pnl}<=0","column_id":"pnl"},"color":CL["r"],"fontWeight":"bold"},
                {"if":{"filter_query":"{ret}>0","column_id":"ret"},"color":CL["g"]},
                {"if":{"filter_query":"{ret}<=0","column_id":"ret"},"color":CL["r"]},
                {"if":{"filter_query":'{type}="Stop Loss"'},"backgroundColor":"rgba(248,113,113,0.08)"},
            ],
            style_filter={"backgroundColor":"#1e293b","color":CL["txt"]},
            export_format="csv")
    else:
        tt=html.P("No trades — adjust parameters",style={"color":CL["mut"]})

    # Architecture
    steps=[
        ("1. Data Pipeline","Real market data via yfinance with caching and fallback. Supports 15+ tickers.",
         "Production: FIX protocol feeds, co-located servers, kernel bypass (DPDK), kdb+/q storage",CL["b"]),
        ("2. Feature Engine","SMA, EMA, RSI, Bollinger Bands, MACD, VWAP, ATR — all vectorized.",
         "Production: Incremental C++ with O(1) per-tick updates on streaming data",CL["p"]),
        ("3. Strategy Engine","MA Crossover with RSI & Bollinger filters. Supports SMA/EMA toggle.",
         "Production: Event-driven C++ engine, sub-microsecond decision latency",CL["g"]),
        ("4. Backtester","Event-driven simulation with stop-loss, commission modeling, and position sizing.",
         "Production: Tick-level with order book impact, partial fills, slippage modeling",CL["y"]),
        ("5. Risk & Analytics","Sharpe, Sortino, Calmar ratios. VaR, drawdown, monthly returns heatmap.",
         "Production: Real-time VaR/CVaR, position limits, circuit breakers, Greeks hedging",CL["r"]),
        ("6. Parameter Optimizer","Grid search across MA combinations. Sharpe ratio heatmap visualization.",
         "Production: Bayesian optimization, walk-forward analysis, out-of-sample testing",CL["c"]),
        ("7. Dashboard","Interactive Plotly Dash with drawing tools, CSV export, live ticker tape.",
         "Production: Low-latency C++ GUI or custom React frontend with WebSocket feeds",CL["o"]),
    ]
    arch=html.Div([
        dbc.Card(dbc.CardBody([
            html.H5(t,style={"color":c,"margin":"0 0 6px","fontWeight":700}),
            html.P(d,style={"color":CL["txt"],"margin":"0 0 4px","fontSize":13}),
            html.P(f"🏭 {p}",style={"color":CL["mut"],"fontSize":11,"margin":0,"fontStyle":"italic"}),
        ]),style={"background":CL["card"],"border":f"1px solid {CL['bdr']}",
            "borderLeft":f"4px solid {c}","borderRadius":10},className="mb-2")
        for t,d,p,c in steps
    ]+[dbc.Card(dbc.CardBody([
        html.H5("🎤 Interview Talking Points",style={"color":CL["i"],"fontWeight":700}),
        html.Ul([
            html.Li([html.Strong("Latency: "),"C++ hot path, lock-free queues, kernel bypass (DPDK/Solarflare), FPGA acceleration"]),
            html.Li([html.Strong("Data: "),"kdb+/q for tick storage, Kafka for streaming, ClickHouse for analytics"]),
            html.Li([html.Strong("Scale: "),"Billions of events/day, horizontal partitioning, cross-asset correlation"]),
            html.Li([html.Strong("Risk: "),"Real-time position limits, circuit breakers, Greeks-based delta hedging"]),
            html.Li([html.Strong("ML/AI: "),"Feature engineering on order flow, reinforcement learning for execution"]),
            html.Li([html.Strong("DevOps: "),"Bazel/Buck2 builds, Docker/K8s, GCP/AWS, Grafana monitoring, CI/CD"]),
        ],style={"color":CL["txt"],"fontSize":13,"lineHeight":2})
    ]),style={"background":"rgba(99,102,241,0.08)","border":"1px solid rgba(99,102,241,0.3)",
        "borderRadius":10},className="mt-3")])

    return (stats,mk_price(df,fm,sm,mat,"bb" in ov,"vwap" in ov),
            mk_equity(eq,bh_eq,m,bh_m),mk_monthly(eq),mk_macd(df),
            mk_trade_scatter(trades),tt,arch,source_badge)


# --- Optimizer callback ---
@app.callback(
    [Output("hm","figure"),Output("opt-result","children")],
    [Input("opt-btn","n_clicks")],
    [State("tk","value"),State("pr","value"),State("sl","value")],
    prevent_initial_call=True
)
def optimize(n,tk,pr,sl):
    df=Data.fetch(tk,pr)
    if df.empty:
        return go.Figure(layout=LO),html.Div("No data")
    results,best=BT.optimize(df,sl=sl/100)
    fig=mk_heatmap(results)
    info=dbc.Card(dbc.CardBody([
        html.H5("🏆 Optimal Parameters Found",style={"color":CL["g"],"fontWeight":700}),
        html.P([
            html.Span(f"Fast MA: {best.get('fast','?')}",style={"color":CL["g"],"fontWeight":700,"marginRight":20}),
            html.Span(f"Slow MA: {best.get('slow','?')}",style={"color":CL["o"],"fontWeight":700,"marginRight":20}),
            html.Span(f"Sharpe: {best.get('sharpe','?')}",style={"color":CL["y"],"fontWeight":700,"marginRight":20}),
            html.Span(f"Return: {best.get('ret','?')}%",style={"color":CL["c"],"fontWeight":700}),
        ],style={"fontSize":14}),
        html.P("⚠️ Caution: Optimized parameters may be overfit to historical data. "
               "Use walk-forward analysis in production.",
               style={"color":CL["mut"],"fontSize":11,"margin":"8px 0 0","fontStyle":"italic"})
    ]),style={"background":"rgba(52,211,153,0.06)","border":"1px solid rgba(52,211,153,0.2)",
        "borderRadius":10},className="mt-3")
    return fig,info


if __name__=="__main__":
    port=int(os.environ.get("PORT",8050))
    app.run(host="0.0.0.0",port=port,debug=False)

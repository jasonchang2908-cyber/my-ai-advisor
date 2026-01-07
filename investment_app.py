import streamlit as st
import yfinance as yf
import pandas as pd
from openai import OpenAI
import google.generativeai as genai
import os 
from datetime import datetime, date, timedelta
import plotly.graph_objects as go 
import plotly.express as px 
import requests 
import xml.etree.ElementTree as ET

# --- 1. 頁面設定 ---
st.set_page_config(page_title="AI 投資指揮中心", layout="wide", initial_sidebar_state="expanded")

# --- 2. Session & Keys ---
if "messages" not in st.session_state: st.session_state.messages = []
if "openai_key" not in st.session_state: st.session_state.openai_key = st.secrets.get("OPENAI_API_KEY", "")
if "gemini_key" not in st.session_state: st.session_state.gemini_key = st.secrets.get("GEMINI_API_KEY", "")
if "tool_results" not in st.session_state:
    st.session_state.tool_results = {"stock_diagnosis": None, "stock_hunter": None, "portfolio_check": None}
if "daily_briefing" not in st.session_state: st.session_state.daily_briefing = None
# 新增：登入狀態與身份
if "user_role" not in st.session_state: st.session_state.user_role = None 

# --- 設定密碼 ---
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "admin123") # 您自己的密碼
GUEST_PASSWORD = "guest"     # 給朋友的密碼

# --- 3. 核心邏輯 (Google Sheets) ---
try:
    from streamlit_gsheets import GSheetsConnection
    try:
        conn = st.connection("gsheets", type=GSheetsConnection)
        CONNECTION_STATUS = True
    except Exception as e:
        CONNECTION_STATUS = False
        CONNECTION_ERROR = str(e)
except ImportError:
    st.error("⚠️ 嚴重錯誤：缺少套件。請在終端機執行 `pip install st-gsheets-connection`")
    st.stop()

def load_data():
    if st.session_state.user_role != "Admin": return pd.DataFrame(columns=["Date", "Account", "Action", "Symbol", "Price", "Shares"])
    if not CONNECTION_STATUS: return pd.DataFrame(columns=["Date", "Account", "Action", "Symbol", "Price", "Shares"])
    try:
        df = conn.read(ttl=0)
        if df.empty: return pd.DataFrame(columns=["Date", "Account", "Action", "Symbol", "Price", "Shares"])
        required_cols = ["Date", "Account", "Action", "Symbol", "Price", "Shares"]
        for col in required_cols:
            if col not in df.columns:
                if col == "Account": df[col] = "TFSA"
                elif col == "Date": df[col] = str(date.today())
                else: df[col] = ""
        df = df[required_cols]
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
        return df
    except: return pd.DataFrame(columns=["Date", "Account", "Action", "Symbol", "Price", "Shares"])

def save_data_to_gsheet(df):
    if st.session_state.user_role != "Admin": return False 
    if not CONNECTION_STATUS: return False
    try:
        df_save = df.copy()
        df_save['Date'] = df_save['Date'].astype(str)
        conn.update(data=df_save)
        return True
    except: return False

def load_local_csv():
    if st.session_state.user_role != "Admin": return None
    local_file = 'my_portfolio.csv'
    if os.path.exists(local_file):
        try:
            df = pd.read_csv(local_file)
            if 'BuyDate' in df.columns: df = df.rename(columns={'BuyDate': 'Date'})
            if 'Cost' in df.columns: df = df.rename(columns={'Cost': 'Price'})
            if 'Action' not in df.columns: df['Action'] = 'Buy'
            required_cols = ["Date", "Account", "Action", "Symbol", "Price", "Shares"]
            for col in required_cols:
                if col not in df.columns:
                    if col == "Account": df[col] = "TFSA"
                    elif col == "Date": df[col] = str(date.today())
                    else: df[col] = ""
            df = df[required_cols]
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
            return df
        except: return None
    return None

def calculate_portfolio(df_transactions):
    if df_transactions.empty: return pd.DataFrame(), 0, 0
    holdings = {}
    realized_pl = 0
    total_dividend = 0 
    stock_dividends = {} 

    df_sorted = df_transactions.sort_values(by="Date")
    for _, row in df_sorted.iterrows():
        sym = str(row['Symbol']).strip().upper()
        if not sym: continue
        try:
            action, shares, price = row['Action'], float(row['Shares']), float(row['Price'])
        except: continue 
        account = row.get('Account', 'TFSA')
        if sym not in holdings: holdings[sym] = {'shares': 0, 'total_cost': 0, 'account': account}
        
        if action == 'Buy':
            holdings[sym]['shares'] += shares
            holdings[sym]['total_cost'] += (shares * price)
            holdings[sym]['account'] = account
        elif action == 'Sell':
            if holdings[sym]['shares'] > 0:
                avg_cost = holdings[sym]['total_cost'] / holdings[sym]['shares']
                realized_pl += (price - avg_cost) * shares
                holdings[sym]['shares'] -= shares
                holdings[sym]['total_cost'] -= (shares * avg_cost)
        elif action == 'Dividend': 
            amt = price * shares
            total_dividend += amt
            stock_dividends[sym] = stock_dividends.get(sym, 0) + amt

    final_data = []
    for sym, data in holdings.items():
        if data['shares'] > 0.001:
            final_data.append({
                "帳戶": data['account'], "代碼": sym, "持股": data['shares'],
                "總成本": data['total_cost'], "均價": data['total_cost']/data['shares'],
                "已領股息": stock_dividends.get(sym, 0)
            })
    return pd.DataFrame(final_data), realized_pl, total_dividend

def get_usd_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1d")
        if hist.empty: return 0
        raw_price = hist['Close'].iloc[-1]
        if symbol.endswith(".TW"): return raw_price * yf.Ticker("TWDUSD=X").history(period="1d")['Close'].iloc[-1]
        elif symbol.endswith(".TO"): return raw_price * yf.Ticker("CADUSD=X").history(period="1d")['Close'].iloc[-1]
        return raw_price
    except: return 0

def get_annual_dividend_rate(symbol):
    try:
        stock = yf.Ticker(symbol)
        div_rate = stock.info.get('dividendRate')
        if div_rate: return div_rate
        divs = stock.dividends
        one_year_ago = pd.Timestamp.now(tz=divs.index.tz) - pd.DateOffset(years=1)
        last_year_divs = divs[divs.index >= one_year_ago]
        return last_year_divs.sum()
    except: return 0

@st.cache_data(ttl=3600)
def get_usdcad_rate():
    try: return yf.Ticker("CAD=X").history(period="1d")['Close'].iloc[-1]
    except: return 1.35

def get_stock_news(symbol):
    news_items = []
    try:
        url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            for item in root.findall('./channel/item')[:5]:
                news_items.append({'title': item.find('title').text, 'link': item.find('link').text, 'date': item.find('pubDate').text})
    except: pass
    return news_items

def scan_missing_dividends(df_trans):
    if df_trans.empty: return []
    symbols = df_trans['Symbol'].unique()
    missing_dividends = []
    recorded_divs = df_trans[df_trans['Action'] == 'Dividend']
    progress_text = st.empty()
    for sym in symbols:
        sym = str(sym).strip().upper()
        if not sym: continue
        progress_text.caption(f"正在掃描 {sym}...")
        try:
            stock = yf.Ticker(sym)
            div_history = stock.dividends
            if div_history.empty: continue
            start_date = pd.Timestamp.now(tz=div_history.index.tz) - pd.DateOffset(years=2)
            recent_divs = div_history[div_history.index >= start_date]
            for div_date, div_amount in recent_divs.items():
                div_date_date = div_date.date()
                past_trans = df_trans[(df_trans['Symbol'] == sym) & (df_trans['Date'] < div_date_date)]
                shares_held = 0
                account_map = {}
                for _, row in past_trans.iterrows():
                    if row['Action'] == 'Buy':
                        shares_held += float(row['Shares'])
                        account_map = row['Account']
                    elif row['Action'] == 'Sell':
                        shares_held -= float(row['Shares'])
                if shares_held > 0:
                    total_payout = shares_held * div_amount
                    is_recorded = False
                    for _, rec in recorded_divs.iterrows():
                        rec_date = rec['Date']
                        if rec['Symbol'] == sym and abs((rec_date - div_date_date).days) <= 5:
                            is_recorded = True
                            break
                    if not is_recorded:
                        missing_dividends.append({
                            "Date": div_date_date, "Account": account_map if account_map else "TFSA",
                            "Action": "Dividend", "Symbol": sym,
                            "Price": round(total_payout, 2), "Shares": 1,
                            "Info": f"US${div_amount} x {shares_held} 股"
                        })
        except: pass
    progress_text.empty()
    return missing_dividends

def forecast_calendar_dividends(df_inventory):
    forecast_data = []
    today = date.today()
    end_date = today + timedelta(days=90)
    progress = st.empty()
    for _, row in df_inventory.iterrows():
        sym = row['代碼']
        shares = row['持股']
        if shares <= 0: continue
        progress.caption(f"🔮 正在計算 {sym} 配息日曆...")
        try:
            stock = yf.Ticker(sym)
            divs = stock.dividends
            if divs.empty: continue
            last_div_date = divs.index[-1].date()
            last_div_amount = divs.iloc[-1]
            if len(divs) >= 2:
                prev_div_date = divs.index[-2].date()
                days_diff = (last_div_date - prev_div_date).days
                if days_diff < 10: days_diff = 30 
                next_div_date = last_div_date + timedelta(days=days_diff)
                while next_div_date < today: next_div_date += timedelta(days=days_diff)
                while next_div_date <= end_date:
                    est_amount = last_div_amount * shares
                    month_str = next_div_date.strftime("%Y-%m")
                    forecast_data.append({
                        "月份": month_str, "代碼": sym,
                        "預測除息日": next_div_date, "預估金額 (USD)": est_amount
                    })
                    next_div_date += timedelta(days=days_diff)
        except: pass
    progress.empty()
    return pd.DataFrame(forecast_data)

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def scan_technical_signals(df_inventory):
    signals = []
    watch_list = df_inventory['代碼'].unique().tolist()
    market_tickers = ["NVDA", "TSLA", "AAPL", "AMD", "SPY", "QQQ"]
    full_list = list(set(watch_list + market_tickers))
    for sym in full_list:
        try:
            stock = yf.Ticker(sym)
            df = stock.history(period="3mo")
            if len(df) < 20: continue
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['RSI'] = calculate_rsi(df)
            current_price = df['Close'].iloc[-1]
            current_rsi = df['RSI'].iloc[-1]
            ma20 = df['MA20'].iloc[-1]
            status = "盤整"
            note = ""
            if current_rsi < 30: 
                status = "超賣(Buy)"
                note = f"RSI={current_rsi:.1f} 低檔鈍化"
            elif current_rsi > 70: 
                status = "超買(Sell)"
                note = f"RSI={current_rsi:.1f} 過熱警戒"
            dist_ma = (current_price - ma20) / ma20 * 100
            if dist_ma > 15: note += f", 乖離率過大({dist_ma:.1f}%)"
            if dist_ma < -10: note += f", 跌深乖離({dist_ma:.1f}%)"
            if status != "盤整" or abs(dist_ma) > 10:
                signals.append(f"{sym}: 現價{current_price:.1f}, {status}, {note}")
        except: pass
    return signals

def generate_briefing(df_inventory, total_mkt_val, total_profit):
    tech_signals = scan_technical_signals(df_inventory)
    signals_text = "\n".join(tech_signals) if tech_signals else "目前無明顯極端訊號"
    summary = df_inventory[['代碼', '市值', '帳面損益']].to_dict('records')
    prompt = f"""
    你是 AI 投資總監。今天是 {date.today()}。
    【市場與持倉數據】：
    - **用戶總資產市值**：US$ {total_mkt_val:,.0f} (這是目前的股票總值)
    - **整體淨獲利**：US$ {total_profit:,.0f}
    - 持倉狀態：{summary}
    - **技術面掃描訊號**：{signals_text}
    請撰寫【晨間戰報】：1. 持倉健檢 2. 🎯 今日焦點買入 3. ⚠️ 今日風險警示。Markdown格式。
    """
    try:
        key = st.session_state.gemini_key if "Gemini" in ai_model else st.session_state.openai_key
        if not key: return "⚠️ 請先在設定頁面輸入 API Key"
        if "OpenAI" in ai_model:
            ans = OpenAI(api_key=key).chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":prompt}]).choices[0].message.content
        else:
            genai.configure(api_key=key)
            ans = genai.GenerativeModel('gemini-2.5-flash').generate_content(prompt).text
        return ans
    except Exception as e: return f"AI 發生錯誤: {str(e)}"

# --- 策略回測核心 ---
def run_backtest(symbol, strategy):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="1y") 
        if len(df) < 50: return None, "資料不足"
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        df['RSI'] = calculate_rsi(df)
        initial_capital = 10000; cash = initial_capital; position = 0; trades = []
        for i in range(200, len(df)):
            price = df['Close'].iloc[i]; date_idx = df.index[i]; signal = 0; reason = ""
            if strategy == "黃金交叉 (MA50 > MA200)":
                if df['MA50'].iloc[i] > df['MA200'].iloc[i] and df['MA50'].iloc[i-1] <= df['MA200'].iloc[i-1]: signal = 1; reason = "黃金交叉"
                elif df['MA50'].iloc[i] < df['MA200'].iloc[i] and df['MA50'].iloc[i-1] >= df['MA200'].iloc[i-1]: signal = -1; reason = "死亡交叉"
            elif strategy == "RSI 極限反轉 (30/70)":
                if df['RSI'].iloc[i] < 30: signal = 1; reason = "RSI 超賣"
                elif df['RSI'].iloc[i] > 70: signal = -1; reason = "RSI 超買"
            if signal == 1 and cash > 0:
                position = cash / price; cash = 0; trades.append({'Date': date_idx, 'Type': 'Buy', 'Price': price, 'Reason': reason})
            elif signal == -1 and position > 0:
                cash = position * price; position = 0; trades.append({'Date': date_idx, 'Type': 'Sell', 'Price': price, 'Reason': reason})
        final_value = cash + (position * df['Close'].iloc[-1])
        roi = (final_value - initial_capital) / initial_capital * 100
        buy_hold_roi = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
        return {'roi': roi, 'buy_hold_roi': buy_hold_roi, 'trades': trades, 'win': roi > 0, 'better_than_hold': roi > buy_hold_roi, 'df': df}, "Success"
    except Exception as e: return None, str(e)

def draw_backtest_chart(symbol, res):
    df = res['df']; trades = res['trades']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='股價'))
    buy_x, buy_y = [], []
    sell_x, sell_y = [], []
    for t in trades:
        if t['Type'] == 'Buy': buy_x.append(t['Date']); buy_y.append(t['Price'])
        else: sell_x.append(t['Date']); sell_y.append(t['Price'])
    fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode='markers', marker_symbol='triangle-up', marker_color='green', marker_size=12, name='買進'))
    fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode='markers', marker_symbol='triangle-down', marker_color='red', marker_size=12, name='賣出'))
    fig.update_layout(title=f"{symbol} 策略回測圖", height=400)
    return fig

def draw_k_line(symbol):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="6mo")
        if df.empty: return None
        df['MA20'] = df['Close'].rolling(window=20).mean()
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange'), name='月線'))
        fig.update_layout(title=f"{symbol} 走勢", height=300, margin=dict(l=10, r=10, t=30, b=10))
        return fig
    except: return None

def draw_radar(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        scores = [
            min((info.get('profitMargins',0)*100)/2, 10), min((info.get('revenueGrowth',0)*100)/2, 10),
            min(100/info.get('trailingPE', 50)*2, 10) if info.get('trailingPE') else 0,
            min((info.get('dividendYield',0) or 0)*100*2, 10), min((info.get('returnOnEquity',0)*100)/2, 10)
        ]
        fig = px.line_polar(r=scores, theta=['獲利', '成長', '估值', '股息', 'ROE'], line_close=True)
        fig.update_traces(fill='toself')
        fig.update_layout(title="基本面雷達", height=250, margin=dict(t=30, b=0, l=30, r=30))
        return fig
    except: return None

# ==========================================
# 側邊欄：登入系統 & 導航
# ==========================================
with st.sidebar:
    st.header("📱 指揮中心")
    
    # ★★★ 魔法連結自動登入 (V28.1 新增) ★★★
    # 檢查網址是否有 ?auth=admin123
    query_params = st.query_params
    auth_token = query_params.get("auth", "")
    
    if st.session_state.user_role is None:
        # 如果網址帶有正確密碼，直接自動登入為 Admin
        if auth_token == ADMIN_PASSWORD:
            st.session_state.user_role = "Admin"
            st.toast(f"🔑 魔法連結驗證成功！歡迎回來。", icon="🚀")
            st.rerun()
        
        # 否則顯示一般登入畫面
        st.info("🔒 請先登入以解鎖功能")
        pwd = st.text_input("輸入密碼", type="password")
        if st.button("登入"):
            if pwd == ADMIN_PASSWORD:
                st.session_state.user_role = "Admin"
                st.success("歡迎回來，指揮官！")
                st.rerun()
            elif pwd == GUEST_PASSWORD:
                st.session_state.user_role = "Guest"
                st.success("訪客模式已啟動")
                st.rerun()
            else:
                st.error("密碼錯誤")
        st.stop() 

    # 登入後顯示的選單
    if st.session_state.user_role == "Admin":
        st.write("👤 身份: **指揮官 (Admin)**")
        if st.button("登出"): 
            st.session_state.user_role = None
            st.rerun()
        page = st.radio("前往", ["🏠 資產總覽", "🛠️ 投資工具箱", "📝 交易紀錄", "⚙️ 設定"])
    else:
        st.write("👤 身份: **訪客 (Guest)**")
        if st.button("登出"): 
            st.session_state.user_role = None
            st.rerun()
        page = st.radio("前往", ["🛠️ 投資工具箱"])

    st.divider()
    ai_model = st.selectbox("AI 模型", ["Gemini", "OpenAI"])
    
    # --- 隨身 AI 顧問 ---
    st.divider()
    with st.expander("💬 AI 隨身顧問", expanded=True):
        if st.session_state.user_role == "Admin" and CONNECTION_STATUS:
            st.caption("AI 已連線至您的資產資料庫")
            df_trans = load_data()
            if not df_trans.empty:
                df_inv, realized_pl, total_dividends = calculate_portfolio(df_trans)
                ai_portfolio_view = df_inv[['代碼', '持股', '均價', '總成本']].to_string(index=False)
                system_context = f"你是隨身AI投資顧問。用戶資產數據：已實現損益{realized_pl}，累計股息{total_dividends}。持倉：\n{ai_portfolio_view}"
            else: system_context = "用戶目前無持倉。"
        else:
            st.caption("AI 處於通用諮詢模式 (無個資)")
            system_context = "你是專業的 AI 投資顧問。請回答用戶關於股票市場、技術分析或投資策略的問題。"

        for msg in st.session_state.messages:
            role_icon = "👤" if msg["role"] == "user" else "🤖"
            st.markdown(f"**{role_icon}** {msg['content']}")

        if prompt := st.chat_input("問我任何事..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()

# --- 處理 AI 回覆 ---
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    try:
        key = st.session_state.gemini_key if "Gemini" in ai_model else st.session_state.openai_key
        if not key: ans = "⚠️ 請先設定 API Key"
        elif "OpenAI" in ai_model:
            messages = [{"role": "system", "content": system_context}] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            ans = OpenAI(api_key=key).chat.completions.create(model="gpt-4o", messages=messages).choices[0].message.content
        else:
            genai.configure(api_key=key)
            full_prompt = system_context + "\n\n用戶歷史對話:\n" + "\n".join([m['content'] for m in st.session_state.messages])
            ans = genai.GenerativeModel('gemini-2.5-flash').generate_content(full_prompt).text
    except Exception as e: ans = f"錯誤: {str(e)}"
    st.session_state.messages.append({"role": "assistant", "content": ans})
    st.rerun()

# ==========================================
# 頁面 1: 🏠 資產總覽 (Admin Only)
# ==========================================
if page == "🏠 資產總覽" and st.session_state.user_role == "Admin":
    st.subheader("💰 資產戰情室 (USD)")
    if not CONNECTION_STATUS: st.error("⚠️ 無法連線至 Google Sheets")
    else:
        with st.container(border=True):
            st.markdown("### 🌞 智能晨間戰報 (含買賣訊號)")
            if st.session_state.daily_briefing is None:
                if st.button("✨ 生成今日戰略", use_container_width=True):
                    temp_df = load_data()
                    if not temp_df.empty:
                        t_inv, t_pl, t_div = calculate_portfolio(temp_df)
                        current_total_mkt = 0
                        for i, r in t_inv.iterrows():
                            p = get_usd_price(r['代碼'])
                            mkt = p * r['持股']
                            t_inv.at[i, '現價'] = p
                            t_inv.at[i, '市值'] = mkt
                            t_inv.at[i, '帳面損益'] = (mkt / r['持股'] - r['均價']) * r['持股'] if r['持股'] > 0 else 0
                            current_total_mkt += mkt
                        total_profit_val = t_pl + t_inv['帳面損益'].sum() + t_div
                        with st.spinner("AI 正在計算技術指標..."):
                            briefing = generate_briefing(t_inv, current_total_mkt, total_profit_val)
                            st.session_state.daily_briefing = briefing
                            st.rerun()
                    else: st.warning("請先新增交易紀錄。")
            if st.session_state.daily_briefing:
                st.markdown(st.session_state.daily_briefing)
                if st.button("🔄 重新分析"): st.session_state.daily_briefing = None; st.rerun()

        if st.button("🔄 更新即時報價", use_container_width=True): st.rerun()

        df_trans = load_data()
        if df_trans.empty: st.info("👋 連線成功！雲端目前是空的。")
        else:
            df_inv, realized_pl, total_dividends = calculate_portfolio(df_trans)
            if not df_inv.empty:
                total_mkt = 0
                df_inv['現價'] = 0.0
                df_inv['市值'] = 0.0
                df_inv['帳面損益'] = 0.0
                with st.spinner('同步美金匯率與股價...'):
                    for i, row in df_inv.iterrows():
                        p = get_usd_price(row['代碼'])
                        df_inv.at[i, '現價'] = p
                        df_inv.at[i, '市值'] = p * row['持股']
                        df_inv.at[i, '帳面損益'] = (p - row['均價']) * row['持股']
                total_mkt = df_inv['市值'].sum()
                total_unrealized = df_inv['帳面損益'].sum()
                total_net = realized_pl + total_unrealized + total_dividends
                usdcad = get_usdcad_rate()
                
                st.container(border=True).metric("總資產淨值 (USD)", f"US$ {total_mkt:,.0f}", f"US$ {total_net:,.0f} (Net Profit)")
                st.caption(f"🇨🇦 約合加幣: CAD$ {total_mkt * usdcad:,.0f}")
                
                st.divider()
                st.markdown("### 🗓️ 股息現金流日曆 (未來 90 天)")
                if st.button("查看完整預測", use_container_width=True):
                    with st.spinner("正在推算每支股票的配息週期..."):
                        df_forecast = forecast_calendar_dividends(df_inv)
                        if not df_forecast.empty:
                            months = sorted(df_forecast['月份'].unique())
                            tabs = st.tabs([f"📅 {m}" for m in months])
                            for i, month in enumerate(months):
                                with tabs[i]:
                                    month_data = df_forecast[df_forecast['月份'] == month]
                                    month_total = month_data['預估金額 (USD)'].sum()
                                    st.metric(f"{month} 預估總收入", f"US$ {month_total:,.2f}")
                                    st.dataframe(month_data[['代碼', '預測除息日', '預估金額 (USD)']], hide_index=True, use_container_width=True)
                        else: st.info("未來 90 天內，您的持倉預計沒有配息。")
                
                st.divider()
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("已落袋", f"US$ {realized_pl:,.0f}")
                col_b.metric("💰 累計股息", f"US$ {total_dividends:,.0f}")
                col_c.metric("帳面", f"US$ {total_unrealized:,.0f}")
                
                st.divider()
                st.write("🔥 持倉明細 (USD)")
                df_inv = df_inv.sort_values(by="帳面損益", ascending=False)
                for _, row in df_inv.iterrows():
                    color = "🟢" if row['帳面損益'] > 0 else "🔴"
                    roi = (row['帳面損益'] / row['總成本'] * 100) if row['總成本']>0 else 0
                    annual_div_per_share = get_annual_dividend_rate(row['代碼'])
                    annual_income = annual_div_per_share * row['持股']
                    received_div = row['已領股息']
                    remaining_cost = row['總成本'] - received_div
                    years_to_breakeven = 999
                    if annual_income > 0: years_to_breakeven = remaining_cost / annual_income
                    payback_pct = min(received_div / row['總成本'], 1.0) if row['總成本'] > 0 else 0
                    with st.expander(f"{color} {row['代碼']} (US${row['現價']:.2f})"):
                        c1, c2 = st.columns(2)
                        c1.metric("市值", f"US$ {row['市值']:,.0f}")
                        c2.metric("損益", f"US$ {row['帳面損益']:,.0f}", f"{roi:.1f}%")
                        st.markdown("---")
                        st.markdown(f"**⏳ 零成本回本進度 (已領 US$ {received_div:,.0f})**")
                        st.progress(payback_pct)
                        if remaining_cost <= 0: st.success("🎉 恭喜！此股票已達成「零成本」(Free Ride)！")
                        elif annual_income > 0:
                            st.caption(f"預估年配息: US$ {annual_income:,.2f} | 剩餘成本: US$ {remaining_cost:,.2f}")
                            st.info(f"🚀 預計再領 **{years_to_breakeven:.1f} 年** 股息可完全回本")
                        else: st.caption("目前無配息。")
            else: st.info("無庫存。")

# ==========================================
# 頁面 2: 🛠️ 投資工具箱 (Admin & Guest)
# ==========================================
elif page == "🛠️ 投資工具箱":
    st.subheader("🛠️ 投資工具箱")
    tab1, tab2, tab3, tab4 = st.tabs(["📊 策略回測", "🔎 個股診斷", "🏹 選股獵人", "🧬 組合健檢"])
    
    if st.session_state.user_role == "Admin":
        df_trans = load_data()
        my_stocks = df_trans['Symbol'].unique().tolist() if not df_trans.empty else []
    else:
        my_stocks = [] # 訪客無持股

    with tab1:
        st.markdown("### 🤖 AI 策略回測實驗室")
        st.caption("驗證策略在過去一年的表現，別再憑感覺下單。")
        c1, c2 = st.columns([1,1])
        with c1: 
            sel_s = st.selectbox("回測標的", [""] + my_stocks + ["NVDA", "TSLA", "AAPL", "AMD", "SPY"])
            inp_s = st.text_input("或輸入代號 (如 2330.TW)", value=sel_s)
        with c2:
            strategy = st.selectbox("選擇策略", ["黃金交叉 (MA50 > MA200)", "RSI 極限反轉 (30/70)"])

        with st.expander("📖 策略操作手冊 (忘記怎麼看時點我)", expanded=True):
            st.markdown("### 🚦 必勝口訣：先看成績，再看訊號")
            col_a, col_b = st.columns(2)
            with col_a: st.info(f"**步驟 1：檢查教練是否合格**\n\n回測跑完後，請務必查看 **「勝過大盤?」**\n\n- ✅ **是**：代表這策略有效，請相信它。\n- ❌ **否**：代表這策略不準，**請忽略訊號，長期持有就好。**")
            with col_b:
                if "黃金交叉" in strategy: st.success("**步驟 2：看圖找買賣點 (趨勢型)**\n\n- 🟢 **綠色三角形**：黃金交叉 (短期穿過長期)，**趨勢向上，買進！**\n- 🔴 **紅色三角形**：死亡交叉 (短期跌破長期)，**趨勢向下，賣出！**")
                else: st.success("**步驟 2：看圖找買賣點 (反轉型)**\n\n- 🟢 **綠色三角形**：RSI < 30 (超賣)，**現在太便宜，撿便宜！**\n- 🔴 **紅色三角形**：RSI > 70 (超買)，**現在太貴了，快逃頂！**")
        
        target = inp_s.upper().strip()
        if st.button("🚀 開始回測", type="primary", use_container_width=True):
            if not target: st.error("請輸入股票代號")
            else:
                with st.spinner(f"正在模擬 {strategy} 交易..."):
                    res, msg = run_backtest(target, strategy)
                    if res:
                        st.success(f"回測完成！ 策略報酬率: {res['roi']:.2f}%")
                        m1, m2, m3 = st.columns(3)
                        m1.metric("策略報酬", f"{res['roi']:.1f}%", delta_color="normal" if res['roi']>0 else "inverse")
                        m2.metric("買進持有 (Benchmark)", f"{res['buy_hold_roi']:.1f}%")
                        m3.metric("勝過大盤?", "✅ 是" if res['better_than_hold'] else "❌ 否")
                        st.plotly_chart(draw_backtest_chart(target, res), use_container_width=True)
                        with st.expander("查看交易明細"): st.table(pd.DataFrame(res['trades']))
                    else: st.error(f"回測失敗: {msg}")

    with tab2: # 個股診斷
        col_sel, col_in = st.columns([1,1])
        with col_sel: sel_stock = st.selectbox("選擇持股", [""] + my_stocks)
        with col_in: inp_stock = st.text_input("或輸入代號", value=sel_stock, key="diag_input")
        target = inp_stock.upper().strip()
        if target:
            if st.button(f"🚀 分析 {target}", type="primary", use_container_width=True, key="diag_btn"):
                with st.spinner(f"正在分析 {target}..."):
                    c1, c2 = st.columns(2)
                    with c1: st.plotly_chart(draw_k_line(target), use_container_width=True)
                    with c2: st.plotly_chart(draw_radar(target), use_container_width=True)
                    news = get_stock_news(target)
                    with st.expander("📰 最新新聞"):
                        for n in news: st.write(f"**{n['title']}**\n{n['date']} | [閱讀]({n['link']})")
                    sys_prompt = f"分析 {target}。請給出：1. 技術面強弱 2. 基本面評分 3. 操作建議。簡短。"
                    try:
                        key = st.session_state.gemini_key if "Gemini" in ai_model else st.session_state.openai_key
                        if "OpenAI" in ai_model:
                            ans = OpenAI(api_key=key).chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":sys_prompt}]).choices[0].message.content
                        else:
                            genai.configure(api_key=key)
                            ans = genai.GenerativeModel('gemini-2.5-flash').generate_content(sys_prompt).text
                        st.session_state.tool_results["stock_diagnosis"] = ans
                        st.session_state["diag_chat"] = [] 
                    except: st.error("請設定 API Key")
        if st.session_state.tool_results["stock_diagnosis"]:
            st.info(st.session_state.tool_results["stock_diagnosis"])
            render_followup_chat("diag_chat", st.session_state.tool_results["stock_diagnosis"])

    with tab3: # 選股獵人
        strategy = st.selectbox("掃描策略", ["價值抄底 (RSI < 30)", "強勢突破 (站上月線)", "高股息定存"])
        if st.button("🔍 開始掃描", type="primary", use_container_width=True):
            st.caption("模擬掃描...")
            prompt = f"請從美股七巨頭和台積電中，根據「{strategy}」策略，推薦 1 支最值得買的股票並說明原因。"
            try:
                key = st.session_state.gemini_key if "Gemini" in ai_model else st.session_state.openai_key
                if "OpenAI" in ai_model:
                    ans = OpenAI(api_key=key).chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":prompt}]).choices[0].message.content
                else:
                    genai.configure(api_key=key)
                    ans = genai.GenerativeModel('gemini-2.5-flash').generate_content(prompt).text
                st.session_state.tool_results["stock_hunter"] = ans
                st.session_state["hunter_chat"] = []
            except: st.error("API Key Error")
        if st.session_state.tool_results["stock_hunter"]:
            st.success(st.session_state.tool_results["stock_hunter"])
            render_followup_chat("hunter_chat", st.session_state.tool_results["stock_hunter"])

    with tab4: # 組合健檢 (Admin Only)
        if st.session_state.user_role == "Admin":
            df_inv, _, _ = calculate_portfolio(load_data())
            if not df_inv.empty:
                st.write("📊 帳戶配置")
                st.plotly_chart(px.pie(df_inv, values='市值', names='帳戶', hole=0.4), use_container_width=True)
                if st.button("⚖️ 取得配倉建議", type="primary", use_container_width=True):
                    with st.spinner("AI 計算中..."):
                        portfolio_summary = df_inv[['代碼','市值', '帳戶']].to_dict('records')
                        prompt = f"用戶持倉(USD)：{portfolio_summary}。請給出再平衡建議。"
                        try:
                            key = st.session_state.gemini_key if "Gemini" in ai_model else st.session_state.openai_key
                            if "OpenAI" in ai_model:
                                ans = OpenAI(api_key=key).chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":prompt}]).choices[0].message.content
                            else:
                                genai.configure(api_key=key)
                                ans = genai.GenerativeModel('gemini-2.5-flash').generate_content(prompt).text
                            st.session_state.tool_results["portfolio_check"] = ans
                            st.session_state["check_chat"] = []
                        except: st.error("API Key Error")
                if st.session_state.tool_results["portfolio_check"]:
                    st.container(border=True).markdown(st.session_state.tool_results['portfolio_check'])
                    render_followup_chat("check_chat", st.session_state.tool_results["portfolio_check"])
            else: st.warning("無庫存")
        else:
            st.warning("🔒 此功能需要 Admin 權限 (涉及個人資產分析)")

# ==========================================
# 頁面 3, 4: 交易紀錄 & 設定 (Admin Only)
# ==========================================
elif (page == "📝 交易紀錄" or page == "⚙️ 設定") and st.session_state.user_role == "Admin":
    if page == "📝 交易紀錄":
        st.subheader("📝 交易流水帳 (USD)")
        if not CONNECTION_STATUS: st.error("無法連線 Google Sheets")
        else:
            df_trans = load_data()
            with st.expander("🔎 自動掃描漏記的股息"):
                st.write("AI 會自動查詢除息紀錄，計算應得美金股息。")
                if st.button("開始掃描", use_container_width=True):
                    with st.spinner("查詢中..."):
                        missing = scan_missing_dividends(df_trans)
                        st.session_state['missing_divs'] = missing
                if 'missing_divs' in st.session_state and st.session_state['missing_divs']:
                    st.success(f"發現 {len(st.session_state['missing_divs'])} 筆漏記！")
                    div_df = pd.DataFrame(st.session_state['missing_divs'])
                    st.dataframe(div_df[['Date', 'Symbol', 'Price', 'Info']], use_container_width=True)
                    if st.button("💾 加入帳本", type="primary", use_container_width=True):
                        new_records = pd.DataFrame(st.session_state['missing_divs']).drop(columns=['Info'])
                        combined_df = pd.concat([df_trans, new_records], ignore_index=True)
                        if save_data_to_gsheet(combined_df):
                            st.success("寫入成功！")
                            del st.session_state['missing_divs']
                            st.rerun()
                elif 'missing_divs' in st.session_state: st.info("無漏記股息。")

            st.divider()
            st.caption("手動輸入 (請輸入美金金額)")
            edited_df = st.data_editor(
                df_trans, num_rows="dynamic",
                column_config={
                    "Date": st.column_config.DateColumn("日期"),
                    "Account": st.column_config.SelectboxColumn("帳戶", options=["TFSA", "USD Cash", "RRSP"]),
                    "Action": st.column_config.SelectboxColumn("動作", options=["Buy", "Sell", "Dividend"]),
                    "Symbol": st.column_config.TextColumn("代碼"),
                    "Price": st.column_config.NumberColumn("成交價/金額 (USD)", format="$%.2f"),
                    "Shares": st.column_config.NumberColumn("股數"),
                }, use_container_width=True, hide_index=True
            )
            if st.button("💾 儲存並同步至雲端", type="primary", use_container_width=True):
                if save_data_to_gsheet(edited_df):
                    st.success("✅ 雲端同步成功！")
                    st.rerun()
                    
    elif page == "⚙️ 設定":
        st.subheader("設定")
        with st.expander("🔑 更新 API 金鑰"):
            new_o = st.text_input("OpenAI Key", value=st.session_state.openai_key, type="password")
            new_g = st.text_input("Gemini Key", value=st.session_state.gemini_key, type="password")
            if st.button("更新金鑰", use_container_width=True):
                st.session_state.openai_key = new_o
                st.session_state.gemini_key = new_g
                st.success("Updated!")
        st.divider()
        st.markdown("### ☁️ 資料同步")
        local_df = load_local_csv()
        if local_df is not None:
            if st.button("📤 上傳舊資料到雲端", type="primary"):
                if save_data_to_gsheet(local_df):
                    st.success("✅ 搬家成功！")
                    st.rerun()
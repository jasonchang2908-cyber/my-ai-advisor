import streamlit as st
import yfinance as yf
import pandas as pd
from openai import OpenAI
import google.generativeai as genai
import os 
from datetime import datetime, date
import plotly.graph_objects as go 
import plotly.express as px 
import requests # 新增：用於抓取 Google News
import xml.etree.ElementTree as ET # 新增：用於解析新聞 RSS

# --- 1. 頁面設定 ---
st.set_page_config(page_title="AI 投資指揮中心", layout="centered", initial_sidebar_state="collapsed")

# --- 2. Session & Keys ---
if "messages" not in st.session_state: st.session_state.messages = []
if "openai_key" not in st.session_state: st.session_state.openai_key = st.secrets.get("OPENAI_API_KEY", "")
if "gemini_key" not in st.session_state: st.session_state.gemini_key = st.secrets.get("GEMINI_API_KEY", "")

# 儲存分析結果
if "tool_results" not in st.session_state:
    st.session_state.tool_results = {"stock_diagnosis": None, "stock_hunter": None, "portfolio_check": None}

# --- 3. 側邊選單 ---
with st.sidebar:
    st.header("📱 指揮中心")
    page = st.radio("前往", ["🏠 資產總覽", "🛠️ 投資工具箱", "📝 交易紀錄", "💬 AI 顧問", "⚙️ 設定"])
    st.divider()
    ai_model = st.selectbox("AI 模型", ["Gemini", "OpenAI"])

# --- 4. 核心邏輯 ---
csv_file = 'my_portfolio.csv'

def load_data():
    if not os.path.exists(csv_file):
        return pd.DataFrame(columns=["Date", "Account", "Action", "Symbol", "Price", "Shares"])
    try:
        df = pd.read_csv(csv_file)
        if 'BuyDate' in df.columns: df = df.rename(columns={'BuyDate': 'Date'})
        if 'Cost' in df.columns: df = df.rename(columns={'Cost': 'Price'})
        if 'Action' not in df.columns: df['Action'] = 'Buy'
        
        required_cols = ["Date", "Account", "Action", "Symbol", "Price", "Shares"]
        for col in required_cols:
            if col not in df.columns:
                if col == "Account": df[col] = "TFSA"
                elif col == "Date": df[col] = date.today()
                else: df[col] = ""
        df = df[required_cols]
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
        return df
    except: return pd.DataFrame(columns=["Date", "Account", "Action", "Symbol", "Price", "Shares"])

def calculate_portfolio(df_transactions):
    if df_transactions.empty: return pd.DataFrame(), 0
    holdings = {}
    realized_pl = 0
    df_sorted = df_transactions.sort_values(by="Date")
    
    for _, row in df_sorted.iterrows():
        sym = str(row['Symbol']).strip().upper()
        if not sym: continue
        action, shares, price = row['Action'], float(row['Shares']), float(row['Price'])
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

    final_data = []
    for sym, data in holdings.items():
        if data['shares'] > 0.001:
            final_data.append({
                "帳戶": data['account'], "代碼": sym, "持股": data['shares'],
                "總成本": data['total_cost'], "均價": data['total_cost']/data['shares']
            })
    return pd.DataFrame(final_data), realized_pl

# --- 工具函數 ---
def get_realtime_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period="1d")
        return hist['Close'].iloc[-1] if not hist.empty else 0
    except: return 0

# ★★★ 新增：Google News RSS 抓取函數 (解決新聞空白問題) ★★★
def get_stock_news(symbol):
    news_items = []
    try:
        # 優先嘗試 Google News RSS (穩定且免費)
        # 針對台灣/中文使用者優化查詢參數
        url = f"https://news.google.com/rss/search?q={symbol}+stock&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            # 抓取前 5 則新聞
            for item in root.findall('./channel/item')[:5]:
                title = item.find('title').text
                link = item.find('link').text
                pub_date = item.find('pubDate').text
                # 簡單格式化日期
                try:
                    dt = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %Z")
                    date_str = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    date_str = pub_date
                
                news_items.append({
                    'title': title,
                    'link': link,
                    'date': date_str
                })
    except Exception as e:
        print(f"News Error: {e}")
    
    return news_items

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
            min((info.get('profitMargins',0)*100)/2, 10),
            min((info.get('revenueGrowth',0)*100)/2, 10),
            min(100/info.get('trailingPE', 50)*2, 10) if info.get('trailingPE') else 0,
            min((info.get('dividendYield',0) or 0)*100*2, 10),
            min((info.get('returnOnEquity',0)*100)/2, 10)
        ]
        fig = px.line_polar(r=scores, theta=['獲利', '成長', '估值', '股息', 'ROE'], line_close=True)
        fig.update_traces(fill='toself')
        fig.update_layout(title="基本面雷達", height=250, margin=dict(t=30, b=0, l=30, r=30))
        return fig
    except: return None

def render_followup_chat(context_key, context_text):
    st.write("---")
    st.caption("🗣️ 與顧問討論此建議")
    if context_key not in st.session_state: st.session_state[context_key] = []
    for msg in st.session_state[context_key]:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    if prompt := st.chat_input(f"有疑問嗎？", key=f"input_{context_key}"):
        st.session_state[context_key].append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("顧問思考中..."):
                full_prompt = f"背景：{context_text}\n用戶問：{prompt}\n請回答。"
                try:
                    key = st.session_state.gemini_key if "Gemini" in ai_model else st.session_state.openai_key
                    if not key: ans = "請先設定 API Key"
                    elif "OpenAI" in ai_model:
                        ans = OpenAI(api_key=key).chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":full_prompt}]).choices[0].message.content
                    else:
                        genai.configure(api_key=key)
                        ans = genai.GenerativeModel('gemini-2.5-flash').generate_content(full_prompt).text
                except Exception as e: ans = str(e)
                st.markdown(ans)
                st.session_state[context_key].append({"role": "assistant", "content": ans})

# ==========================================
# 頁面 1: 🏠 資產總覽
# ==========================================
if page == "🏠 資產總覽":
    st.subheader("💰 資產戰情室")
    if st.button("🔄 更新報價", use_container_width=True): st.rerun()

    df_trans = load_data()
    df_inv, realized_pl = calculate_portfolio(df_trans)

    if not df_inv.empty:
        total_mkt = 0
        df_inv['現價'] = 0.0
        df_inv['市值'] = 0.0
        df_inv['帳面損益'] = 0.0
        
        with st.spinner('同步市場數據...'):
            for i, row in df_inv.iterrows():
                p = get_realtime_price(row['代碼'])
                df_inv.at[i, '現價'] = p
                df_inv.at[i, '市值'] = p * row['持股']
                df_inv.at[i, '帳面損益'] = (p - row['均價']) * row['持股']
        
        total_mkt = df_inv['市值'].sum()
        total_unrealized = df_inv['帳面損益'].sum()
        total_net = realized_pl + total_unrealized
        
        c1, c2 = st.columns(2)
        c1.metric("總市值", f"${total_mkt:,.0f}")
        c2.metric("淨獲利", f"${total_net:,.0f}", delta_color="normal" if total_net>0 else "inverse")
        st.caption(f"已實現: ${realized_pl:,.0f} | 帳面: ${total_unrealized:,.0f}")
        st.divider()

        st.write("🔥 持倉明細")
        df_inv = df_inv.sort_values(by="帳面損益", ascending=False)
        for _, row in df_inv.iterrows():
            color = "🟢" if row['帳面損益'] > 0 else "🔴"
            roi = (row['帳面損益'] / row['總成本'] * 100) if row['總成本']>0 else 0
            with st.expander(f"{color} {row['代碼']} (${row['現價']:.2f})"):
                c1, c2 = st.columns(2)
                c1.metric("市值", f"${row['市值']:,.0f}")
                c2.metric("損益", f"${row['帳面損益']:,.0f}", f"{roi:.1f}%")

# ==========================================
# 頁面 2: 🛠️ 投資工具箱
# ==========================================
elif page == "🛠️ 投資工具箱":
    st.subheader("🛠️ 投資工具箱")
    
    tab1, tab2, tab3 = st.tabs(["🔎 個股診斷", "🏹 選股獵人", "🧬 組合健檢"])
    
    # --- Tab 1: 個股診斷 ---
    with tab1:
        df_trans = load_data()
        my_stocks = df_trans['Symbol'].unique().tolist() if not df_trans.empty else []
        col_sel, col_in = st.columns([1,1])
        with col_sel: sel_stock = st.selectbox("選擇持股", [""] + my_stocks)
        with col_in: inp_stock = st.text_input("或輸入代號", value=sel_stock)
        target = inp_stock.upper().strip()
        
        if target:
            if st.button(f"🚀 分析 {target}", type="primary", use_container_width=True):
                with st.spinner(f"正在分析 {target}..."):
                    c1, c2 = st.columns(2)
                    with c1: 
                        fig_k = draw_k_line(target)
                        if fig_k: st.plotly_chart(fig_k, use_container_width=True)
                    with c2:
                        fig_r = draw_radar(target)
                        if fig_r: st.plotly_chart(fig_r, use_container_width=True)
                    
                    # ★★★ 新增：新聞摺疊區塊 ★★★
                    news_items = get_stock_news(target)
                    st.write("📰 **相關新聞**")
                    if news_items:
                        for news in news_items:
                            # 使用 expander 讓點擊標題展開
                            with st.expander(f"📅 {news['date']} | {news['title']}"):
                                st.write("點擊下方連結閱讀全文：")
                                st.markdown(f"[閱讀全文]({news['link']})")
                    else:
                        st.info("暫無新聞，請點擊下方搜尋：")
                        st.markdown(f"[🔍 Google 搜尋 {target}](https://www.google.com/search?q={target}+stock)")

                    sys_prompt = f"分析 {target}。請給出：1. 技術面強弱 2. 基本面評分 3. 操作建議 (短/中/長)。簡短白話。"
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


    # --- Tab 2: 選股獵人 (含詳細解釋) ---
    with tab2:
        st.write("🤖 AI 自動掃描市場機會")
        strategy = st.selectbox("選擇策略", ["價值抄底 (RSI < 30)", "強勢突破 (站上月線)", "高股息定存"])
        
        # ★★★ 新增：策略白話文解釋與操作建議 ★★★
        if strategy == "價值抄底 (RSI < 30)":
            with st.container(border=True):
                st.markdown("### 📉 價值抄底策略")
                st.write("**這是什麼意思？**")
                st.caption("股價短時間跌太兇了，市場過度恐慌，就像百貨公司跳樓大拍賣。RSI 指標低於 30 代表「超賣」。")
                st.write("**💰 操作建議**")
                st.caption("1. **不要一次梭哈**：因為跌勢可能還沒停，建議分批買進。\n2. **設定停損**：如果買進後再跌 5-10%，先跑再說。")
        
        elif strategy == "強勢突破 (站上月線)":
            with st.container(border=True):
                st.markdown("### 🚀 強勢突破策略")
                st.write("**這是什麼意思？**")
                st.caption("股價原本在整理，現在突然衝破關鍵的 20 日均線 (月線)，代表主力資金進場，趨勢轉強。")
                st.write("**💰 操作建議**")
                st.caption("1. **順勢交易**：現在氣勢正旺，適合追價買進。\n2. **防守點**：如果股價跌回月線之下，代表假突破，要立刻停損。")

        elif strategy == "高股息定存":
            with st.container(border=True):
                st.markdown("### 🛡️ 高股息定存策略")
                st.write("**這是什麼意思？**")
                st.caption("挑選那些賺錢穩定、每年都大方發錢給股東的好公司 (如可口可樂、銀行股)。")
                st.write("**💰 操作建議**")
                st.caption("1. **長期持有**：不用太在意股價每天的漲跌。\n2. **股息再投入**：領到的股息不要花掉，拿去買更多股數，享受複利效果。")

        if st.button("🔍 開始掃描", type="primary", use_container_width=True):
            st.caption("模擬掃描黃金池...")
            prompt = f"請從美股七巨頭和台積電中，根據「{strategy}」策略，推薦 1 支最值得買的股票並說明原因。200字內。"
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

    # --- Tab 3: 組合健檢 ---
    with tab3:
        df_inv, _ = calculate_portfolio(load_data())
        if not df_inv.empty:
            for i, r in df_inv.iterrows(): df_inv.at[i, '市值'] = get_realtime_price(r['代碼']) * r['持股']
            
            st.write("📊 帳戶配置")
            fig = px.pie(df_inv, values='市值', names='帳戶', hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
            
            if st.button("⚖️ 取得配倉調整建議", type="primary", use_container_width=True):
                with st.spinner("AI 計算中..."):
                    portfolio_summary = df_inv[['代碼','市值', '帳戶']].to_dict('records')
                    prompt = f"""
                    用戶持倉：{portfolio_summary}。
                    請給出「再平衡 (Rebalancing)」建議：
                    1. 哪些股票或帳戶佔比過高，建議減碼？
                    2. 建議增持哪類資產以達到平衡？
                    請給出具體操作指令。
                    """
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
                st.container(border=True).markdown(f"### ⚖️ 再平衡建議\n{st.session_state.tool_results['portfolio_check']}")
                render_followup_chat("check_chat", st.session_state.tool_results["portfolio_check"])
        else: st.warning("無庫存")

# ==========================================
# 頁面 3: 📝 交易紀錄
# ==========================================
elif page == "📝 交易紀錄":
    st.subheader("📝 交易流水帳")
    df_trans = load_data()
    edited_df = st.data_editor(
        df_trans, num_rows="dynamic",
        column_config={
            "Date": st.column_config.DateColumn("日期"),
            "Account": st.column_config.SelectboxColumn("帳戶", options=["TFSA", "USD Cash", "RRSP"]),
            "Action": st.column_config.SelectboxColumn("動作", options=["Buy", "Sell"]),
            "Symbol": st.column_config.TextColumn("代碼"),
            "Price": st.column_config.NumberColumn("成交價", format="$%.2f"),
            "Shares": st.column_config.NumberColumn("股數"),
        }, use_container_width=True, hide_index=True
    )
    if st.button("💾 儲存", type="primary", use_container_width=True):
        save_df = edited_df.copy()
        save_df['Date'] = save_df['Date'].fillna(date.today())
        save_df.to_csv(csv_file, index=False)
        st.success("已更新！")
        st.rerun()

# ==========================================
# 頁面 4: 💬 AI 顧問
# ==========================================
elif page == "💬 AI 顧問":
    st.subheader("🤖 AI 聊天室")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    if prompt := st.chat_input("輸入問題..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    key = st.session_state.gemini_key if "Gemini" in ai_model else st.session_state.openai_key
                    if not key: ans = "請先設定 API Key"
                    elif "OpenAI" in ai_model:
                        ans = OpenAI(api_key=key).chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":prompt}]).choices[0].message.content
                    else:
                        genai.configure(api_key=key)
                        ans = genai.GenerativeModel('gemini-2.5-flash').generate_content(prompt).text
                except Exception as e: ans = str(e)
                st.markdown(ans)
                st.session_state.messages.append({"role": "assistant", "content": ans})

# ==========================================
# 頁面 5: ⚙️ 設定
# ==========================================
elif page == "⚙️ 設定":
    st.subheader("設定")
    new_o = st.text_input("OpenAI Key", value=st.session_state.openai_key, type="password")
    new_g = st.text_input("Gemini Key", value=st.session_state.gemini_key, type="password")
    if st.button("更新金鑰", use_container_width=True):
        st.session_state.openai_key = new_o
        st.session_state.gemini_key = new_g
        st.success("Updated!")
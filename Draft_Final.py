
import streamlit as st
import requests
import pandas as pd
import numpy as np
from scipy.stats import norm
from datetime import datetime
import os
import json

# --- Wide line ---
st.set_page_config(layout="wide") 
# --------------------------

# --- PATH CONFIGURATION ---
BASE_PATH = os.getcwd() 
TRADES_FILE = os.path.join(BASE_PATH, "paper_trades.json")

# Change this:
def call_dhan(endpoint, payload):
    headers = {
        'access-token': config.ACCESS_TOKEN, 
        'client-id': str(config.CLIENT_ID), 
        'Content-Type': 'application/json'
    }

# To this:
def call_dhan(endpoint, payload):
    headers = {
        'access-token': st.secrets["ACCESS_TOKEN"], 
        'client-id': str(st.secrets["CLIENT_ID"]), 
        'Content-Type': 'application/json'
    }

def save_data():
    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
    data = {
        "paper_positions": st.session_state.paper_positions,
        "trade_history": st.session_state.trade_history
    }
    with open(TRADES_FILE, "w") as f:
        json.dump(data, f, indent=4)

def load_data():
    if os.path.exists(TRADES_FILE):
        try:
            with open(TRADES_FILE, "r") as f:
                data = json.load(f)
                st.session_state.paper_positions = data.get("paper_positions", [])
                st.session_state.trade_history = data.get("trade_history", [])
        except Exception as e:
            st.error(f"Error loading saved trades: {e}")


# --- INITIALIZATION & SCRIP LOOKUP ---
@st.cache_data(ttl=3600)
def load_scrip_lookup():
    url = "https://images.dhan.co/api-data/api-scrip-master.csv"
    try:
        df = pd.read_csv(url, low_memory=False)
        # Filter for Index Options (Nifty/BankNifty)
        fno_df = df[df['SEM_INSTRUMENT_NAME'] == 'OPTIDX'].copy()
        fno_df['MATCH_DATE'] = pd.to_datetime(fno_df['SEM_EXPIRY_DATE'], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')
        
        lookup = {}
        for _, row in fno_df.iterrows():
            # Logic: Match DD MMM to date and CE/PE to CALL/PUT
            mapped_type = "CALL" if row['SEM_OPTION_TYPE'] == "CE" else "PUT"
            key = f"{row['MATCH_DATE']}_{float(row['SEM_STRIKE_PRICE'])}_{mapped_type}"
            lookup[key] = {
                "sec_id": str(row['SEM_SMST_SECURITY_ID']), 
                "lot": int(row['SEM_LOT_UNITS'])
            }
        return lookup
    except Exception as e:
        st.error(f"Failed to load scrip master: {e}")
        return {}


def get_fast_mapping(lookup_dict, expiry, strike, opt_type):
    # Standardize opt_type to CALL/PUT for the key
    target_type = "CALL" if opt_type.upper() in ["CE", "CALL"] else "PUT"
    key = f"{expiry}_{float(strike)}_{target_type}"
    return lookup_dict.get(key)

scrip_lookup = load_scrip_lookup()

# --- STATE MANAGEMENT ---
if 'initialized' not in st.session_state:
    st.session_state.paper_positions = []
    st.session_state.trade_history = []
    st.session_state.scanning_active = False  # <--- ADD THIS LINE
    load_data()  
    st.session_state.initialized = True

# Safety check (add this below the block above)
if 'scanning_active' not in st.session_state:
    st.session_state.scanning_active = False

# --- MATH ENGINE ---
def calculate_pop_exact(S_fwd, K, LTP, T, sigma, opt_type):
    try:
        if T <= 0 or sigma <= 0 or LTP <= 0: return 50.0
        breakeven = K + LTP if opt_type.lower() == 'ce' else K - LTP
        d2 = (np.log(S_fwd / breakeven) - (0.5 * sigma**2 * T)) / (sigma * np.sqrt(T))
        pop = norm.cdf(d2) if opt_type.lower() == 'pe' else norm.cdf(-d2)
        return pop * 100
    except: return 50.0

def call_dhan(endpoint, payload):
    headers = {'access-token': config.ACCESS_TOKEN, 'client-id': str(config.CLIENT_ID), 'Content-Type': 'application/json'}
    url = f"https://api.dhan.co/v2/{endpoint}"
    try:
        return requests.post(url, headers=headers, json=payload).json()
    except: return {"status": "error"}

def get_live_vix():
    vix_res = call_dhan("marketfeed/ltp", {"SecurityId": "17", "ExchangeSegment": "IDX_I"})
    vix_val = vix_res.get("data", {}).get("last_price", 0)
    return float(vix_val) if vix_val > 0 else 11.30

def get_prices_from_chain(expiry):
    """Fetches the entire option chain to get LTP for all strikes at once."""
    res = call_dhan("optionchain", {"UnderlyingScrip": 13, "UnderlyingSeg": "IDX_I", "Expiry": expiry})
    price_map = {}
    if res.get("status") == "success":
        oc_data = res.get("data", {}).get("oc", {})
        for strike, data in oc_data.items():
            # Force strike to float string to ensure match (e.g., '26500.0_CALL')
            strike_val = f"{float(strike)}"
            price_map[f"{strike_val}_CALL"] = data.get("ce", {}).get("last_price", 0)
            price_map[f"{strike_val}_PUT"] = data.get("pe", {}).get("last_price", 0)
    return price_map
    
# --- AUTO SCAN & DEPLOY LOGIC ---
def run_auto_scan_and_deploy(expiry_list, min_pop, dte_range, ltp_range, min_eff): # 1. Added min_eff to arguments
    new_deployments = 0
    for exp in expiry_list:
        expiry_dt = datetime.strptime(exp, '%Y-%m-%d')
        dte = (expiry_dt - datetime.now()).days
        if dte_range[0] <= dte <= dte_range[1]:
            res = call_dhan("optionchain", {"UnderlyingScrip": 13, "UnderlyingSeg": "IDX_I", "Expiry": exp})
            if res.get("status") == "success":
                data = res.get("data", {})
                spot, oc_dict = data.get("last_price", 0), data.get("oc", {})
                
                # Math for Synthetic Future
                atm_strike = round(spot / 50) * 50
                atm_data = oc_dict.get(str(float(atm_strike)), {})
                synth_fut = atm_strike + atm_data.get("ce", {}).get("last_price", 0) - atm_data.get("pe", {}).get("last_price", 0)
                T = max((expiry_dt - datetime.now()).total_seconds(), 0.5) / (365 * 24 * 3600)

                for strike_key, d in oc_dict.items():
                    k = float(strike_key)
                    if k % 100 == 0:
                        for side in ['ce', 'pe']:
                            leg = d.get(side, {})
                            ltp = leg.get("last_price", 0)
                            
                            # --- NEW: 2. CALCULATE EFFICIENCY ---
                            greeks = leg.get("greeks", {})
                            theta = abs(greeks.get("theta", 0))
                            gamma = greeks.get("gamma", 0)
                            efficiency = (theta / (gamma * 100)) if gamma > 0 else 0
                            
                            if ltp_range[0] <= ltp <= ltp_range[1]:
                                pop = calculate_pop_exact(synth_fut, k, ltp, T, leg.get("implied_volatility", 0)/100, side)
                                
                                # --- NEW: 3. ADD EFFICIENCY TO THE IF CONDITION ---
                                if pop >= min_pop and efficiency >= min_eff:
                                    mapping = get_fast_mapping(scrip_lookup, exp, k, side)
                                    sec_id = mapping['sec_id'] if mapping else "N/A"
                                    qty = mapping['lot'] if mapping else 50
                                    opt_type = "CALL" if side == 'ce' else "PUT"
                                    
                                    # CHECK IF TRADE ALREADY EXISTS
                                    exists = any(p['Strike'] == k and p['Type'] == opt_type and p['Expiry'] == exp for p in st.session_state.paper_positions)
                                    
                                    if not exists:
                                        trade = {
                                            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                            "Expiry": exp, "DTE": dte, "Strike": k,
                                            "Type": opt_type, "Entry_Price": ltp, 
                                            "POP": f"{pop:.1f}%", "SecurityID": sec_id, "Qty": qty,
                                            "Efficiency": round(efficiency, 2) # Added to record the score
                                        }
                                        st.session_state.paper_positions.append(trade)
                                        save_data() 
                                        new_deployments += 1
    return new_deployments

# --- DASHBOARD TABLE ---
def render_dashboard_table(expiry):
    res = call_dhan("optionchain", {"UnderlyingScrip": 13, "UnderlyingSeg": "IDX_I", "Expiry": expiry})
    if res.get("status") == "success":
        data = res.get("data", {})
        spot, oc_dict = data.get("last_price", 0), data.get("oc", {})
        sorted_strikes = sorted([float(s) for s in oc_dict.keys()])
        atm_strike = min(sorted_strikes, key=lambda x: abs(x - spot))
        atm_key = next((k for k in oc_dict.keys() if float(k) == atm_strike), str(float(atm_strike)))
        atm_data = oc_dict.get(atm_key, {})
        synth_fut = atm_strike + atm_data.get("ce", {}).get("last_price", 0) - atm_data.get("pe", {}).get("last_price", 0)
        T = max((datetime.strptime(expiry, '%Y-%m-%d') - datetime.now()).total_seconds(), 0.5) / (365 * 24 * 3600)

        m1, m2, m3 = st.columns(3)
        m1.metric("Synth Fut", f"₹{synth_fut:.2f}")
        m2.metric("Nifty Spot", f"₹{spot}")
        m3.metric("ATM Strike", atm_strike)

        rows = []
        for strike_key, d in oc_dict.items():
            k = float(strike_key)
            ce, pe = d.get("ce", {}), d.get("pe", {})
            ce_ltp, pe_ltp = ce.get("last_price", 0), pe.get("last_price", 0)
            if ce_ltp > 10 or pe_ltp > 10:
                ce_g, pe_g = ce.get("greeks", {}), pe.get("greeks", {})
                rows.append({
                    "CE_POP": calculate_pop_exact(synth_fut, k, ce_ltp, T, ce.get("implied_volatility", 0)/100, 'ce'),
                    "CE_Th": ce_g.get("theta", 0), "CE_Ga": ce_g.get("gamma", 0), "CE_Delta": ce_g.get("delta", 0),
                    "CE_IV": ce.get("implied_volatility", 0), "CE_OI": ce.get("oi", 0),
                    "CE_Chg_OI": ce.get("oi", 0) - ce.get("previous_oi", 0), "CE_LTP": ce_ltp,
                    "Strike": k,
                    "PE_LTP": pe_ltp, "PE_Chg_OI": pe.get("oi", 0) - pe.get("previous_oi", 0),
                    "PE_OI": pe.get("oi", 0), "PE_IV": pe.get("implied_volatility", 0), "PE_Delta": pe_g.get("delta", 0),
                    "PE_Ga": pe_g.get("gamma", 0), "PE_Th": pe_g.get("theta", 0),
                    "PE_POP": calculate_pop_exact(synth_fut, k, pe_ltp, T, pe.get("implied_volatility", 0)/100, 'pe'),
                })
        df = pd.DataFrame(rows).sort_values("Strike")
        view_cols = ["CE_POP", "CE_Th", "CE_Ga", "CE_Delta", "CE_IV", "CE_OI", "CE_Chg_OI", "CE_LTP", "Strike", "PE_LTP", "PE_Chg_OI", "PE_OI", "PE_IV", "PE_Delta", "PE_Ga", "PE_Th", "PE_POP"]
        st.dataframe(df[view_cols], use_container_width=True, hide_index=True)

# --- MULTI-EXPIRY SCREENER (Logic updated for ID/Qty) ---
def get_all_screener_trades(expiry_list, min_eff):
    all_matches = []
    for exp in expiry_list:
        expiry_dt = datetime.strptime(exp, '%Y-%m-%d')
        dte = (expiry_dt - datetime.now()).days
        if entry_dte_range[0] <= dte <= entry_dte_range[1]:
            res = call_dhan("optionchain", {"UnderlyingScrip": 13, "UnderlyingSeg": "IDX_I", "Expiry": exp})
            if res.get("status") == "success":
                data = res.get("data", {})
                spot, oc_dict = data.get("last_price", 0), data.get("oc", {})
                
                atm_strike = round(spot / 50) * 50
                atm_data = oc_dict.get(str(float(atm_strike)), {})
                synth_fut = atm_strike + atm_data.get("ce", {}).get("last_price", 0) - atm_data.get("pe", {}).get("last_price", 0)
                T = max((expiry_dt - datetime.now()).total_seconds(), 0.5) / (365 * 24 * 3600)
                
                for strike_key, d in oc_dict.items():
                    k = float(strike_key)
                    
                    if k % 100 == 0: 
                        for side in ['ce', 'pe']:
                    # --- NEW: CHECK IF TRADE IS ALREADY OPEN ---
                            opt_type_mapped = "CALL" if side == 'ce' else "PUT"
                            is_already_open = any(
                            p['Strike'] == k and 
                            p['Type'] == opt_type_mapped and 
                            p['Expiry'] == exp 
                            for p in st.session_state.paper_positions
                            )
                    
                            if is_already_open:
                                continue  # This skips the trade so it vanishes from the Screener
                    # --------------------------------------------

                            leg = d.get(side, {})
                            ltp = leg.get("last_price", 0)  

                            pop = calculate_pop_exact(synth_fut, k, ltp, T, leg.get("implied_volatility", 0)/100, side)
                            
                            # --- CALCULATE EFFICIENCY FIRST ---
                            g = leg.get("greeks", {})
                            theta, gamma = abs(g.get("theta", 0)), g.get("gamma", 0)
                            efficiency = round(theta / (gamma * 100), 2) if gamma > 0 else 0
                            
                            # --- UPDATED CONDITION: Added 'and efficiency >= min_eff' ---
                            if pop >= min_pop_filter and ltp_range_val[0] <= ltp <= ltp_range_val[1] and efficiency >= min_eff:
                                mapping = get_fast_mapping(scrip_lookup, exp, k, side)
                                
                                all_matches.append({
                                    "Expiry": exp, "DTE": dte, "Strike": k,
                                    "Type": "CALL" if side == 'ce' else "PUT",
                                    "LTP": ltp, "POP": f"{pop:.1f}%", 
                                    "SecurityID": mapping['sec_id'] if mapping else "N/A",
                                    "Qty": mapping['lot'] if mapping else 50,
                                    "Efficiency": efficiency
                                })
    return all_matches

with st.sidebar:
    st.header("🎯 Strategy Filters")
    
    # --- 1. ROBUST EXPIRY FETCHING ---
    # We store the list in session_state so it doesn't disappear on every slider move
    if 'full_expiry_list' not in st.session_state or not st.session_state.full_expiry_list:
        expiry_res = call_dhan("optionchain/expirylist", {"UnderlyingScrip": 13, "UnderlyingSeg": "IDX_I"})
        if expiry_res.get("status") == "success":
            st.session_state.full_expiry_list = expiry_res.get("data", [])
        else:
            st.session_state.full_expiry_list = []

    # Assign to the local variable used by the rest of your script
    full_expiry_list = st.session_state.full_expiry_list
    
    # 2. Check lock state safely
    is_locked = st.session_state.get('scanning_active', False)
    
    # --- 3. UI FEEDBACK FOR MISSING DATA ---
    if not full_expiry_list:
        st.error("❌ No Expiry Data Found")
        st.info("Check your Dhan Access Token or Network.")
        if st.button("🔄 Retry Connection"):
            st.rerun()
    
    # 4. Sliders (Keep your existing slider code here)
    min_efficiency = st.slider("Min Efficiency Score", 0, 100, 50, disabled=is_locked)
    min_pop_filter = st.slider("Minimum POP %", 0, 100, 72, disabled=is_locked)
    entry_dte_range = st.slider("Entry DTE Range", 0, 100, (45, 90), disabled=is_locked)
    ltp_range_val = st.slider("Select LTP Range", 0, 1000, (127, 1000), disabled=is_locked)
    exit_dte_threshold = st.number_input("Hard Exit DTE (21-Day Rule)", value=21, disabled=is_locked)
    
    st.divider()

    # 4. START/STOP Button Logic
    if not is_locked:
        if st.button("🚀 START SCANNER", use_container_width=True, type="primary"):
            if full_expiry_list:
                st.session_state.scanning_active = True
                # Run one scan immediately so you see trades right away
# Replace 0.7 with whatever your desired min_eff value is
                run_auto_scan_and_deploy(full_expiry_list, min_pop_filter, entry_dte_range, ltp_range_val, 50)
                st.rerun()
            else:
                st.error("Cannot start: No Expiry data from Dhan.")
    else:
        if st.button("🛑 STOP & UNLOCK", use_container_width=True):
            st.session_state.scanning_active = False
            st.rerun()
    
    # 5. Visual Status
    status_icon = "🟢" if is_locked else "🔴"
    status_label = "RUNNING" if is_locked else "STOPPED"
    st.subheader(f"Status: {status_icon} {status_label}")
    
    # Safely show last scan time
    last_scan = st.session_state.get('last_scan_time', 'Never')
    st.caption(f"Last Scan: {last_scan}")
    

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🔍 Screener", "💼 Positions", "📜 Trade Report", "🛡️ Exit Check"])

# ... (Tab 1 Dashboard code remains largely the same) ...
with tab1:
    sel_exp = st.selectbox("📅 Select Expiry", full_expiry_list, key="dash_sel")
    if sel_exp: render_dashboard_table(sel_exp)

with tab2:
    st.subheader("🔍 Strategy Screener")
    
    # 1. ADD A REFRESH BUTTON
    if st.button("🔄 Refresh Screener Matches", use_container_width=True):
        with st.spinner("Searching for matches..."):
            # This calls your function and saves results to session state
            # Pass the expiry list AND a value for min_eff (e.g., 0.5)
            matches = get_all_screener_trades(full_expiry_list, 50)
            st.session_state.scan_results = matches
            if not matches:
                st.warning("No trades found matching your filters. Try lowering Min POP or widening LTP range.")

    # 2. DISPLAY THE TABLE
    # We look for 'scan_results' which we just filled above
    scan_data = st.session_state.get('scan_results', [])
    
    if not scan_data:
        st.info("Click 'Refresh Screener Matches' to see available trades based on your filters.")
    else:
        df_scan = pd.DataFrame(scan_data)
        
        # Ensure 'Select' checkbox column exists
        if "Select" not in df_scan.columns:
            df_scan.insert(0, "Select", False)

        # 3. THE DATA EDITOR (The Table)
        edited_scan = st.data_editor(
            df_scan,
            key="screener_editor_v2", # Changed key to force refresh
            hide_index=True,
            use_container_width=True,
            column_config={
                "Select": st.column_config.CheckboxColumn("Deploy", default=False),
                "LTP": st.column_config.NumberColumn(format="%.2f"),
                "POP": st.column_config.TextColumn("POP")
            },
            disabled=[c for c in df_scan.columns if c != "Select"]
        )

# 4. DEPLOY BUTTON (PAPER VERSION)
        if st.button("🚀 Deploy Selected Trades"):
            selected_rows = edited_scan[edited_scan["Select"] == True]
            
            if not selected_rows.empty:
                for _, row in selected_rows.iterrows():
                    # No API call here—just local recording
                    trade = {
                        "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Expiry": row['Expiry'], 
                        "DTE": row['DTE'], 
                        "Strike": row['Strike'],
                        "Type": row['Type'], 
                        "Entry_Price": row['LTP'], 
                        "POP": row['POP'], 
                        "SecurityID": row['SecurityID'], 
                        "Qty": row['Qty'],
                        "Efficiency": row.get('Efficiency', 0)
                    }
                    st.session_state.paper_positions.append(trade)
                    st.toast(f"Paper Trade Added: {row['Strike']} {row['Type']}")
                
                save_data()
                st.success(f"Successfully deployed {len(selected_rows)} trades to Paper Positions.")
                st.rerun()
                
with tab3:
    st.subheader("💼 Active Paper Positions")

    if not st.session_state.get('paper_positions'):
        st.info("No active positions.")
    else:
        # 1. SYNC DATA FRAGMENT
        @st.fragment(run_every=180)
        def sync_data_and_header():
            df_temp = pd.DataFrame(st.session_state.paper_positions)
            with st.spinner("Updating Live Prices..."):
                prices = {exp: get_prices_from_chain(exp) for exp in df_temp['Expiry'].unique()}
                st.session_state['current_prices_cache'] = prices
            
            def calc_pnl(r):
                p = prices.get(r['Expiry'], {}).get(f"{float(r['Strike'])}_{r['Type']}", r['Entry_Price'])
                return (r['Entry_Price'] - p) * r['Qty']
            
            total_val = df_temp.apply(calc_pnl, axis=1).sum()
            
            # --- BIG COLORING FOR TOTAL P&L ---
            # We use HTML to force the big number to be Red or Green
            color = "#28a745" if total_val >= 0 else "#ff4b4b"  # Green or Red
            
            st.markdown(f"Total Unrealized P&L")
            st.markdown(f"<h1 style='color: {color}; margin-top: -15px;'>₹{total_val:.2f}</h1>", unsafe_allow_html=True)
            st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

        sync_data_and_header()

        # 2. PREPARE DATA
        df_pos = pd.DataFrame(st.session_state.paper_positions)
        all_current_prices = st.session_state.get('current_prices_cache', {})
        
        def calculate_live_row_data(r):
            # 1. Calculate REAL current DTE dynamically based on today's date
            try:
                expiry_dt = datetime.strptime(r['Expiry'], '%Y-%m-%d')
                current_dte = (expiry_dt - datetime.now()).days
            except:
                current_dte = r['DTE'] # Fallback if date parsing fails

            # 2. Fetch Live Price from the synced cache
            expiry_prices = all_current_prices.get(r['Expiry'], {})
            lookup_key = f"{float(r['Strike'])}_{r['Type']}"
            ltp = expiry_prices.get(lookup_key, r.get('Entry_Price'))
            
            # 3. Calculate P&L and Days remaining until the Hard Exit Rule (e.g., 21 days)
            pnl = (r['Entry_Price'] - ltp) * r['Qty']
            days_until_exit_rule = current_dte - exit_dte_threshold
            
            return pd.Series([ltp, pnl, days_until_exit_rule])

        # Apply the update to all three columns at once
        df_pos[['LTP', 'P&L', 'Days_to_Exit']] = df_pos.apply(calculate_live_row_data, axis=1)

        # --- TABLE INDICATORS ---
        def get_pnl_indicator(val):
            return f"🟢 ₹{val:,.2f}" if val >= 0 else f"🔴 ₹{val:,.2f}"

        def get_exit_indicator(days):
            # This now correctly alerts when you hit the 21-day mark
            return "🚨 EXIT NOW" if days <= 0 else f"⏳ {int(days)} Days"

        df_pos['P&L_Live'] = df_pos['P&L'].apply(get_pnl_indicator)
        df_pos['Time_to_Exit'] = df_pos['Days_to_Exit'].apply(get_exit_indicator)

        # 3. SELECT ALL LOGIC
        is_all_selected = st.checkbox("Select All", key="master_select_all")
        df_pos_editor = df_pos.copy()
        df_pos_editor.insert(0, "Exit", is_all_selected)

        # 4. THE TABLE
        edited_df = st.data_editor(
            df_pos_editor,
            key=f"pos_editor_{is_all_selected}",
            hide_index=True,
            use_container_width=True,
            column_config={
                "Exit": st.column_config.CheckboxColumn("Exit", default=False),
                "P&L_Live": st.column_config.TextColumn("P&L (Live)"),
                "Time_to_Exit": st.column_config.TextColumn("Time to Exit"),
                "LTP": st.column_config.NumberColumn("LTP", format="%.2f"),
                # Hide original math columns
                "P&L": None,
                "Days_to_Exit": None,
            },
            disabled=[c for c in df_pos_editor.columns if c != "Exit"]
        )
# 5. Exit Button Logic (Tab 3)
        if st.button("Exit Selected Positions", type="primary"):
            selected_indices = edited_df[edited_df["Exit"] == True].index.tolist()
            if selected_indices:
                new_history = []
            remaining_positions = []
        
            for i, pos in enumerate(st.session_state.paper_positions):
                if i in selected_indices:
                    # Capture ACTUAL data at the moment of exit
                    exit_price = fetch_row_ltp(pos) # Gets current live price
                    pos['Exit_price'] = exit_price
                    pos['Exit_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    pos['Status'] = "CLOSED"
                    st.session_state.trade_history.append(pos)
                else:
                    remaining_positions.append(pos)
        
                    st.session_state.paper_positions = remaining_positions
                    save_data()
                    st.success(f"Exited {len(selected_indices)} positions and moved to Report.")
                    st.rerun()

# ... (Tab 4, 5 and Background monitor) ...

with tab4:
    st.subheader("📜 Completed Trade Cycles")
    
    if st.session_state.get('trade_history'):
        report_data = []
        for trade in st.session_state.trade_history:
            # Expiry formatting to 'DD MMM'
            try:
                dt_obj = datetime.strptime(trade['Expiry'], '%Y-%m-%d')
                display_expiry = dt_obj.strftime('%d %b').upper()
            except:
                display_expiry = trade['Expiry']

            entry_p = trade.get('Entry_Price', 0)
            exit_p = trade.get('Exit_price', 0) 
            qty = trade.get('Qty', 50)
            
            # P&L Calculation for Shorts: (Entry - Exit) * Qty
            pnl_val = (entry_p - exit_p) * qty
            
            report_data.append({
                "Select": False,
                "Expiry": display_expiry,
                "Strike": trade.get('Strike'),
                "Type": trade.get('Type'),
                "Entry Time": trade.get('Time'),
                "Exit Time": trade.get('Exit_time', "N/A"),
                "Entry": entry_p,
                "Exit": exit_p,
                "Qty": qty,
                "P&L": pnl_val,
                "Eff.": trade.get('Efficiency', 0)
            })

        df_report = pd.DataFrame(report_data)

        # --- NEW: Realized P&L Metric ---
        total_realized = df_report['P&L'].sum()
        color = "#28a745" if total_realized >= 0 else "#ff4b4b"
        st.markdown(f"### Total Realized P&L: <span style='color:{color}'>₹{total_realized:,.2f}</span>", unsafe_allow_html=True)
        st.divider()

        # --- Updated Data Editor ---
        edited_report = st.data_editor(
            df_report,
            key="final_report_viewer_v3",
            hide_index=True,
            use_container_width=True,
            column_config={
                "Select": st.column_config.CheckboxColumn("Select", default=False),
                "Entry": st.column_config.NumberColumn("Entry Price", format="₹%.2f"),
                "Exit": st.column_config.NumberColumn("Exit Price", format="₹%.2f"),
                "P&L": st.column_config.NumberColumn("Profit/Loss", format="₹%.2f"),
                "Eff.": st.column_config.NumberColumn("Eff.", format="%.2f"),
                "Entry Time": st.column_config.TextColumn("Entry Time"),
                "Exit Time": st.column_config.TextColumn("Exit Time"),
            },
            disabled=[c for c in df_report.columns if c != "Select"]
        )

        # --- Action Buttons ---
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🗑️ Clear Selected", type="primary"):
                selected_indices = edited_report[edited_report["Select"] == True].index.tolist()
                if selected_indices:
                    st.session_state.trade_history = [
                        item for i, item in enumerate(st.session_state.trade_history) 
                        if i not in selected_indices
                    ]
                    save_data()
                    st.rerun()
        with col2:
            if st.button("🧹 Wipe All History"):
                st.session_state.trade_history = []
                save_data()
                st.rerun()
    else:
        st.info("No completed trade cycles found. Exit an active position to generate a report.")

# --- REPLACE THE CODE INSIDE TAB 5 WITH THIS ---
with tab5:
    st.subheader("🛡️ Global Exit Check")
    today = datetime.now()
    alerts = []
    
    for exp in full_expiry_list:
        try:
            exp_date = datetime.strptime(exp, '%Y-%m-%d')
            current_dte = (exp_date - today).days
            
            # This calculates how many days are left UNTIL you hit the 21-day rule
            days_until_hard_exit = current_dte - exit_dte_threshold
            
            # We only show alerts for expiries that are within 10 days of hitting the rule
            # OR have already passed the rule (days_until_hard_exit <= 0)
            if days_until_hard_exit <= 10:
                alerts.append((exp, days_until_hard_exit))
        except ValueError:
            continue

    if alerts:
        for exp, days in alerts:
            if days <= 0:
                st.error(f"🚨 CRITICAL: Expiry {exp} has already hit the {exit_dte_threshold}-Day Exit Rule!")
            else:
                st.warning(f"⚠️ approaching: Expiry {exp} will hit the {exit_dte_threshold}-Day Rule in {days} days.")
    else:
        st.success(f"All expiries are currently safely above the {exit_dte_threshold}-day exit threshold.")


def is_market_open():
    now = datetime.now().time()
    market_start = datetime.strptime("09:15", "%H:%M").time()
    market_end = datetime.strptime("15:30", "%H:%M").time()
    return market_start <= now <= market_end
# --- IMPROVED BACKGROUND MONITOR ---
# This runs in the background. Removing st.rerun() stops the full-page "blowing" effect.
@st.fragment(run_every=60) # Changed to 60 seconds for faster response
def background_monitor():
    # Fixes the "Never" issue by updating the sidebar timestamp
    st.session_state.last_scan_time = datetime.now().strftime("%H:%M:%S")
    
    if st.session_state.get('scanning_active', False):
        # We pass 'min_efficiency' (the variable from your slider) instead of '50'
        run_auto_scan_and_deploy(
            full_expiry_list, 
            min_pop_filter, 
            entry_dte_range, 
            ltp_range_val, 
            min_efficiency # Use the dynamic slider value here
        )
    
    # 1. AUTOMATED EXIT LOGIC
# Inside background_monitor()
# PASTE THIS REPLACEMENT:
    if st.session_state.paper_positions:
        active = []
        initial_count = len(st.session_state.paper_positions)
        
        for p in st.session_state.paper_positions:
            # We calculate current DTE by subtracting today from the Expiry date
            try:
                exp_dt = datetime.strptime(p['Expiry'], '%Y-%m-%d')
                live_dte = (exp_dt - datetime.now()).days
            except:
                # If date format fails, use the saved DTE as backup
                live_dte = p.get('DTE', 99)

            # Check if actual live DTE has hit your threshold (e.g., 21 days)
            if live_dte <= exit_dte_threshold:
                prices = get_prices_from_chain(p['Expiry'])
                p['Exit_price'] = prices.get(f"{float(p['Strike'])}_{p['Type']}", p['Entry_Price'])
                p['Exit_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                p['Status'] = "CLOSED"
                st.session_state.trade_history.append(p)
                st.toast(f"Auto-exit triggered: {p['Strike']} {p['Type']}")
            else:
                active.append(p)
        
        # Save changes if any trades were moved to history
        if len(active) != initial_count:
            st.session_state.paper_positions = active
            save_data()

    # 2. AUTOMATED ENTRY LOGIC (Only if scanner is active)
    if st.session_state.get('scanning_active', False):
        run_auto_scan_and_deploy(full_expiry_list, min_pop_filter, entry_dte_range, ltp_range_val)
    
    # --- CHANGE HERE: REMOVED st.rerun() ---

    # By removing it, the sidebar and other tabs will remain stable.

import MetaTrader5 as mt5
from datetime import datetime, timedelta
import time
import pandas as pd
import numpy as np

# --- Configuration ---
ACCOUNT_LOGIN = 240453935 # Replace with your MT5 Login
ACCOUNT_PASSWORD = "Chubby@123"  # Replace with your MT5 Password
ACCOUNT_SERVER = "Exness-MT5Trial6"  # Replace with your MT5 Server
SYMBOL = "EURUSD"
LOT_SIZE = 0.01

# Timeframes (Crucial for SMC)
TIMEFRAME_HTF = mt5.TIMEFRAME_M15 # Higher timeframe for structure & POIs
TIMEFRAME_LTF = mt5.TIMEFRAME_M1  # Lower timeframe for entry confirmation

# --- Strategy Parameters ---
# SL_PIPS = 10         # Default SL - Now primarily determined by structure/POI
TP_RR = 3            # Take Profit based on Risk:Reward ratio
MAX_SPREAD_POINTS = 5
LOOKBACK_CANDLES_HTF = 500 # Increased lookback for better structure mapping
LOOKBACK_CANDLES_LTF = 300 # Increased lookback for LTF analysis
SWING_ORDER = 5      # Number of candles to define swing points
SL_BUFFER_PCT = 0.1  # Percentage buffer for SL below/above POI/Structure (e.g., 0.1 = 10%)

# --- Global State Variables (for HTF structure tracking) ---
market_state_htf = {
    "trend": 0, # 1 for bullish, -1 for bearish, 0 for undetermined
    "last_confirmed_high": np.nan,
    "last_confirmed_low": np.nan,
    "last_high_index": None, # Timestamp index
    "last_low_index": None,  # Timestamp index
    "idm_level": np.nan,
    "idm_time_index": None, # Timestamp index
    "idm_taken": False,
    "last_bos_choch_time_index": None # Timestamp index
}

# --- MT5 Connection ---
def start_mt5():
    """Initializes connection to MetaTrader 5"""
    if not mt5.initialize(login=ACCOUNT_LOGIN, password=ACCOUNT_PASSWORD, server=ACCOUNT_SERVER):
        print(f"initialize() failed, error code = {mt5.last_error()}")
        return False
    terminal_info = mt5.terminal_info()
    if not terminal_info:
         print(f"Terminal info failed, error code = {mt5.last_error()}")
         return False
    print(f"Connected to MT5: {terminal_info.name}")
    print(f"MT5 Version: {mt5.version()}")
    if not mt5.symbol_select(SYMBOL, True):
         print(f"Failed to select symbol {SYMBOL}, error: {mt5.last_error()}")
         mt5.shutdown()
         return False
    print(f"Symbol {SYMBOL} selected.")
    return True

def stop_mt5():
    """Shuts down connection to MetaTrader 5"""
    mt5.shutdown()
    print("MT5 connection closed.")

def get_symbol_info(symbol):
    """Gets information about the symbol"""
    info = mt5.symbol_info(symbol)
    # --- Robust checking and enabling ---
    retry_count = 0
    while info is None or not info.visible:
        if retry_count > 2 :
            print(f"Failed to get visible info for {symbol} after multiple attempts.")
            return None
        print(f"Symbol {symbol} info missing or not visible. Attempting to select/enable...")
        if not mt5.symbol_select(symbol, True):
             print(f"symbol_select({symbol}) failed, error: {mt5.last_error()}")
             time.sleep(1) # Wait before retrying
        else:
            print(f"Symbol {symbol} selected/enabled.")
        info = mt5.symbol_info(symbol)
        retry_count += 1
        time.sleep(0.5)
    return info


# --- Market Data Functions ---
def get_rates(symbol, timeframe, count):
    """Fetches historical price data and returns a pandas DataFrame"""
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            print(f"No rates returned for {symbol} on {timeframe}, error: {mt5.last_error()}")
            # Attempt to fetch again after a short delay
            time.sleep(1)
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None or len(rates) == 0:
                 print(f"Still no rates after retry for {symbol} on {timeframe}.")
                 return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df[['open', 'high', 'low', 'close', 'tick_volume']] = df[['open', 'high', 'low', 'close', 'tick_volume']].apply(pd.to_numeric)
        df['idx_num'] = range(len(df)) # Numerical index for iloc-based logic
        return df
    except Exception as e:
        print(f"Error getting rates for {symbol} on {timeframe}: {e}")
        return None

# --- SMC Core Logic Functions ---

def identify_swing_points(df, order=SWING_ORDER):
    """Identifies swing highs and lows using the specified order."""
    df['is_swing_high'] = False
    df['is_swing_low'] = False
    # Use rolling window to check 'order' candles left and right
    df['min_low'] = df['low'].rolling(window=2*order+1, center=True).min()
    df['max_high'] = df['high'].rolling(window=2*order+1, center=True).max()
    df.loc[df['low'] == df['min_low'], 'is_swing_low'] = True
    df.loc[df['high'] == df['max_high'], 'is_swing_high'] = True
    df.drop(columns=['min_low', 'max_high'], inplace=True) # Clean up helper columns
    return df

def find_valid_pullback_idm(df, swing_candle_index, lookback_limit, is_bullish_trend):
    """
    Finds the first valid pullback (Inducement - IDM) before a potential swing point.
    swing_candle_index: Timestamp index of the potential swing high/low candle.
    Returns: (idm_level, idm_time_index) or (np.nan, None)
    """
    try:
        swing_candle_loc = df.index.get_loc(swing_candle_index)
    except KeyError:
        print(f"Error: Swing candle index {swing_candle_index} not found in DataFrame.")
        return np.nan, None

    if swing_candle_loc < 1: return np.nan, None
    limit_loc = max(0, swing_candle_loc - lookback_limit)

    # Iterate backwards using iloc for numerical index safety
    for i in range(swing_candle_loc - 1, limit_loc - 1, -1):
        if i < 1: continue
        current_candle = df.iloc[i]
        prev_candle = df.iloc[i-1]

        if is_bullish_trend: # Looking for IDM (low) below potential SH
            if current_candle['low'] < prev_candle['low']:
                return current_candle['low'], df.index[i] # Return level and timestamp index
        else: # Looking for IDM (high) above potential SL
            if current_candle['high'] > prev_candle['high']:
                return current_candle['high'], df.index[i] # Return level and timestamp index

    return np.nan, None

def update_market_structure(df):
    """
    Updates the global market_state_htf based on BOS and CHOCH on the given DataFrame.
    More robust structure mapping attempt.
    """
    global market_state_htf
    if df is None or len(df) < SWING_ORDER * 2 + 1:
        print("DEBUG: Not enough data for structure mapping.")
        return df

    df = identify_swing_points(df.copy(), order=SWING_ORDER)
    swing_highs = df[df['is_swing_high']]
    swing_lows = df[df['is_swing_low']]

    if swing_highs.empty or swing_lows.empty:
        # print("DEBUG: No swing points identified in the current data slice.")
        return df

    # --- Iterate through candles to update state more dynamically ---
    # This simplified version still relies heavily on the latest swings found in the batch
    # A true robust implementation would process candle-by-candle historically.

    last_sh_index = swing_highs.index[-1]
    last_sl_index = swing_lows.index[-1]
    last_sh_val = swing_highs['high'].iloc[-1]
    last_sl_val = swing_lows['low'].iloc[-1]

    # --- IDM Identification based on last swing ---
    new_idm_level = np.nan
    new_idm_time_index = None
    idm_trend_context = 0 # 1 if IDM is below SH, -1 if IDM is above SL

    if last_sh_index > last_sl_index: # Last swing was high, look for IDM below it
        new_idm_level, new_idm_time_index = find_valid_pullback_idm(df, last_sh_index, 50, is_bullish_trend=True)
        idm_trend_context = 1
    else: # Last swing was low, look for IDM above it
        new_idm_level, new_idm_time_index = find_valid_pullback_idm(df, last_sl_index, 50, is_bullish_trend=False)
        idm_trend_context = -1

    # Update IDM if a newer one is found
    if new_idm_time_index is not None and (market_state_htf["idm_time_index"] is None or new_idm_time_index > market_state_htf["idm_time_index"]):
        market_state_htf["idm_level"] = new_idm_level
        market_state_htf["idm_time_index"] = new_idm_time_index
        market_state_htf["idm_taken"] = False # Reset taken status
        # print(f"DEBUG HTF: New IDM identified at {new_idm_time_index}, level {new_idm_level:.5f}")

    # --- Check if IDM has been taken ---
    if not market_state_htf["idm_taken"] and market_state_htf["idm_time_index"] is not None:
        candles_after_idm = df[df.index > market_state_htf["idm_time_index"]]
        if not candles_after_idm.empty:
            if idm_trend_context == 1: # IDM is a low
                if (candles_after_idm['low'] < market_state_htf["idm_level"]).any():
                    market_state_htf["idm_taken"] = True
                    # print(f"DEBUG HTF: IDM level {market_state_htf['idm_level']:.5f} taken.")
            elif idm_trend_context == -1: # IDM is a high
                 if (candles_after_idm['high'] > market_state_htf["idm_level"]).any():
                    market_state_htf["idm_taken"] = True
                    # print(f"DEBUG HTF: IDM level {market_state_htf['idm_level']:.5f} taken.")

    # --- Confirm Highs/Lows (only after IDM is taken) ---
    # This part needs careful sequencing: IDM taken -> confirms the swing that *preceded* it.
    if market_state_htf["idm_taken"]:
        # If IDM was below the last SH, confirm that SH
        if idm_trend_context == 1 and last_sh_index > market_state_htf["idm_time_index"]:
             if np.isnan(market_state_htf["last_confirmed_high"]) or last_sh_val > market_state_htf["last_confirmed_high"]:
                 market_state_htf["last_confirmed_high"] = last_sh_val
                 market_state_htf["last_high_index"] = last_sh_index
                 # print(f"DEBUG HTF: Confirmed High at {last_sh_val:.5f} ({last_sh_index})")
                 # Reset IDM state for next confirmation cycle
                 market_state_htf["idm_taken"] = False
                 market_state_htf["idm_level"] = np.nan
                 market_state_htf["idm_time_index"] = None

        # If IDM was above the last SL, confirm that SL
        elif idm_trend_context == -1 and last_sl_index > market_state_htf["idm_time_index"]:
             if np.isnan(market_state_htf["last_confirmed_low"]) or last_sl_val < market_state_htf["last_confirmed_low"]:
                 market_state_htf["last_confirmed_low"] = last_sl_val
                 market_state_htf["last_low_index"] = last_sl_index
                 # print(f"DEBUG HTF: Confirmed Low at {last_sl_val:.5f} ({last_sl_index})")
                  # Reset IDM state for next confirmation cycle
                 market_state_htf["idm_taken"] = False
                 market_state_htf["idm_level"] = np.nan
                 market_state_htf["idm_time_index"] = None

    # --- Check for BOS/CHOCH (Breaks of *confirmed* levels) ---
    df['BOS'] = 0
    df['CHOCH'] = 0
    break_occurred = False

    # Check candles after the last structure break event
    start_check_index = market_state_htf["last_bos_choch_time_index"] or df.index[0]
    candles_to_check = df[df.index > start_check_index]

    for current_index, current_candle in candles_to_check.iterrows():
        # Check break of CONFIRMED high
        if not np.isnan(market_state_htf["last_confirmed_high"]) and market_state_htf["last_high_index"] is not None:
            if current_candle['close'] > market_state_htf["last_confirmed_high"]:
                if market_state_htf["trend"] == 1: # BOS
                    df.loc[current_index, 'BOS'] = 1
                    print(f"HTF BOS High confirmed at {current_index}")
                else: # CHOCH
                    df.loc[current_index, 'CHOCH'] = 1
                    print(f"HTF CHOCH High confirmed at {current_index}")
                # Update state
                market_state_htf["trend"] = 1
                market_state_htf["last_confirmed_low"] = market_state_htf["last_confirmed_high"] # Old high becomes new low reference
                market_state_htf["last_low_index"] = market_state_htf["last_high_index"]
                market_state_htf["last_confirmed_high"] = np.nan # Reset high
                market_state_htf["last_high_index"] = None
                market_state_htf["idm_taken"] = False # Reset IDM
                market_state_htf["idm_level"] = np.nan
                market_state_htf["idm_time_index"] = None
                market_state_htf["last_bos_choch_time_index"] = current_index
                break_occurred = True
                break

        # Check break of CONFIRMED low
        if not np.isnan(market_state_htf["last_confirmed_low"]) and market_state_htf["last_low_index"] is not None:
             if current_candle['close'] < market_state_htf["last_confirmed_low"]:
                if market_state_htf["trend"] == -1: # BOS
                    df.loc[current_index, 'BOS'] = -1
                    print(f"HTF BOS Low confirmed at {current_index}")
                else: # CHOCH
                    df.loc[current_index, 'CHOCH'] = -1
                    print(f"HTF CHOCH Low confirmed at {current_index}")
                 # Update state
                market_state_htf["trend"] = -1
                market_state_htf["last_confirmed_high"] = market_state_htf["last_confirmed_low"] # Old low becomes new high reference
                market_state_htf["last_high_index"] = market_state_htf["last_low_index"]
                market_state_htf["last_confirmed_low"] = np.nan # Reset low
                market_state_htf["last_low_index"] = None
                market_state_htf["idm_taken"] = False # Reset IDM
                market_state_htf["idm_level"] = np.nan
                market_state_htf["idm_time_index"] = None
                market_state_htf["last_bos_choch_time_index"] = current_index
                break_occurred = True
                break
        if break_occurred:
            break # Only process one break per function call for clarity

    return df


def identify_fvg(df):
    """Identifies Fair Value Gaps (FVG) / Imbalances."""
    df = df.copy() # Ensure working with a copy
    bullish_fvg = (df['low'] > df['high'].shift(2))
    bearish_fvg = (df['high'] < df['low'].shift(2))
    df['fvg_type'] = 0
    df.loc[bullish_fvg, 'fvg_type'] = 1
    df.loc[bearish_fvg, 'fvg_type'] = -1
    df['fvg_bottom'] = np.nan
    df['fvg_top'] = np.nan
    df.loc[bullish_fvg, 'fvg_bottom'] = df['high'].shift(2)
    df.loc[bullish_fvg, 'fvg_top'] = df['low']
    df.loc[bearish_fvg, 'fvg_bottom'] = df['low'].shift(2)
    df.loc[bearish_fvg, 'fvg_top'] = df['high']
    return df

def identify_order_blocks(df):
    """Identifies potential Order Blocks (OB)."""
    df = identify_fvg(df.copy()) # Need FVG info
    df['ob_type'] = 0
    df['ob_top'] = np.nan
    df['ob_bottom'] = np.nan

    # Use iloc for safer index access
    for i in range(3, len(df) - 1):
        current_candle = df.iloc[i]
        prev_candle = df.iloc[i-1]
        ob_candle = df.iloc[i-2]
        ob_candle_prev = df.iloc[i-3]

        # Check for valid price data before proceeding
        if pd.isna(ob_candle_prev['low']) or pd.isna(ob_candle_prev['high']): continue

        is_bullish_move = current_candle['close'] > current_candle['open'] and current_candle['close'] > prev_candle['high']
        is_bearish_move = current_candle['close'] < current_candle['open'] and current_candle['close'] < prev_candle['low']

        if is_bullish_move and ob_candle['close'] < ob_candle['open']: # Potential Bullish OB
            swept_liquidity = ob_candle['low'] < ob_candle_prev['low']
            created_fvg = current_candle['low'] > ob_candle['high']
            if swept_liquidity and created_fvg:
                df.iloc[i-2, df.columns.get_loc('ob_type')] = 1
                df.iloc[i-2, df.columns.get_loc('ob_top')] = ob_candle['high']
                df.iloc[i-2, df.columns.get_loc('ob_bottom')] = ob_candle['low']

        elif is_bearish_move and ob_candle['close'] > ob_candle['open']: # Potential Bearish OB
            swept_liquidity = ob_candle['high'] > ob_candle_prev['high']
            created_fvg = current_candle['high'] < ob_candle['low']
            if swept_liquidity and created_fvg:
                df.iloc[i-2, df.columns.get_loc('ob_type')] = -1
                df.iloc[i-2, df.columns.get_loc('ob_top')] = ob_candle['high']
                df.iloc[i-2, df.columns.get_loc('ob_bottom')] = ob_candle['low']
    return df

def find_htf_poi_with_mitigation(htf_data):
    """
    Identifies high-probability Points of Interest (POI) on the HTF data,
    filtering out mitigated Order Blocks.
    """
    if htf_data is None or len(htf_data) < 10: return []
    htf_data = identify_order_blocks(htf_data.copy())
    valid_obs = htf_data[htf_data['ob_type'] != 0].copy()
    if valid_obs.empty: return []

    valid_obs['mitigated'] = False
    # Iterate using index for safety
    for idx in valid_obs.index:
        try:
            ob_index_loc = htf_data.index.get_loc(idx)
            ob = valid_obs.loc[idx] # Get the row data
            ob_top = ob['ob_top']
            ob_bottom = ob['ob_bottom']

            # Check candles *after* the OB candle using iloc
            for check_loc in range(ob_index_loc + 1, len(htf_data)):
                check_candle = htf_data.iloc[check_loc]
                # Mitigation check: If candle's high/low range overlaps OB range
                # Ensure values are not NaN before comparison
                if not (pd.isna(check_candle['low']) or pd.isna(ob_bottom) or pd.isna(check_candle['high']) or pd.isna(ob_top)):
                    if max(check_candle['low'], ob_bottom) <= min(check_candle['high'], ob_top):
                        valid_obs.loc[idx, 'mitigated'] = True
                        break # Stop checking for this OB once mitigated
        except KeyError:
            print(f"Warning: Index {idx} not found during mitigation check.")
            continue # Skip if index somehow becomes invalid

    unmitigated_obs = valid_obs[~valid_obs['mitigated']]
    pois = []
    last_bullish_poi = unmitigated_obs[unmitigated_obs['ob_type'] == 1]
    last_bearish_poi = unmitigated_obs[unmitigated_obs['ob_type'] == -1]

    # Return the most recent unmitigated POIs
    if not last_bullish_poi.empty: pois.append(last_bullish_poi.iloc[-1])
    if not last_bearish_poi.empty: pois.append(last_bearish_poi.iloc[-1])
    return pois

# --- LTF Structure Mapping Helpers ---
def map_ltf_structure(ltf_df):
    """Simplified structure mapping for LTF confirmation (less state-dependent)."""
    ltf_df = identify_swing_points(ltf_df.copy(), order=3) # Use smaller order for LTF
    ltf_df['ltf_bos'] = 0
    ltf_df['ltf_choch'] = 0
    last_ltf_sh_val = np.nan
    last_ltf_sl_val = np.nan
    ltf_trend = 0 # 1 bullish, -1 bearish

    swing_highs = ltf_df[ltf_df['is_swing_high']]
    swing_lows = ltf_df[ltf_df['is_swing_low']]

    if not swing_highs.empty: last_ltf_sh_val = swing_highs['high'].iloc[-1]
    if not swing_lows.empty: last_ltf_sl_val = swing_lows['low'].iloc[-1]

    # Basic trend determination
    if not swing_highs.empty and not swing_lows.empty:
         if swing_highs.index[-1] > swing_lows.index[-1] and last_ltf_sh_val > (swing_highs['high'].iloc[-2] if len(swing_highs)>1 else 0):
              if last_ltf_sl_val > (swing_lows['low'].iloc[-2] if len(swing_lows)>1 else -np.inf):
                   ltf_trend = 1 # Higher high, higher low
         elif swing_lows.index[-1] > swing_highs.index[-1] and last_ltf_sl_val < (swing_lows['low'].iloc[-2] if len(swing_lows)>1 else np.inf):
              if last_ltf_sh_val < (swing_highs['high'].iloc[-2] if len(swing_highs)>1 else np.inf):
                   ltf_trend = -1 # Lower low, lower high

    # Find breaks of the *last* swing high/low
    for i in range(1, len(ltf_df)): # Check from second candle onwards
        current_close = ltf_df['close'].iloc[i]
        current_index = ltf_df.index[i]

        # Check break high
        if not np.isnan(last_ltf_sh_val) and current_close > last_ltf_sh_val:
            if ltf_trend == 1: ltf_df.loc[current_index, 'ltf_bos'] = 1
            else: ltf_df.loc[current_index, 'ltf_choch'] = 1
            # Update for subsequent checks within this run
            ltf_trend = 1
            if not swing_lows[swing_lows.index < current_index].empty:
                 last_ltf_sl_val = swing_lows[swing_lows.index < current_index]['low'].iloc[-1] # Update last low before break
            last_ltf_sh_val = ltf_df['high'].iloc[i] # Update high to current break level temporarily

        # Check break low
        elif not np.isnan(last_ltf_sl_val) and current_close < last_ltf_sl_val:
            if ltf_trend == -1: ltf_df.loc[current_index, 'ltf_bos'] = -1
            else: ltf_df.loc[current_index, 'ltf_choch'] = -1
             # Update for subsequent checks
            ltf_trend = -1
            if not swing_highs[swing_highs.index < current_index].empty:
                 last_ltf_sh_val = swing_highs[swing_highs.index < current_index]['high'].iloc[-1] # Update last high before break
            last_ltf_sl_val = ltf_df['low'].iloc[i] # Update low to current break level temporarily

    return ltf_df

def find_ltf_idm_after_choch(ltf_df, choch_index, is_bullish_choch):
    """Finds the first valid pullback (IDM) on LTF *after* an LTF CHOCH."""
    try:
        choch_loc = ltf_df.index.get_loc(choch_index)
    except KeyError:
        return np.nan, None, False # IDM level, time_index, taken_status

    idm_level = np.nan
    idm_time_index = None
    idm_taken = False

    # Look for pullback starting from the candle *after* CHOCH
    for i in range(choch_loc + 1, len(ltf_df) -1): # Iterate until second to last candle
        current_candle = ltf_df.iloc[i]
        prev_candle = ltf_df.iloc[i-1]
        next_candle = ltf_df.iloc[i+1] # Needed to check if IDM is taken immediately

        if is_bullish_choch: # CHOCH broke a high, now look for first low pullback
            if current_candle['low'] < prev_candle['low']:
                idm_level = current_candle['low']
                idm_time_index = ltf_df.index[i]
                # Check if the *next* candle takes this IDM low
                if next_candle['low'] < idm_level:
                    idm_taken = True
                # print(f"DEBUG LTF: IDM found after Bullish CHOCH at {idm_time_index}, Level: {idm_level:.5f}, Taken: {idm_taken}")
                return idm_level, idm_time_index, idm_taken
        else: # CHOCH broke a low, now look for first high pullback
             if current_candle['high'] > prev_candle['high']:
                idm_level = current_candle['high']
                idm_time_index = ltf_df.index[i]
                # Check if the *next* candle takes this IDM high
                if next_candle['high'] > idm_level:
                     idm_taken = True
                # print(f"DEBUG LTF: IDM found after Bearish CHOCH at {idm_time_index}, Level: {idm_level:.5f}, Taken: {idm_taken}")
                return idm_level, idm_time_index, idm_taken

    return np.nan, None, False # No IDM found

def find_entry_zone_after_idm(ltf_df, idm_taken_index, is_bullish_entry):
    """Finds the first unmitigated LTF OB or FVG after LTF IDM is taken."""
    try:
        idm_taken_loc = ltf_df.index.get_loc(idm_taken_index)
    except KeyError:
        return None # No entry zone found

    ltf_df = identify_order_blocks(ltf_df.copy()) # Includes FVG check

    # Look for OBs/FVGs from the IDM taken candle onwards
    for i in range(idm_taken_loc, len(ltf_df)):
        candle_index = ltf_df.index[i]
        candle = ltf_df.iloc[i]

        # Check for unmitigated OB in the desired direction
        if is_bullish_entry and candle['ob_type'] == 1:
            # Basic mitigation check: has price traded below its low since it formed?
            if not (ltf_df['low'].iloc[i+1:] < candle['ob_bottom']).any():
                 print(f"DEBUG LTF: Found Bullish OB entry zone at {candle_index}")
                 return candle # Return the OB candle Series
        elif not is_bullish_entry and candle['ob_type'] == -1:
             if not (ltf_df['high'].iloc[i+1:] > candle['ob_top']).any():
                 print(f"DEBUG LTF: Found Bearish OB entry zone at {candle_index}")
                 return candle

        # Check for unmitigated FVG in the desired direction (use FVG created *before* current candle)
        prev_candle = ltf_df.iloc[i-1] if i > 0 else None
        if prev_candle is not None:
             if is_bullish_entry and prev_candle['fvg_type'] == 1:
                  # Basic mitigation check: has price traded below FVG bottom?
                  if not (ltf_df['low'].iloc[i:] < prev_candle['fvg_bottom']).any():
                       print(f"DEBUG LTF: Found Bullish FVG entry zone ending at {candle_index}")
                       return prev_candle # Return the candle row containing FVG info
             elif not is_bullish_entry and prev_candle['fvg_type'] == -1:
                  if not (ltf_df['high'].iloc[i:] > prev_candle['fvg_top']).any():
                       print(f"DEBUG LTF: Found Bearish FVG entry zone ending at {candle_index}")
                       return prev_candle

    return None # No suitable entry zone found

# --- Advanced LTF Confirmation ---
def check_ltf_confirmation_advanced(ltf_data, poi, signal_type):
    """
    Checks for ADVANCED LTF confirmation patterns after price mitigates an HTF POI.
    signal_type: 'BUY' or 'SELL' expected based on HTF POI type.
    Returns: (confirmation_type, entry_price, sl_price)
    Confirmation types: 'sweep', 'choch_idm', 'flip', None
    """
    # print(f"DEBUG: Checking ADVANCED LTF confirmation near POI {poi.name} for {signal_type}")
    if ltf_data is None or len(ltf_data) < 30: # Need more data for structure
        return None, None, None

    poi_top = poi['ob_top']
    poi_bottom = poi['ob_bottom']
    poi_size = poi_top - poi_bottom

    # 1. Find Mitigation Start Point (more robustly)
    mitigation_start_index = None
    mitigation_start_loc = -1
    # Check last N candles for entry into POI
    check_range = min(50, len(ltf_data))
    for i in range(len(ltf_data) - 1, len(ltf_data) - check_range - 1, -1):
        candle = ltf_data.iloc[i]
        # Check if candle's range overlaps with POI range
        if max(candle['low'], poi_bottom) <= min(candle['high'], poi_top):
            mitigation_start_index = ltf_data.index[i]
            mitigation_start_loc = i
            # print(f"DEBUG: HTF POI Mitigation detected around {mitigation_start_index}")
            break # Found the most recent mitigation

    if mitigation_start_index is None:
        return None, None, None # No recent mitigation

    # Analyze LTF data from mitigation point onwards
    ltf_since_mitigation = ltf_data.iloc[mitigation_start_loc:].copy()
    if len(ltf_since_mitigation) < 5: # Need a few candles after mitigation starts
         return None, None, None

    # --- Check for Confirmation Patterns ---

    # Pattern 1: Liquidity Sweep Entry
    ltf_since_mitigation['prev_low'] = ltf_since_mitigation['low'].shift(1)
    ltf_since_mitigation['prev_high'] = ltf_since_mitigation['high'].shift(1)
    for i in range(1, len(ltf_since_mitigation)):
        current = ltf_since_mitigation.iloc[i]
        previous = ltf_since_mitigation.iloc[i-1]
        # Check if the sweep happened *within* the POI
        in_poi = max(current['low'], poi_bottom) <= min(current['high'], poi_top)

        if signal_type == 'BUY' and in_poi:
            swept_low = current['low'] < previous['low']
            is_bullish_close = current['close'] > current['open'] and current['close'] > (current['open'] + current['low']) / 2 # Closes in upper half
            if swept_low and is_bullish_close:
                print(f"DEBUG: LTF Buy Confirmation: Sweep Entry at {current.name}")
                entry_price = current['close']
                sl_price = current['low'] - poi_size * SL_BUFFER_PCT # SL below the sweep low + buffer
                return 'sweep', entry_price, sl_price

        elif signal_type == 'SELL' and in_poi:
            swept_high = current['high'] > previous['high']
            is_bearish_close = current['close'] < current['open'] and current['close'] < (current['open'] + current['high']) / 2 # Closes in lower half
            if swept_high and is_bearish_close:
                print(f"DEBUG: LTF Sell Confirmation: Sweep Entry at {current.name}")
                entry_price = current['close']
                sl_price = current['high'] + poi_size * SL_BUFFER_PCT # SL above the sweep high + buffer
                return 'sweep', entry_price, sl_price

    # --- Pattern 2: LTF CHOCH + IDM Entry ---
    ltf_struct_df = map_ltf_structure(ltf_since_mitigation)
    choch_signals = ltf_struct_df[ltf_struct_df['ltf_choch'] != 0]

    if not choch_signals.empty:
        first_choch = choch_signals.iloc[0]
        choch_index = choch_signals.index[0]
        choch_type = first_choch['ltf_choch'] # 1 for bullish, -1 for bearish

        # Check if CHOCH is in the desired direction
        if (signal_type == 'BUY' and choch_type == 1) or \
           (signal_type == 'SELL' and choch_type == -1):

            is_bullish_choch = (choch_type == 1)
            # Find the IDM formed *after* this CHOCH
            idm_level, idm_time_index, idm_taken = find_ltf_idm_after_choch(ltf_struct_df, choch_index, is_bullish_choch)

            if idm_time_index is not None and idm_taken:
                print(f"DEBUG LTF: CHOCH confirmed at {choch_index}, subsequent IDM at {idm_time_index} was taken.")
                # Find the entry zone (OB or FVG) created after CHOCH, before or during IDM take
                entry_zone_candle = find_entry_zone_after_idm(ltf_struct_df, idm_time_index, is_bullish_choch)

                if entry_zone_candle is not None:
                    entry_price = None
                    sl_price = None
                    if not pd.isna(entry_zone_candle['ob_type']): # Entry based on OB
                        entry_price = entry_zone_candle['ob_top'] if is_bullish_choch else entry_zone_candle['ob_bottom'] # Enter at OB edge
                        sl_price = entry_zone_candle['ob_bottom'] if is_bullish_choch else entry_zone_candle['ob_top']
                        sl_buffer = abs(entry_zone_candle['ob_top'] - entry_zone_candle['ob_bottom']) * SL_BUFFER_PCT
                        sl_price = sl_price - sl_buffer if is_bullish_choch else sl_price + sl_buffer
                        print(f"DEBUG LTF: CHOCH+IDM Entry identified using OB at {entry_zone_candle.name}")

                    elif not pd.isna(entry_zone_candle['fvg_type']): # Entry based on FVG
                         entry_price = entry_zone_candle['fvg_top'] if is_bullish_choch else entry_zone_candle['fvg_bottom'] # Enter at FVG edge
                         sl_price = entry_zone_candle['fvg_bottom'] if is_bullish_choch else entry_zone_candle['fvg_top'] # SL at other end of FVG
                         sl_buffer = abs(entry_zone_candle['fvg_top'] - entry_zone_candle['fvg_bottom']) * SL_BUFFER_PCT
                         sl_price = sl_price - sl_buffer if is_bullish_choch else sl_price + sl_buffer
                         print(f"DEBUG LTF: CHOCH+IDM Entry identified using FVG ending at {entry_zone_candle.name}")

                    if entry_price is not None and sl_price is not None:
                         return 'choch_idm', entry_price, sl_price


    # --- Pattern 3: Flip Entry ---
    # Detect failed reaction + break of reaction structure + entry on flip zone
    # Simplified Flip Check: Look for an OB within the POI that gets violated,
    # then look for a *new* OB forming in the signal direction after the violation.
    ltf_obs = identify_order_blocks(ltf_since_mitigation.copy())
    flip_zone = None
    flip_break_candle = None

    for i in range(1, len(ltf_obs)): # Start checking from 2nd candle in mitigation zone
        current_idx = ltf_obs.index[i]
        ob_candle = ltf_obs.iloc[i-1] # Potential initial reaction OB

        if signal_type == 'BUY': # Looking for bullish entry after bearish POI mitigation
            # Check if a bearish OB inside POI failed (price closed above it)
            if ob_candle['ob_type'] == -1 and max(ob_candle['low'], poi_bottom) <= min(ob_candle['high'], poi_top):
                # Did price close above this bearish OB?
                if (ltf_obs['close'].iloc[i:] > ob_candle['ob_top']).any():
                     # Find the bullish OB created *after* this failure
                     failed_ob_idx = ltf_obs.index[i-1]
                     subsequent_bullish_obs = ltf_obs[(ltf_obs.index > failed_ob_idx) & (ltf_obs['ob_type'] == 1)]
                     if not subsequent_bullish_obs.empty:
                          flip_zone = subsequent_bullish_obs.iloc[0] # First bullish OB after failure
                          print(f"DEBUG LTF: Potential Bullish Flip Zone found at {flip_zone.name}")
                          # Basic entry: Enter on retest of flip zone (more complex: add IDM check after flip)
                          entry_price = flip_zone['ob_top']
                          sl_price = flip_zone['ob_bottom'] - abs(flip_zone['ob_top'] - flip_zone['ob_bottom']) * SL_BUFFER_PCT
                          return 'flip', entry_price, sl_price
                          break # Found potential flip

        elif signal_type == 'SELL': # Looking for bearish entry after bullish POI mitigation
            # Check if a bullish OB inside POI failed (price closed below it)
            if ob_candle['ob_type'] == 1 and max(ob_candle['low'], poi_bottom) <= min(ob_candle['high'], poi_top):
                 # Did price close below this bullish OB?
                 if (ltf_obs['close'].iloc[i:] < ob_candle['ob_bottom']).any():
                     # Find the bearish OB created *after* this failure
                     failed_ob_idx = ltf_obs.index[i-1]
                     subsequent_bearish_obs = ltf_obs[(ltf_obs.index > failed_ob_idx) & (ltf_obs['ob_type'] == -1)]
                     if not subsequent_bearish_obs.empty:
                          flip_zone = subsequent_bearish_obs.iloc[0] # First bearish OB after failure
                          print(f"DEBUG LTF: Potential Bearish Flip Zone found at {flip_zone.name}")
                          # Basic entry: Enter on retest of flip zone
                          entry_price = flip_zone['ob_bottom']
                          sl_price = flip_zone['ob_top'] + abs(flip_zone['ob_top'] - flip_zone['ob_bottom']) * SL_BUFFER_PCT
                          return 'flip', entry_price, sl_price
                          break # Found potential flip
        if flip_zone is not None: break


    return None, None, None # No confirmation found


# --- Trading Logic ---
def generate_signals_advanced(symbol, htf, ltf):
    """
    Generates trading signals based on HTF POI mitigation and ADVANCED LTF confirmation.
    Returns: (signal_type, entry_price, sl_price)
    """
    global market_state_htf # Use global state for HTF structure
    # print(f"\n--- Generating Signals V4 for {symbol} at {datetime.now()} ---")

    htf_data = get_rates(symbol, htf, LOOKBACK_CANDLES_HTF)
    ltf_data = get_rates(symbol, ltf, LOOKBACK_CANDLES_LTF)

    if htf_data is None or ltf_data is None or htf_data.empty or ltf_data.empty:
        print("Failed to get sufficient market data.")
        return None, None, None

    # 1. Update HTF Market Structure
    htf_data = update_market_structure(htf_data)
    # print(f"DEBUG HTF State: Trend={market_state_htf['trend']}, Conf High={market_state_htf['last_confirmed_high']:.5f}, Conf Low={market_state_htf['last_confirmed_low']:.5f}, IDM Taken={market_state_htf['idm_taken']}")

    # 2. Identify Unmitigated HTF POIs
    htf_pois = find_htf_poi_with_mitigation(htf_data)
    if not htf_pois:
        # print("No suitable unmitigated HTF POIs found.")
        return None, None, None

    # 3. Check each POI for LTF confirmation
    for poi in htf_pois:
        poi_type = poi['ob_type']
        signal_type = 'BUY' if poi_type == 1 else 'SELL' if poi_type == -1 else None

        if signal_type:
            confirmation_type, entry_price, sl_price = check_ltf_confirmation_advanced(ltf_data, poi, signal_type)

            if confirmation_type is not None:
                print(f"--- Signal Generated: {signal_type} (Confirmation: {confirmation_type}) ---")
                if entry_price is None or sl_price is None:
                     print("Error: Confirmation found but entry/sl price is None.")
                     continue
                if pd.isna(entry_price) or pd.isna(sl_price):
                     print("Error: Confirmation found but entry/sl price is NaN.")
                     continue

                # Final SL/Entry validation
                if signal_type == 'BUY' and sl_price >= entry_price:
                    print(f"Error: BUY SL {sl_price:.5f} >= Entry {entry_price:.5f}. Invalid logic.")
                    continue
                if signal_type == 'SELL' and sl_price <= entry_price:
                    print(f"Error: SELL SL {sl_price:.5f} <= Entry {entry_price:.5f}. Invalid logic.")
                    continue

                return signal_type, entry_price, sl_price

    # print("No confirmed entry signals found.")
    return None, None, None

# --- Order Execution Functions ---
def calculate_tp(order_type, entry_price, sl_price, symbol_info):
    """Calculates TP based on SL distance and RR ratio"""
    point = symbol_info.point
    digits = symbol_info.digits
    if entry_price is None or sl_price is None or pd.isna(entry_price) or pd.isna(sl_price): return None

    sl_distance_price = abs(entry_price - sl_price)
    if sl_distance_price < point: # Handle zero or near-zero SL distance
        print(f"Warning: SL distance very small ({sl_distance_price:.{digits}f}). Using minimum stop level for TP calc.")
        sl_distance_price = max(sl_distance_price, symbol_info.trade_stops_level * point)
        if sl_distance_price < point: sl_distance_price = 5 * point # Absolute minimum fallback

    tp_distance_price = sl_distance_price * TP_RR

    if order_type == mt5.ORDER_TYPE_BUY:
        tp = entry_price + tp_distance_price
    elif order_type == mt5.ORDER_TYPE_SELL:
        tp = entry_price - tp_distance_price
    else:
        return None
    return round(tp, digits)

def place_order(signal, symbol, lot, entry_price, sl_price, symbol_info):
    """Places a market order with specified entry, SL, and calculated TP"""
    if entry_price is None or sl_price is None or pd.isna(entry_price) or pd.isna(sl_price):
        print("Error: Cannot place order with invalid entry or SL price.")
        return None

    point = symbol_info.point
    digits = symbol_info.digits
    order_type = None
    # Use the entry price determined by the confirmation logic
    price = round(entry_price, digits)
    sl = round(sl_price, digits)

    # Validate SL placement relative to entry price
    if signal == 'BUY':
        order_type = mt5.ORDER_TYPE_BUY
        if sl >= price:
            print(f"Error: BUY SL ({sl}) must be below entry price ({price}). Order rejected.")
            return None
    elif signal == 'SELL':
        order_type = mt5.ORDER_TYPE_SELL
        if sl <= price:
            print(f"Error: SELL SL ({sl}) must be above entry price ({price}). Order rejected.")
            return None
    else:
        print("Invalid signal type for placing order.")
        return None

    # Check Spread at the time of order placement
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        print(f"Failed to get tick for spread check on {symbol}. Order rejected.")
        return None
    spread = (tick.ask - tick.bid) / point
    if spread > MAX_SPREAD_POINTS:
        print(f"Spread too high ({spread:.1f} points). Max allowed: {MAX_SPREAD_POINTS}. Skipping trade.")
        return None

    # Calculate TP
    tp = calculate_tp(order_type, price, sl, symbol_info)
    if tp is None:
        print("Failed to calculate TP. Order rejected.")
        return None

    # Check Minimum Stop Level Distance required by broker
    min_stop_level_price = symbol_info.trade_stops_level * point
    if abs(price - sl) < min_stop_level_price:
        print(f"Error: SL distance {abs(price-sl):.{digits}f} is less than minimum {min_stop_level_price:.{digits}f}. Order rejected.")
        # Consider adjusting SL automatically here if desired, but rejecting is safer initially
        return None
    if abs(price - tp) < min_stop_level_price:
        print(f"Warning: TP distance {abs(price-tp):.{digits}f} is less than minimum {min_stop_level_price:.{digits}f}. TP might not be placed correctly by broker.")
        # Allow order attempt, but be aware TP might be adjusted/rejected by server

    deviation = 20 # Slippage allowance
    request = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": lot,
        "type": order_type,
        "price": price, # Use the calculated entry price from confirmation
        "sl": sl, "tp": tp,
        "deviation": deviation, "magic": 234004, # New Magic Number for v4
        "comment": "Python SMC Bot v4", "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC, # Or FOK
    }

    print(f"\nAttempting to send {signal} order...")
    print(f"Request: {request}")
    result = mt5.order_send(request)

    # Process Result
    if result is None:
        print(f"order_send failed, returned None. Last error: {mt5.last_error()}")
        return None
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order Send Failed. Retcode: {result.retcode} - {result.comment}")
        print(f"Last Error: {mt5.last_error()}")
        return None
    else:
        print(f"\n***********************************")
        print(f"Order Sent Successfully! Ticket: {result.order}")
        print(f"Details: {signal} {result.volume} {symbol} @ {result.price:.{digits}f}")
        print(f"SL: {result.sl:.{digits}f}, TP: {result.tp:.{digits}f}")
        print(f"***********************************\n")
        return result


# --- Main Bot Loop ---
def run_bot():
    """Main execution loop for the trading bot"""
    if not start_mt5(): return
    symbol_info = get_symbol_info(SYMBOL)
    if not symbol_info:
        stop_mt5()
        return

    print(f"\nStarting SMC Trading Bot v4 for {SYMBOL}...")
    print(f"Lot Size: {LOT_SIZE}, TP Ratio: {TP_RR}R")
    print(f"HTF: {TIMEFRAME_HTF}, LTF: {TIMEFRAME_LTF}")
    print(f"Max Spread: {MAX_SPREAD_POINTS} points")

    while True:
        try:
            # Check connection & symbol info periodically
            if mt5.terminal_info() is None:
                 print("MT5 Terminal disconnected. Attempting to reconnect...")
                 if not start_mt5(): break
                 symbol_info = get_symbol_info(SYMBOL) # Re-fetch symbol info
                 if not symbol_info: break
                 print("Reconnected successfully.")

            # --- Check for Open Positions ---
            positions = mt5.positions_get(symbol=SYMBOL, magic=234004) # Use correct magic number
            if positions is None:
                 print(f"Error getting positions: {mt5.last_error()}. Retrying...")
                 time.sleep(10); continue

            # --- Logic if NO Open Position ---
            if len(positions) == 0:
                signal, entry_price, sl_price = generate_signals_advanced(SYMBOL, TIMEFRAME_HTF, TIMEFRAME_LTF)

                if signal and entry_price and sl_price:
                    print(f"Signal '{signal}' received. Attempting order...")
                    place_order(signal, SYMBOL, LOT_SIZE, entry_price, sl_price, symbol_info)
                    print("\nWaiting after trade attempt...")
                    time.sleep(120) # Wait longer after an attempt
                else:
                    # print("\nWaiting for next check (no signal)...")
                    time.sleep(30) # Check more frequently if no signal

            # --- Logic if Position IS Open ---
            else:
                print(f"Position already open ({len(positions)}) for {SYMBOL}. Monitoring...")
                # TODO: Implement position management logic (e.g., trailing SL, BE)
                time.sleep(60) # Check open position every minute

        except KeyboardInterrupt:
            print("Bot stopped by user.")
            break
        except Exception as e:
            print(f"!!! An unexpected error occurred in the main loop: {e} !!!")
            import traceback
            traceback.print_exc()
            print("Waiting 60 seconds before retrying...")
            time.sleep(60)

    stop_mt5()

# --- Start Execution ---
if __name__ == "__main__":
    run_bot()

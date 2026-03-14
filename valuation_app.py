import streamlit as st
import joblib
import pandas as pd
import os
from preprocessor import FootballPreprocessor

# ── File paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load models and data (cached so they only load once) ──────────────────────
@st.cache_resource
def load_models():
    pipeline = joblib.load(os.path.join(BASE_DIR, 'football_transfer_pipeline.joblib'))
    rf_no_mv = joblib.load(os.path.join(BASE_DIR, 'RandomForest_without_market_value_compressed.joblib'))
    return pipeline, rf_no_mv

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, 'raw_data.csv'))

pipeline, rf_no_mv = load_models()
raw_data = load_data()

# ── Analysis function ─────────────────────────────────────────────────────────
def player_transfer_analysis(player_name):
    player_data = raw_data[raw_data['player_name'].str.lower() == player_name.lower()]

    if player_data.empty:
        return None

    X = player_data.drop(columns=['transfer_fee'])
    player_info = player_data[['player_name', 'role', 'season_age', 'transfer_season',
                                'from_club_name', 'to_club_name', 'transfer_fee']].copy()

    X_preprocessed = pipeline.named_steps['preprocessor'].transform(X)
    X_no_mv = X_preprocessed.drop(columns=['market_value_in_eur'], errors='ignore')

    player_info['predicted_fee'] = rf_no_mv.predict(X_no_mv)
    player_info['value_diff']    = player_info['predicted_fee'] - player_info['transfer_fee']
    player_info['assessment']    = player_info['value_diff'].apply(
                                       lambda x: '🟢 Undervalued' if x > 0 else '🔴 Overvalued')

    for col in ['transfer_fee', 'predicted_fee', 'value_diff']:
        player_info[col] = player_info[col].apply(lambda x: f'€{x:,.0f}')

    return player_info

# ── Page layout ───────────────────────────────────────────────────────────────
st.title('⚽ Football Transfer Analyser')
st.write('Enter a player name to see their transfer history and whether they were undervalued or overvalued.')

# ── Search box ────────────────────────────────────────────────────────────────
all_players = sorted(raw_data['player_name'].dropna().unique().tolist())
player_name = st.selectbox('Search for a player:', [''] + all_players)

# ── Results ───────────────────────────────────────────────────────────────────
if player_name:
    results = player_transfer_analysis(player_name)

    if results is None:
        st.error(f"No transfers found for '{player_name}'")
    else:
        st.subheader(f'Transfer History: {player_name}')
        st.dataframe(results, use_container_width=True)

        # ── Summary metrics ───────────────────────────────────────────────────
        total_transfers    = len(results)
        undervalued_count  = results['assessment'].str.contains('Undervalued').sum()
        overvalued_count   = total_transfers - undervalued_count

        col1, col2, col3 = st.columns(3)
        col1.metric('Total Transfers', total_transfers)
        col2.metric('🟢 Undervalued', undervalued_count)
        col3.metric('🔴 Overvalued', overvalued_count)

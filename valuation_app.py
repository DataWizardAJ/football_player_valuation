import streamlit as st
import joblib
import pandas as pd
import os
from preprocessor import FootballPreprocessor

# ── File paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load models and data ──────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    pipeline = joblib.load(os.path.join(BASE_DIR, 'football_transfer_pipeline.joblib'))
    rf_no_mv = joblib.load(os.path.join(BASE_DIR, 'RandomForest_without_market_value_compressed.joblib'))
    return pipeline, rf_no_mv

@st.cache_data
def load_data():
    raw_data     = pd.read_csv(os.path.join(BASE_DIR, 'paid_player_transfers_with_stats.csv'))
    stats_2025   = pd.read_csv(os.path.join(BASE_DIR, 'full_2025_season_player_stats.csv'))
    return raw_data, stats_2025

pipeline, rf_no_mv   = load_models()
raw_data, stats_2025 = load_data()

# ── Feature 1: Player Transfer History ───────────────────────────────────────
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

# ── Feature 2: Transfer Fee Predictor ────────────────────────────────────────
def predict_transfer_fee(player_name, buying_club_name):
    player_data = stats_2025[stats_2025['player_name'].str.lower() == player_name.lower()]

    if player_data.empty:
        return None, f"No 2025 stats found for '{player_name}'"

    club_row = raw_data[raw_data['to_club_name'].str.lower() == buying_club_name.lower()]

    if club_row.empty:
        return None, f"Buying club '{buying_club_name}' not found in training data"

    buying_club_id = club_row['to_club_id'].iloc[0]
    player_row     = player_data.iloc[-1:].copy()

    player_row['to_club_id']   = buying_club_id
    player_row['to_club_name'] = buying_club_name

    X              = player_row.drop(columns=['transfer_fee'], errors='ignore')
    X_preprocessed = pipeline.named_steps['preprocessor'].transform(X)
    X_no_mv        = X_preprocessed.drop(columns=['market_value_in_eur'], errors='ignore')

    predicted_fee  = rf_no_mv.predict(X_no_mv)[0]
    return predicted_fee, None

# ── Page layout ───────────────────────────────────────────────────────────────
st.title('⚽ Football Transfer Analyser')

tab1, tab2 = st.tabs(['📋 Player Transfer History', '🔮 Transfer Fee Predictor'])

# ── Tab 1: Transfer History ───────────────────────────────────────────────────
with tab1:
    st.write('Search for a player to see their transfer history and whether they were undervalued or overvalued.')

    all_players  = sorted(raw_data['player_name'].dropna().unique().tolist())
    player_name  = st.selectbox('Search for a player:', [''] + all_players, key='tab1_player')

    if player_name:
        results = player_transfer_analysis(player_name)

        if results is None:
            st.error(f"No transfers found for '{player_name}'")
        else:
            st.subheader(f'Transfer History: {player_name}')
            st.dataframe(results, use_container_width=True)

            total_transfers   = len(results)
            undervalued_count = results['assessment'].str.contains('Undervalued').sum()
            overvalued_count  = total_transfers - undervalued_count

            col1, col2, col3 = st.columns(3)
            col1.metric('Total Transfers', total_transfers)
            col2.metric('🟢 Undervalued', undervalued_count)
            col3.metric('🔴 Overvalued', overvalued_count)

# ── Tab 2: Fee Predictor ──────────────────────────────────────────────────────
with tab2:
    st.write('Select a player and a buying club to predict the transfer fee based on their stats from last season.')

    all_2025_players = sorted(stats_2025['player_name'].dropna().unique().tolist())
    all_clubs        = sorted(raw_data['to_club_name'].dropna().unique().tolist())

    player_name_pred = st.selectbox('Select a player:', [''] + all_2025_players, key='tab2_player')
    buying_club      = st.selectbox('Select buying club:', [''] + all_clubs, key='tab2_club')

    if player_name_pred and buying_club:
        if st.button('Predict Transfer Fee'):
            predicted_fee, error = predict_transfer_fee(player_name_pred, buying_club)

            if error:
                st.error(error)
            else:
                st.success(f'Predicted Transfer Fee: €{predicted_fee:,.0f}')

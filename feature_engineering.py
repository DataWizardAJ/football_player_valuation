import pandas as pd
import numpy as np
import requests
import io
import os
import zipfile
import streamlit as st
import kagglehub

def load_raw_data():
    os.environ['KAGGLE_USERNAME'] = st.secrets['KAGGLE_USERNAME']
    os.environ['KAGGLE_KEY']      = st.secrets['KAGGLE_KEY']

    path = kagglehub.dataset_download('davidcariboo/player-scores')

    # Only load columns you actually need - saves significant memory
    appearances = pd.read_csv(f'{path}/appearances.csv', low_memory=False, usecols=[
        'game_id', 'player_id', 'player_club_id', 'yellow_cards', 'red_cards',
        'goals', 'assists', 'minutes_played'
    ])

    games = pd.read_csv(f'{path}/games.csv', low_memory=False, usecols=[
        'game_id', 'competition_id', 'season', 'home_club_id', 'away_club_id',
        'home_club_goals', 'away_club_goals'
    ])

    players = pd.read_csv(f'{path}/players.csv', low_memory=False, usecols=[
        'player_id', 'name', 'date_of_birth'
    ])

    lineups = pd.read_csv(f'{path}/game_lineups.csv', low_memory=False, usecols=[
        'game_id', 'player_id', 'club_id', 'player_name', 'type', 'position'
    ])

    return appearances, games, players, lineups



def merge_data(appearances, games, players, lineups):
    """Merge all dataframes together"""
    # Merge lineups with appearances
    combined = pd.merge(
        lineups,
        appearances,
        on=['game_id', 'player_id'],
        how='inner'
    )

    # Merge in game level data
    combined = pd.merge(
        combined,
        games,
        on='game_id',
        how='left'
    )

    # Filter out players who didn't play
    combined = combined[combined['minutes_played'] != 0]

    return combined


def add_match_stats(df):
    """Add team scored, conceded and clean sheet columns"""
    df['team_scored'] = np.where(
        df['club_id'] == df['home_club_id'],
        df['home_club_goals'],
        df['away_club_goals']
    )

    df['team_conceded'] = np.where(
        df['club_id'] == df['home_club_id'],
        df['away_club_goals'],
        df['home_club_goals']
    )

    df['clean_sheet'] = np.where(df['team_conceded'] == 0, 1, 0)

    return df


def add_appearance_flags(df):
    """Add started and appearance columns"""
    df['started']    = np.where(df['type'] == 'starting_lineup', 1, 0)
    df['appearance'] = 1
    return df


def simplify_positions(df):
    """Map granular positions to four main roles"""
    position_map = {
        'Goalkeeper':        'Goalkeeper',
        'Centre-Back':       'Defender',
        'Left-Back':         'Defender',
        'Right-Back':        'Defender',
        'Sweeper':           'Defender',
        'Defender':          'Defender',
        'Defensive Midfield':'Midfielder',
        'Attacking Midfield':'Midfielder',
        'Central Midfield':  'Midfielder',
        'Right Midfield':    'Midfielder',
        'Left Midfield':     'Midfielder',
        'midfield':          'Midfielder',
        'Midfield':          'Midfielder',
        'Left Winger':       'Forward',
        'Right Winger':      'Forward',
        'Centre-Forward':    'Forward',
        'Second Striker':    'Forward',
        'Attack':            'Forward',
    }
    df['role'] = df['position'].map(position_map)
    df = df.dropna(subset=['role'])
    return df


def convert_season_format(df):
    """Convert season from xx/yy format to 20yy integer"""
    df['season'] = df['season'].str.split('/').str[1].apply(lambda x: int('20' + x))
    return df


def add_age(df, players):
    """Extract birth year from players df and calculate season age"""
    players['birth_year'] = pd.to_datetime(players['date_of_birth'], errors='coerce').dt.year
    df = df.merge(players[['player_id', 'birth_year']], on='player_id', how='left')
    df['season_age'] = df['season'] - df['birth_year']
    df = df.drop(columns=['birth_year'])
    return df


def aggregate_to_season_level(df):
    """Aggregate match level data up to one row per player per season"""
    stat_cols = ['yellow_cards', 'red_cards', 'goals', 'assists', 'minutes_played',
                 'team_scored', 'team_conceded', 'started', 'appearance', 'clean_sheet']

    # Most common role and club per player per season
    agg_dict = {col: 'sum' for col in stat_cols}
    agg_dict['role']       = lambda x: x.mode()[0] if not x.mode().empty else np.nan
    agg_dict['club_id']    = lambda x: x.mode()[0] if not x.mode().empty else np.nan
    agg_dict['season_age'] = 'max'

    df = (df.groupby(['player_id', 'player_name', 'season'])
            .agg(agg_dict)
            .reset_index())

    return df


def add_time_features(df):
    """Add prior year, rolling 3 year and career total columns"""
    stat_cols = ['yellow_cards', 'red_cards', 'goals', 'assists', 'minutes_played',
                 'team_scored', 'team_conceded', 'started', 'appearance', 'clean_sheet']

    df = df.sort_values(['player_name', 'season']).reset_index(drop=True)

    # Prior year
    for col in stat_cols:
        df[f'{col}_prior_year'] = df.groupby('player_name')[col].shift(1)

    # Rolling 3 year total
    for col in stat_cols:
        df[f'{col}_rolling_3yr'] = (df.groupby('player_name')[col]
                                      .transform(lambda x: x.rolling(3, min_periods=1).sum()))

    # Career total
    for col in stat_cols:
        df[f'{col}_career'] = (df.groupby('player_name')[col]
                                 .transform(lambda x: x.expanding().sum()))

    # Fill prior year NAs with 0 (first season of career)
    prior_year_cols = [col for col in df.columns if col.endswith('_prior_year')]
    df[prior_year_cols] = df[prior_year_cols].fillna(0)

    return df


def run_pipeline():
    
    print('Loading raw data...')
    appearances, games, players, lineups = load_raw_data()

    # Only keep last 4 seasons - enough for rolling 3yr stats + prior year
    print('Filtering to recent seasons...')
    current_season = games['season'].max()
    recent_seasons = [current_season, current_season-1, current_season-2, current_season-3]
    games    = games[games['season'].isin(recent_seasons)]
    game_ids = games['game_id'].unique()
    lineups      = lineups[lineups['game_id'].isin(game_ids)]
    appearances  = appearances[appearances['game_id'].isin(game_ids)]

    # rest of pipeline continues as before...
    """Run the full feature engineering pipeline and return processed dataframe"""
    print('Loading raw data...')
    appearances, games, players, lineups = load_raw_data()

    print('Merging data...')
    df = merge_data(appearances, games, players, lineups)

    print('Adding match stats...')
    df = add_match_stats(df)
    df = add_appearance_flags(df)

    print('Cleaning positions...')
    df = simplify_positions(df)

    print('Converting season format...')
    df = convert_season_format(df)

    print('Adding age...')
    df = add_age(df, players)

    print('Aggregating to season level...')
    df = aggregate_to_season_level(df)

    print('Adding time features...')
    df = add_time_features(df)

    print('Done!')
    return df


if __name__ == '__main__':
    df = run_pipeline()
    df.to_csv('player_stats_live.csv', index=False)
    print(f'Saved {len(df)} rows to player_stats_live.csv')

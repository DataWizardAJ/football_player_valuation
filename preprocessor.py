from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class FootballPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.target_encoding_maps = {}
        self.drop_cols = ['player_name', 'from_club_name', 'transfer_season',
                          'to_club_name', 'transfer_date', 'player_id']
        self.high_cardinality_cols = ['from_club_id', 'to_club_id', 'club_id',
                                      'country_of_citizenship', 'domestic_competition_id']
        self.final_columns = None  # stores column structure seen at fit time

    def fit(self, X, y=None):
        temp = X.copy()
        temp['transfer_fee'] = y
        for col in self.high_cardinality_cols:
            self.target_encoding_maps[col] = temp.groupby(col)['transfer_fee'].mean()
        
        # Store the column structure after fitting so transform can always match it
        temp = temp.drop(columns=['transfer_fee'])
        temp = self._base_transform(temp)
        self.final_columns = temp.columns.tolist()
        return self

    
    def _base_transform(self, df):
        df = df.drop(columns=[c for c in self.drop_cols if c in df.columns])
        df = pd.get_dummies(df, columns=['role'], drop_first=False)
    
        for col in self.high_cardinality_cols:
            if col in df.columns:
                df[col] = df[col].map(self.target_encoding_maps[col])
            else:
                # Use mean of top 20% highest paying clubs as default
                encoding_map    = self.target_encoding_maps[col]
                top_20_threshold = encoding_map.quantile(0.90)
                top_20_mean      = encoding_map[encoding_map >= top_20_threshold].mean()
                df[col]          = top_20_mean
    
        return df

    def transform(self, X, y=None):
        df = X.copy()
        df = self._base_transform(df)
        
        # Reindex to match training columns exactly - fills any missing cols with 0
        df = df.reindex(columns=self.final_columns, fill_value=0)
        return df

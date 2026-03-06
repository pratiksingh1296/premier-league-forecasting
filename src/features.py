import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/Pratik/DS/premier-league-ml/data/processed/PL_processed.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Calculating Points Per Game (PPG) for home and away teams
def ppg(team, date, df, n_games):
    h_past_games = df[((df['HomeTeam'] == team) & (df['Date'] < date))].tail(n_games)
    a_past_games = df[((df['AwayTeam'] == team) & (df['Date'] < date))].tail(n_games)
    
    h_points = 0
    a_points = 0

    for _, match in h_past_games.iterrows():
            if match['FTR'] == 'H':
                h_points += 3
            elif match['FTR'] == 'D':
                h_points += 1

    for _, match in a_past_games.iterrows():
            if match['FTR'] == 'A':
                a_points += 3
            elif match['FTR'] == 'D':
                a_points += 1

    if len(h_past_games) > 0:
        h_ppg = h_points / len(h_past_games)
    else:
        h_ppg = np.nan
    if len(a_past_games) > 0:
        a_ppg = a_points / len(a_past_games)
    else:
        a_ppg = np.nan
    
    return h_ppg, a_ppg

# Calculating Goals For and Against Per Game
def goals_pg(team, date, df, n_games):
    h_past_games = df[((df['HomeTeam'] == team) & (df['Date'] < date))].tail(n_games)
    a_past_games = df[((df['AwayTeam'] == team) & (df['Date'] < date))].tail(n_games)
    
    h_GF = h_past_games['FTHG'].sum()
    h_GA = h_past_games['FTAG'].sum()
    a_GF = a_past_games['FTAG'].sum()
    a_GA = a_past_games['FTHG'].sum()

    h_GF_pg = h_GF / len(h_past_games) if len(h_past_games) > 0 else np.nan
    h_GA_pg = h_GA / len(h_past_games) if len(h_past_games) > 0 else np.nan
    a_GF_pg = a_GF / len(a_past_games) if len(a_past_games) > 0 else np.nan
    a_GA_pg = a_GA / len(a_past_games) if len(a_past_games) > 0 else np.nan

    return h_GF_pg, h_GA_pg, a_GF_pg, a_GA_pg

# Generating feature table
def feature_table(df):
    
    df.sort_values(by=['Date'], ascending=True, inplace=True)
    
    feature_rows = []
    
    # Filtering past 5 games and calculating PPG
    for _, matchrow in df.iterrows():
        h_team = matchrow['HomeTeam']
        a_team = matchrow['AwayTeam']
        match_date = matchrow['Date']
        
        # Calculate features for home and away teams
        # Points per game
        h_ppg_5, _ = ppg(h_team, match_date, df, 5)
        _, a_ppg_5 = ppg(a_team, match_date, df, 5)
        
        h_ppg_15, _ = ppg(h_team, match_date, df, 15)
        _, a_ppg_15 = ppg(a_team, match_date, df, 15)

        # Goals for and against per game
        h_GF_5, h_GA_5, _ , _ = goals_pg(h_team, match_date, df, 5)
        _ , _ , a_GF_5, a_GA_5 = goals_pg(a_team, match_date, df, 5)
        h_GF_15, h_GA_15, _ , _ = goals_pg(h_team, match_date, df, 15)
        _ , _ , a_GF_15, a_GA_15 = goals_pg(a_team, match_date, df, 15)

        # Total Goals Per Match
        Total_GF_5 = h_GF_5 + a_GF_5
        Total_GF_15 = h_GF_15 + a_GF_15

        # Absolute PPG Difference
        Abs_PPG_Diff_15 = abs(h_ppg_15 - a_ppg_15)
        Abs_PPG_Diff_5 = abs(h_ppg_5 - a_ppg_5)

        # Binary Trigger to help think how close games are
        Is_Close_5  = (Abs_PPG_Diff_5 < 0.3) 
        Is_Close_15 = (Abs_PPG_Diff_15 < 0.3)
        
        # Goal differences
        h_GD_5 = h_GF_5 - h_GA_5
        a_GD_5 = a_GF_5 - a_GA_5
        h_GD_15 = h_GF_15 - h_GA_15
        a_GD_15 = a_GF_15 - a_GA_15

        # Difference in PPG
        Diff_PPG_5 = h_ppg_5 - a_ppg_5
        Diff_PPG_15 = h_ppg_15 - a_ppg_15

        # Difference in Goals for and against
        Diff_GF_5 = h_GF_5 - a_GF_5 
        Diff_GA_5 = h_GA_5 - a_GA_5
        Diff_GF_15 = h_GF_15 - a_GF_15
        Diff_GA_15 = h_GA_15 - a_GA_15

        # Difference in Goal Differences
        Diff_GD_5 = h_GD_5 - a_GD_5
        Diff_GD_15 = h_GD_15 - a_GD_15

        # Games played 
        Home_GP_5 = len(df[(df['HomeTeam'] == h_team) & (df['Date'] < match_date)].tail(5))
        Away_GP_5 = len(df[(df['AwayTeam'] == a_team) & (df['Date'] < match_date)].tail(5))
        Home_GP_15 = len(df[(df['HomeTeam'] == h_team) & (df['Date'] < match_date)].tail(15))
        Away_GP_15 = len(df[(df['AwayTeam'] == a_team) & (df['Date'] < match_date)].tail(15))

        feature_rows.append({
            'Date': match_date,
            'HomeTeam': h_team,
            'AwayTeam': a_team,
            'Home_PPG_5': h_ppg_5,
            'Away_PPG_5': a_ppg_5,
            'Home_PPG_15': h_ppg_15,
            'Away_PPG_15': a_ppg_15,
            'Home_GF_5': h_GF_5,
            'Home_GA_5': h_GA_5,    
            'Away_GF_5': a_GF_5,
            'Away_GA_5': a_GA_5,
            'Home_GD_5': h_GD_5,
            'Away_GD_5': a_GD_5,
            'Home_GF_15': h_GF_15,
            'Home_GA_15': h_GA_15,
            'Away_GF_15': a_GF_15,
            'Away_GA_15': a_GA_15,
            'Home_GD_15': h_GD_15,
            'Away_GD_15': a_GD_15,
            'Diff_PPG_5': Diff_PPG_5,
            'Diff_PPG_15': Diff_PPG_15,
            'Diff_GF_5': Diff_GF_5,
            'Diff_GA_5': Diff_GA_5,
            'Diff_GF_15': Diff_GF_15,
            'Diff_GA_15': Diff_GA_15,
            'Diff_GD_5': Diff_GD_5,
            'Diff_GD_15': Diff_GD_15,
            'Home_GP_5': Home_GP_5,
            'Away_GP_5': Away_GP_5,
            'Home_GP_15': Home_GP_15,
            'Away_GP_15': Away_GP_15,
            'Total_GF_5': Total_GF_5,
            'Total_GF_15': Total_GF_15,
            'Abs_PPG_Diff_5':Abs_PPG_Diff_5,
            'Abs_PPG_Diff_15':Abs_PPG_Diff_15,
            'Is_Close_5':Is_Close_5,
            'Is_Close_15':Is_Close_15,
            'FTR': matchrow['FTR']
        })
    features = pd.DataFrame(feature_rows)
    return features

# Generate and save feature table
features_v1 = feature_table(df)
features_v1['Is_Close_5'] = features_v1['Is_Close_5'].astype(int) # Convert boolean to int for easier modeling
features_v1['Is_Close_15'] = features_v1['Is_Close_15'].astype(int) 
features_v1.to_csv("C:/Users/Pratik/DS/premier-league-ml/data/processed/features_v1.csv", index=False)
import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/Pratik/DS/premier-league-ml/reports/tables/match_probabilities_xgb_calibrated.csv")

print(df.head(),'\n')

# Expected Points Calculation
df['ExpPoints_Home'] = (df['P_Home'] * 3) + (df['P_Draw'] * 1)
df['ExpPoints_Away'] = (df['P_Away'] * 3) + (df['P_Draw'] * 1)

# Aggregate Points
home = df.groupby('HomeTeam')['ExpPoints_Home'].sum()
away = df.groupby('AwayTeam')['ExpPoints_Away'].sum()

# Expected Points Table
xP_table = (
    home.add(away, fill_value=0).reset_index().rename(columns={0: 'xPts', 'HomeTeam': 'Team'}).sort_values(by='xPts', ascending=False)
)

print(xP_table.head(10),'\n')

# Actual Points Calculation
def calculate_actual_points(row):
    if row['FTR'] == 'H':
        return 3, 0
    elif row['FTR'] == 'A':
        return 0, 3
    else:
        return 1, 1
df[['ActualPoints_Home', 'ActualPoints_Away']] = df.apply(calculate_actual_points, axis=1, result_type='expand')

# Aggregate Actual Points
home_actual = df.groupby('HomeTeam')['ActualPoints_Home'].sum()
away_actual = df.groupby('AwayTeam')['ActualPoints_Away'].sum()

# Actual Points Table
actual_table = (
    home_actual.add(away_actual, fill_value=0).reset_index().rename(columns={0: 'ActualPts', 'HomeTeam': 'Team'}).sort_values(by='ActualPts', ascending=False)
)
print(actual_table.head(10),'\n')

# Merging Expected and Actual Points
league_table = pd.merge(xP_table, actual_table, on='Team')
league_table['Difference'] = league_table['xPts'] - league_table['ActualPts']
league_table = league_table.sort_values(by='xPts', ascending=False).reset_index(drop=True)
league_table.index += 1  # Start index from 1 since it's a league table
print("\nLeague Table based on Expected Points vs Actual Points:\n")
print(league_table)
league_table.to_csv("C:/Users/Pratik/DS/premier-league-ml/reports/tables/league_table_expected_vs_actual.csv", index_label='Position')
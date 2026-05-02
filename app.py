import streamlit as st
import pandas as pd

match_probs = pd.read_csv("reports/tables/match_probabilities_xgb_calibrated.csv")
season_summary = pd.read_csv("reports/tables/monte_carlo_season_summary.csv", index_col=0)

# Get Team Lists
home_teams = match_probs["HomeTeam"].unique()
away_teams = match_probs["AwayTeam"].unique()
all_teams = sorted(season_summary.index.tolist())

st.title("Premier League Predictor")
st.markdown("Probabilistic match predictions and Monte Carlo season simulations.")


# Tabs
tab1, tab2 = st.tabs(["Match Predictor", "Season Simulation"])

with tab1:
    st.subheader("Match Outcome - Test Season Predictions")
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home Team", home_teams)
    with col2:
        away_team = st.selectbox("Away Team", [t for t in away_teams if t != home_team])

    matches = match_probs[
        (match_probs["HomeTeam"] == home_team) &
        (match_probs["AwayTeam"] == away_team)
    ]


    if len(matches) > 0:
        row = matches.iloc[0]

        st.markdown("---")
        st.subheader(f"{home_team} vs {away_team}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{home_team} Win", f"{row['P_Home']:.1%}")
        with col2:
            st.metric("Draw", f"{row['P_Draw']:.1%}")
        with col3:
            st.metric(f"{away_team} Win", f"{row['P_Away']:.1%}")

        # Show actual results for context
        result_map = {"H": f"{home_team} Won", "D": "Draw", "A": f"{away_team} Won"}
        actual = result_map.get(row["FTR"], "Unknown")
        predicted = result_map.get(row["Predicted"], "Unknown")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Model Predicted:** {predicted}")
        with col2:
            if predicted == actual:
                st.success(f"**Actual Result:** {actual}")
            else:
                st.error(f"**Actual Result:** {actual}")

    else:
        st.info("This fixture wasn't in the test season.")


with tab2:
    st.subheader("Monte Carlo Season Simulation")
    st.markdown("Based on 10,000 simulated seasons using calibrated probabilities.")

    selected_team = st.selectbox("Select a Team to View Season Simulation", all_teams)

    if selected_team:
        team_data = season_summary.loc[selected_team]

        st.markdown("---")
        st.subheader(f"{selected_team} - Season Outlook")   

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Expected Points", f"{team_data['Expected_Points']:.1f}")
            st.caption(f"± {team_data['Std_Points']:.1f} std dev")
        with col2:
            st.metric("Top 4 Probability", f"{team_data['Top4_Prob']:.1%}")
        with col3:
            st.metric("Relegation Risk", f"{team_data['Relegation_Prob']:.1%}")

        # Title Probability if relevant
        if team_data['Title_Prob'] > 0.01:
            st.metric("Title Probability", f"{team_data['Title_Prob']:.1%}")
        
        # Context
        st.markdown("---")
        st.markdown("Probabilities are estimated from 10,000 Monte Carlo simulations using XGBoost calibrated match outcome probabilities.")

        # Show Full Season Summary Table
        st.markdown("### Full League Table")
        display_df = season_summary[["Expected_Points", "Top4_Prob", "Relegation_Prob", "Title_Prob"]].copy()
        display_df.columns = ["Exp. Points", "Top 4 %", "Relegation %", "Title %"]
        # Format percentages and points
        display_df["Top 4 %"] = (display_df["Top 4 %"] * 100).round(1).astype(str) + "%"
        display_df["Relegation %"] = (display_df["Relegation %"] * 100).round(1).astype(str) + "%"
        display_df["Title %"] = (display_df["Title %"] * 100).round(1).astype(str) + "%"
        display_df["Exp. Points"] = (display_df["Exp. Points"].round(1))
        st.dataframe(display_df, use_container_width=True)

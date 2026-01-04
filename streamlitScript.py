import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

with open('ufc_model.pkl', 'rb') as f:
    best = pickle.load(f)
df_fighters = pd.read_csv("fighters_lookup.csv")

# Static features to show on the radar
static_cols = ["height", "reach", "splm", "str_acc", "sapm", "str_def", "td_avg","td_avg_acc","td_def", "sub_avg"]

# Precompute global min/max for normalisation (0–1 scale)
stats_min = df_fighters[static_cols].min()
stats_max = df_fighters[static_cols].max()

def get_normalised_stats(row: pd.Series) -> pd.Series:
    """Return 0-1 normalised static stats for a fighter."""
    vals = (row[static_cols] - stats_min) / (stats_max - stats_min)
    return vals.fillna(0).clip(0, 1)

required_cols = {"name", "division"}
missing = required_cols - set(df_fighters.columns)
if missing:
    raise ValueError(f"fighters_with_division.csv is missing required columns: {missing}")

# MultiIndex: (name, division)
df_fighters = df_fighters.set_index(["name", "division"])
df_fighters = df_fighters.sort_index()

# List of all fighter names & divisions for dropdowns
all_fighters = sorted(df_fighters.index.get_level_values("name").unique())
all_divisions = sorted(df_fighters.index.get_level_values("division").unique())

# All Features
feature_names = [
    'r_height',
    'r_weight',
    'r_reach',
    'r_splm',
    'r_str_acc',
    'r_sapm',
    'r_str_def',
    'r_td_avg',
    'r_td_avg_acc',
    'r_td_def',
    'r_sub_avg',
    'b_height',
    'b_weight',
    'b_reach',
    'b_splm',
    'b_str_acc',
    'b_sapm',
    'b_str_def',
    'b_td_avg',
    'b_td_avg_acc',
    'b_td_def',
    'b_sub_avg',
    'r_roll_kd',
    'r_roll_sig_str_landed',
    'r_roll_sig_str_atmpted',
    'r_roll_sig_str_acc',
    'r_roll_total_str_landed',
    'r_roll_total_str_atmpted',
    'r_roll_total_str_acc',
    'r_roll_td_landed',
    'r_roll_td_atmpted',
    'r_roll_td_acc',
    'r_roll_sub_att',
    'r_roll_ctrl',
    'r_roll_head_landed',
    'r_roll_head_atmpted',
    'r_roll_head_acc',
    'r_roll_body_landed',
    'r_roll_body_atmpted',
    'r_roll_body_acc',
    'r_roll_leg_landed',
    'r_roll_leg_atmpted',
    'r_roll_leg_acc',
    'r_roll_dist_landed',
    'r_roll_dist_atmpted',
    'r_roll_dist_acc',
    'r_roll_clinch_landed',
    'r_roll_clinch_atmpted',
    'r_roll_clinch_acc',
    'r_roll_ground_landed',
    'r_roll_ground_atmpted',
    'r_roll_ground_acc',
    'r_roll_landed_head_per',
    'r_roll_landed_body_per',
    'r_roll_landed_leg_per',
    'r_roll_landed_dist_per',
    'r_roll_landed_clinch_per',
    'r_roll_landed_ground_per',
    'b_roll_kd',
    'b_roll_sig_str_landed',
    'b_roll_sig_str_atmpted',
    'b_roll_sig_str_acc',
    'b_roll_total_str_landed',
    'b_roll_total_str_atmpted',
    'b_roll_total_str_acc',
    'b_roll_td_landed',
    'b_roll_td_atmpted',
    'b_roll_td_acc',
    'b_roll_sub_att',
    'b_roll_ctrl',
    'b_roll_head_landed',
    'b_roll_head_atmpted',
    'b_roll_head_acc',
    'b_roll_body_landed',
    'b_roll_body_atmpted',
    'b_roll_body_acc',
    'b_roll_leg_landed',
    'b_roll_leg_atmpted',
    'b_roll_leg_acc',
    'b_roll_dist_landed',
    'b_roll_dist_atmpted',
    'b_roll_dist_acc',
    'b_roll_clinch_landed',
    'b_roll_clinch_atmpted',
    'b_roll_clinch_acc',
    'b_roll_ground_landed',
    'b_roll_ground_atmpted',
    'b_roll_ground_acc',
    'b_roll_landed_head_per',
    'b_roll_landed_body_per',
    'b_roll_landed_leg_per',
    'b_roll_landed_dist_per',
    'b_roll_landed_clinch_per',
    'b_roll_landed_ground_per',
    'r_stance_Orthodox',
    'r_stance_Sideways',
    'r_stance_Southpaw',
    'r_stance_Switch',
    'b_stance_Orthodox',
    'b_stance_Sideways',
    'b_stance_Southpaw',
    'b_stance_Switch'
]


# FOR VISUALS
def plot_radar_chart(fighter_red, fighter_blue, division, static_cols):
    """
    Build and return a Plotly radar chart comparing two fighters on static stats.
    """

    # Get rows
    r_row = get_fighter_row(fighter_red, division)
    b_row = get_fighter_row(fighter_blue, division)

    # Normalise
    r_norm = get_normalised_stats(r_row)
    b_norm = get_normalised_stats(b_row)

    # Build radar chart
    categories = static_cols
    theta = categories + [categories[0]]

    r_values = r_norm.tolist() + [r_norm.iloc[0]]
    b_values = b_norm.tolist() + [b_norm.iloc[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=r_values,
        theta=theta,
        name=fighter_red,
        line=dict(color="red"),
        fill="toself",
        opacity=0.6
    ))

    fig.add_trace(go.Scatterpolar(
        r=b_values,
        theta=theta,
        name=fighter_blue,
        line=dict(color="royalblue"),
        fill="toself",
        opacity=0.6
    ))

    fig.update_layout(
        title=f"Profile Comparison: {fighter_red} vs {fighter_blue}",
        polar=dict(radialaxis=dict(visible=False)),
        showlegend=True
    )

    return fig


# Fetch a fighter row for (name, division)
def get_fighter_row(name: str, division: str) -> pd.Series:
    idx = (name, division)
    if idx not in df_fighters.index:
        if name not in df_fighters.index.get_level_values("name"):
            raise ValueError(f"Fighter '{name}' not found in fighters_with_division.csv.")
        else:
            available_divs = df_fighters.loc[name].index.unique().tolist()
            raise ValueError(
                f"Fighter '{name}' has no data in division '{division}'. "
                f"Available divisions: {available_divs}"
            )
    row = df_fighters.loc[idx]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return row


# Create a single-row dataframe with the same columns as the model 
def build_match_row(
    fighter_red: str,
    division_red: str,
    fighter_blue: str,
    division_blue: str
) -> pd.DataFrame:
    if division_red != division_blue:
        raise ValueError(
            f"Division mismatch: '{fighter_red}' ({division_red}) vs "
            f"'{fighter_blue}' ({division_blue}). "
            "Fights should be within the same division."
        )

    r = get_fighter_row(fighter_red, division_red)
    b = get_fighter_row(fighter_blue, division_blue)

    row = {}
    for col in feature_names:
        if col.startswith("r_"):
            base = col[2:]
            row[col] = r.get(base, np.nan)
        elif col.startswith("b_"):
            base = col[2:]
            row[col] = b.get(base, np.nan)
        else:
            row[col] = 0

    X_input = pd.DataFrame([row])
    X_input = X_input[feature_names]       
    X_input = X_input.fillna(0)              
    return X_input
    
def predict_match(
    fighter1: str,
    division1: str,
    fighter2: str,
    division2: str,
):
    # Basic sanity check: same fighter
    if (fighter1 == fighter2):
        raise ValueError(
            f"Invalid matchup: '{fighter1}' cannot fight themselves "
            f"in division '{division1}'. Please choose two different fighters."
        )

    # 1) First ordering: fighter1 = red, fighter2 = blue
    X_12 = build_match_row(fighter1, division1, fighter2, division2)
    p_red_12 = best.predict_proba(X_12)[0, 1]   # P(red wins) = P(fighter1 wins in this ordering)

    # 2) Second ordering: fighter2 = red, fighter1 = blue
    X_21 = build_match_row(fighter2, division2, fighter1, division1)
    p_red_21 = best.predict_proba(X_21)[0, 1]   # P(red wins) = P(fighter2 wins in this ordering)

    # Symmetric probability that fighter1 wins
    p_f1 = (p_red_12 + (1.0 - p_red_21)) / 2.0
    p_f2 = 1.0 - p_f1

    if p_f1 >= 0.5:
        winner = fighter1
        confidence = p_f1
    else:
        winner = fighter2
        confidence = p_f2

    return winner, float(confidence)


st.set_page_config(
    page_title="UFC Fight Predictor",
    layout="centered",
)

st.title("UFC Fight Predictor")
# Division selection
all_divisions = sorted(df_fighters.index.get_level_values("division").unique())
division = st.selectbox("Select Division", all_divisions)

# Fighters available in this division
fighters_in_div = sorted(df_fighters.xs(division, level="division").index.unique())

col1, col2 = st.columns(2)
with col1:
    fighter_red = st.selectbox("Red Corner", fighters_in_div, key="red")
with col2:
    # filter so blue options exclude the chosen red fighter
    blue_options = [f for f in fighters_in_div if f != fighter_red]
    fighter_blue = st.selectbox("Blue Corner", blue_options, key="blue")

st.markdown("---")

if st.button("Predict"):
    try:
        winner, confidence = predict_match(fighter_red, division, fighter_blue, division)
        st.success(f"**Predicted Winner:** {winner}")
        st.metric("Probability", f"{confidence*100:.2f}%")

        fig = plot_radar_chart(fighter_red, fighter_blue, division, static_cols)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("splm - Significant Strikes Landed per Minute")
        st.markdown("str_acc - Significant Striking Accuracy")
        st.markdown("sapm - Significant Strikes Absorbed per Minute")
        st.markdown("str_def - Significant Strike Defence (the % of opponents strikes that did not land)")
        st.markdown("td_avg - Average Takedowns Landed per 15 minutes")
        st.markdown("td_avg_acc - Average Takedown Accuracy")
        st.markdown("td_def - Takedown Defense (the % of opponents TD attempts that did not land)")
        st.markdown("sub_avg - Average Submissions Attempted per 15 minutes")
        st.markdown("height - Height of a Fighter (cm)")
        st.markdown("reach - Measurement of a Fighter's wingspan (cm)")

    except ValueError as e:
        st.error(str(e))
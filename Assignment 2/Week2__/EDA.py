
# ============================================================================
# CLEAN DATASET CREATION - WINDOWS FIX
# ============================================================================
# This script uses CORRECT file paths for Windows!
# ============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os

print("\n" + "=" * 80)
print("CREATING CLEAN DATASET - WINDOWS VERSION")
print("=" * 80)

# ============================================================================
# IMPORTANT: HOW TO SET FILE PATHS ON WINDOWS
# ============================================================================


# ============================================================================
# STEP 0: SET YOUR FILE PATHS (MODIFY THESE!)
# ============================================================================

# Change these paths to YOUR folder location!
BASE_PATH = 'C:/Users/cbsen/OneDrive/Desktop/FIFA Project/Code'

# File paths using FORWARD SLASHES (Windows-safe)
decision_path = f'{BASE_PATH}/decision.csv'
fifa_results_path = f'{BASE_PATH}/FIFA Results.csv'
penalty_path = f'{BASE_PATH}/penality kick.csv'

# Output path
output_path = f'{BASE_PATH}/clean_fifa_dataset.csv'

print(f"\n📁 File Paths Being Used:")
print(f"  Decision: {decision_path}")
print(f"  FIFA Results: {fifa_results_path}")
print(f"  Penalty: {penalty_path}")
print(f"  Output: {output_path}")

# Check if files exist
print(f"\n✓ Checking if files exist...")
print(f"  Decision exists: {os.path.exists(decision_path)}")
print(f"  FIFA Results exists: {os.path.exists(fifa_results_path)}")
print(f"  Penalty exists: {os.path.exists(penalty_path)}")

# ============================================================================
# STEP 1: LOAD ALL 3 FILES
# ============================================================================
print("\n[STEP 1] Loading Raw Files...")

try:
    decision = pd.read_csv(decision_path)
    print(f"✓ decision.csv loaded: {decision.shape}")
except Exception as e:
    print(f"❌ Error loading decision.csv: {e}")
    exit()

try:
    fifa_results = pd.read_csv(fifa_results_path)
    print(f"✓ FIFA Results.csv loaded: {fifa_results.shape}")
except Exception as e:
    print(f"❌ Error loading FIFA Results.csv: {e}")
    exit()

try:
    penalty = pd.read_csv(penalty_path)
    print(f"✓ penality kick.csv loaded: {penalty.shape}")
except Exception as e:
    print(f"❌ Error loading penality kick.csv: {e}")
    exit()

# ============================================================================
# STEP 2: CLEAN DECISION.CSV (MATCH RESULTS)
# ============================================================================
print("\n[STEP 2] Cleaning Match Results...")

print(f"  Before: {decision.shape}")

# Convert date
decision['date'] = pd.to_datetime(decision['date'], errors='coerce')

# Remove missing values
decision = decision.dropna(subset=['home_team', 'away_team', 'home_score', 'away_score'])

# Convert scores to integers
decision['home_score'] = pd.to_numeric(decision['home_score'], errors='coerce').astype('Int64')
decision['away_score'] = pd.to_numeric(decision['away_score'], errors='coerce').astype('Int64')

# Remove rows where conversion failed
decision = decision.dropna(subset=['home_score', 'away_score'])

print(f"  After: {decision.shape}")
print(f"  ✓ Cleaned!")

# ============================================================================
# STEP 3: CLEAN FIFA RESULTS.CSV (GOALS)
# ============================================================================
print("\n[STEP 3] Cleaning Goals Data...")

print(f"  Before: {fifa_results.shape}")

# Convert date
fifa_results['date'] = pd.to_datetime(fifa_results['date'], errors='coerce')

# Remove missing
fifa_results = fifa_results.dropna(subset=['team', 'home_team', 'away_team'])

# Fix boolean columns
if 'penalty' in fifa_results.columns:
    fifa_results['penalty'] = fifa_results['penalty'].astype(str).str.lower() == 'true'
if 'own_goal' in fifa_results.columns:
    fifa_results['own_goal'] = fifa_results['own_goal'].astype(str).str.lower() == 'true'

print(f"  After: {fifa_results.shape}")
print(f"  ✓ Cleaned!")

# ============================================================================
# STEP 4: CLEAN PENALTY.CSV (SHOOTOUTS)
# ============================================================================
print("\n[STEP 4] Cleaning Penalty Shootouts...")

print(f"  Before: {penalty.shape}")

# Convert date
penalty['date'] = pd.to_datetime(penalty['date'], errors='coerce')

# Remove missing
penalty = penalty.dropna(subset=['home_team', 'winner'])

print(f"  After: {penalty.shape}")
print(f"  ✓ Cleaned!")

# ============================================================================
# STEP 5: CREATE MATCH FEATURES
# ============================================================================
print("\n[STEP 5] Creating Match Features...")

all_teams = set(decision['home_team'].unique()) | set(decision['away_team'].unique())
print(f"  Total teams: {len(all_teams)}")

match_features = []

for team in sorted(all_teams):
    # HOME
    home = decision[decision['home_team'] == team]
    h_wins = len(home[home['home_score'] > home['away_score']])
    h_draws = len(home[home['home_score'] == home['away_score']])
    h_losses = len(home[home['home_score'] < home['away_score']])
    h_gf = home['home_score'].sum()
    h_ga = home['away_score'].sum()
    
    # AWAY
    away = decision[decision['away_team'] == team]
    a_wins = len(away[away['away_score'] > away['home_score']])
    a_draws = len(away[away['away_score'] == away['home_score']])
    a_losses = len(away[away['away_score'] < away['home_score']])
    a_gf = away['away_score'].sum()
    a_ga = away['home_score'].sum()
    
    # TOTAL
    total_matches = len(home) + len(away)
    total_wins = h_wins + a_wins
    total_draws = h_draws + a_draws
    total_losses = h_losses + a_losses
    total_gf = h_gf + a_gf
    total_ga = h_ga + a_ga
    
    if total_matches > 0:
        match_features.append({
            'team': team,
            'total_matches': total_matches,
            'total_wins': total_wins,
            'total_draws': total_draws,
            'total_losses': total_losses,
            'match_win_rate': round(total_wins / total_matches, 3),
            'goals_for_per_match': round(total_gf / total_matches, 2),
            'goals_against_per_match': round(total_ga / total_matches, 2),
            'goal_difference_per_match': round((total_gf - total_ga) / total_matches, 2),
        })

match_df = pd.DataFrame(match_features)
print(f"✓ Match features: {len(match_df)} teams")

# ============================================================================
# STEP 6: CREATE GOAL FEATURES
# ============================================================================
print("\n[STEP 6] Creating Goal Features...")

goal_features = []

for team in sorted(all_teams):
    goals = fifa_results[fifa_results['team'] == team]
    
    total_goals = len(goals)
    penalty_goals = len(goals[goals['penalty'] == True]) if 'penalty' in goals.columns else 0
    own_goals = len(goals[goals['own_goal'] == True]) if 'own_goal' in goals.columns else 0
    
    goal_features.append({
        'team': team,
        'total_goals_scored': total_goals,
        'penalty_goals': penalty_goals,
        'own_goals': own_goals,
    })

goal_df = pd.DataFrame(goal_features)
print(f"✓ Goal features: {len(goal_df)} teams")

# ============================================================================
# STEP 7: CREATE SHOOTOUT FEATURES
# ============================================================================
print("\n[STEP 7] Creating Shootout Features...")

shootout_features = []

# Find opponent column
opponent_col = 'apponent_team' if 'apponent_team' in penalty.columns else 'away_team'

for team in sorted(all_teams):
    # HOME
    home_shoot = penalty[penalty['home_team'] == team]
    home_wins = len(home_shoot[home_shoot['winner'] == team])
    home_total = len(home_shoot)
    
    # AWAY
    away_shoot = penalty[penalty[opponent_col] == team]
    away_wins = len(away_shoot[away_shoot['winner'] == team])
    away_total = len(away_shoot)
    
    # TOTAL
    total_shootouts = home_total + away_total
    total_shoot_wins = home_wins + away_wins
    
    shoot_wr = total_shoot_wins / total_shootouts if total_shootouts > 0 else 0
    
    shootout_features.append({
        'team': team,
        'total_shootouts': total_shootouts,
        'shootout_wins': total_shoot_wins,
        'shootout_losses': total_shootouts - total_shoot_wins,
        'shootout_win_rate': round(shoot_wr, 3),
    })

shootout_df = pd.DataFrame(shootout_features)
print(f"✓ Shootout features: {len(shootout_df)} teams")

# ============================================================================
# STEP 8: MERGE ALL FEATURES
# ============================================================================
print("\n[STEP 8] Merging All Features...")

master_df = match_df.merge(goal_df, on='team', how='left').merge(shootout_df, on='team', how='left')
master_df = master_df.fillna(0)

print(f"✓ Master dataset: {master_df.shape}")
print(f"  Teams: {len(master_df)}")
print(f"  Features: {len(master_df.columns)}")

# ============================================================================
# STEP 9: QUALITY CHECK
# ============================================================================
print("\n[STEP 9] Quality Check...")

print(f"  Missing values: {master_df.isnull().sum().sum()}")
print(f"  Duplicates: {master_df.duplicated().sum()}")

# ============================================================================
# STEP 10: SAVE CLEAN DATASET
# ============================================================================
print("\n[STEP 10] Saving Clean Dataset...")

master_df.to_csv(output_path, index=False)

print(f"\n✅ SUCCESS!")
print(f"✅ File saved: {output_path}")
print(f"\nDataset Info:")
print(f"  Teams: {len(master_df)}")
print(f"  Features: {len(master_df.columns)}")
print(f"  Rows × Cols: {master_df.shape}")

print(f"\nTop 5 teams by win rate:")
print(master_df.nlargest(5, 'match_win_rate')[['team', 'match_win_rate', 'total_matches']])

print(f"\nFirst 10 teams:")
print(master_df.head(10))

print("\n" + "=" * 80)
print("✅ CLEAN DATASET READY FOR ML!")
print("=" * 80)
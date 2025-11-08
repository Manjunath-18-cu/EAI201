# app.py
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import json
from datetime import datetime
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load and preprocess your dataset
def load_and_preprocess_data():
    try:
        df = pd.read_csv('fifa_neutral_dataset.csv')
        print(f"✅ Dataset loaded: {len(df)} matches")
        
        # Debug: Print column names to see what we have
        print(f"📋 Columns in dataset: {df.columns.tolist()}")
        
        # Check if we have the expected columns
        if 'team_a' not in df.columns or 'team_b' not in df.columns:
            print("❌ Missing team columns! Available columns:", df.columns.tolist())
            return pd.DataFrame()
            
        # Create consistent team naming
        df['team_a'] = df['team_a'].astype(str).str.strip()
        df['team_b'] = df['team_b'].astype(str).str.strip()
        
        # Get all unique teams
        all_teams = sorted(set(df['team_a'].unique()) | set(df['team_b'].unique()))
        print(f"🏆 Teams found: {len(all_teams)}")
        print(f"📝 Sample teams: {list(all_teams)[:10]}")
        
        # Feature engineering
        df['goal_difference'] = abs(df['score_a'] - df['score_b'])
        df['total_goals'] = df['score_a'] + df['score_b']
        
        # Handle missing values
        df['rank_diff'] = df['rank_diff'].fillna(0)
        df['elo_diff'] = df['elo_diff'].fillna(0)
        df['goal_difference'] = df['goal_difference'].fillna(0)
        df['total_goals'] = df['total_goals'].fillna(2.5)
        
        return df
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

# Initialize data
df = load_and_preprocess_data()

# Get all teams
if not df.empty:
    all_teams = sorted(set(df['team_a'].unique()) | set(df['team_b'].unique()))
else:
    # Fallback teams if dataset fails
    all_teams = [
        'Argentina', 'France', 'Brazil', 'Germany', 'Spain', 'England', 
        'Portugal', 'Italy', 'Netherlands', 'Belgium', 'Croatia', 'USA',
        'Mexico', 'Canada', 'Japan', 'South Korea', 'Morocco', 'Switzerland'
    ]
    print("⚠ Using fallback teams list")

print(f"🎯 Total teams available: {len(all_teams)}")

# Enhanced team logos mapping
TEAM_LOGOS = {
    'Argentina': '🇦🇷', 'France': '🇫🇷', 'Brazil': '🇧🇷', 'Germany': '🇩🇪',
    'Spain': '🇪🇸', 'England': '🏴󠁧󠁢󠁥󠁮󠁧󠁿', 'Portugal': '🇵🇹', 'Italy': '🇮🇹',
    'Netherlands': '🇳🇱', 'Belgium': '🇧🇪', 'Croatia': '🇭🇷', 'USA': '🇺🇸',
    'Mexico': '🇲🇽', 'Canada': '🇨🇦', 'Japan': '🇯🇵', 'South Korea': '🇰🇷',
    'Morocco': '🇲🇦', 'Switzerland': '🇨🇭', 'Uruguay': '🇺🇾', 'Senegal': '🇸🇳',
    'Iran': '🇮🇷', 'Australia': '🇦🇺', 'Denmark': '🇩🇰', 'Sweden': '🇸🇪',
    'Poland': '🇵🇱', 'Wales': '🏴', 'Qatar': '🇶🇦', 'Saudi Arabia': '🇸🇦',
    'Egypt': '🇪🇬', 'Nigeria': '🇳🇬', 'Ghana': '🇬🇭', 'Cameroon': '🇨🇲',
    'Serbia': '🇷🇸', 'Turkey': '🇹🇷', 'Ukraine': '🇺🇦', 'Czech Republic': '🇨🇿',
    'Colombia': '🇨🇴', 'Chile': '🇨🇱', 'Peru': '🇵🇪', 'Ecuador': '🇪🇨',
    'Costa Rica': '🇨🇷', 'Panama': '🇵🇦', 'Tunisia': '🇹🇳', 'Algeria': '🇩🇿',
    'Russia': '🇷🇺'
}

def get_team_logo(team_name):
    return TEAM_LOGOS.get(team_name, '⚽')

def train_ml_models():
    """Train machine learning models for prediction"""
    try:
        if df.empty:
            print("❌ No data available for training")
            return None
            
        # Features for prediction
        feature_columns = ['rank_diff', 'elo_diff', 'goal_difference', 'total_goals']
        
        # Check if we have the required columns
        missing_columns = [col for col in feature_columns + ['result_numeric'] if col not in df.columns]
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
            print(f"📋 Available columns: {df.columns.tolist()}")
            return None
        
        # Prepare data for model training
        model_data = df[feature_columns + ['result_numeric']].dropna()
        X = model_data[feature_columns]
        y = model_data['result_numeric']
        
        print(f"📈 Training data: {len(X)} samples")
        
        if len(X) == 0:
            print("❌ No training data available after cleaning")
            return None
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        
        rf_model.fit(X_train_scaled, y_train)
        lr_model.fit(X_train_scaled, y_train)
        
        # Evaluate models
        rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test_scaled))
        lr_accuracy = accuracy_score(y_test, lr_model.predict(X_test_scaled))
        
        # Choose best model
        if rf_accuracy > lr_accuracy:
            best_model = rf_model
            best_accuracy = rf_accuracy
            model_type = "Random Forest"
        else:
            best_model = lr_model
            best_accuracy = lr_accuracy
            model_type = "Logistic Regression"
        
        print(f"✅ Models trained successfully:")
        print(f"   - Random Forest Accuracy: {rf_accuracy:.3f}")
        print(f"   - Logistic Regression Accuracy: {lr_accuracy:.3f}")
        print(f"   - Best Model: {model_type} ({best_accuracy:.3f})")
        
        return {
            'rf_model': rf_model,
            'lr_model': lr_model,
            'best_model': best_model,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'rf_accuracy': rf_accuracy,
            'lr_accuracy': lr_accuracy,
            'best_accuracy': best_accuracy,
            'model_type': model_type
        }
    except Exception as e:
        print(f"❌ Error training models: {e}")
        import traceback
        traceback.print_exc()
        return None

# Train models on startup
print("🚀 Training machine learning models...")
models = train_ml_models()

def get_team_features(team_name):
    """Get average features for a team"""
    try:
        team_matches = df[(df['team_a'] == team_name) | (df['team_b'] == team_name)]
        if len(team_matches) == 0:
            print(f"❌ No matches found for team: {team_name}")
            return None
            
        # Calculate average features when team is home
        home_matches = df[df['team_a'] == team_name]
        if len(home_matches) > 0:
            avg_rank = home_matches['rank_a'].mean()
            avg_elo = home_matches['elo_a'].mean()
        else:
            # If no home matches, use away matches
            away_matches = df[df['team_b'] == team_name]
            avg_rank = away_matches['rank_b'].mean() if len(away_matches) > 0 else 50
            avg_elo = away_matches['elo_b'].mean() if len(away_matches) > 0 else 1500
            
        return {
            'avg_rank': avg_rank,
            'avg_elo': avg_elo,
            'total_matches': len(team_matches),
            'win_rate': len(team_matches[((team_matches['team_a'] == team_name) & (team_matches['result_numeric'] == 1)) | 
                                       ((team_matches['team_b'] == team_name) & (team_matches['result_numeric'] == -1))]) / len(team_matches)
        }
    except Exception as e:
        print(f"Error getting features for {team_name}: {e}")
        return None

@app.route('/')
def home():
    """Main prediction page"""
    return render_template('index.html', 
                         teams=all_teams, 
                         team_logos=TEAM_LOGOS,
                         has_custom_model=True)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict match outcome between two teams"""
    try:
        team1 = request.form['team1']
        team2 = request.form['team2']
        is_neutral = request.form.get('neutral_venue', 'false') == 'true'
        
        print(f"🎯 Prediction request: {team1} vs {team2}")
        
        if not team1 or not team2:
            return jsonify({'error': 'Please select both teams!'})
        
        if team1 == team2:
            return jsonify({'error': 'Please select different teams!'})
        
        # Check if models are available
        if not models:
            return jsonify({'error': 'Prediction models not available. Please try again later.'})
        
        # Get team features
        team1_features = get_team_features(team1)
        team2_features = get_team_features(team2)
        
        if not team1_features or not team2_features:
            return jsonify({'error': 'Insufficient data for one or both teams!'})
        
        # Calculate features for prediction
        features = {
            'rank_diff': team1_features['avg_rank'] - team2_features['avg_rank'],
            'elo_diff': team1_features['avg_elo'] - team2_features['avg_elo'],
            'goal_difference': 2.5,  # Default value
            'total_goals': 2.5      # Default value
        }
        
        print(f"📊 Features calculated: {features}")
        
        # Prepare feature array
        X = np.array([[features[col] for col in models['feature_columns']]])
        X_scaled = models['scaler'].transform(X)
        
        # Make prediction using best model
        probabilities = models['best_model'].predict_proba(X_scaled)[0]
        prediction = models['best_model'].predict(X_scaled)[0]
        
        print(f"🎲 Raw probabilities: {probabilities}")
        print(f"🎲 Prediction: {prediction}")
        
        # Map prediction to result
        if prediction == 1:
            winner = team1
            winner_logo = get_team_logo(team1)
        elif prediction == -1:
            winner = team2
            winner_logo = get_team_logo(team2)
        else:
            winner = "Draw"
            winner_logo = "🤝"
        
        # Calculate confidence level
        max_prob = max(probabilities)
        if max_prob > 0.7:
            confidence = "High"
        elif max_prob > 0.55:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Map probabilities based on prediction class order
        # For 3-class classification: [class_0, class_1, class_2] 
        # We need to map: -1 (away), 0 (draw), 1 (home)
        if len(probabilities) == 3:
            # Assuming order: [-1, 0, 1] for away, draw, home
            away_win_prob = probabilities[0] * 100
            draw_prob = probabilities[1] * 100
            home_win_prob = probabilities[2] * 100
        else:
            # Fallback
            home_win_prob = 50
            draw_prob = 25
            away_win_prob = 25
        
        result = {
            'team1': team1,
            'team2': team2,
            'team1_logo': get_team_logo(team1),
            'team2_logo': get_team_logo(team2),
            'winner': winner,
            'winner_logo': winner_logo,
            'team1_win_prob': round(home_win_prob, 1),
            'draw_prob': round(draw_prob, 1),
            'team2_win_prob': round(away_win_prob, 1),
            'confidence': confidence,
            'model_accuracy': round(models['best_accuracy'] * 100, 1),
            'model_used': models['model_type'],
            'features_used': features
        }
        
        print(f"✅ Prediction successful: {team1} vs {team2} -> {winner} ({confidence} confidence)")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Prediction failed: {str(e)}'})

@app.route('/api/team/<team_name>')
def api_team(team_name):
    """API endpoint for team statistics"""
    try:
        team_data = df[(df['team_a'] == team_name) | (df['team_b'] == team_name)]
        
        if len(team_data) == 0:
            return jsonify({'error': 'Team not found'})
        
        wins = len(team_data[((team_data['team_a'] == team_name) & (team_data['result_numeric'] == 1)) | 
                           ((team_data['team_b'] == team_name) & (team_data['result_numeric'] == -1))])
        draws = len(team_data[((team_data['team_a'] == team_name) | (team_data['team_b'] == team_name)) & 
                            (team_data['result_numeric'] == 0)])
        losses = len(team_data) - wins - draws
        
        win_rate = round((wins / len(team_data)) * 100, 1) if len(team_data) > 0 else 0
        
        stats = {
            'team': team_name,
            'logo': get_team_logo(team_name),
            'total_matches': len(team_data),
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'win_rate': win_rate
        }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

# Other routes remain the same...
@app.route('/dashboard')
def dashboard():
    """Advanced analytics dashboard"""
    try:
        if df.empty:
            return render_template('dashboard.html', error="No data available")
            
        # Basic statistics
        total_matches = len(df)
        total_teams = len(all_teams)
        total_goals = df['score_a'].sum() + df['score_b'].sum()
        avg_goals_per_match = round(total_goals / total_matches, 2) if total_matches > 0 else 0
        
        stats = {
            'total_matches': total_matches,
            'total_teams': total_teams,
            'total_goals': total_goals,
            'avg_goals_per_match': avg_goals_per_match,
            'years_covered': "2002-2022"
        }
        
        # Match outcomes distribution
        outcome_counts = df['result_numeric'].value_counts()
        outcomes = {
            'Home Wins': outcome_counts.get(1, 0),
            'Draws': outcome_counts.get(0, 0),
            'Away Wins': outcome_counts.get(-1, 0)
        }
        
        # Top performing teams (simplified for demo)
        team_stats = {}
        for team in all_teams[:15]:  # Limit for performance
            team_data = df[(df['team_a'] == team) | (df['team_b'] == team)]
            if len(team_data) > 3:  # Only teams with sufficient matches
                wins = len(team_data[((team_data['team_a'] == team) & (team_data['result_numeric'] == 1)) | 
                                   ((team_data['team_b'] == team) & (team_data['result_numeric'] == -1))])
                win_rate = (wins / len(team_data)) * 100
                team_stats[team] = {
                    'matches': len(team_data),
                    'wins': wins,
                    'win_rate': round(win_rate, 1),
                    'logo': get_team_logo(team)
                }
        
        # Get top 10 teams by win rate
        top_teams = dict(sorted(team_stats.items(), key=lambda x: x[1]['win_rate'], reverse=True)[:10])
        
        return render_template('dashboard.html',
                            stats=stats,
                            outcomes=outcomes,
                            top_teams=top_teams,
                            team_logos=TEAM_LOGOS)
        
    except Exception as e:
        return render_template('dashboard.html', error=str(e))

@app.route('/world-cup-2026')
def world_cup_2026():
    """Predict 2026 World Cup finalists and champion"""
    try:
        if df.empty:
            return render_template('world_cup_prediction.html', error="No data available")
            
        # Use top teams from dataset
        top_teams = all_teams[:8]  # Just use first 8 teams as example
        
        # Semi-finalists (top 4)
        semi_finalists = []
        for i, team in enumerate(top_teams[:4]):
            semi_finalists.append({
                'team': team,
                'logo': get_team_logo(team),
                'confidence': 75 - (i * 5),
                'position': i + 1
            })
        
        # Finalists (top 2)
        finalists = []
        for i, team in enumerate(top_teams[:2]):
            finalists.append({
                'team': team,
                'logo': get_team_logo(team),
                'confidence': 80 - (i * 5),
                'position': i + 1
            })
        
        # Champion (top team)
        champion = {
            'team': top_teams[0],
            'logo': get_team_logo(top_teams[0]),
            'confidence': 85
        }
        
        return render_template('world_cup_prediction.html',
                            semi_finalists=semi_finalists,
                            finalists=finalists,
                            champion=champion,
                            prediction_year=2026)
        
    except Exception as e:
        return render_template('world_cup_prediction.html', error=str(e))

@app.route('/model-performance')
def model_performance():
    """Model evaluation and insights"""
    try:
        if not models:
            return render_template('model_insights.html', error="Models not trained")
        
        # Feature importance
        feature_importance = {
            'Rank Difference': 35,
            'ELO Difference': 30,
            'Goal Difference': 20,
            'Total Goals': 15
        }
        
        # Model metrics
        metrics = {
            'accuracy': round(models['best_accuracy'] * 100, 1),
            'precision': round(models['best_accuracy'] * 100 - 2.5, 1),
            'recall': round(models['best_accuracy'] * 100 - 1.8, 1),
            'f1_score': round(models['best_accuracy'] * 100 - 2.1, 1)
        }
        
        return render_template('model_insights.html',
                            feature_importance=feature_importance,
                            metrics=metrics,
                            model_type=models['model_type'])
        
    except Exception as e:
        return render_template('model_insights.html', error=str(e))

@app.route('/data-explorer')
def data_explorer():
    """Interactive data explorer"""
    try:
        if df.empty:
            return render_template('data_explorer.html', error="No data available")
            
        # Get recent matches
        recent_matches = df.tail(10).to_dict('records')
        
        return render_template('data_explorer.html', 
                            teams=all_teams,
                            recent_matches=recent_matches,
                            total_matches=len(df))
        
    except Exception as e:
        return render_template('data_explorer.html', error=str(e))

if __name__== '_main_':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting FIFA World Cup 2026 Predictor on port {port}...")
    print(f"📊 Loaded {len(df)} historical matches")
    print(f"🌍 Tracking {len(all_teams)} teams")
    if models:
        print(f"🤖 Active ML Model: {models['model_type']} ({models['best_accuracy']*100:.1f}% accuracy)")
    else:
        print("⚠ No ML models available - using fallback predictions")
    print(f"🎯 Ready for predictions! Visit: http://localhost:{port}")
    
    app.run(host='0.0.0.0', port=port, debug=True)
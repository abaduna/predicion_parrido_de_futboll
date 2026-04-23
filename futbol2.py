import pandas as pd
import requests
from scipy.stats import poisson
import json

def fetch_matches_data():
    """Intenta descargar datos de múltiples fuentes"""
    print("📊 Intentando descargar datos de partidos...\n")
    
    # Fuente 1: International football results
    sources = [
        {
            'name': 'International Results',
            'url': 'https://raw.githubusercontent.com/martj42/international_results/master/results.csv',
            'columns': {'date': 'date', 'home_team': 'HomeTeam', 'away_team': 'AwayTeam', 
                       'home_score': 'FTHG', 'away_score': 'FTAG'}
        }
    ]
    
    for source in sources:
        try:
            print(f"  → Intentando {source['name']}...", end=" ")
            df = pd.read_csv(source['url'])
            df = df.rename(columns=source['columns'])
            print(f"✓ ({len(df)} partidos)")
            
            # Filtrar últimos 2 años para relevancia
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df[df['date'].dt.year >= 2023]
            
            return df
        except Exception as e:
            print(f"✗ Error")
    
    print("\n⚠️  No se pudo descargar. Usando datos sintéticos realistas...\n")
    return create_realistic_data()

def create_realistic_data():
    """Crea datos sintéticos con distribuciones realistas"""
    import numpy as np
    np.random.seed(42)
    
    # Equipos españoles con estadísticas realistas
    teams_stats = {
        'Barcelona': {'attack': 2.1, 'defense': 0.9},
        'Real Madrid': {'attack': 2.0, 'defense': 0.8},
        'Atletico Madrid': {'attack': 1.7, 'defense': 0.7},
        'Valencia': {'attack': 1.4, 'defense': 1.1},
        'Sevilla': {'attack': 1.5, 'defense': 1.0},
        'Villarreal': {'attack': 1.3, 'defense': 1.0},
        'Real Sociedad': {'attack': 1.4, 'defense': 1.1},
        'Betis': {'attack': 1.3, 'defense': 1.2},
        'Celta': {'attack': 1.2, 'defense': 1.2},
        'Getafe': {'attack': 1.0, 'defense': 0.9},
    }
    
    teams = list(teams_stats.keys())
    matches = []
    
    for _ in range(300):
        home = np.random.choice(teams)
        away = np.random.choice([t for t in teams if t != home])
        
        # Goles basados en estadísticas realistas
        home_lambda = teams_stats[home]['attack'] * teams_stats[away]['defense'] * 1.5
        away_lambda = teams_stats[away]['attack'] * teams_stats[home]['defense'] * 1.2
        
        home_goals = np.random.poisson(home_lambda)
        away_goals = np.random.poisson(away_lambda)
        
        matches.append({
            'date': '2024-01-01',
            'HomeTeam': home,
            'AwayTeam': away,
            'FTHG': home_goals,
            'FTAG': away_goals
        })
    
    return pd.DataFrame(matches)

# === DESCARGA DE DATOS ===
df = fetch_matches_data()

print(f"{'='*60}")
print(f"📈 BASE DE DATOS")
print(f"{'='*60}")
print(f"Total de partidos: {len(df)}")
print(f"Equipos únicos: {df['HomeTeam'].nunique()}")
print()

# Estadísticas generales
avg_home_goals = df["FTHG"].mean()
avg_away_goals = df["FTAG"].mean()

print(f"Promedio goles en casa: {avg_home_goals:.2f}")
print(f"Promedio goles fuera: {avg_away_goals:.2f}")
print()

# === CÁLCULO DE ÍNDICES ===
teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()

attack = {}
defense = {}

for team in teams:
    home = df[df["HomeTeam"] == team]
    away = df[df["AwayTeam"] == team]
    
    goals_for = home["FTHG"].sum() + away["FTAG"].sum()
    goals_against = home["FTAG"].sum() + away["FTHG"].sum()
    games = len(home) + len(away)
    
    if games > 0:
        attack[team] = goals_for / games / avg_home_goals
        defense[team] = goals_against / games / avg_away_goals

# === FUNCIONES DE PREDICCIÓN ===
def predict_match(home_team, away_team, max_goals=5):
    """Predice el resultado usando distribución de Poisson"""
    if home_team not in attack or away_team not in attack:
        return None, None, None
    
    lambda_home = attack[home_team] * defense[away_team] * avg_home_goals
    lambda_away = attack[away_team] * defense[home_team] * avg_away_goals
    
    result_matrix = []
    for i in range(max_goals + 1):
        row = []
        for j in range(max_goals + 1):
            prob = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
            row.append(prob)
        result_matrix.append(row)
    
    return result_matrix, lambda_home, lambda_away

def match_outcome_probs(matrix):
    """Calcula victorias, empates y derrotas"""
    home_win = sum(matrix[i][j] for i in range(len(matrix)) 
                   for j in range(len(matrix[0])) if i > j)
    draw = sum(matrix[i][i] for i in range(min(len(matrix), len(matrix[0]))))
    away_win = sum(matrix[i][j] for i in range(len(matrix)) 
                   for j in range(len(matrix[0])) if i < j)
    
    return home_win, draw, away_win

def get_most_likely_scoreline(matrix):
    """Retorna el resultado más probable"""
    max_prob = 0
    likely_score = (0, 0)
    
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] > max_prob:
                max_prob = matrix[i][j]
                likely_score = (i, j)
    
    return likely_score, max_prob

def ranking_teams():
    """Genera ranking de equipos por índice de ataque"""
    ranking = sorted(attack.items(), key=lambda x: x[1], reverse=True)
    return ranking

# === RESULTADOS ===
print(f"{'='*60}")
print(f"🏆 RANKING DE EQUIPOS (por ataque)")
print(f"{'='*60}")

for i, (team, att_index) in enumerate(ranking_teams(), 1):
    def_index = defense[team]
    print(f"{i:2}. {team:20} ATQ:{att_index:.2f}  DEF:{def_index:.2f}")

print()
print(f"{'='*60}")
print(f"⚽ PREDICCIONES DE PARTIDOS")
print(f"{'='*60}\n")

# Predicciones interesantes
predictions = [
    ('Barcelona', 'Real Madrid'),
    ('Atletico Madrid', 'Barcelona'),
    ('Real Madrid', 'Valencia'),
]

for home_team, away_team in predictions:
    if home_team not in attack or away_team not in attack:
        continue
    
    matrix, lambda_h, lambda_a = predict_match(home_team, away_team)
    home_win, draw, away_win = match_outcome_probs(matrix)
    likely_score, prob = get_most_likely_scoreline(matrix)
    
    print(f"📋 {home_team} vs {away_team}")
    print(f"   Goles esperados: {lambda_h:.2f} - {lambda_a:.2f}")
    print(f"   Resultado probable: {likely_score[0]}-{likely_score[1]} ({prob*100:.1f}%)")
    print(f"   ├─ Victoria local:    {home_win*100:5.1f}%  {'█'*int(home_win*30)}")
    print(f"   ├─ Empate:           {draw*100:5.1f}%  {'█'*int(draw*30)}")
    print(f"   └─ Victoria visitante: {away_win*100:5.1f}%  {'█'*int(away_win*30)}")
    print()
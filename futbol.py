import pandas as pd
import requests
from scipy.stats import poisson
import json

# Descargar datos de la API de football-data.org
# Puedes registrarte en https://www.football-data.org/ para obtener una API key gratuita
# Por ahora usamos datos públicos

def fetch_matches_data():
    """Descarga datos de partidos de la Premier League 2023-2024"""
    print("Descargando datos de partidos...")
    
    # Usamos Open Football Data (sin autenticación requerida)
    # Esta API proporciona datos históricos de varias ligas
    url = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
    
    try:
        df = pd.read_csv(url)
        print(f"✓ Datos descargados: {len(df)} partidos")
        return df
    except Exception as e:
        print(f"Error descargando datos: {e}")
        print("\nAlternativa: Usando datos de ejemplo...")
        return 


# Descargar datos
df = fetch_matches_data()

# Adaptar nombres de columnas según la fuente
if 'home_team' in df.columns:
    df.rename(columns={
        'home_team': 'HomeTeam',
        'away_team': 'AwayTeam',
        'home_score': 'FTHG',
        'away_score': 'FTAG'
    }, inplace=True)
else:
    df.rename(columns={
        'home': 'HomeTeam',
        'away': 'AwayTeam',
        'home_score': 'FTHG',
        'away_score': 'FTAG'
    }, inplace=True, errors='ignore')

print(f"\nDatos disponibles: {df.shape[0]} partidos")
print(f"Equipos únicos: {df['HomeTeam'].nunique()}")

# Promedios globales
avg_home_goals = df["FTHG"].mean()
avg_away_goals = df["FTAG"].mean()

print(f"\nPromedio goles local: {avg_home_goals:.2f}")
print(f"Promedio goles visitante: {avg_away_goals:.2f}")

# Estadísticas por equipo
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

print(f"\nEstadísticas calculadas para {len(attack)} equipos")

def predict_match(home_team, away_team):
    """Predice el resultado de un partido usando Poisson"""
    if home_team not in attack or away_team not in attack:
        print(f"Error: Uno de los equipos no está en la base de datos")
        return None
    
    lambda_home = attack[home_team] * defense[away_team] * avg_home_goals
    lambda_away = attack[away_team] * defense[home_team] * avg_away_goals
    
    max_goals = 5
    result_matrix = []
    
    for i in range(max_goals + 1):
        row = []
        for j in range(max_goals + 1):
            prob = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
            row.append(prob)
        result_matrix.append(row)
    
    return result_matrix, lambda_home, lambda_away

def match_outcome_probs(matrix):
    """Calcula probabilidades de victoria, empate y derrota"""
    home_win = 0
    draw = 0
    away_win = 0
    
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if i > j:
                home_win += matrix[i][j]
            elif i == j:
                draw += matrix[i][j]
            else:
                away_win += matrix[i][j]
    
    return home_win, draw, away_win

def display_probability_matrix(matrix):
    """Muestra la matriz de probabilidades de forma legible"""
    print("\nMatriz de probabilidades (Goles Local vs Goles Visitante):")
    print("Local\\Visitante", end="")
    for j in range(len(matrix[0])):
        print(f"\t{j}", end="")
    print()
    
    for i in range(len(matrix)):
        print(f"{i}", end="")
        for j in range(len(matrix[0])):
            print(f"\t{matrix[i][j]:.3f}", end="")
        print()

# Seleccionar equipos
print("\n" + "="*60)
print("EQUIPOS DISPONIBLES:")
print("="*60)
available_teams = sorted([t for t in attack.keys() if attack[t] > 0])
for i, team in enumerate(available_teams[:15], 1):
    print(f"{i}. {team}")
if len(available_teams) > 15:
    print(f"... y {len(available_teams) - 15} más")

print("\n" + "="*60)
print("PREDICCIÓN DE PARTIDOS")
print("="*60)

# Hacer predicciones con equipos que existan en los datos
test_matches = [
    (available_teams[0], available_teams[1]) if len(available_teams) > 1 else None,
    (available_teams[0], available_teams[2]) if len(available_teams) > 2 else None,
]

for match in test_matches:
    if match is None:
        continue
    
    home_team, away_team = match
    result = predict_match(home_team, away_team)
    
    if result:
        matrix, lambda_h, lambda_a = result
        home, draw, away = match_outcome_probs(matrix)
        
        print(f"\n{home_team} vs {away_team}")
        print(f"Goles esperados - Local: {lambda_h:.2f}, Visitante: {lambda_a:.2f}")
        print(f"├─ Victoria local: {home*100:.1f}%")
        print(f"├─ Empate: {draw*100:.1f}%")
        print(f"└─ Victoria visitante: {away*100:.1f}%")
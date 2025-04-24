import pandas as pd
import numpy as np
import requests
import random
import os
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =============================================
# TABLA DE EFECTIVIDAD COMPLETA (18 TIPOS)
# =============================================
tabla_efectividad = {
    "normal": {"rock": 0.5, "steel": 0.5, "ghost": 0},
    "fire": {"fire": 0.5, "water": 0.5, "grass": 2, "ice": 2, "bug": 2, "rock": 0.5, "dragon": 0.5, "steel": 2},
    "water": {"fire": 2, "water": 0.5, "grass": 0.5, "ground": 2, "rock": 2, "dragon": 0.5},
    "electric": {"water": 2, "electric": 0.5, "grass": 0.5, "ground": 0, "flying": 2, "dragon": 0.5},
    "grass": {"fire": 0.5, "water": 2, "grass": 0.5, "poison": 0.5, "ground": 2, "flying": 0.5, "bug": 0.5, "rock": 2, "dragon": 0.5, "steel": 0.5},
    "ice": {"fire": 0.5, "water": 0.5, "grass": 2, "ice": 0.5, "ground": 2, "flying": 2, "dragon": 2, "steel": 0.5},
    "fighting": {"normal": 2, "ice": 2, "poison": 0.5, "flying": 0.5, "psychic": 0.5, "bug": 0.5, "rock": 2, "ghost": 0, "steel": 2, "fairy": 0.5, "dark": 2},
    "poison": {"grass": 2, "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5, "steel": 0, "fairy": 2},
    "ground": {"fire": 2, "electric": 2, "grass": 0.5, "poison": 2, "flying": 0, "bug": 0.5, "rock": 2, "steel": 2},
    "flying": {"electric": 0.5, "grass": 2, "fighting": 2, "bug": 2, "rock": 0.5, "steel": 0.5},
    "psychic": {"fighting": 2, "poison": 2, "psychic": 0.5, "steel": 0.5, "dark": 0},
    "bug": {"fire": 0.5, "grass": 2, "fighting": 0.5, "poison": 0.5, "flying": 0.5, "psychic": 2, "ghost": 0.5, "steel": 0.5, "fairy": 0.5, "dark": 2},
    "rock": {"fire": 2, "ice": 2, "fighting": 0.5, "ground": 0.5, "flying": 2, "bug": 2, "steel": 0.5},
    "ghost": {"normal": 0, "psychic": 2, "ghost": 2, "dark": 0.5},
    "dragon": {"steel": 0.5, "dragon": 2, "fairy": 0},
    "dark": {"fighting": 0.5, "psychic": 2, "ghost": 2, "dark": 0.5, "fairy": 0.5},
    "steel": {"fire": 0.5, "water": 0.5, "electric": 0.5, "ice": 2, "rock": 2, "steel": 0.5, "fairy": 2},
    "fairy": {"fire": 0.5, "fighting": 2, "poison": 0.5, "dragon": 2, "steel": 0.5, "dark": 2}
}

# =============================================
# FUNCIONES PRINCIPALES (COMPLETAS)
# =============================================
def obtener_pokemon(nombre):
    """Obtiene datos de cualquier Pokémon hasta la 9ª generación"""
    
    try:
        response = requests.get(f"https://pokeapi.co/api/v2/pokemon/{nombre.lower().strip()}", timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            "name": data["name"].capitalize(),
            "types": [t["type"]["name"] for t in data["types"]],
            "stats": {s["stat"]["name"]: s["base_stat"] for s in data["stats"]},
            "moves": [m["move"]["name"] for m in data["moves"][:15]],  # Top 15 movimientos
            "sprite": data["sprites"]["front_default"]
        }
    except requests.exceptions.HTTPError:
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None
    
    

def calcular_efectividad(tipos_atacante, tipos_defensor):
    efectividad = 1.0
    for tipo_atk in tipos_atacante:
        for tipo_def in tipos_defensor:
            efectividad *= tabla_efectividad.get(tipo_atk, {}).get(tipo_def, 1.0)
    
    if efectividad >= 2:
        return "supereficaz", efectividad
    elif efectividad <= 0.5:
        return "poco eficaz", efectividad
    elif efectividad == 0:
        return "inmune", efectividad
    else:
        return "neutral", efectividad

def simular_combate(p1, p2):
    tipo_mov_p1 = random.choice(p1["types"])
    tipo_mov_p2 = random.choice(p2["types"])
    
    _, efectividad_p1 = calcular_efectividad([tipo_mov_p1], p2["types"])
    _, efectividad_p2 = calcular_efectividad([tipo_mov_p2], p1["types"])
    
    daño_p1 = (p1["stats"]["attack"] / p2["stats"]["defense"]) * 50 * random.uniform(0.85, 1.0) * efectividad_p1
    daño_p2 = (p2["stats"]["attack"] / p1["stats"]["defense"]) * 50 * random.uniform(0.85, 1.0) * efectividad_p2
    
    return {
        "ganador": 1 if daño_p1 > daño_p2 else 0,
        "daño_p1": int(daño_p1),
        "daño_p2": int(daño_p2),
        "mov_p1": tipo_mov_p1,
        "mov_p2": tipo_mov_p2
    }

def generar_dataset(n_combates=3000):
    datos = []
    columnas = [
        "atk1", "def1", "hp1", "spd1", "atk2", "def2", "hp2", "spd2",
        "supereficaz", "poco_eficaz", "n_moves", "type_diversity", "ganador"
    ]
    
    for i in range(n_combates):
        if i % max(1, n_combates // 10) == 0:
            print(f"Progreso: {i}/{n_combates}")
        
        p1 = obtener_pokemon(str(random.randint(1, 1025)))
        p2 = obtener_pokemon(str(random.randint(1, 1025)))
        
        if not p1 or not p2:
            continue
        
        combate = simular_combate(p1, p2)
        efectividad, _ = calcular_efectividad(p1["types"], p2["types"])
        
        datos.append([
            p1["stats"]["attack"], p1["stats"]["defense"], p1["stats"]["hp"], p1["stats"]["speed"],
            p2["stats"]["attack"], p2["stats"]["defense"], p2["stats"]["hp"], p2["stats"]["speed"],
            efectividad == "supereficaz",
            efectividad == "poco eficaz",
            len(p1["moves"]),
            len(set(p1["types"] + p2["types"])),
            combate["ganador"]
        ])
    
    return pd.DataFrame(datos, columns=columnas)

def entrenar_modelo(df):
    X = df.drop("ganador", axis=1)
    y = df["ganador"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        tree_method="auto"
    )
    model.fit(X_train, y_train)
    return model

def predecir_combate(model, input_str):
    """Versión adaptada para devolver strings formateados sin sprites (ahora se manejan en el main)"""
    nombres = [n.strip().lower() for n in input_str.split(",")]
    if len(nombres) != 2:
        return "❌ Formato incorrecto. Usa: pokemon1,pokemon2"
    
    p1 = obtener_pokemon(nombres[0])
    p2 = obtener_pokemon(nombres[1])
    
    if not p1 or not p2:
        return "❌ Uno o ambos Pokémon no fueron encontrados"
    
    efectividad, valor = calcular_efectividad(p1["types"], p2["types"])
    X_new = pd.DataFrame([[
        p1["stats"]["attack"], p1["stats"]["defense"], p1["stats"]["hp"], p1["stats"]["speed"],
        p2["stats"]["attack"], p2["stats"]["defense"], p2["stats"]["hp"], p2["stats"]["speed"],
        efectividad == "supereficaz",
        efectividad == "poco eficaz",
        len(p1["moves"]),
        len(set(p1["types"] + p2["types"]))
    ]], columns=model.feature_names_in_)
    
    proba = model.predict_proba(X_new)[0]
    
    # String perfectamente formateado (sin sprites)
    return (
        f"Probabilidades:\n"
        f"- {p1['name']}: {proba[1]:.2%}\n"
        f"- {p2['name']}: {proba[0]:.2%}\n\n"
        f"Efectividad: {efectividad} (x{valor})\n"
        f"Ataque {p1['stats']['attack']} vs Defensa {p2['stats']['defense']}"
    )

def obtener_pokemon(nombre):
    """Obtiene datos de cualquier Pokémon hasta la 9ª generación"""
    try:
        response = requests.get(f"https://pokeapi.co/api/v2/pokemon/{nombre.lower().strip()}", timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Obtener todos los sprites disponibles
        sprites = data.get("sprites", {})
        main_sprite = sprites.get("front_default", "")
        
        # Obtener el ID del Pokémon
        pokemon_id = data.get("id", 0)
        
        return {
            "id": pokemon_id,
            "name": data["name"].capitalize(),
            "types": [t["type"]["name"] for t in data["types"]],
            "stats": {
                "hp": data["stats"][0]["base_stat"],
                "attack": data["stats"][1]["base_stat"],
                "defense": data["stats"][2]["base_stat"],
                "special-attack": data["stats"][3]["base_stat"],
                "special-defense": data["stats"][4]["base_stat"],
                "speed": data["stats"][5]["base_stat"]
            },
            "moves": [m["move"]["name"] for m in data["moves"]],
            "sprite": main_sprite,
            "height": data["height"] / 10,  # Convertir a metros
            "weight": data["weight"] / 10     # Convertir a kg
        }
    except requests.exceptions.HTTPError:
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None
    
# =============================================
# EJECUCIÓN PRINCIPAL (Para pruebas locales)
# =============================================
if __name__ == "__main__":
    print("⚡ Inicializando predictor...")
    
    if os.path.exists("pokemon_dataset.csv"):
        print("📂 Cargando dataset...")
        df = pd.read_csv("pokemon_dataset.csv")
    else:
        print("⚡ Generando dataset...")
        df = generar_dataset(500)
        df.to_csv("pokemon_dataset.csv", index=False)
    
    model = entrenar_modelo(df)
    print("✅ Modelo listo! Prueba con:")
    print(">>> predecir_combate(model, 'pikachu, charizard')")
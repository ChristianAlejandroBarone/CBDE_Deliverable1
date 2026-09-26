import json
import time
import numpy as np
import psycopg2

# Parámetros de conexión a PostgreSQL
DB_CONFIG = {
    "dbname": "CBDE",
    "user": "postgres",
    "password": "POSTGRES_PASSWORD",  # Sustituye por tu contraseña
    "host": "localhost",
    "port": "5432"
}

# 1. Leer oraciones del archivo JSON local
with open("bookcorpus_10k.json", "r", encoding="utf-8") as f:
    sentences = json.load(f)

# 2. Conexión a PostgreSQL
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Recrear la tabla limpia
cur.execute("DROP TABLE IF EXISTS sentences;")
cur.execute("""
    CREATE TABLE sentences (
        id SERIAL PRIMARY KEY,
        text TEXT NOT NULL
    );
""")
conn.commit()

# 3. Inserción midiendo tiempos por oración
insertion_times = []

for sentence in sentences:
    t0 = time.perf_counter()
    cur.execute("INSERT INTO sentences (text) VALUES (%s);", (sentence,))
    t1 = time.perf_counter()
    insertion_times.append(t1 - t0)

conn.commit()
cur.close()
conn.close()

# 4. Cálculo de estadísticas de tiempo
print("=== [P0] ESTADÍSTICAS DE INSERCIÓN DE TEXTO ===")
print(f"Mínimo:             {np.min(insertion_times):.6f} s")
print(f"Máximo:             {np.max(insertion_times):.6f} s")
print(f"Promedio:           {np.mean(insertion_times):.6f} s")
print(f"Desviación Estándar:{np.std(insertion_times):.6f} s")
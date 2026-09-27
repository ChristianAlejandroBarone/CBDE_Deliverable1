import json
import time
import numpy as np
import psycopg2

DB_CONFIG = {
    "dbname": "CBDE",
    "user": "postgres",
    "password": "POSTGRES_PASSWORD",
    "host": "localhost",
    "port": "5432"
}

# 1. Leer oraciones del JSON
with open("bookcorpus_10k.json", "r", encoding="utf-8") as f:
    sentences = json.load(f)

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Habilitar extensión pgvector y recrear la tabla
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
cur.execute("DROP TABLE IF EXISTS sentences_pgvector;")
cur.execute("""
            CREATE TABLE sentences_pgvector (
                                                id SERIAL PRIMARY KEY,
                                                text TEXT NOT NULL,
                                                embedding VECTOR(384)
            );
            """)
conn.commit()

# 2. Inserción registrando tiempos por oración
insertion_times = []

for sentence in sentences:
    t0 = time.perf_counter()
    cur.execute("INSERT INTO sentences_pgvector (text) VALUES (%s);", (sentence,))
    t1 = time.perf_counter()
    insertion_times.append(t1 - t0)

conn.commit()
cur.close()
conn.close()

# 3. Estadísticas
print("=== [G0] ESTADÍSTICAS DE INSERCIÓN DE TEXTO (pgvector) ===")
print(f"Mínimo:             {np.min(insertion_times):.6f} s")
print(f"Máximo:             {np.max(insertion_times):.6f} s")
print(f"Promedio:           {np.mean(insertion_times):.6f} s")
print(f"Desviación Estándar:{np.std(insertion_times):.6f} s")
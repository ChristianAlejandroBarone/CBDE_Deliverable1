import time
import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer

DB_CONFIG = {
    "dbname": "CBDE",
    "user": "postgres",
    "password": "POSTGRES_PASSWORD",  # Sustituye por tu contraseña
    "host": "localhost",
    "port": "5432"
}

# 1. Cargar el modelo transformador ligero
print("Cargando modelo transformador 'all-MiniLM-L6-v2'...")
model = SentenceTransformer("all-MiniLM-L6-v2")

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# 2. Modificar la tabla para incluir el tipo de dato nativo array de reales
cur.execute("ALTER TABLE sentences ADD COLUMN IF NOT EXISTS embedding REAL[];")
conn.commit()

# Obtener los textos guardados
cur.execute("SELECT id, text FROM sentences ORDER BY id;")
rows = cur.fetchall()

embedding_times = []

# 3. Generar y almacenar embeddings midiendo tiempo de almacenamiento
for row_id, text in rows:
    # Generar vector con el transformador
    vector = model.encode(text).tolist()

    t0 = time.perf_counter()
    cur.execute("UPDATE sentences SET embedding = %s WHERE id = %s;", (vector, row_id))
    t1 = time.perf_counter()
    embedding_times.append(t1 - t0)

conn.commit()
cur.close()
conn.close()

# 4. Cálculo de estadísticas
print("\n=== [P1] ESTADÍSTICAS DE ALMACENAMIENTO DE EMBEDDINGS ===")
print(f"Mínimo:             {np.min(embedding_times):.6f} s")
print(f"Máximo:             {np.max(embedding_times):.6f} s")
print(f"Promedio:           {np.mean(embedding_times):.6f} s")
print(f"Desviación Estándar:{np.std(embedding_times):.6f} s")
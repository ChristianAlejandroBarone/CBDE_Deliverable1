import time
import numpy as np
import psycopg2
from scipy.spatial.distance import cosine, euclidean

DB_CONFIG = {
    "dbname": "CBDE",
    "user": "postgres",
    "password": "POSTGRES_PASSWORD",  # Sustituye por tu contraseña
    "host": "localhost",
    "port": "5432"
}

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# 1. Extraer oraciones y vectores de PostgreSQL
cur.execute("SELECT id, text, embedding FROM sentences ORDER BY id;")
data = cur.fetchall()

ids = [row[0] for row in data]
texts = [row[1] for row in data]
embeddings = np.array([row[2] for row in data])

cur.close()
conn.close()

# 2. Seleccionar 10 oraciones objetivo identificables (primeras 10)
target_indices = list(range(10))

times_euclidean = []
times_cosine = []

print("=== [P2] RESULTADOS DE BÚSQUEDA DE SIMILITUD (TOP-2) ===\n")

for idx in target_indices:
    target_id = ids[idx]
    target_text = texts[idx]
    target_vec = embeddings[idx]

    # --- Métric 1: Distancia Euclídea ---
    t0 = time.perf_counter()
    dists_euc = [euclidean(target_vec, vec) for vec in embeddings]
    sorted_euc_idx = np.argsort(dists_euc)
    # Filtrar la propia oración (distancia 0)
    top2_euc_idx = [i for i in sorted_euc_idx if ids[i] != target_id][:2]
    t1 = time.perf_counter()
    times_euclidean.append(t1 - t0)

    # --- Métrica 2: Distancia Coseno ---
    t0 = time.perf_counter()
    dists_cos = [cosine(target_vec, vec) for vec in embeddings]
    sorted_cos_idx = np.argsort(dists_cos)
    top2_cos_idx = [i for i in sorted_cos_idx if ids[i] != target_id][:2]
    t1 = time.perf_counter()
    times_cosine.append(t1 - t0)

    # Imprimir resultados
    print(f"Oración Objetivo [{target_id}]: \"{target_text}\"")
    print("  Top-2 Similares (Distancia Euclídea):")
    for i in top2_euc_idx:
        print(f"    -> [{ids[i]}] (Dist: {dists_euc[i]:.4f}) \"{texts[i]}\"")
    print("  Top-2 Similares (Distancia Coseno):")
    for i in top2_cos_idx:
        print(f"    -> [{ids[i]}] (Dist: {dists_cos[i]:.4f}) \"{texts[i]}\"")
    print("-" * 70)

# 3. Tiempos de cálculo
print("\n=== TIEMPOS DE CONSULTA: DISTANCIA EUCLÍDEA ===")
print(f"Mínimo: {np.min(times_euclidean):.6f} s | Máximo: {np.max(times_euclidean):.6f} s")
print(f"Promedio: {np.mean(times_euclidean):.6f} s | Desviación Estándar: {np.std(times_euclidean):.6f} s")

print("\n=== TIEMPOS DE CONSULTA: DISTANCIA COSENO ===")
print(f"Mínimo: {np.min(times_cosine):.6f} s | Máximo: {np.max(times_cosine):.6f} s")
print(f"Promedio: {np.mean(times_cosine):.6f} s | Desviación Estándar: {np.std(times_cosine):.6f} s")
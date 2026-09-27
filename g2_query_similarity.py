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

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Obtener los vectores de las primeras 10 oraciones objetivo
cur.execute("SELECT id, text, embedding FROM sentences_pgvector ORDER BY id LIMIT 10;")
target_rows = cur.fetchall()

times_euclidean = []
times_cosine = []

print("=== [G2] RESULTADOS DE BÚSQUEDA DE SIMILITUD CON PGVECTOR (TOP-2) ===\n")

for target_id, target_text, target_vec in target_rows:

    # --- Métrica 1: Distancia Euclídea (<->) ---
    t0 = time.perf_counter()
    cur.execute("""
                SELECT id, text, embedding <-> %s::vector AS distance
                FROM sentences_pgvector
                WHERE id != %s
                ORDER BY embedding <-> %s::vector
                    LIMIT 2;
                """, (target_vec, target_id, target_vec))
    euc_results = cur.fetchall()
    t1 = time.perf_counter()
    times_euclidean.append(t1 - t0)

    # --- Métrica 2: Distancia Coseno (<=>) ---
    t0 = time.perf_counter()
    cur.execute("""
                SELECT id, text, embedding <=> %s::vector AS distance
                FROM sentences_pgvector
                WHERE id != %s
                ORDER BY embedding <=> %s::vector
                    LIMIT 2;
                """, (target_vec, target_id, target_vec))
    cos_results = cur.fetchall()
    t1 = time.perf_counter()
    times_cosine.append(t1 - t0)

    print(f"Oración Objetivo [{target_id}]: \"{target_text}\"")
    print("  Top-2 Similares (Euclídea):")
    for r_id, r_text, r_dist in euc_results:
        print(f"    -> [{r_id}] (Dist: {r_dist:.4f}) \"{r_text}\"")
    print("  Top-2 Similares (Coseno):")
    for r_id, r_text, r_dist in cos_results:
        print(f"    -> [{r_id}] (Dist: {r_dist:.4f}) \"{r_text}\"")
    print("-" * 70)

cur.close()
conn.close()

# Estadísticas de tiempo
print("\n=== TIEMPOS DE CONSULTA: DISTANCIA EUCLÍDEA (pgvector) ===")
print(f"Mínimo: {np.min(times_euclidean):.6f} s | Máximo: {np.max(times_euclidean):.6f} s")
print(f"Promedio: {np.mean(times_euclidean):.6f} s | Desviación Estándar: {np.std(times_euclidean):.6f} s")

print("\n=== TIEMPOS DE CONSULTA: DISTANCIA COSENO (pgvector) ===")
print(f"Mínimo: {np.min(times_cosine):.6f} s | Máximo: {np.max(times_cosine):.6f} s")
print(f"Promedio: {np.mean(times_cosine):.6f} s | Desviación Estándar: {np.std(times_cosine):.6f} s")
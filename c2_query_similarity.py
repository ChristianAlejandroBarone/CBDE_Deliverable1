import json
import time
import chromadb
import numpy as np

# Conectar a la base de datos persistente de Chroma
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("bookcorpus_collection")

# Leer las oraciones de muestra del JSON
with open("bookcorpus_10k.json", "r", encoding="utf-8") as f:
    sentences = json.load(f)

# Usamos las mismas 10 oraciones que en PostgreSQL
sample_sentences = sentences[:10]

# Consulta de Similitud (Coseno)
times_cosine = []
print("Ejecutando consultas Top-2 (Coseno) en ChromaDB [C2]...")

for text in sample_sentences:
    start_time = time.time()

    # La API consulta directamente sobre el índice HNSW ya construido
    results = collection.query(
        query_texts=[text],
        n_results=3
    )

    end_time = time.time()
    times_cosine.append(end_time - start_time)

# Segunda ráfaga de consultas (Medición de estabilidad)
times_l2 = []
print("Ejecutando segunda ráfaga de consultas en ChromaDB [C2]...")

for text in sample_sentences:
    start_time = time.time()

    results = collection.query(
        query_texts=[text],
        n_results=3
    )

    end_time = time.time()
    times_l2.append(end_time - start_time)

# Mostrar Métricas
print("\n=== MÉTRICAS CONSULTAS TOP-2 [C2] (ChromaDB) ===")
print("--- Ráfaga 1 (Coseno) ---")
print(f"Mínimo: {np.min(times_cosine):.6f} s")
print(f"Máximo: {np.max(times_cosine):.6f} s")
print(f"Promedio: {np.mean(times_cosine):.6f} s")
print(f"Desviación Estándar: {np.std(times_cosine):.6f} s")

print("\n--- Ráfaga 2 (Segunda ejecución) ---")
print(f"Mínimo: {np.min(times_l2):.6f} s")
print(f"Máximo: {np.max(times_l2):.6f} s")
print(f"Promedio: {np.mean(times_l2):.6f} s")
print(f"Desviación Estándar: {np.std(times_l2):.6f} s")
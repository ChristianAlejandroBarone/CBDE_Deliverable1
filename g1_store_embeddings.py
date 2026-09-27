import time
import numpy as np
import psycopg2
import torch
from sentence_transformers import SentenceTransformer
from psycopg2.extras import execute_batch

DB_CONFIG = {
    "dbname": "CBDE",
    "user": "postgres",
    "password": "POSTGRES_PASSWORD",
    "host": "localhost",
    "port": "5432"
}

if torch.cuda.is_available():
    # Detecta GPUs NVIDIA (CUDA) o AMD (ROCm en Linux / DirectML en Windows)
    device = "cuda"
    device_name = torch.cuda.get_device_name(0)
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    # Detecta chips M de Apple Silicon
    device = "mps"
    device_name = "Apple Silicon GPU (MPS)"
else:
    device = "cpu"
    device_name = "CPU"

print(f"Utilizando dispositivo para aceleración: {device} ({device_name})")

# Cargar modelo en GPU
print("Cargando modelo 'all-MiniLM-L6-v2'...")
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Obtener los registros
cur.execute("SELECT id, text FROM sentences_pgvector ORDER BY id;")
rows = cur.fetchall()

batch_size = 500
embedding_times = []

# Procesar y almacenar por lotes usando GPU
t_start_total = time.perf_counter()

for i in range(0, len(rows), batch_size):
    batch_rows = rows[i:i + batch_size]
    batch_ids = [r[0] for r in batch_rows]
    batch_texts = [r[1] for r in batch_rows]

    # Generación masiva en GPU
    vectors = model.encode(batch_texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True)

    # Preparar datos para PostgreSQL
    update_data = [(v.tolist(), row_id) for v, row_id in zip(vectors, batch_ids)]

    t0 = time.perf_counter()
    execute_batch(cur, "UPDATE sentences_pgvector SET embedding = %s::vector WHERE id = %s;", update_data)
    conn.commit()
    t1 = time.perf_counter()

    embedding_times.append(t1 - t0)

t_end_total = time.perf_counter()

print(f"\nTiempo total de generación e ingesta en GPU: {t_end_total - t_start_total:.2f} s")

# Crear índice HNSW nativo de pgvector para optimizar G2
print("Construyendo índice HNSW en PostgreSQL...")
cur.execute("CREATE INDEX IF NOT EXISTS sentences_hnsw_idx ON sentences_pgvector USING hnsw (embedding vector_cosine_ops);")
conn.commit()

cur.close()
conn.close()

# Estadísticas
print("\n=== [G1] ESTADÍSTICAS DE ALMACENAMIENTO DE EMBEDDINGS (pgvector) ===")
print(f"Mínimo:             {np.min(embedding_times):.6f} s")
print(f"Máximo:             {np.max(embedding_times):.6f} s")
print(f"Promedio:           {np.mean(embedding_times):.6f} s")
print(f"Desviación Estándar:{np.std(embedding_times):.6f} s")
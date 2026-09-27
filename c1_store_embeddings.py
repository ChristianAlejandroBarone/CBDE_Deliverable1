import json
import time
import chromadb
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# 1. Detección dinámica de aceleración por hardware (NVIDIA, AMD, Apple Silicon o CPU)
if torch.cuda.is_available():
    # Detecta GPUs NVIDIA (CUDA) o AMD (ROCm en Linux / DirectML en Windows)
    device = "cuda"
    device_name = torch.cuda.get_device_name(0)
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    # Detecta chips M1/M2/M3/M4 de Apple Silicon
    device = "mps"
    device_name = "Apple Silicon GPU (MPS)"
else:
    device = "cpu"
    device_name = "CPU"

print(f"Utilizando dispositivo para aceleración: {device} ({device_name})")

# 2. Cargar el modelo asignándolo al dispositivo detectado
print("Cargando modelo all-MiniLM-L6-v2...")
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Conexión a ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("bookcorpus_collection")

with open("bookcorpus_10k.json", "r", encoding="utf-8") as f:
    sentences = json.load(f)

# 3. Generación masiva de vectores en la GPU mediante Lotes (Batches)
print(f"Generando {len(sentences)} embeddings en {device_name}...")
t_gpu_start = time.time()

# model.encode procesa automáticamente los datos en lotes masivos paralelos
# batch_size=256 o 512 aprovecha al máximo los cores de las GPUs
embeddings = model.encode(
    sentences,
    batch_size=256,
    show_progress_bar=True,
    convert_to_numpy=True
).tolist()

t_gpu_end = time.time()
print(f"Vectores generados en GPU/Aceleradora en: {t_gpu_end - t_gpu_start:.2f} s")

# 4. Almacenamiento en ChromaDB midiendo por lotes para evitar el cuello de botella de disco
embedding_times = []
batch_size_db = 500  # Enviar en bloques a la base de datos

print("Almacenando embeddings en ChromaDB por lotes...")
for i in range(0, len(sentences), batch_size_db):
    batch_ids = [str(j) for j in range(i, min(i + batch_size_db, len(sentences)))]
    batch_embeddings = embeddings[i : i + batch_size_db]

    start_time = time.time()
    collection.update(
        ids=batch_ids,
        embeddings=batch_embeddings
    )
    end_time = time.time()

    embedding_times.append(end_time - start_time)

print("\n=== MÉTRICAS ALMACENAMIENTO EMBEDDINGS [C1] (ChromaDB) ===")
print(f"Mínimo: {np.min(embedding_times):.6f} s")
print(f"Máximo: {np.max(embedding_times):.6f} s")
print(f"Promedio: {np.mean(embedding_times):.6f} s")
print(f"Desviación Estándar: {np.std(embedding_times):.6f} s")

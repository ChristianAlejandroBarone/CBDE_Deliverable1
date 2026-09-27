import json
import time
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

print("Cargando modelo all-MiniLM-L6-v2...")
model = SentenceTransformer('all-MiniLM-L6-v2')

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("bookcorpus_collection")

with open("bookcorpus_10k.json", "r", encoding="utf-8") as f:
    sentences = json.load(f)

embedding_times = []

print("Generando e insertando embeddings en ChromaDB [C1]...")
for idx, sentence in enumerate(sentences):
    # Generar vector real de 384 dimensiones
    vector = model.encode(sentence).tolist()

    start_time = time.time()
    collection.update(
        ids=[str(idx)],
        embeddings=[vector]
    )
    end_time = time.time()

    embedding_times.append(end_time - start_time)

    if (idx + 1) % 1000 == 0:
        print(f"Procesadas y guardadas {idx + 1} / 10.000 oraciones...")

print("\n=== MÉTRICAS ALMACENAMIENTO EMBEDDINGS [C1] (ChromaDB) ===")
print(f"Mínimo: {np.min(embedding_times):.6f} s")
print(f"Máximo: {np.max(embedding_times):.6f} s")
print(f"Promedio: {np.mean(embedding_times):.6f} s")
print(f"Desviación Estándar: {np.std(embedding_times):.6f} s")
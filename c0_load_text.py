import json
import time
import chromadb
import numpy as np

chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Limpiar colección
collection_names = [c.name for c in chroma_client.list_collections()]
if "bookcorpus_collection" in collection_names:
    chroma_client.delete_collection("bookcorpus_collection")

collection = chroma_client.create_collection(
    name="bookcorpus_collection",
    embedding_function=None,
    metadata={"hnsw:space": "cosine"}
)

with open("bookcorpus_10k.json", "r", encoding="utf-8") as f:
    sentences = json.load(f)

insertion_times = []

# Crear un vector nulo de 384 dimensiones para fijar el esquema correcto
dummy_embedding_384 = [0.0] * 384

print("Cargando texto en ChromaDB [C0]...")
for idx, sentence in enumerate(sentences):
    start_time = time.time()

    # Insertar la oración con un embedding nulo para establecer el esquema
    collection.add(
        documents=[sentence],
        ids=[str(idx)],
        embeddings=[dummy_embedding_384]
    )

    end_time = time.time()
    insertion_times.append(end_time - start_time)

    if (idx + 1) % 2500 == 0:
        print(f"Cargadas {idx + 1} / 10.000 oraciones...")

print("\n=== MÉTRICAS CARGA TEXTO [C0] (ChromaDB) ===")
print(f"Mínimo: {np.min(insertion_times):.6f} s")
print(f"Máximo: {np.max(insertion_times):.6f} s")
print(f"Promedio: {np.mean(insertion_times):.6f} s")
print(f"Desviación Estándar: {np.std(insertion_times):.6f} s")
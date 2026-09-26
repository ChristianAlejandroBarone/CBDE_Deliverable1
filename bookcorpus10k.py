import json
from datasets import load_dataset

print("Descargando muestra de 10.000 oraciones...")

sentences = []

try:
    # 1. Intentar descargar desde el mirror público de BookCorpus
    dataset = load_dataset("rojagtap/bookcorpus", split="train", streaming=True)
    for item in dataset.take(10000):
        text = item.get("text", "").strip()
        if text:
            sentences.append(text)
    print("Muestra obtenida correctamente desde 'rojagtap/bookcorpus'.")

except Exception as e:
    print(f"Aviso: No se pudo acceder al mirror de BookCorpus ({e}).")
    print("Cargando corpus alternativo de texto plano (WikiText)...")

    # 2. Rescate con un dataset de texto 100% público y abierto
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # Filtrar líneas vacías o encabezados muy cortos
    sentences = [line.strip() for line in dataset["text"] if len(line.strip()) > 25][:10000]

print(f"Total de oraciones listas: {len(sentences)}")

# 3. Guardar el archivo JSON localmente para usar en los scripts [P0] y [C0]
with open("bookcorpus_10k.json", "w", encoding="utf-8") as f:
    json.dump(sentences, f, ensure_ascii=False, indent=2)

print("¡Archivo 'bookcorpus_10k.json' generado con éxito en la raíz del proyecto!")
import os
import random
from itertools import combinations

# Caminho da pasta com os arquivos
folder = "/workspace/toy_dataset_tv/Download"

# Lista de arquivos .wav
files = [f for f in os.listdir(folder) if f.endswith(".wav")]

# Agrupar arquivos por locutor (nome antes do "_")
speakers = {}
for f in files:
    name = f.split("_")[0]
    speakers.setdefault(name, []).append(f)

# ----------------
# Pares positivos (mesmo locutor) → label 1
# ----------------
positive_pairs = []
for spk, spk_files in speakers.items():
    for f1, f2 in combinations(spk_files, 2):
        positive_pairs.append((1, f1, f2))

# ----------------
# Pares negativos (locutores diferentes) → label 0
# ----------------
negative_pairs = []
speaker_names = list(speakers.keys())
for i in range(len(speaker_names)):
    for j in range(i + 1, len(speaker_names)):
        spk1, spk2 = speaker_names[i], speaker_names[j]
        for f1 in speakers[spk1]:
            for f2 in speakers[spk2]:
                negative_pairs.append((0, f1, f2))

# Embaralhar
random.shuffle(positive_pairs)
random.shuffle(negative_pairs)

# # (Opcional) balancear a quantidade
# num_pairs = min(len(positive_pairs), len(negative_pairs))
# positive_pairs = positive_pairs[:num_pairs]
# negative_pairs = negative_pairs[:num_pairs]

# ----------------
# Salvar em TXT
# ----------------
output_file = "pairs.txt"
with open(output_file, "w") as f:
    for label, f1, f2 in positive_pairs + negative_pairs:
        f.write(f"{label} {f1} {f2}\n")

print(f"Arquivo salvo em: {output_file}")
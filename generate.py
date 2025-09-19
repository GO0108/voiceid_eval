import os
import random
import itertools
from pathlib import Path
from collections import defaultdict

# --- CONFIGURAÇÕES ---
# Caminho para a pasta principal do seu dataset
DATASET_ROOT = Path("/workspace/dataset/16000_pcm_speeches")

# Nome do arquivo de saída que será gerado
OUTPUT_FILE = DATASET_ROOT.parent / "veri_pairs_generated.txt"

# Pastas a serem ignoradas
IGNORE_DIRS = ["_background_noise_", "other"]
# --- FIM DAS CONFIGURAÇÕES ---


def main():
    """
    Gera um arquivo de texto com pares de áudio para verificação de locutor.
    """
    print(f"Analisando o diretório: {DATASET_ROOT}")
    
    # 1. Encontrar todos os arquivos de áudio e agrupá-los por locutor
    speaker_files = defaultdict(list)
    for speaker_dir in DATASET_ROOT.iterdir():
        if not speaker_dir.is_dir() or speaker_dir.name in IGNORE_DIRS:
            continue
        
        speaker_id = speaker_dir.name
        # Encontra apenas arquivos .wav e armazena o nome do arquivo
        wavs = [f.name for f in speaker_dir.glob("*.wav")]
        
        # Apenas considera locutores com pelo menos 2 áudios para formar pares
        if len(wavs) >= 2:
            speaker_files[speaker_id] = wavs
            
    if not speaker_files:
        print("Erro: Nenhum locutor com áudios suficientes foi encontrado.")
        return

    speaker_list = list(speaker_files.keys())
    print(f"Encontrados {len(speaker_list)} locutores com 2 ou mais áudios.")

    intra_pairs = []
    inter_pairs = []

    # 2. Gerar pares intra-locutor (label 1 - mesma pessoa)
    for speaker_id, files in speaker_files.items():
        # itertools.combinations garante que cada par seja único (a,b) e não (b,a)
        for file1, file2 in itertools.combinations(files, 2):
            path1 = f"{speaker_id}/{file1}"
            path2 = f"{speaker_id}/{file2}"
            intra_pairs.append(f"1 {path1} {path2}")

    print(f"Gerados {len(intra_pairs)} pares intra-locutor (mesma pessoa).")

    # 3. Gerar pares inter-locutor (label 0 - pessoas diferentes)
    # Vamos gerar um número equivalente de pares 'impostores' para balancear
    num_inter_pairs_to_generate = len(intra_pairs)
    
    if len(speaker_list) < 2:
        print("Aviso: Não há locutores suficientes para criar pares inter-locutores.")
    else:
        while len(inter_pairs) < num_inter_pairs_to_generate:
            # Escolhe dois locutores diferentes aleatoriamente
            spk1_id, spk2_id = random.sample(speaker_list, 2)
            
            # Escolhe um arquivo aleatório de cada um dos locutores
            file1 = random.choice(speaker_files[spk1_id])
            file2 = random.choice(speaker_files[spk2_id])
            
            path1 = f"{spk1_id}/{file1}"
            path2 = f"{spk2_id}/{file2}"
            inter_pairs.append(f"0 {path1} {path2}")

        print(f"Gerados {len(inter_pairs)} pares inter-locutor (pessoas diferentes).")

    # 4. Combinar, embaralhar e salvar no arquivo
    all_pairs = intra_pairs + inter_pairs
    random.shuffle(all_pairs)
    
    print(f"\nTotal de pares gerados: {len(all_pairs)}")
    
    with open(OUTPUT_FILE, "w") as f:
        f.write("\n".join(all_pairs))
        
    print(f"Arquivo de pares salvo com sucesso em: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
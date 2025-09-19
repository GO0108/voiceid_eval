import csv
from pathlib import Path
import numpy as np
import sherpa_onnx
import librosa
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import * # Supondo que 'extract_embedding' e 'evaluate_speaker_verification' estão aqui
import torchaudio

# -------- Configurações de Paths --------
ROOT_DIR = Path("/workspace/toy_dataset_tv/Download/")
TXT_PATH = Path("/workspace/toy_dataset_tv/pairs.txt")
MODELS_DIR = Path("/workspace/models")
OUTPUT_DIR = Path("/workspace/results")
PLOTS_DIR = OUTPUT_DIR / "plots"
SCORES_DIR = OUTPUT_DIR / "scores"
CSV_PATH = OUTPUT_DIR / "evaluation_results.csv"

SAMPLE_RATE = 16000
MAX_PAIRS = 1000 # Limite de pares para testar

def main():
    # -------- Criação dos diretórios de saída --------
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_DIR.mkdir(parents=True, exist_ok=True)

    # -------- Inicializa o arquivo CSV com o cabeçalho --------
    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model_name', 'eer', 'mindcf'])

    # -------- Itera sobre todos os modelos na pasta --------
    model_paths = list(MODELS_DIR.glob("*.onnx"))
    for model_path in model_paths:
        model_name = model_path.stem
        print(f"\n{'='*20}\nProcessing model: {model_name}\n{'='*20}")

        # -------- Inicializa o extrator sherpa para o modelo atual --------
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(model_path), num_threads=1, debug=False, provider="cpu"
        )
        assert config.validate(), f"Configuração inválida para o modelo {model_name}"
        extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

        scores = []
        labels = []

        # -------- Lê arquivo txt, extrai scores para o modelo atual --------
        with open(TXT_PATH, "r") as f:
            lines = f.readlines()
            for line in tqdm(lines, desc=model_name):
                line = line.strip()
                if not line:
                    continue
                
                label_str, wav1_rel, wav2_rel = line.split()
                label = int(label_str)

                wav1_path = ROOT_DIR / wav1_rel
                wav2_path = ROOT_DIR / wav2_rel

                if not (wav1_path.is_file() and wav2_path.is_file()):
                    print(f"AVISO: Arquivo não encontrado: {wav1_path} ou {wav2_path}")
                    continue

                # y1, _ = librosa.load(wav1_path, sr=SAMPLE_RATE, mono=True)
                # y2, _ = librosa.load(wav2_path, sr=SAMPLE_RATE, mono=True)
                y1, _ = torchaudio.load(wav1_path, normalize=True)
                y2, _ = torchaudio.load(wav2_path, normalize=True)
                y1 = y1.mean(dim=0).numpy().squeeze()
                y2 = y2.mean(dim=0).numpy().squeeze()
                try:
                    emb1 = extract_embedding(y1, SAMPLE_RATE, extractor)
                    emb2 = extract_embedding(y2, SAMPLE_RATE, extractor)

                    

                    if np.isnan(emb1).any() or np.isnan(emb2).any():
                        continue

                    score = cosine_similarity([emb1], [emb2])[0][0]
                    scores.append(score)
                    labels.append(label)

                except Exception as e:
                    print(f"Erro ao processar {wav1_path} e {wav2_path}: {e}")
                    continue
        scores = np.array(scores)
        labels = np.array(labels)

        # -------- Avalia, plota e salva os resultados --------
        if len(scores) == 0:
            print(f"Nenhum score válido gerado para o modelo {model_name}. Pulando.")
            continue

        # 1. Avalia e gera o gráfico
        results = evaluate_speaker_verification(
            scores_intra=scores[labels == 1],
            scores_inter=scores[labels == 0],
            title=f"Evaluation for {model_name}"
        )
        print(f"EER: {results['eer']*100:.2f}%")
        print(f"MinDCF: {results['mindcf']:.3f}")

        # 2. Salva o gráfico PNG
        plot_path = PLOTS_DIR / f"{model_name}_eer.png"
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close() # Libera a figura da memória
        print(f"Gráfico salvo em: {plot_path}")

        # 3. Salva os scores
        scores_path = SCORES_DIR / f"{model_name}_scores.npz"
        np.savez_compressed(scores_path, scores=scores, labels=labels)
        print(f"Scores salvos em: {scores_path}")

        # 4. Anota os resultados no CSV
        with open(CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([model_name, results['eer'], results['mindcf']])
        print(f"Resultados de {model_name} adicionados ao CSV.")

if __name__ == "__main__":
    main()
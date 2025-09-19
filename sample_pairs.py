import random
from pathlib import Path

# --- CONFIGURAÇÕES ---

# Diretório onde os arquivos de pares estão localizados
# (Assumindo a mesma estrutura da pergunta anterior)
DATA_DIR = Path("/workspace/dataset")

# Arquivo de entrada (a lista completa que você gerou)
INPUT_FILE = DATA_DIR / "veri_pairs_generated.txt"

# Arquivo de saída (a nova lista com 1000 amostras)
OUTPUT_FILE = DATA_DIR / "veri_pairs_1000_samples.txt"

# Número de amostras aleatórias que você deseja
NUM_SAMPLES = 1000

# --- FIM DAS CONFIGURAÇÕES ---


def main():
    """
    Lê uma lista grande de pares e cria uma nova lista menor
    com uma amostragem aleatória de linhas.
    """
    print(f"Script de amostragem iniciado.")
    
    # Verifica se o arquivo de entrada existe
    if not INPUT_FILE.is_file():
        print(f"Erro: O arquivo de entrada não foi encontrado em '{INPUT_FILE}'")
        print("Por favor, gere o arquivo primeiro ou verifique o caminho.")
        return

    # Lê todas as linhas do arquivo original para a memória
    print(f"Lendo todas as linhas de '{INPUT_FILE}'...")
    with open(INPUT_FILE, 'r') as f:
        all_lines = f.readlines()
    
    total_lines = len(all_lines)
    print(f"Leitura concluída. Total de {total_lines} linhas encontradas.")

    # Verifica se há linhas suficientes para amostrar
    if total_lines < NUM_SAMPLES:
        print(f"Aviso: O arquivo original tem apenas {total_lines} linhas, "
              f"que é menos do que as {NUM_SAMPLES} solicitadas.")
        print("Todas as linhas serão copiadas para o novo arquivo.")
        sampled_lines = all_lines
    else:
        # Seleciona N amostras aleatórias da lista de linhas
        print(f"Selecionando {NUM_SAMPLES} amostras aleatórias...")
        sampled_lines = random.sample(all_lines, NUM_SAMPLES)

    # Escreve as linhas amostradas no novo arquivo
    print(f"Escrevendo {len(sampled_lines)} linhas amostradas em '{OUTPUT_FILE}'...")
    with open(OUTPUT_FILE, 'w') as f:
        f.writelines(sampled_lines)

    print("\nProcesso concluído com sucesso!")
    print(f"Arquivo de amostragem salvo em: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
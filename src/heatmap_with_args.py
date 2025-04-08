import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Detecção simples de veículos em vídeo aéreo")
    parser.add_argument('--input', type=str, required=True, help='Caminho para o vídeo de entrada')
    parser.add_argument('--output', type=str, required=True, help='Caminho para salvar a imagem de saída')
    parser.add_argument('--csv', type=str, required=True, help='Caminho para salvar o CSV com as detecções')
    return parser.parse_args()


def main():
    args = parse_args()

    # Configurações
    CSV_FILE = args.csv
    VIDEO_FILE = args.input
    HEATMAP_IMAGE = args.output

    # Ler os dados
    df = pd.read_csv(CSV_FILE)

    # Usar centro da caixa como posição do jogador
    df['x_center'] = (df['x1'] + df['x2']) / 2
    df['y_center'] = (df['y1'] + df['y2']) / 2

    # Carregar um frame base (opcional)
    cap = cv2.VideoCapture(VIDEO_FILE)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise Exception("Não foi possível carregar o frame do vídeo.")

    # Converter para RGB para usar com matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    # Criar o gráfico
    plt.figure(figsize=(12, 7))
    plt.imshow(frame_rgb)
    sns.kdeplot(
        data=df,
        x='x_center',
        y='y_center',
        fill=True,
        cmap='Reds',
        alpha=0.6,
        thresh=0.01,
        levels=100
    )
    plt.title("Mapa de Calor da Presença dos Jogadores")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(HEATMAP_IMAGE, dpi=300)
    plt.show()


if __name__ == '__main__':
    main()

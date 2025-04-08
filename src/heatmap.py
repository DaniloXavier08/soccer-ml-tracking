import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

# Configurações
CSV_FILE = 'csv/tracking-roi.csv'
VIDEO_FILE = 'videos/campo.mp4' 
HEATMAP_IMAGE = 'img/heatmap'

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

def save_image(cmap):
    # Criar o gráfico
    plt.figure(figsize=(12, 7))
    plt.imshow(frame_rgb)
    sns.kdeplot(
        data=df,
        x='x_center',
        y='y_center',
        fill=True,
        cmap=cmap, # 'Reds'
        alpha=0.6,
        thresh=0.01,
        levels=100
    )
    plt.title("Mapa de Calor da Presença dos Jogadores")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{HEATMAP_IMAGE}_{cmap}.png", dpi=300)
    #plt.show()

cmaps = ['Reds', 'viridis' , 'magma', 'rocket' , 'inferno']

for cmap in cmaps:
    save_image(cmap)

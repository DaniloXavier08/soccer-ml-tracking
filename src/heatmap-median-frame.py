import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

# Configurações
CSV_FILE = 'csv/tracking-soccer.csv'
VIDEO_FILE = 'videos/campo.mp4' 
HEATMAP_IMAGE = 'img/heatmap_medio.png'

# Ler os dados
df = pd.read_csv(CSV_FILE)

# Usar centro da caixa como posição do jogador
df['x_center'] = (df['x1'] + df['x2']) / 2
df['y_center'] = (df['y1'] + df['y2']) / 2

# Carregar um frame base (opcional)
cap = cv2.VideoCapture(VIDEO_FILE)
ret, frame = cap.read()
cap.release()

f = cv2.imread('img/frame_medio.png')

if not ret:
    raise Exception("Não foi possível carregar o frame do vídeo.")

# Converter para RGB para usar com matplotlib
frame_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
h, w, _ = f.shape

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

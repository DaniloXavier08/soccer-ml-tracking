import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

# Configurações
CSV_FILE = 'csv/tracking-high.csv'
HEATMAP_IMAGE = 'img/heatmap_three_players'

# Ler os dados
df = pd.read_csv(CSV_FILE)

# Usar centro da caixa como posição do jogador
df['x_center'] = (df['x1'] + df['x2']) / 2
df['y_center'] = (df['y1'] + df['y2']) / 2

f = cv2.imread('img/frame_medio.png')
frame_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
h, w, _ = f.shape

# Tamanho da figura com 3 subplots lado a lado
img_height, img_width, _ = frame_rgb.shape
player_ids = [3, 4, 9]
plt.figure(figsize=(18, 6))

for i, player_id in enumerate(player_ids):
    plt.subplot(1, 3, i+1)
    plt.imshow(frame_rgb)
    
    sns.kdeplot(
        data=df[df['id'] == player_id],
        x='x_center',
        y='y_center',
        fill=True,
        cmap='viridis',  
        alpha=0.6,
        thresh=0.01,
        levels=100,
        clip=((0, img_width), (0, img_height))
    )
    plt.title(f"Jogador ID {player_id}")
    plt.xlim(0, img_width)
    plt.ylim(img_height, 0) 
    plt.axis('off')

plt.suptitle("Mapas de Calor por Jogador", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.88)  # Ajuste para não sobrepor o título
plt.savefig(f"{HEATMAP_IMAGE}.png", dpi=300)
plt.show()
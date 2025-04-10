import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

# Configurações
CSV_FILE = 'csv/tracking-high.csv'
HEATMAP_IMAGE = 'img/heatmap_medio'

# Ler os dados
df = pd.read_csv(CSV_FILE)

# Usar centro da caixa como posição do jogador
df['x_center'] = (df['x1'] + df['x2']) / 2
df['y_center'] = (df['y1'] + df['y2']) / 2

f = cv2.imread('img/frame_medio.png')
frame_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
h, w, _ = f.shape

# Criar o gráfico
def save_image(cmap, df, id=None):
    # Criar o gráfico
    plt.figure(figsize=(12, 7))
    plt.imshow(frame_rgb)
    sns.kdeplot(
        data=df,
        x='x_center',
        y='y_center',
        fill=True,
        cmap=cmap,
        alpha=0.6,
        thresh=0.01,
        levels=100
    )
    plt.title("Mapa de Calor da Presença dos Jogadores")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{HEATMAP_IMAGE}_{id if id is not None else ''}.png", dpi=300)
    #plt.show()

save_image('viridis', df)
save_image('viridis', df[df['id'] == 3], id=3)
save_image('viridis', df[df['id'] == 9], id=9)
save_image('viridis', df[df['id'] == 4], id=4)
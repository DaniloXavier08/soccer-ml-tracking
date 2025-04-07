import cv2
import numpy as np

# Caminho para o vídeo de entrada
VIDEO = "videos/Estradas.mp4"

# Carregar o vídeo
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

# Obter informações do vídeo
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Configurar o vídeo de saída
out = cv2.VideoWriter("videos/Estabilizado.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

# Parâmetros de inicialização
_, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
transforms = []

# Processar cada quadro
while True:
    print("Processando quadro %d de %d" % (cap.get(cv2.CAP_PROP_POS_FRAMES), frame_count))
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar pontos de interesse no quadro anterior
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

    # Calcular o fluxo óptico para os pontos detectados
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)

    # Selecionar pontos válidos
    good_prev = prev_pts[status == 1]
    good_next = next_pts[status == 1]

    # Calcular a transformação afim entre os quadros
    matrix, _ = cv2.estimateAffinePartial2D(good_prev, good_next)

    # Extrair os parâmetros de translação
    dx = matrix[0, 2]
    dy = matrix[1, 2]
    da = np.arctan2(matrix[1, 0], matrix[0, 0])

    # Armazenar os deslocamentos
    transforms.append([dx, dy, da])

    # Atualizar o quadro anterior
    prev_gray = gray.copy()

# Suavizar os deslocamentos usando média móvel
smooth_transforms = []
for i in range(len(transforms)):
    print("Suavizando deslocamento %d de %d" % (i, len(transforms)))
    start = max(0, i - 5)
    end = min(len(transforms), i + 5)
    smooth_transforms.append(np.mean(transforms[start:end], axis=0))

# Aplicar os deslocamentos suavizados para estabilizar o vídeo
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reiniciar o vídeo
for i, smoothed_transform in enumerate(smooth_transforms):
    print("Estabilizando quadro %d de %d" % (i, len(smooth_transforms)))
    ret, frame = cap.read()
    if not ret:
        break

    # Aplicar a transformação inversa para compensar o movimento
    dx, dy, da = smoothed_transform
    transform_matrix = np.array([
        [np.cos(da), -np.sin(da), dx],
        [np.sin(da), np.cos(da), dy]
    ])

    stabilized_frame = cv2.warpAffine(frame, transform_matrix, (width, height))

    # Escrever o quadro estabilizado no arquivo de saída
    out.write(stabilized_frame)

    # Exibir o vídeo estabilizado em tempo real (opcional)
    # cv2.imshow("Estabilização", stabilized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()

print("Vídeo estabilizado salvo com sucesso.")

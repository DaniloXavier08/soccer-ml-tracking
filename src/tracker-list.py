import cv2
import numpy as np

# Carregar vídeo
video = cv2.VideoCapture('videos/qatar.mp4')

# Verificar se o vídeo foi carregado corretamente
if not video.isOpened():
    print("Erro ao abrir o vídeo!")
    exit()

# Ler o primeiro quadro
ret, frame = video.read()
if not ret:
    print("Não foi possível ler o vídeo!")
    exit()


# Selecionar ROI (Região de Interesse) para limitar a área
roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
# (1056, 660, 0, 0)
print(roi)

# Criar máscara binária com base na ROI
mask = np.zeros_like(frame[:, :, 0])  # Criar máscara do tamanho do quadro em tons de cinza
x, y, w, h = roi
mask[y:y+h, x:x+w] = 255  # Preencher a área da ROI com 255 (branco)

# turn grey, create the mask and detect
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#points = cv2.goodFeaturesToTrack(gray, mask=None, maxCorners=100, qualityLevel=0.5, minDistance=7, blockSize=7)
points = cv2.goodFeaturesToTrack(gray, mask=mask, maxCorners=5, qualityLevel=0.5, minDistance=7, blockSize=7)
print(f"Número de pontos detectados: {len(points)}")

# Criar o rastreador
trackers = []
for point in points[-5:]:
    x, y = point.ravel()
    t = cv2.TrackerCSRT_create()
    t.init(frame, (int(x), int(y), 20, 20))
    trackers.append(t)

# Processar o vídeo quadro a quadro
while True:
    ret, frame = video.read()
    if not ret:
        break

    for tracker in trackers:
        # Atualizar rastreador
        success, box = tracker.update(frame)

        if success:
            # Desenhar o retângulo ao redor do objeto
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Falha no rastreamento", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Mostrar o vídeo
    cv2.imshow("Rastreamento", frame)

    # Sair ao pressionar a tecla 'q'
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

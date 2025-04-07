import cv2

# Carregar vídeo
video = cv2.VideoCapture('videos/liverpool.mp4')

# Verificar se o vídeo foi carregado corretamente
if not video.isOpened():
    print("Erro ao abrir o vídeo!")
    exit()

# Ler o primeiro quadro
ret, frame = video.read()
if not ret:
    print("Não foi possível ler o vídeo!")
    exit()

# Selecionar ROI (Região de Interesse) para rastreamento
roi = cv2.selectROI("Selecione o objeto", frame, fromCenter=False, showCrosshair=True)
print(roi)

# Criar o rastreador
tracker = cv2.TrackerCSRT_create()
tracker.init(frame, roi)

# Processar o vídeo quadro a quadro
while True:
    ret, frame = video.read()
    if not ret:
        break

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

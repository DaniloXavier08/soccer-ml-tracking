import cv2

def resize_video(input_path, output_path, width, height):
    """
    Redimensiona um vídeo para a resolução especificada.

    Args:
        input_path (str): Caminho do vídeo de entrada.
        output_path (str): Caminho para salvar o vídeo redimensionado.
        width (int): Largura desejada.
        height (int): Altura desejada.
    """
    # Abre o vídeo de entrada
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo!")
        return

    # Obtém o codec e a taxa de quadros do vídeo original
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para salvar o vídeo
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Cria o objeto VideoWriter para salvar o vídeo redimensionado
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensiona o frame
        resized_frame = cv2.resize(frame, (width, height))

        # Escreve o frame redimensionado no arquivo de saída
        out.write(resized_frame)

    # Libera os recursos
    cap.release()
    out.release()
    print(f"Vídeo redimensionado salvo em: {output_path}")


if __name__ == "__main__":
    # Caminho do vídeo de entrada e saída
    input_video = "videos/campo_3.mp4"
    output_video = "videos/campo_3_resized.mp4"

    # Resolução desejada
    target_width = 640
    target_height = 360

    # Redimensiona o vídeo
    resize_video(input_video, output_video, target_width, target_height)
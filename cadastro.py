import cv2
import face_recognition
import os

# Verifica se a pasta 'usuarios' existe; se não, cria
if not os.path.exists('usuarios'):
    os.makedirs('usuarios')

nome = input("Digite seu nome: ").strip().replace(" ", "_")

cam = cv2.VideoCapture(0)
print("Pressione 'Espaço' para capturar a imagem ou 'Esc' para sair.")

# Reduz a resolução da câmera para melhorar o desempenho
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

rosto_detectado_consecutivo = 0
FRAMES_NECESSARIOS = 5  # Número de frames consecutivos necessários

while True:
    ret, frame = cam.read()
    if not ret:
        print("Falha ao acessar a câmera.")
        break

    # Redimensiona o frame para acelerar o processamento
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Converte para RGB (necessário para face_recognition)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Detecta os rostos
    loc_rostos = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=1)
    num_rostos = len(loc_rostos)
    # print(f"Número de rostos detectados: {num_rostos}")  # Pode ser descomentado para debug

    # Verifica se há exatamente um rosto detectado
    if num_rostos == 1:
        rosto_detectado_consecutivo += 1
        cor_retangulo = (0, 255, 0)  # Verde
    else:
        rosto_detectado_consecutivo = 0
        cor_retangulo = (0, 0, 255)  # Vermelho

    # Escala as coordenadas de volta para o tamanho original
    for (top, right, bottom, left) in loc_rostos:
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        cv2.rectangle(frame, (left, top), (right, bottom), cor_retangulo, 2)

    # Exibe uma mensagem na tela
    if rosto_detectado_consecutivo >= FRAMES_NECESSARIOS:
        cv2.putText(frame, "Rosto estável detectado", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Detectando rosto...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Cadastro", frame)

    key = cv2.waitKey(1) & 0xFF  # Captura a tecla pressionada
    if key != 255:
        # print(f"Tecla pressionada: {key}")  # Pode ser descomentado para debug
        if key == 27:  # Tecla Esc
            print("Encerrando...")
            break
        elif key == 32:  # Tecla Espaço
            if rosto_detectado_consecutivo >= FRAMES_NECESSARIOS:
                # Salva a imagem
                try:
                    cv2.imwrite(f'usuarios/{nome}.jpg', frame)
                    print("Cadastro realizado com sucesso!")
                except Exception as e:
                    print(f"Erro ao salvar a imagem: {e}")
                break
            else:
                print("Rosto não estável detectado. Mantenha-se imóvel por alguns instantes.")

cam.release()
cv2.destroyAllWindows()

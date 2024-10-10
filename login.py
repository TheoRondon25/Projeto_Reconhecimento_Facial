import cv2
import face_recognition
import dlib
import numpy as np
import os

# Carrega os modelos
predictor_model = r"C:\Projetos\Projeto_Reconhecimento_Facial\Dlib\shape_predictor_68_face_landmarks.dat"
face_rec_model = r"C:\Projetos\Projeto_Reconhecimento_Facial\Dlib\dlib_face_recognition_resnet_model_v1.dat"

if not os.path.exists(predictor_model) or not os.path.exists(face_rec_model):
    print("Os modelos necessários não foram encontrados. Certifique-se de que os arquivos .dat estão no mesmo diretório do script.")
    exit()

face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(predictor_model)
face_recognizer = dlib.face_recognition_model_v1(face_rec_model)

# Carrega as imagens dos usuários cadastrados
imagens_cadastradas = []
nomes_cadastrados = []

for arquivo in os.listdir('usuarios'):
    imagem = face_recognition.load_image_file(f'usuarios/{arquivo}')
    # Detecta rostos na imagem
    faces = face_detector(imagem, 1)
    if len(faces) == 0:
        print(f"Atenção: Nenhum rosto detectado na imagem {arquivo}.")
        continue
    # Assume o primeiro rosto detectado
    face = faces[0]
    shape = shape_predictor(imagem, face)
    face_descriptor = face_recognizer.compute_face_descriptor(imagem, shape)
    encoding = np.array(face_descriptor)
    imagens_cadastradas.append(encoding)
    nomes_cadastrados.append(os.path.splitext(arquivo)[0])

# Inicializa a captura de vídeo
cam = cv2.VideoCapture(0)
print("Olhe para a câmera para login. Pressione 'Espaço' para tentar ou 'Esc' para sair.")

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
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detecta rostos no frame atual
    faces = face_detector(rgb_small_frame, 1)

    # Verifica se há exatamente um rosto detectado
    if len(faces) == 1:
        rosto_detectado_consecutivo += 1
        cor_retangulo = (0, 255, 0)  # Verde
    else:
        rosto_detectado_consecutivo = 0
        cor_retangulo = (0, 0, 255)  # Vermelho

    # Desenha retângulos ao redor dos rostos detectados
    for face in faces:
        left = face.left() * 2
        top = face.top() * 2
        right = face.right() * 2
        bottom = face.bottom() * 2
        cv2.rectangle(frame, (left, top), (right, bottom), cor_retangulo, 2)

    # Exibe uma mensagem na tela
    if rosto_detectado_consecutivo >= FRAMES_NECESSARIOS:
        cv2.putText(frame, "Rosto estável detectado", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Detectando rosto...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Login", frame)

    key = cv2.waitKey(1) & 0xFF  # Captura a tecla pressionada
    if key != 255:
        if key == 27:  # Tecla Esc
            print("Encerrando...")
            break
        elif key == 32:  # Tecla Espaço
            if rosto_detectado_consecutivo >= FRAMES_NECESSARIOS:
                if len(faces) == 0:
                    print("Nenhum rosto detectado. Tente novamente.")
                else:
                    reconhecido = False
                    # Processa o rosto detectado
                    face = faces[0]
                    shape = shape_predictor(rgb_small_frame, face)
                    face_descriptor = face_recognizer.compute_face_descriptor(rgb_small_frame, shape)
                    encoding = np.array(face_descriptor)

                    # Compara com os encodings cadastrados
                    distancias = face_recognition.face_distance(imagens_cadastradas, encoding)
                    melhor_indice = np.argmin(distancias)
                    if distancias[melhor_indice] < 0.6:
                        nome = nomes_cadastrados[melhor_indice]
                        print(f"Bem-vindo, {nome}!")
                        reconhecido = True
                    else:
                        print("Usuário não reconhecido.")
                    break
            else:
                print("Rosto não estável detectado. Mantenha-se imóvel por alguns instantes.")

cam.release()
cv2.destroyAllWindows()

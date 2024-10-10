import cv2
import face_recognition
import os

# Carrega as imagens dos usuários cadastrados
imagens_cadastradas = []
nomes_cadastrados = []

for arquivo in os.listdir('usuarios'):
    imagem = face_recognition.load_image_file(f'usuarios/{arquivo}')
    encoding = face_recognition.face_encodings(imagem)[0]
    imagens_cadastradas.append(encoding)
    nomes_cadastrados.append(os.path.splitext(arquivo)[0])

# Captura a imagem para login
cam = cv2.VideoCapture(0)
print("Olhe para a câmera para login.")
ret, frame = cam.read()
cam.release()

if ret:
    # Converte a imagem de BGR (OpenCV) para RGB (face_recognition)
    rgb_frame = frame[:, :, ::-1]
    # Localiza e codifica o rosto na imagem capturada
    loc_rostos = face_recognition.face_locations(rgb_frame)
    encodings_rostos = face_recognition.face_encodings(rgb_frame, loc_rostos)

    for encoding in encodings_rostos:
        resultados = face_recognition.compare_faces(imagens_cadastradas, encoding)
        if True in resultados:
            indice = resultados.index(True)
            nome = nomes_cadastrados[indice]
            print(f"Bem-vindo, {nome}!")
            break
        else:
            print("Usuário não reconhecido.")
else:
    print("Falha ao capturar imagem.")

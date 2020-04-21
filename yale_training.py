import cv2
import os
import numpy as np
from PIL import Image

eigenface = cv2.face.EigenFaceRecognizer_create(40, 8000)
fisherface = cv2.face.FisherFaceRecognizer_create(12, 2000)
lbph = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=2, grid_x=7, grid_y=7, threshold=50)


def get_image_com_id():
    caminhos = [os.path.join('yalefaces/treinamento', f) for f in os.listdir('yalefaces/treinamento')]
    faces_array = []
    cod = []
    for caminhoImagem in caminhos:
        imagem_face = Image.open(caminhoImagem).convert('L')
        imagem_np = np.array(imagem_face, 'uint8')
        id = int(os.path.split(caminhoImagem)[1].split(".")[0].replace("subject", ""))
        cod.append(id)
        faces_array.append(imagem_np)

    return np.array(cod), faces_array


ids, faces = get_image_com_id()

print("Treinando...")
eigenface.train(faces, ids)
eigenface.write('classificadorEigenYale.yml')

fisherface.train(faces, ids)
fisherface.write('classificadorFisherYale.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPHYale.yml')

print("Treinamento realizado")

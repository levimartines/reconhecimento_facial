import cv2
import os
import numpy as np

# num_components -> No de componentes principais
# threshold -> limite de confianca/distancia ( vizinho mais próximo - KNN )
#
# radius = 1,
# neighbors = 8,
# grid_x = 8,
# grid_y = 8,
# threshold = DBL_MAX


eigenface = cv2.face.EigenFaceRecognizer_create(num_components=50, threshold=3)
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()


def get_image_com_id():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    faces = []
    cods = []

    for caminhoImagem in caminhos:
        imagem_face = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        cod = int(os.path.split(caminhoImagem)[-1].split('.')[1])
        print("(IMG): " + os.path.split(caminhoImagem)[-1].split('.')[2] + " | COD: " + str(cod))
        cods.append(cod)
        faces.append(imagem_face)
    return np.array(cods), faces


ids, faces = get_image_com_id()

# TREINAMENTO
print("COMECANDO TREINAMENTO")

eigenface.train(faces, ids)
eigenface.write('classificadorEigen.yml')

fisherface.train(faces, ids)
fisherface.write('classificadorFisher.yml')

lbph.train(faces, ids)
lbph.write('classificadorLbph.yml')

print('TREINAMENTO REALIZADO')

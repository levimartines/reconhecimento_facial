import cv2
import os
import numpy as np
from PIL import Image
import sys

detector_face = cv2.CascadeClassifier("original.xml")
# reconhecedor = cv2.face.EigenFaceRecognizer_create()
# reconhecedor.read("classificadorEigenYale.yml")
reconhecedor = cv2.face.FisherFaceRecognizer_create()
reconhecedor.read("classificadorFisherYale.yml")
# reconhecedor = cv2.face.LBPHFaceRecognizer_create()
# reconhecedor.read("classificadorLBPHYale.yml")

total_acertos = 0
percentual_acerto = 0.0
total_confianca = 0.0

caminhos = [os.path.join('yalefaces/teste', f) for f in os.listdir('yalefaces/teste')]

for caminho_imagem in caminhos:
    imagem_face = Image.open(caminho_imagem).convert('L')
    imagem_cinza = np.array(imagem_face, 'uint8')
    faces_detectadas = detector_face.detectMultiScale(imagem_cinza)
    for (x, y, l, a) in faces_detectadas:
        id_previsto, confianca = reconhecedor.predict(imagem_cinza)
        id_atual = int(os.path.split(caminho_imagem)[1].split(".")[0].replace("subject", ""))
        print(str(id_atual) + " foi classificado como - " + str(id_previsto) + " confianca: " + str(confianca))
        if id_previsto == id_atual:
            total_acertos += 1
            total_confianca += confianca

        # cv2.rectangle(imagem_cinza, (x, y), (x + l, y + a), (0, 0, 255), 2)
        # cv2.imshow("Face", imagem_cinza)
        # cv2.waitKey(1000)
percentual_acerto = (total_acertos / 30) * 100
total_confianca = total_confianca / total_acertos
print("Percentual acertos: " + str(percentual_acerto))
print("Total confianca: " + str(total_confianca))

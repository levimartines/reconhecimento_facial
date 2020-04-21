import cv2
import numpy as np

url = "http://192.168.25.60:4747/video"
classificador = cv2.CascadeClassifier("original.xml")
classificadorOlho = cv2.CascadeClassifier("cascade_eye.xml")
camera = cv2.VideoCapture(url)
amosta = 1
numeroAmostras = 10
id = input('Digite seu id: ')
larg, alt = 220, 220

while True:
    conectado, imagem = camera.read()
    # No caso para o processamento é indicado processar em uma escala de cinza 
    # para melhor precisão e desempenho
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    # print(np.average(imagemCinza))

    facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(100, 100))

    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
        regiao = imagem[y:y + a, x:x + l]
        regiaoCinzaOlho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
        olhosDetectados = classificadorOlho.detectMultiScale(regiaoCinzaOlho)

        for (ox, oy, ol, oa) in olhosDetectados:
            cv2.rectangle(regiao, (ox, oy), (ox + 50, oy + 50), (0, 255, 0), 2)

            if cv2.waitKey(1) == ord('q'):
                print(np.average(imagemCinza))
                if np.average(imagemCinza) > 100:
                    imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (larg, alt))
                    cv2.imwrite("fotos/pessoa." + str(id) + "." + str(amosta) + ".jpg", imagemFace)
                    print("Foto capturada com sucesso - " + str(amosta))
                    amosta += 1

    cv2.imshow("Face", imagem)
    cv2.waitKey(1)
    if amosta >= numeroAmostras:
        break
camera.release()
cv2.destroyAllWindows()
exit(-1)

import cv2

detectorFace = cv2.CascadeClassifier("original.xml")
reconhecedor = cv2.face.EigenFaceRecognizer_create()
reconhecedor.read("classificadorLbph.yml")
largura, altura = 220, 220
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

url = "http://192.168.25.60:4747/video"
camera = cv2.VideoCapture(url)

while True:
    conectado, imagem = camera.read()
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    faces_detectadas = detectorFace.detectMultiScale(imagem_cinza, scaleFactor=1.5, minSize=(30, 30))

    for (x, y, l, a) in faces_detectadas:
        imagem_face = cv2.resize(imagem_cinza[y:y + a, x:x + l], (largura, altura))
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
        id, confianca = reconhecedor.predict(imagem_face)
        if id == 1:
            nome = "Renan"
        elif id == 2:
            nome = "Levi"
        elif id == 3:
            nome = "Solano"
        else:
            nome = "Desconhecido"

        cv2.putText(imagem, nome, (x, y + (a + 30)), font, 2, (0, 0, 255))
        cv2.putText(imagem, str(confianca), (x, y + (a + 50)), font, 1, (0, 0, 255))

    cv2.imshow("Face", imagem)
    if cv2.waitKey(1) == ord('q'):
        break
camera.release()
cv2.destroyAllWindows()
exit(0)

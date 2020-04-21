import cv2
import urllib.request
import numpy as np


def url_to_image(image_url):
    resp = urllib.request.urlopen(image_url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, 1)
    return image


classificador = cv2.CascadeClassifier("original.xml")
amostra = 1
cod = input('Digite seu id: ')
larg, alt = 220, 220

urls = ["https://scontent.fsod3-1.fna.fbcdn.net/v/t1.0-9/s960x960/51031504_2238308409591178_6470855778931572736_o.jpg"
        "?_nc_cat=108&_nc_sid=dd7718&_nc_eui2=AeGwAIiFET76e1bKJgAe3IppRVcdP_K2WPtFVx0_8rZY"
        "-855Wt_NL7m2j9gRGz42zMv1o3_4lQoP-BBC9aYhcpc8&_nc_oc=AQl"
        "-DKVd0oI3UqWAVnXqiujxID_TAMkK_hAiZBvZwbv_NL8fpYCyH39dtaTbpf-4Z5pqo_gBzBHDMuIXffTkDzBL&_nc_ht=scontent.fsod3"
        "-1.fna&_nc_tp=7&oh=48e02afff09dcfc786e2249f77ca32a7&oe=5EC1FFA9",
        "https://scontent.fsod3-1.fna.fbcdn.net/v/t1.0-9/s960x960/39121872_1946007348821287_7135044940517605376_o.jpg"
        "?_nc_cat=102&_nc_sid=dd7718&_nc_eui2=AeG6y-1EmdOKYuIy8HZg2oR-hSw69cX0tyuFLDr1xfS3K4KyADn"
        "-iXjEiVN7nHfXEG4Hr7tZOth2zvbt5-JJI6_0&_nc_oc=AQn7B3_ji7nWtBlAcUnZrFuH2Gl2zpGAYnN9F"
        "-_szz3gjkTZGkQEfcnmhKegn7utrPQxUt_AltFD5AFfWMd4uMzK&_nc_ht=scontent.fsod3-1.fna&_nc_tp=7&oh"
        "=ebb2924be8f3ae5cd868528b5d433f93&oe=5EC347FC",
        "https://scontent.fsod3-1.fna.fbcdn.net/v/t31.0-8/p960x960/13975489_1072513962837301_8410466348303106071_o.jpg"
        "?_nc_cat=106&_nc_sid=daf655&_nc_eui2"
        "=AeFZYHrNN3vDnhi0ipnjM8XY5f34B5xLS43l_fgHnEtLjdCo3q5JX8fTtVRe2N_B3bdwD39HJ1TScr1ENARQ9pxm&_nc_oc=AQnWTrmG"
        "-zbjljWUJAjHLujL-jiYzUkTRGtxKA_vk2rd7dU2ioUfECXaR-Fr2ZGPbEjSUche42E6q1-zZKbt-Xmw&_nc_ht=scontent.fsod3-1.fna"
        "&_nc_tp=6&oh=d7d8eb05f54e6bd6431288adc025e08a&oe=5EC5A5A2"]
for url in urls:
    imagem = url_to_image(url)
    # No caso para o processamento é indicado processar em uma escala de cinza 
    # para melhor precisão e desempenho
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.5, minSize=(100, 100))
    print(amostra)
    for (x, y, l, a) in facesDetectadas:
        print(amostra)
        imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (larg, alt))
        cv2.imwrite("fotos/pessoa." + str(cod) + "." + str(amostra) + ".jpg", imagemFace)
        amostra += 1

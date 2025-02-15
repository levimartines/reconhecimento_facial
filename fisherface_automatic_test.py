import os  # importing the OS for path
import cv2  # importing the OpenCV library
import numpy as np  # importing Numpy library
from PIL import Image  # importing Image library
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('original.xml')
path = 'yalefaces/treinamento'  # path to the photos
img = Image.open('yalefaces/teste/subject01.gif').convert('L')


def getImageWithID(imagePath):
    imagePaths = [os.path.join(imagePath, f) for f in os.listdir(imagePath)]
    FaceList = []
    IDs = []

    for imagePath in imagePaths:
        faceImage = Image.open(imagePath).convert('L')  # Open image and convert to gray
        # print(str((faceImage.size)))
        faceImage = faceImage.resize((110, 110))  # resize the image so the EIGEN recogniser can be trained

        faceNP = np.array(faceImage, 'uint8')  # convert the image to Numpy array
        print(str(faceNP.shape))

        ID = int(os.path.split(imagePath)[1].split(".")[0].replace("subject", ""))  # Retreave the ID of the array
        print(ID)
        FaceList.append(faceNP)  # Append the Numpy Array to the list
        IDs.append(ID)  # Append the ID to the IDs list

    return np.array(IDs), FaceList  # The IDs are converted in to a Numpy array


face_number = 1
IDs, FaceList = getImageWithID(path)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert the Camera to gray
gray = np.array(img, 'uint8')
faces = face_cascade.detectMultiScale(gray, 1.3, 4)  # Detect the faces and store the positions
Info = open('SaveData/FISHER_TRAINER.txt', "w+")
for (x, y, w, h) in faces:
    Face = cv2.resize((gray[y: y + h, x: x + w]), (110, 110))
    Lev = 1
    fish_ID = []
    fish_conf = []
    for _ in range(55):
        recog = cv2.face.FisherFaceRecognizer_create(Lev)  # creating FISHER FACE RECOGNISER
        print('TRAINING FOR ' + str(Lev) + ' ......')
        recog.train(FaceList, IDs)  # The recongniser is trained using the images
        print('FISHERFACE FACE RECOGNISER COMPLETE')
        ID, conf = recog.predict(Face)
        fish_ID.append(ID)
        fish_conf.append(conf)
        Info.write(str(ID) + "," + str(conf) + "\n")
        print(_)
        'FOR ' + str(Lev) + ' COMPONENTS ID: ' + str(ID) + ' CONFIDENT: ' + str(conf)
        Lev = Lev + 1

    fig = plt.gcf()
    fig.canvas.set_window_title('RESULTS FOR FACE ' + str(face_number))
    plt.subplot(2, 1, 1)
    plt.plot(fish_ID)
    plt.title('ID against Number of Components', fontsize=10)
    plt.axis([0, Lev, 0, 25])
    plt.ylabel('ID', fontsize=8)
    plt.xlabel('Number of Components', fontsize=8)
    p2 = plt.subplot(2, 1, 2)
    plt.plot(fish_conf, 'red')
    plt.title('Confidence against Number of Components', fontsize=10)
    p2.set_xlim(xmin=0)
    p2.set_xlim(xmax=Lev)
    plt.ylabel('Confidence', fontsize=8)
    plt.xlabel('Number of Components', fontsize=8)
    plt.tight_layout()
    print(_)
    ' SHOW RESULTS FOR FACE ' + str(face_number)
    # NameFind.tell_time_passed()
    cv2.imshow('FACE' + str(face_number), Face)
    plt.show()
    face_number = face_number + 1

Info.close()
cv2.destroyAllWindows()

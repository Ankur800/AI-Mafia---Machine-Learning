import numpy as np
import cv2

smile_cascade = cv2.CascadeClassifier('./data/haarcascade_smile.xml')

video_capture = cv2.VideoCapture(0)
scale_factor = 1.3

while 1:
    ret, img = video_capture.read()

    smiles = smile_cascade.detectMultiScale(img, scale_factor, 20)

    for (sx, sy, sw, sh) in smiles:
        cv2.rectangle(img, (sx, sy), ((sx + sw), (sy + sh)), (0, 255, 0), 5)

    cv2.imshow("Smile detected", img)

    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
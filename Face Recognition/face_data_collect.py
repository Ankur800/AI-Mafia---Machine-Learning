# Write a Python Script that captures images from your webcam video stream
# Extracts all Faces from the image frame (using haarcascades)
# Stores the Face information into numpy arrays

# 1. Read and show video stream, capture images
# 2. Detect Faces and show bounding box (haarcascade)
# 3. Flatten the largest face image(gray scale) and save in a numpy array
# 4. Repeat the above for multiple people to generate training data

# IMPORTING LIBRARIES
import cv2
import numpy as np

# INIT WEB CAM
cap = cv2.VideoCapture(0)                   # id means camera number in case of multiple cams

# FACE DETECTION
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

face_data = []
dataset_path = "./data/"
file_name = input("Enter the name of the person: ")


while True:
    ret, frame = cap.read()                 # ret to read frame

    if not ret:                             # if ret == false: -> means frame has not been captured
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # gray frame

    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(faces) == 0:
        continue

    faces = sorted(faces, key=lambda f: f[2]*f[3])      # sorting based upon area

    # pick the last face (because it has the largest area)
    for face in faces[-1:]:
        # draw bounding box or rectangle
        x, y, w, h = face
        cv2.rectangle(gray_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

        # extract (crop out the required face) : region of interest
        offset = 10
        face_section = gray_frame[y - offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))
        face_data.append(face_section)
        print(len(face_section))

    # cv2.imshow("Frame", frame)              # if frame has been captured
    cv2.imshow("gray_frame", gray_frame)    # showing gray frame

    key_pressed = cv2.waitKey(1) & 0xFF     # key press detection
    if key_pressed == ord('q'):             # ord function given the ASCII values of a character
        break                               # if 'q' has been pressed then quit

# convert face data list into numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

# save this data into file system
np.save(dataset_path + file_name + '.npy', face_data)
print("Data Saved Successfully!")

cap.release()                               # to release the object
cv2.destroyAllWindows()                     # destroys all the created windows
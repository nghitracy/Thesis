import cv2
import numpy as np

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('C:\Python27\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    cv2.putText(frame, "Nghi", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 0, 255), 3)

    cv2.putText(frame, "dx: {}, dy: {}" ,(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (0, 0, 255), 1)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
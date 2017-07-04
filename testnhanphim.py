import cv2

cap= cv2.VideoCapture(0)
while (True):
    ret,frame= cap\
        .read()
    cv2.imshow('frame', frame)
    k = cv2.waitKey(3) & 0xff
    if k == ord('a'):
        print"A press"
    elif k == ord('w'):
        print "D press"
    elif k == 27:
        break

cap.release()
cv2.destroyAllWindows()
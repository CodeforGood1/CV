import cv2

cap = cv2.VideoCapture(0)
helmet_cascade = cv2.CascadeClassifier('haarcascade_helmet.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    helmets = helmet_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in helmets:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('Safety Monitoring', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

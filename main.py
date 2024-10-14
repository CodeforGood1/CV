

import cv2

cap = cv2.VideoCapture('cars.mp4')
car_cascade = cv2.CascadeClassifier('haarcascade_cars.xml')

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Failed to capture frame or frame is empty")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 3)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        crop_img = frame[y:y + h, x:x + w]

    cv2.imshow('video', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

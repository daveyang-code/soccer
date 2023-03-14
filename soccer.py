import cv2
from ultralytics import YOLO
from collections import deque

model = YOLO("custom.pt")
model.fuse()

cap = cv2.VideoCapture("Portugal_France.mp4")

buffer = 1024

pts = deque(maxlen=buffer)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.predict(frame)

    for i in range(1, len(pts)):
        cv2.circle(frame, pts[i][0], 1, pts[i][1], -1)

    for r in results:

        boxes = r.boxes

        for box in boxes:

            b = box.xyxy[0]

            if box.cls == 3:
                pts.appendleft([(int((b[0]+b[2])/2),int((b[1]+b[3])/2)),(255,)])
            elif box.cls == 4:
                pts.appendleft([(int((b[0]+b[2])/2),int((b[1]+b[3])/2)),(0,255,)])

    cv2.imshow("", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

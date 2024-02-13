import torch
import cv2
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

current_dir = os.getcwd()  
model_dir = os.path.join(current_dir, '..', 'yolov5')

model = torch.hub.load(repo_or_dir=model_dir, model='yolov5s', source='local')

classNames = model.names


colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (0, 255, 255), (255, 255, 0)]  

while True:
    success, frame = cap.read()
    if not success:
        break


    results = model(frame)

  
    for det in results.pred[0]:
        box, class_id, confidence = det[:4], int(det[5]), det[4]
        
        x1, y1, x2, y2 = map(int, box)
        color = colors[class_id % len(colors)] 
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        
        org = (x1, y1 - 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        cv2.putText(frame, f"{classNames[class_id]} ({confidence:.2f})", org, font, fontScale, color, 2)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import paddlex as pdx
from playsound import playsound

# 修改模型所在位置
predictor = pdx.deploy.Predictor('D:\\project\\python\\cigarette\\inference_model')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        result = predictor.predict(frame)
        score = result[0]['score']
        if score >= 0.3:
            print("*"*100)
            # 修改音频所在位置
            playsound('D:\\project\\python\\cigarette\\cigarette.mp3')
        # print(result)
        vis_img = pdx.det.visualize(frame, result, threshold=0.6, save_dir=None)
        cv2.imshow('cigarette', vis_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
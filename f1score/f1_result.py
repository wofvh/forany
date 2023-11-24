import torch
import cv2
import numpy as np
import datetime
import time
import os

def detect_ppe(image, model, timestamp):
    results = model(image)
    # names: [ "scaffolds","worker","hardhat","hook","opened_hatch","closed_hatch"]
    person = []
    hat = []
    hook = []
    person_count = 0
    hook_count = 0
    o_hatch = 0
    c_hatch = 0
    finalStatus = ""
    # missing_hooks = max(person_count - hook_count, 0)
    if np.shape(results.xyxy[0].cpu().numpy())[0] > 0:
        for (x0, y0, x1, y1, confi, clas) in results.xyxy[0].cpu().numpy():
            if confi > 0.5:
                # print(x0, y0, x1, y1, confi, clas)
                box = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
                box2 = [int(x0), int(y0), int(x1), int(y1)]
                box3 = [int(x0), int(y0), int(x1), int(y1)]
                if int(clas) == 3:
                    cv2.rectangle(image, box, (0, 130, 0), 2)
                    cv2.putText(image, "hook {:.2f}".format(confi), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 200, 0), 2)
                    hook.append(box3)
                elif int(clas) == 2:
                    cv2.rectangle(image, box, (0, 255, 0), 2)
                    cv2.putText(image, "Hard Hat {:.2f}".format(confi), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0), 2)
                    hat.append(box2)  # Add the box coordinates to the person arrays

                elif int(clas) == 4 :
                    o_hatch = o_hatch + 1
                    cv2.rectangle(image, box, (0, 0, 255), 2)
                    cv2.putText(image, "opened_hatch {:.2f}".format(confi), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255), 2)
                    
                elif int(clas) == 5:
                    c_hatch = c_hatch + 1
                    cv2.rectangle(image, box, (0, 255, 0), 2)
                    cv2.putText(image, "closed_hatch {:.2f}".format(confi), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0), 2)
                elif int(clas) == 1 :
                    person.append(box2)  

        class_worker_count = (results.xyxy[0].cpu().numpy()[:, -1] == 1).sum()
        class_helmet_count = (results.xyxy[0].cpu().numpy()[:, -1] == 2).sum()
        class_hook_count = (results.xyxy[0].cpu().numpy()[:, -1] == 3).sum()
        class_opened_hatch = (results.xyxy[0].cpu().numpy()[:, -1] == 4).sum()
        class_closed_hatch = (results.xyxy[0].cpu().numpy()[:, -1] == 5).sum()

        if class_hook_count >= class_worker_count:
            missing_hooks = 0
        else:
            missing_hooks = abs(class_worker_count - class_hook_count)

        if class_helmet_count >= class_worker_count:
            missing_helmet = 0
        else:
            missing_helmet = abs(class_worker_count - class_helmet_count) #

        if missing_helmet == 0:
            finalStatus = "Safe"
        else:
            finalStatus = "UnSafe"

        if missing_hooks == 0:
            finalStatus = "Safe"
        else:
            finalStatus = "UnSafe"

        if class_opened_hatch == 1:
            finalStatus = "UnSafe"
        elif class_closed_hatch == 1:
            finalStatus = "Safe"

        hatDetected = False
        for perBox in person:
            hatDetected = False
            for hatBox in hat:
                if int(hatBox[0]) > int(perBox[0]) and int(hatBox[2]) < int(perBox[2]):
                    if hatBox[1] >= perBox[1] - 20:
                        hatDetected = True
            if hatDetected :
                cv2.rectangle(image,
                              [int(perBox[0]), int(perBox[1]), int(perBox[2] - perBox[0]), int(perBox[3] - perBox[1])],
                              (0, 180, 0), 2)
                cv2.putText(image, "Person with helmet", (perBox[0], perBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (0, 180, 0), 2)
            else:
                finalStatus = "UnSafe"   
                cv2.rectangle(image,
                              [int(perBox[0]), int(perBox[1]), int(perBox[2] - perBox[0]), int(perBox[3] - perBox[1])],
                              (0, 0, 255), 2)
                cv2.putText(image, "Person without helmet", (perBox[0], perBox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                            (0, 0, 255), 2)
                

        # if finalStatus != "Safe":
        color_status = (0, 0, 255) if finalStatus != "Safe" else (0, 255, 0)
        color_hooks = (0, 255, 0) if missing_hooks == 0 else (0, 0, 255)
        color_helmet = (0, 255, 0) if missing_helmet == 0 else (0, 0, 255)

        cv2.putText(image, f"{finalStatus}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_status, 3)
        cv2.putText(image, f"[Missing {missing_hooks} hooks]", (135, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_hooks, 2)
        cv2.putText(image, f"[Missing {missing_helmet} helmet]", (135, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_helmet, 2)


    height, width, _ = image.shape
    formatted_datetime = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    # cv2.putText(image, f"{formatted_datetime} ", (width-200, height-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return image, results.xyxy[0].cpu().numpy(), timestamp, finalStatus

image_folder_path = "C:/f1score/test/GroundTruth/"  
output_folder_path = "C:/f1score/test/Prediction/"

ppe_model = torch.hub.load('WongKinYiu/yolov7', 'custom', './last.pt', force_reload=False)
# 폴더 내 모든 이미지 파일을 처리
for filename in os.listdir(image_folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # 이미지 파일 확장자 확인
        image_path = os.path.join(image_folder_path, filename)
        output_path = os.path.join(output_folder_path, f"result_{filename}") 

        frame = cv2.imread(image_path)  # 이미지 읽기

        timestamp = datetime.datetime.now()
        status = ""

        frame = cv2.resize(frame, (640,640))  # 이미지 크기 조정

        ppe_result_image, bbox_lists, timestamp, status = detect_ppe(frame, ppe_model, timestamp)

        if status == "UnSafe":
            output_path = os.path.join(output_folder_path,
                                        filename + "_ABCUNSAFE.jpg")
        else:
            output_path = os.path.join(output_folder_path,
                                        filename + "_XYZSAFE.jpg")

        cv2.imwrite(output_path, ppe_result_image)  # 결과 이미지 저장

        # cv2.imshow("PPE Detection", ppe_result_image)
        cv2.waitKey(0)
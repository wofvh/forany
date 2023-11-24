import torch
import cv2
import numpy as np
import datetime
import time
#클래스별 숫자입니다 scaffolds 부터 0으로 시작하면 됩니다  scaffolds = 0 ,worker = 1 boots = 2 이런식 
# names: [ "scaffolds","worker","boots","hardhat","safety_vest","hook","robot_dog","opened_hatch","closed_hatch",]
def detect_ppe(image, model, timestamp):
    results = model(image)
    person = []
    hat = []
    hook = []
    finalStatus = ""
    # missing_hooks = max(person_count - hook_count, 0)
    if np.shape(results.xyxy[0].cpu().numpy())[0] > 0:
        for (x0, y0, x1, y1, confi, clas) in results.xyxy[0].cpu().numpy():
            if confi > 0.3:
                # print(x0, y0, x1, y1, confi, clas)
                box = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
                box2 = [int(x0), int(y0), int(x1), int(y1)]
                box3 = [int(x0), int(y0), int(x1), int(y1)]
                if int(clas) == 5: #hook
                    cv2.rectangle(image, box, (0, 130, 0), 2)
                    cv2.putText(image, "hook {:.2f}".format(confi), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 200, 0), 2)
                    hook.append(box3)
                elif int(clas) == 3: #Hard Hat
                    cv2.rectangle(image, box, (0, 255, 0), 2)
                    cv2.putText(image, "Hard Hat {:.2f}".format(confi), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0), 2)
                    hat.append(box2)  # Add the box coordinates to the person arrays

                elif int(clas) == 7 : #opened_hatch
                    cv2.rectangle(image, box, (0, 0, 255), 2)
                    cv2.putText(image, "opened_hatch {:.2f}".format(confi), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255), 2)
                    
                elif int(clas) == 8: #closed_hatch
                    cv2.rectangle(image, box, (0, 255, 0), 2)
                    cv2.putText(image, "closed_hatch {:.2f}".format(confi), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0), 2)
                elif int(clas) == 1 :  # 1 은 사람 
                    person.append(box2)
          
        # 탐지하고자 하는 클래스(객체)숫자를 구하는 부분 
        class_worker_count = (results.xyxy[0].cpu().numpy()[:, -1] == 1).sum()
        class_hook_count = (results.xyxy[0].cpu().numpy()[:, -1] == 5).sum()
        class_helmet_count = (results.xyxy[0].cpu().numpy()[:, -1] == 3).sum()
        class_opened_hatch = (results.xyxy[0].cpu().numpy()[:, -1] == 7).sum()
        class_closed_hatch = (results.xyxy[0].cpu().numpy()[:, -1] == 8).sum()
        # HOOK 나 안전모 경우 객체를 탐지하고 사람 숫자와 비교하는 부분 
        if class_hook_count >= class_worker_count:
            missing_hooks = 0
        else:
            missing_hooks = abs(class_worker_count - class_hook_count)

        if class_helmet_count >= class_worker_count:
            missing_helmet = 0
        else:
            missing_helmet = abs(class_worker_count - class_helmet_count) #
        #비교해서 탐지 객체가 사람수 보다 부족할경우 unsafe가 나오게 끔 하는 부분 (탐지된 사람모두 hook 와 안전모를 착용하고 있어야 합니다)
        if missing_helmet == 0:
            finalStatus = "Safe"
        else:
            finalStatus = "UnSafe"

        if missing_hooks == 0:
            finalStatus = "Safe"
        else:
            finalStatus = "UnSafe"
        # class_opened_hatch 와 class_closed_hatch 는 각각 클래스가 다르게 훈련 되서 만약
        # class_opened_hatch가 탐지되면 unsafe class_closed_hatch 가 발견되면 safe 입니다 
        if class_opened_hatch == 1:
            finalStatus = "UnSafe"
        elif class_closed_hatch == 1:
            finalStatus = "Safe"
        #이부분은 안전모가 탐지되었을때 사람 바운딩 박스에 정확히 들어 가있는지 구하는 부분입니다
        hatDetected = False
        for perBox in person:
            hatDetected = False
            for hatBox in hat:
                if int(hatBox[0]+10) > int(perBox[0]) and int(hatBox[2]-10) < int(perBox[2]):
                    if hatBox[1] >= perBox[1] - 20:
                        hatDetected = True
            if hatDetected :
                cv2.rectangle(image,
                              [int(perBox[0]), int(perBox[1]), int(perBox[2] - perBox[0]), int(perBox[3] - perBox[1])],
                              (0, 180, 0), 2)
                cv2.putText(image, "Person with helmet", (perBox[0], perBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 180, 0), 2)
            else:
                finalStatus = "UnSafe"   
                cv2.rectangle(image,
                              [int(perBox[0]), int(perBox[1]), int(perBox[2] - perBox[0]), int(perBox[3] - perBox[1])],
                              (0, 0, 255), 2)
                cv2.putText(image, "Person without helmet", (perBox[0], perBox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)
        #최종 결과가 나오고 영상에 어떤 상황인지 어떤 객체가 부족한지 적어주는 부분입니다 
        # if finalStatus != "Safe":
        color_status = (0, 0, 255) if finalStatus != "Safe" else (0, 255, 0)
        color_hooks = (0, 255, 0) if missing_hooks == 0 else (0, 0, 255)
        color_helmet = (0, 255, 0) if missing_helmet == 0 else (0, 0, 255)

        cv2.putText(image, f"{finalStatus}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color_status, 3)
        cv2.putText(image, f"[Missing {missing_hooks} hooks]", (170, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_hooks, 2)
        cv2.putText(image, f"[Missing {missing_helmet} helmet]", (170, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_helmet, 2)

    # height, width, _ = image.shape
    # formatted_datetime = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    # cv2.putText(image, f"{formatted_datetime} ", (width-200, height-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return image, results.xyxy[0].cpu().numpy(), timestamp, finalStatus

#밑에부분은 조건 문으로 정해진 상황에 비디오를 저장하는 부분입니다 
if __name__ == "__main__":
    recording = False
    video_path = "./D_20230922_30.mp4"
    unsafe_threshold = 5  # Number of seconds to consider status as "UnSafe" # x = 5  # seconds
    y = 1   # minutes
    output_fps = 30

    ppe_model = torch.hub.load('WongKinYiu/yolov7', 'custom', './best_ppe.pt', force_reload=False)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    desired_fps = 30  # Change it to the desired value
    # cap.set(cv2.CAP_PROP_FPS, desired_fps)

    prev_time = time.time()
    video_count = 1
    out = None  # Initialize the 'out' variabl

    unsafe_timer = 0.0
    y_time_interval = y*60
    percent_count = 5
    mean_save = 70.0
    unsafe_count = 0
    safe_count = 0
    save_video = 1
    video_writer = None

    while True: 
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = datetime.datetime.now()
        status = ""
        
        if out is None:
            video_name = "./unsafe/video_{}.avi".format(video_count)
            height, width, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            out = cv2.VideoWriter(video_name, fourcc, output_fps, (width, height))

        ppe_result_image, bbox_lists, timestamp, status = detect_ppe(frame, ppe_model, timestamp)

        cv2.imshow("PPE Detection", ppe_result_image)
        if status == "UnSafe":
            unsafe_count += 1
        if status == "" or status == "Safe":
            status ="Safe"
            safe_count += 1

        current_time = time.time()
        elapsed_time = current_time - prev_time
        print("Time difference", elapsed_time)

        if elapsed_time >= y_time_interval and recording :
            if out is not None:
                out.release()  # Release the previous video writer
                out = None
                video_count += 1
                recording = False
                prev_time = current_time
                percent_count = 0
                print("stop take vdieo")

        if elapsed_time >= percent_count :  # Print percentages every 5 seconds
            total_frames = unsafe_count + safe_count
            unsafe_percent = (unsafe_count / total_frames) * 100.0
            safe_percent = (safe_count / total_frames) * 100.0
            print(f"UnSafe: {unsafe_percent:.2f}%  Safe: {safe_percent:.2f}%")
            percent_count += 5
            if status == "UnSafe" and recording == False:
                if out is not None :
                    out.write(ppe_result_image)
                    if unsafe_percent >= mean_save:
                        prev_time = current_time
                        percent_count = 0
                        recording = True
                        print("unsafe")
                        print("strat saving vdieo")
                    else:
                        if out is not None :
                            out = None
            unsafe_count = 0
            safe_count = 0

        if elapsed_time <= y_time_interval:# and elapsed_time >= unsafe_threshold:
            if out is not None:
                out.write(ppe_result_image)
        else:
            if out is not None :
                out = None
                
        if elapsed_time >= y_time_interval:
            prev_time = current_time
            percent_count = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if out is not None:
        out.release()  # Release the video writer if it's still open
        out = None

    cap.release()
    cv2.destroyAllWindows()

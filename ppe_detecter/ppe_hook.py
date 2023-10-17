import torch
import cv2
import numpy as np
import datetime
import time

def detect_ppe(image, model, timestamp, cls_detection=[True, True, True]):
    results = model(image)

    # LABELS = ['scaffold', 'worker', 'hat', 'hook']  
    person = []
    hat = []
    hook = []
    finalStatus = ""
    if np.shape(results.xyxy[0].cpu().numpy())[0] > 0:
        for (x0, y0, x1, y1, confi, clas) in results.xyxy[0].cpu().numpy():
            if confi > 0.3:
                # print(x0, y0, x1, y1, confi, clas)
                box = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
                box2 = [int(x0), int(y0), int(x1), int(y1)]
                box3 = [int(x0), int(y0), int(x1), int(y1)]
                if int(clas) == 5 and cls_detection[0] == True:
                    cv2.rectangle(image, box, (0, 130, 0), 2)
                    cv2.putText(image, "hook {:.2f}".format(confi), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 200, 0), 2)
                    hook.append(box3)
                elif int(clas) == 3 and cls_detection[1] == True:
                    cv2.rectangle(image, box, (0, 255, 0), 2)
                    cv2.putText(image, "Hard Hat {:.2f}".format(confi), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0), 2)
                    hat.append(box2)  # Add the box coordinates to the person array
                elif int(clas) == 1 and cls_detection[2] == True:
                    person.append(box2)  # Add the box coordinates to the person array

        for perBox in person:
            hatDetected = False
            hookDetected = False
            for hatBox in hat:
                if int(hatBox[0]) > int(perBox[0]) and int(hatBox[2]) < int(perBox[2]):
                    if hatBox[1] >= perBox[1] - 20:
                        hatDetected = True
            for hookbox in hook:
               if int(hookbox[0]) < int(perBox[0]) and int(hookbox[2]) < int(perBox[2]):
                    if hookbox[1] >= perBox[1] - 20:
                        hookDetected = True
            if hatDetected and hookDetected:
                cv2.rectangle(image,
                              [int(perBox[0]), int(perBox[1]), int(perBox[2] - perBox[0]), int(perBox[3] - perBox[1])],
                              (0, 180, 0), 2)
                cv2.putText(image, "Person with ppe ", (perBox[0], perBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 180, 0), 2)
                            
                if finalStatus != "UnSafe":
                    finalStatus = "Safe"
            else:
                finalStatus = "UnSafe"
                cv2.rectangle(image,
                              [int(perBox[0]), int(perBox[1]), int(perBox[2] - perBox[0]), int(perBox[3] - perBox[1])],
                              (0, 0, 255), 2)
                cv2.putText(image, "Person without ppe ", (perBox[0], perBox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)

        if finalStatus != "Safe":
            cv2.putText(image, f"{finalStatus} ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            cv2.putText(image, f"{finalStatus} ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    height, width, _ = image.shape
    formatted_datetime = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    # cv2.putText(image, f"{formatted_datetime} ", (width-200, height-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return image, results.xyxy[0].cpu().numpy(), timestamp, finalStatus

recording = False
video_path = "C:/logic/Simplatform/A_20230918_26.mp4"
if __name__ == "__main__":
    unsafe_threshold = 5  # Number of seconds to consider status as "UnSafe" # x = 5  # seconds
    y = 1   # minutes
    output_fps = 30

    ppe_model = torch.hub.load('WongKinYiu/yolov7', 'custom', './PPE/best_ppe04.pt', force_reload=False)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    desired_fps = 30  # Change it to the desired value
    # cap.set(cv2.CAP_PROP_FPS, desired_fps)

    prev_time = time.time()
    video_count = 1
    out = None  # Initialize the 'out' variable
    video_writer = None # saving vdeio
    unsafe_timer = 0.0
    y_time_interval = y*60
    percent_count = 0
    mean_save = 20.0
    unsafe_count = 0
    safe_count = 0
    save_video = 1
    while True: 
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = datetime.datetime.now()
        status = ""
        frame = cv2.resize(frame, (640,640))
        
        if video_writer is None:
            save_video += 2
            video_name = "./vdieo_save/video_{}.mp4".format(save_video)
            height, width, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(video_name, fourcc, output_fps, (width, height))

        if out is None:
            video_name = "./unsafe/video_{}.avi".format(video_count)
            height, width, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            out = cv2.VideoWriter(video_name, fourcc, output_fps, (width, height))

        ppe_result_image, bbox_lists, timestamp, status = detect_ppe(frame, ppe_model, timestamp, [True, True, True])
        video_writer.write(ppe_result_image)

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
    if video_writer is not None:
        video_writer.release()
        video_writer = None


    cap.release()
    cv2.destroyAllWindows()

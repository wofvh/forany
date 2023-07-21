import torch
import cv2
import numpy as np
import datetime
import time

def detect_ppe(image, model, timestamp, cls_detection=[True, True, True, True]):
    results = model(image)

    # LABELS = ['Boots', 'Hardhat', 'Person', 'Dog', 'Vest']   0 = Boots, 1 = Hardhat, 2 = Person, 4= Vest
    person = []
    hat = []
    finalStatus = ""
    if np.shape(results.xyxy[0].cpu().numpy())[0] > 0:
        for (x0, y0, x1, y1, confi, clas) in results.xyxy[0].cpu().numpy():
            if confi > 0.6:
                # print(x0, y0, x1, y1, confi, clas)
                box = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
                box2 = [int(x0), int(y0), int(x1), int(y1)]
                if int(clas) == 0 and cls_detection[0] == True:
                    cv2.rectangle(image, box, (0, 0, 255), 2)
                    cv2.putText(image, "Boots {:.2f}".format(confi), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255), 2)
                elif int(clas) == 1 and cls_detection[1] == True:
                    cv2.rectangle(image, box, (0, 255, 0), 2)
                    cv2.putText(image, "Hard Hat {:.2f}".format(confi), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0), 2)
                    hat.append(box2)  # Add the box coordinates to the person array
                elif int(clas) == 2 and cls_detection[2] == True:

                    person.append(box2)  # Add the box coordinates to the person array
                elif int(clas) == 4 and cls_detection[3] == True:
                    cv2.rectangle(image, box, (255, 255, 0), 2)
                    cv2.putText(image, "Vest {:.2f}".format(confi), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 0), 2)
        hatDetected = False
        for perBox in person:
            hatDetected = False
            for hatBox in hat:
                if int(hatBox[0]) > int(perBox[0]) and int(hatBox[2]) < int(perBox[2]):
                    if hatBox[1] >= perBox[1] - 20:
                        hatDetected = True
            if hatDetected:

                cv2.rectangle(image,
                              [int(perBox[0]), int(perBox[1]), int(perBox[2] - perBox[0]), int(perBox[3] - perBox[1])],
                              (0, 128, 0), 2)
                cv2.putText(image, "Person with helmet ", (perBox[0], perBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 128, 0), 2)
                if finalStatus != "UnSafe":
                    finalStatus = "Safe"
            else:
                finalStatus = "UnSafe"

                cv2.rectangle(image,
                              [int(perBox[0]), int(perBox[1]), int(perBox[2] - perBox[0]), int(perBox[3] - perBox[1])],
                              (0, 0, 255), 2)
                cv2.putText(image, "Person without helmet ", (perBox[0], perBox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)

        if finalStatus != "Safe":
            cv2.putText(image, f"{finalStatus} ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        else:
            cv2.putText(image, f"{finalStatus} ", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 0), 2)
    height, width, _ = image.shape
    formatted_datetime = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(image, f"{formatted_datetime} ", (width-200, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return image, results.xyxy[0].cpu().numpy(), timestamp, finalStatus

if __name__ == "__main__":
    unsafe_threshold = 5  # Number of seconds to consider status as "UnSafe" # x = 5  # seconds
    y = 1   # minutes
    output_fps = 30

    ppe_model = torch.hub.load('WongKinYiu/yolov7', 'custom', './PPE/best-1.pt', force_reload=False)
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    desired_fps = 30  # Change it to the desired value
    cap.set(cv2.CAP_PROP_FPS, desired_fps)

    video_count = 1
    out = None  # Initialize the 'out' variable
    unsafe_timer = 0.0
    y_time_interval = y*60
    prev_time = time.time()
    unsafe_count = 0 
    safe_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = datetime.datetime.now()
        status = ""

        if out is None:
            video_name = "video_{}.avi".format(video_count)
            height, width, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*"MJPG") #코덱
            out = cv2.VideoWriter(video_name, fourcc, output_fps, (width, height))

        ppe_result_image, bbox_lists, timestamp, status = detect_ppe(frame, ppe_model, timestamp, [True, True, True, True])

        cv2.imshow("PPE Detection", ppe_result_image)
        if status == "":
            status ="Safe"

        # Calculate UnSafe and Safe counts
        if status == "UnSafe":
            unsafe_count += 1
            # 추가: UnSafe 비디오 저장 조건 검사
            # if unsafe_count >= unsafeframe_limit:
            #     save_unsafe_video = True
        elif status == "Safe":
            safe_count += 1

        current_time = time.time()
        elapsed_time = current_time - prev_time
        print("Time difference", elapsed_time)
        
        # Calculate percentages for UnSafe and Safe
        if elapsed_time >= 5 :  # Print percentages every 5 seconds
            total_frames = unsafe_count + safe_count
            unsafe_percent = (unsafe_count / total_frames) * 100
            safe_percent = (safe_count / total_frames) * 100
            print(f"UnSafe: {unsafe_percent:.2f}%  Safe: {safe_percent:.2f}%")
            unsafe_count = 0
            safe_count = 0
            prev_time = current_time

            # Check if UnSafe percentage is 70% or more
            if unsafe_percent >= 70:
                # Save the previous video for 1 minute (60 seconds)
                if out is not None:
                    out.release()  # Release the previous video writer
                video_count += 1
                video_name = "video_{}_1min_{}.avi".format(video_count, timestamp.strftime("%Y%m%d%H%M%S"))
                height, width, _ = frame.shape
                out = cv2.VideoWriter(video_name, fourcc, output_fps, (width, height))
                prev_time = current_time


        if elapsed_time >= y_time_interval:
            if out is not None:
                out.release()  # Release the previous video writer
                out = None
                video_count += 1
                prev_time = current_time

                # Reset counts for the next interval
                unsafe_count = 0
                safe_count = 0
                
                if status == "UnSafe":
                    print("Current Status: UnSafe")
                else:
                    print("Current Status: Safe")


        if status == "UnSafe" and elapsed_time <= unsafe_threshold:

            if out is not None:
                out.write(ppe_result_image)
        elif status == "Safe" and elapsed_time <= unsafe_threshold:
            if out is not None:
                out = None
            prev_time = current_time
        elif elapsed_time <= y_time_interval: # and elapsed_time >= unsafe_threshold:
            if out is not None:
                out.write(ppe_result_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if out is not None:
        out.release()  # Release the video writer if it's still open
        out = None

    cap.release()
    cv2.destroyAllWindows()

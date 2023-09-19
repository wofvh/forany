import cv2
import os

# 동영상 파일들이 있는 폴더 경로
video_folder = 'F:/total_dataset\data_20230918/'
output_folder = 'F:/total_dataset\data_20230918\image/'

video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi', '.mkv'))]


for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"{video_path} 동영상 파일을 열 수 없습니다.")
        continue

    # 해당 동영상의 이미지 저장 폴더 생성
    video_name = os.path.splitext(video_file)[0]

    # 해당 동영상의 이미지 저장 폴더 생성 (동영상 파일 이름과 동일한 폴더 생성)
    video_output_folder = os.path.join(output_folder, video_name)
    os.makedirs(video_output_folder, exist_ok=True)

    # 이미지 저장을 위한 카운터 초기화
    frame_count = 1

    while True:
        # 프레임 읽기
        ret, frame = cap.read()

        if not ret:
            break

        # 이미지 파일 이름 생성
        image_name = f'{video_name}_{frame_count:03d}.jpg'
        image_path = os.path.join(video_output_folder, image_name)

        cv2.imwrite(image_path, frame)
        print(f'{image_name} 이미지로 저장했습니다.')

        frame_count += 1

    # 작업 완료 후 해제
    cap.release()
    cv2.destroyAllWindows()

print("작업 완료")

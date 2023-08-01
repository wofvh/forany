import math

# YOLO 객체 탐지 결과로부터 바운딩 박스 좌표값을 추출한 리스트
detections = [
    {"x": 100, "y": 200, "width": 50, "height": 50},
    {"x": 300, "y": 400, "width": 60, "height": 60}
    # 다른 바운딩 박스들...
]

def euclidean_distance(p1, p2):
    return math.sqrt((p2["x"] - p1["x"]) ** 2 + (p2["y"] - p1["y"]) ** 2)

# 두 개의 바운딩 박스 사이의 거리 계산
distance_between_boxes = euclidean_distance(detections[0], detections[1])
print("Distance between boxes:", distance_between_boxes)

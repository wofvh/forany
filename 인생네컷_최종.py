from PIL import Image, ImageDraw, ImageFilter,ImageFont
import cv2
import numpy as np
import matplotlib as plt


image = cv2.imread('C:/allmodel/opencv/html/26.jpg')
color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
top_left = np.copy(color_coverted)
top_right = np.copy(color_coverted)
bottom_left = np.copy(color_coverted)
bottom_right = np.copy(color_coverted)

top_left_resized = cv2.resize(top_left, (640, 460))
top_right_resized = cv2.resize(top_right, (640, 460))
bottom_left_resized = cv2.resize(bottom_left, (640, 500))
bottom_right_resized = cv2.resize(bottom_right, (640, 500))

# 결과 이미지를 생성합니다.
combined_image = np.zeros((960, 1280, 3), dtype=np.uint8)

# 작은 이미지를 결과 이미지에 복사합니다.
combined_image[:460, :640] = top_left_resized
combined_image[:460, 640:1280] = top_right_resized
combined_image[460:960, :640] = bottom_left_resized
combined_image[460:960, 640:1280] = bottom_right_resized

print(combined_image.shape)

dst = cv2.resize(combined_image, dsize=(640, 480), interpolation=cv2.INTER_AREA)

d1=Image.fromarray(dst)
ft = ImageFont.truetype("./html/KGPrimaryWhimsy.ttf", 35) #폰트 제일 괜찮았음 
d = ImageDraw.Draw(d1)

width = 640
height = 480
top_border_width = 10
mid_width = 5
mid2_width = 5
bottom_border_width = 120
left_border_width = 10
right_border_width = 10

d.line([(width-1, 0), (width-1, height-1)], fill="#ff9d73", width=right_border_width)
d.line([(0, 230), (640, 230)], fill="#ff9d73", width=mid_width)
d.line([(320, 0), (320, 480)], fill="#ff9d73", width=mid2_width)
d.line([(0, 0), (width-1, 0)], fill="#ff9d73", width=top_border_width)
d.line([(width-1, height-1), (0, height-1)], fill="#ff9d73", width=bottom_border_width)
d.line([(0, height-1), (0, 0)], fill="#ff9d73", width=left_border_width)
d.text((200,430), "have a nice day!", font=ft, fill="white")
# d.line([10, 10, 10 + 500, 10, 10 + 150, 10 + 20, 10, 10 + 20, 10, 10], width=25,  fill="#ff9d73")
# d.rectangle([(75, 300), (100,300)], fill="#ff9d73")
# d.rectangle([(0, 0), (width-border_width, height-border_height)], outline="#ff9d73", width=10)

numpy_image = np.array(d1)  
img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

# 결과 이미지를 화면에 표시합니다.
cv2.imshow('Combined Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("./html/20.jpg", img)

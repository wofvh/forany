from PIL import Image, ImageDraw, ImageFilter,ImageFont
import cv2
import numpy as np
import pixellib
from pixellib.tune_bg import alter_bg
from tensorflow.python.keras.layers import BatchNormalization
# 원본 이미지 열기
# original_image = Image.open("./html/26.jpg")

original_image = cv2.imread("./html/26.jpg")
color_coverted = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

#이미지 4분할로 나누기 160 * 120 (4)          (640 * 480 reshape 해줘야함 같은이미지로 나누고 몇등분 할껀지)
#나눈후 저장 ? or 바로 나눌수 있는지 

#나눈 4개 이미지 다시 한게 이미지 총 640 * 480 합처서 붙혀줘야함   


## PIL로 변경 해주고 이미에 d.line 2개 더 추가해서 합친이미지에다가 분할 해야함 
d1=Image.fromarray(color_coverted)
ft = ImageFont.truetype("./html/KGPrimaryWhimsy.ttf", 50)
d = ImageDraw.Draw(d1)
# canvas = Image.new("RGB", (300, 300), color="#fff")
width = 640
height = 480
top_border_width = 25
bottom_border_width = 190
left_border_width = 30
right_border_width = 30

d.line([(0, 0), (width-1, 0)], fill="#ff9d73", width=top_border_width)
d.line([(width-1, 0), (width-1, height-1)], fill="#ff9d73", width=right_border_width)
d.line([(width-1, height-1), (0, height-1)], fill="#ff9d73", width=bottom_border_width)
d.line([(0, height-1), (0, 0)], fill="#ff9d73", width=left_border_width)
d.text((190,400), "have a nice day!", font=ft, fill="white")
# d.line([10, 10, 10 + 500, 10, 10 + 150, 10 + 20, 10, 10 + 20, 10, 10], width=25,  fill="#ff9d73")
# d.rectangle([(75, 300), (100,300)], fill="#ff9d73")
# d.rectangle([(0, 0), (width-border_width, height-border_height)], outline="#ff9d73", width=10)

numpy_image = np.array(d1)  
img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
# d.line([10, 10, 290, 10, 290, 290, 10, 290, 10, 10], width=25,  fill="#ff9d73")


cv2.imshow("PIL2OpenCV",img)
cv2.waitKey()
cv2.imwrite("./html/18.jpg", img)



# change_bg = alter_bg()
# change_bg.load_pascalvoc_model("./html/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5")
# output = change_bg.change_bg_img(f_image_path = "./html/18.jpg",b_image_path = "./html/yellow.jpg",output_image_name="./html/00.jpg")
# cv2.imwrite( output)




# d1.show()
# print(d)


# cv2.putText(img, text,font,(50,300),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),1,cv2.LINE_AA)
# cv2.imshow("img",d)
# cv2.waitKey()
# cv2.destroyAllWindows()

# width = 10
# height = 10

# border_color = (255,0,0)
# border_width = 10

# # 배경 이미지 생성
# background_image = Image.new("RGB", (width + border_width * 2, height + border_width * 2), border_color)
# final_image = Image.alpha_composite(background_image)

# final_image.save("./html/00.jpg")
# # # 이미지 채널 분리
# # r, g, b = original_image.split()
# # _, _, a = original_image.split() if original_image == "RGBA" else (255, 255, 255)  # 이미지가 RGBA 모드인 경우 알파 채널 분리

# # # 채널 병합
# # merged_image = Image.merge("RGBA", (r, g, b, a))

# # # 새로운 이미지 생성
# # final_image = Image.alpha_composite(background_image, merged_image)

# # # 이미지 저장
# # final_image.save("./html/00.jpg")





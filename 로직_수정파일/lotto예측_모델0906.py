import pandas as pd

path = "C:/all_data/"

data = pd.read_csv(path + "lotto.csv", index_col=0)

print(data.head())
# 0  1083    3    7   14   15   22   38   17  1713084525      15  59482102      72
# 1  1082   21   26   27   32   34   42   31  3720489643       7  70009214      62
# 2  1081    1    9   16   23   24   38   17  2343892944      11  46708012      92
# 3  1080   13   16   23   31   36   44   38  3639444429       7  51780714      82

xyxy_1 = hat['xyxy']
center_1_y = (xyxy_1[1] + xyxy_1[3]) / 2

import torch
print(torch.__version__) # 설치된 PyTorch 버전을 확인합니다. 1.12 이상이어야 합니다.

print(torch.backends.mps.is_built()) # MPS 장치를 지원하도록 빌드되어있는지 확인합니다. True여야 합니다.

import torch

mps_device = torch.device("mps")

# MPS 장치에 바로 tensor를 생성합니다.
x = torch.ones(5, device=mps_device)
# 또는
x = torch.ones(5, device="mps")

# GPU 상에서 연산을 진행합니다.
y = x * 2

# 또는, 다른 장치와 마찬가지로 MPS로 이동할 수도 있습니다.
model = YourFavoriteNet()  # 어떤 모델의 객체를 생성한 뒤,
model.to(mps_device)       # MPS 장치로 이동합니다.

# 이제 모델과 텐서를 호출하면 GPU에서 연산이 이뤄집니다.
pred = model(x)
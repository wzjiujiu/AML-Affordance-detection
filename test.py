import torch

#This part used to check the right version of torch for correspnding the kaolin
print(torch.__version__)  # PyTorch 版本
print(torch.version.cuda) # CUDA 版本
print(torch.cuda.is_available())  # 检查 CUDA 是否可用
import torch
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 빌드 버전: {torch.version.cuda}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
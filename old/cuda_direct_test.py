#!/usr/bin/env python3
"""
PyTorch 없이 직접 CUDA 라이브러리를 테스트하는 스크립트
"""

import ctypes
import os
import sys

def test_cuda_library_direct():
    """CUDA 라이브러리를 직접 로드해서 테스트"""
    print("=== 직접 CUDA 라이브러리 테스트 ===")
    
    try:
        # libcuda.so 직접 로드
        cuda_lib = ctypes.CDLL('libcuda.so.1')
        print("✅ libcuda.so.1 로드 성공")
        
        # cuInit 함수 호출
        cu_init = cuda_lib.cuInit
        cu_init.argtypes = [ctypes.c_uint]
        cu_init.restype = ctypes.c_int
        
        result = cu_init(0)
        if result == 0:  # CUDA_SUCCESS
            print("✅ CUDA 초기화 성공")
            
            # GPU 개수 확인
            cu_device_get_count = cuda_lib.cuDeviceGetCount
            cu_device_get_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
            cu_device_get_count.restype = ctypes.c_int
            
            device_count = ctypes.c_int()
            result = cu_device_get_count(ctypes.byref(device_count))
            
            if result == 0:
                print(f"✅ GPU 개수: {device_count.value}")
                return True
            else:
                print(f"❌ GPU 개수 확인 실패: {result}")
        else:
            print(f"❌ CUDA 초기화 실패: {result}")
            
    except Exception as e:
        print(f"❌ CUDA 라이브러리 테스트 실패: {e}")
    
    return False

def test_environment_fix():
    """환경 변수 설정 후 PyTorch 재테스트"""
    print("\n=== 환경 변수 설정 후 PyTorch 테스트 ===")
    
    # 환경 변수 설정
    os.environ['CUDA_HOME'] = '/usr/local/cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # LD_LIBRARY_PATH 설정
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    new_paths = [
        '/usr/local/cuda/lib64',
        '/usr/local/cuda/targets/x86_64-linux/lib',
        '/usr/lib/x86_64-linux-gnu'
    ]
    
    all_paths = new_paths + ([current_ld_path] if current_ld_path else [])
    os.environ['LD_LIBRARY_PATH'] = ':'.join(all_paths)
    
    print(f"→ CUDA_HOME: {os.environ['CUDA_HOME']}")
    print(f"→ CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    print(f"→ LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}")
    
    # PyTorch 다시 import (새로운 프로세스에서만 효과적)
    print("\n⚠️  환경 변수 변경 후에는 Python을 재시작해야 효과적입니다.")
    print("다음 명령어로 새 Python 세션에서 테스트해보세요:")
    print("python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\"")

def check_container_nvidia_setup():
    """컨테이너의 NVIDIA 설정 확인"""
    print("\n=== 컨테이너 NVIDIA 설정 확인 ===")
    
    # nvidia-ml-py3 라이브러리로 테스트
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        print(f"✅ pynvml을 통한 GPU 개수: {device_count}")
        
        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            print(f"✅ GPU 이름: {name}")
            return True
            
    except ImportError:
        print("→ pynvml 라이브러리가 설치되지 않음")
        print("설치: pip install pynvml")
    except Exception as e:
        print(f"❌ pynvml 테스트 실패: {e}")
    
    return False

def main():
    """메인 테스트 함수"""
    print("🔍 CUDA 직접 테스트를 시작합니다...\n")
    
    # 1. 직접 CUDA 라이브러리 테스트
    cuda_direct_ok = test_cuda_library_direct()
    
    # 2. 환경 변수 설정
    test_environment_fix()
    
    # 3. 컨테이너 NVIDIA 설정 확인
    nvidia_ok = check_container_nvidia_setup()
    
    print("\n=== 테스트 결과 요약 ===")
    print(f"직접 CUDA 라이브러리: {'✅' if cuda_direct_ok else '❌'}")
    print(f"NVIDIA 라이브러리: {'✅' if nvidia_ok else '❌'}")
    
    if cuda_direct_ok:
        print("\n💡 CUDA 라이브러리는 정상이므로 환경 설정 문제일 가능성이 높습니다.")
        print("다음 명령어들을 실행해보세요:")
        print("1. export CUDA_HOME=/usr/local/cuda")
        print("2. export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH")
        print("3. export CUDA_VISIBLE_DEVICES=0")
        print("4. python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\"")
    else:
        print("\n💡 컨테이너가 GPU에 제대로 접근하지 못하고 있습니다.")
        print("컨테이너 실행 시 --gpus all 옵션이나 NVIDIA Container Runtime이 필요합니다.")

if __name__ == "__main__":
    main()
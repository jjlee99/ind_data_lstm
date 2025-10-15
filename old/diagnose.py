#!/usr/bin/env python3
"""
CUDA 환경 진단 스크립트
PyTorch에서 CUDA를 사용할 수 없는 문제를 진단합니다.
"""

import os
import subprocess
import sys

def run_command(cmd):
    """명령어를 실행하고 결과를 반환"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), -1

def check_nvidia_driver():
    """NVIDIA 드라이버 확인"""
    print("=== 1. NVIDIA 드라이버 확인 ===")
    stdout, stderr, code = run_command("nvidia-smi --query-gpu=driver_version --format=csv,noheader")
    if code == 0:
        print(f"✅ NVIDIA 드라이버: {stdout}")
    else:
        print(f"❌ nvidia-smi 실행 실패: {stderr}")
    print()

def check_cuda_installation():
    """CUDA 설치 확인"""
    print("=== 2. CUDA 설치 확인 ===")
    
    # nvcc 확인
    stdout, stderr, code = run_command("nvcc --version")
    if code == 0:
        print("✅ nvcc 사용 가능")
        version_line = [line for line in stdout.split('\n') if 'release' in line.lower()]
        if version_line:
            print(f"→ {version_line[0].strip()}")
    else:
        print("❌ nvcc 사용 불가 (CUDA toolkit 미설치 또는 PATH 문제)")
    
    # CUDA 라이브러리 경로 확인
    cuda_paths = [
        "/usr/local/cuda",
        "/usr/local/cuda-12.8",
        "/usr/local/cuda-12",
        "/opt/cuda"
    ]
    
    print("\n→ CUDA 설치 경로 확인:")
    for path in cuda_paths:
        if os.path.exists(path):
            print(f"  ✅ {path}")
        else:
            print(f"  ❌ {path}")
    print()

def check_environment_variables():
    """환경 변수 확인"""
    print("=== 3. 환경 변수 확인 ===")
    
    env_vars = [
        "CUDA_HOME",
        "CUDA_ROOT",
        "CUDA_PATH",
        "PATH",
        "LD_LIBRARY_PATH",
        "CUDA_VISIBLE_DEVICES"
    ]
    
    for var in env_vars:
        value = os.environ.get(var, "설정되지 않음")
        if var == "PATH":
            # PATH에서 CUDA 관련 경로만 추출
            cuda_paths = [p for p in value.split(':') if 'cuda' in p.lower()]
            if cuda_paths:
                print(f"→ {var} (CUDA 관련): {':'.join(cuda_paths)}")
            else:
                print(f"→ {var}: CUDA 관련 경로 없음")
        elif var == "LD_LIBRARY_PATH":
            if value != "설정되지 않음":
                cuda_paths = [p for p in value.split(':') if 'cuda' in p.lower()]
                if cuda_paths:
                    print(f"→ {var} (CUDA 관련): {':'.join(cuda_paths)}")
                else:
                    print(f"→ {var}: CUDA 관련 경로 없음")
            else:
                print(f"→ {var}: {value}")
        else:
            print(f"→ {var}: {value}")
    print()

def check_pytorch_installation():
    """PyTorch 설치 확인"""
    print("=== 4. PyTorch 설치 확인 ===")
    
    try:
        import torch
        print(f"✅ PyTorch 버전: {torch.__version__}")
        print(f"→ CUDA 빌드 버전: {torch.version.cuda}")
        print(f"→ cuDNN 버전: {torch.backends.cudnn.version()}")
        
        # CUDA 사용 가능 여부 (경고 메시지 포함)
        print("\n→ CUDA 사용 가능 여부 테스트:")
        cuda_available = torch.cuda.is_available()
        print(f"  torch.cuda.is_available(): {cuda_available}")
        
        if cuda_available:
            print(f"  장치 수: {torch.cuda.device_count()}")
            print(f"  현재 장치: {torch.cuda.current_device()}")
            print(f"  장치 이름: {torch.cuda.get_device_name(0)}")
        
    except ImportError:
        print("❌ PyTorch가 설치되지 않음")
    except Exception as e:
        print(f"❌ PyTorch 확인 중 오류: {e}")
    print()

def check_cuda_libraries():
    """CUDA 라이브러리 확인"""
    print("=== 5. CUDA 라이브러리 확인 ===")
    
    libraries = [
        "libcuda.so",
        "libcublas.so",
        "libcurand.so",
        "libcusparse.so"
    ]
    
    for lib in libraries:
        stdout, stderr, code = run_command(f"ldconfig -p | grep {lib}")
        if code == 0 and stdout:
            print(f"✅ {lib} 발견")
            # 첫 번째 라이브러리 경로만 표시
            first_lib = stdout.split('\n')[0].split('=>')[-1].strip()
            print(f"  → {first_lib}")
        else:
            print(f"❌ {lib} 없음")
    print()

def main():
    """메인 진단 함수"""
    print("🔍 CUDA 환경 진단을 시작합니다...\n")
    
    check_nvidia_driver()
    check_cuda_installation()
    check_environment_variables()
    check_pytorch_installation()
    check_cuda_libraries()
    
    print("=== 진단 완료 ===")
    print("위 정보를 바탕으로 문제를 해결할 수 있습니다.")
    print("\n주요 해결 방법:")
    print("1. CUDA Toolkit 설치 확인")
    print("2. 환경 변수 설정 (CUDA_HOME, PATH, LD_LIBRARY_PATH)")
    print("3. PyTorch 재설치 (올바른 CUDA 버전)")
    print("4. 시스템 재부팅")

if __name__ == "__main__":
    main()
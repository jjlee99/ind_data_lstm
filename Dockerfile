# Base image with CUDA support (Ubuntu 22.04, CUDA 11.8, cuDNN 8)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# Install system dependencies, Python, pip, and Git
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 the default python and pip
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    pip install --upgrade pip

# Install Python libraries
# Using a specific PyTorch version compatible with CUDA 11.8
# First, install general packages from PyPI
RUN pip install \
    jupyterlab \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    prophet

# Then, install PyTorch and related packages from the specified index for CUDA compatibility
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set working directory inside the container
WORKDIR /workspace


# Set working directory inside the container
WORKDIR /workspace

# Expose JupyterLab port
EXPOSE 8888

# Start JupyterLab when container launches
# Allows root access and disables token/password for convenience in a dev environment.
# For production, ensure you configure security appropriately.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]


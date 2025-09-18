#!/bin/bash

# GPU Setup Script for AI Content Creator
# This script helps set up NVIDIA GPU support for Docker

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[GPU Setup] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[GPU Setup] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[GPU Setup] ERROR: $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[GPU Setup] $1${NC}"
}

# Check if NVIDIA GPU is available
check_gpu() {
    log "Checking for NVIDIA GPU..."

    if command -v nvidia-smi &> /dev/null; then
        log "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    else
        error "nvidia-smi not found. Please install NVIDIA drivers first."
    fi
}

# Check NVIDIA Docker support
check_nvidia_docker() {
    log "Checking NVIDIA Docker support..."

    # Check if nvidia-docker2 is installed
    if dpkg -l | grep -q nvidia-docker2; then
        log "nvidia-docker2 is installed"
    else
        warn "nvidia-docker2 not found. Will attempt to install."
        install_nvidia_docker
    fi

    # Check if nvidia-container-runtime is available
    if docker info | grep -q nvidia; then
        log "NVIDIA Container Runtime is configured"
    else
        warn "NVIDIA Container Runtime not configured properly"
        configure_nvidia_docker
    fi
}

# Install NVIDIA Docker
install_nvidia_docker() {
    log "Installing NVIDIA Docker support..."

    # Remove old versions
    sudo apt-get remove -y docker docker-engine docker.io containerd runc nvidia-docker

    # Update package list
    sudo apt-get update

    # Install prerequisites
    sudo apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        lsb-release

    # Add Docker GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

    # Add Docker repository
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    # Add NVIDIA Docker repository
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

    # Update package list
    sudo apt-get update

    # Install Docker and NVIDIA Docker
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io nvidia-docker2

    # Add user to docker group
    sudo usermod -aG docker $USER

    log "NVIDIA Docker installed successfully"
}

# Configure NVIDIA Docker
configure_nvidia_docker() {
    log "Configuring NVIDIA Docker..."

    # Create or update daemon.json
    sudo mkdir -p /etc/docker

    if [ -f /etc/docker/daemon.json ]; then
        # Backup existing config
        sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.backup
    fi

    # Create new daemon.json with NVIDIA runtime
    sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "100m",
        "max-file": "3"
    }
}
EOF

    # Restart Docker
    sudo systemctl restart docker

    log "NVIDIA Docker configured successfully"
}

# Test GPU in Docker
test_gpu_docker() {
    log "Testing GPU access in Docker..."

    # Test NVIDIA runtime
    if docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi; then
        log "GPU test successful!"
    else
        error "GPU test failed. Please check your configuration."
    fi

    # Test PyTorch GPU support
    log "Testing PyTorch GPU support..."
    docker run --rm --gpus all -v $(pwd):/workspace \
        pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime \
        python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('CUDA not available')
"
}

# Optimize GPU settings
optimize_gpu() {
    log "Optimizing GPU settings for AI workloads..."

    # Set persistence mode
    sudo nvidia-smi -pm 1

    # Set power limit (adjust as needed)
    # sudo nvidia-smi -pl 300  # Set 300W power limit

    # Set memory and GPU clocks to maximum
    sudo nvidia-smi -ac $(nvidia-smi --query-supported-clocks=memory,graphics --format=csv,noheader,nounits | tail -1 | tr ',' ' ')

    log "GPU optimization completed"
}

# Create GPU monitoring script
create_monitoring() {
    log "Creating GPU monitoring script..."

    cat > gpu-monitor.sh << 'EOF'
#!/bin/bash

# GPU Monitoring Script for AI Content Creator

echo "GPU Monitoring - Press Ctrl+C to exit"
echo "=================================="

while true; do
    clear
    echo "$(date)"
    echo "=================================="

    # GPU status
    nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv

    echo ""
    echo "Docker GPU Containers:"
    docker ps --format "table {{.Names}}\t{{.Image}}\t{{.Status}}" | grep -E "(ai-content|nvidia|cuda|pytorch)"

    echo ""
    echo "Press Ctrl+C to exit..."
    sleep 5
done
EOF

    chmod +x gpu-monitor.sh
    log "GPU monitoring script created: ./gpu-monitor.sh"
}

# Main setup function
main() {
    case "${1:-setup}" in
        "setup")
            info "Setting up GPU support for AI Content Creator..."
            check_gpu
            check_nvidia_docker
            configure_nvidia_docker
            test_gpu_docker
            optimize_gpu
            create_monitoring
            log "GPU setup completed successfully!"
            log "You may need to logout and login again for group changes to take effect."
            ;;
        "test")
            test_gpu_docker
            ;;
        "monitor")
            if [ -f gpu-monitor.sh ]; then
                ./gpu-monitor.sh
            else
                error "Monitoring script not found. Run 'setup' first."
            fi
            ;;
        "optimize")
            optimize_gpu
            ;;
        "status")
            if command -v nvidia-smi &> /dev/null; then
                nvidia-smi
            else
                error "nvidia-smi not available"
            fi
            ;;
        "help"|"--help"|"-h")
            echo "GPU Setup Script for AI Content Creator"
            echo
            echo "Usage: $0 [COMMAND]"
            echo
            echo "Commands:"
            echo "  setup     Full GPU setup (default)"
            echo "  test      Test GPU in Docker"
            echo "  monitor   Monitor GPU usage"
            echo "  optimize  Optimize GPU settings"
            echo "  status    Show GPU status"
            echo "  help      Show this help"
            ;;
        *)
            error "Unknown command: $1. Use '$0 help' for usage."
            ;;
    esac
}

# Run with provided arguments
main "$@"
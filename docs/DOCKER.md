# Docker Deployment Guide - AI Content Creator

This guide covers Docker containerization and deployment options for the AI Content Creator platform.

## 🐳 Quick Start

### Prerequisites

1. **Docker & Docker Compose**
   ```bash
   # Install Docker (Ubuntu/Debian)
   curl -fsSL https://get.docker.com -o get-docker.sh
   sh get-docker.sh

   # Install Docker Compose
   sudo apt-get install docker-compose-plugin
   ```

2. **NVIDIA Docker (for GPU support)**
   ```bash
   # Run GPU setup script
   ./scripts/setup-gpu.sh
   ```

3. **System Requirements**
   - 8GB+ RAM (16GB recommended)
   - 20GB+ free disk space
   - GPU with 4GB+ VRAM (optional but recommended)

### One-Command Deployment

```bash
# Clone and deploy
git clone https://github.com/your-username/ai-content-creator.git
cd ai-content-creator
./scripts/deploy.sh deploy
```

## 🏗️ Architecture Overview

### Multi-Service Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│     Nginx       │────│   FastAPI API    │────│    Workers      │
│  Load Balancer  │    │   (Web + API)    │    │ (GPU Processing)│
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         │              ┌─────────────────┐               │
         │              │     Redis       │               │
         └──────────────│ (Cache + Queue) │───────────────┘
                        └─────────────────┘
                                 │
                        ┌─────────────────┐
                        │   PostgreSQL    │
                        │   (Metadata)    │
                        └─────────────────┘
```

### Container Images

| Service | Base Image | Purpose | GPU |
|---------|------------|---------|-----|
| `api` | `nvidia/cuda:11.8-devel-ubuntu22.04` | Main API server | ✓ |
| `worker` | Same as API | Background processing | ✓ |
| `nginx` | `nginx:alpine` | Reverse proxy | ✗ |
| `redis` | `redis:7-alpine` | Cache + task queue | ✗ |
| `postgres` | `postgres:15-alpine` | Metadata storage | ✗ |

## 🚀 Deployment Options

### 1. Production Deployment

```bash
# Full production setup
./scripts/deploy.sh deploy

# Services will be available at:
# - Web Interface: http://localhost
# - API Documentation: http://localhost/docs
# - Monitoring: http://localhost:3000
```

**Features:**
- Multi-stage Docker builds for optimized images
- NGINX reverse proxy with SSL support
- Redis for caching and task queuing
- PostgreSQL for metadata persistence
- Prometheus + Grafana monitoring
- Auto-scaling workers

### 2. Development Setup

```bash
# Development environment with hot reload
./scripts/deploy.sh dev

# Additional services:
# - Jupyter Notebooks: http://localhost:8888
# - Direct API access: http://localhost:8000
```

**Features:**
- Code hot reloading
- Jupyter notebooks for experimentation
- Debug logging enabled
- Development tools included

### 3. GPU-Optimized Deployment

```bash
# Setup GPU support first
./scripts/setup-gpu.sh setup

# Deploy with GPU acceleration
./scripts/deploy.sh deploy

# Monitor GPU usage
./scripts/setup-gpu.sh monitor
```

## 📋 Configuration

### Environment Variables

Copy and customize the environment file:

```bash
cp .env.docker .env
# Edit .env with your settings
```

**Key Configuration Options:**

```bash
# === Performance ===
MAX_WORKERS=4                    # Number of worker processes
MAX_CONCURRENT_TASKS=2          # Concurrent GPU tasks
PROCESSING_TIMEOUT=3600         # Task timeout (seconds)

# === GPU Settings ===
CUDA_VISIBLE_DEVICES=0          # GPU device to use
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# === Storage ===
DATA_PATH=/app/data             # Data directory
OUTPUT_PATH=/app/data/output    # Output files
TEMP_PATH=/app/data/temp        # Temporary files

# === External APIs ===
AZURE_SPEECH_KEY=your_key       # Azure TTS (optional)
OPENAI_API_KEY=your_key         # OpenAI API (optional)
```

### Volume Mounts

```yaml
volumes:
  - ./data:/home/app/data           # Persistent data
  - ./logs:/home/app/logs           # Log files
  - ai_models:/home/app/.cache      # Cached AI models
```

## 🛠️ Management Commands

### Deployment Script

```bash
./scripts/deploy.sh COMMAND [OPTIONS]
```

**Available Commands:**

| Command | Description |
|---------|-------------|
| `deploy` | Full deployment (build + start + health check) |
| `start` | Start all services |
| `stop` | Stop all services |
| `restart` | Restart services |
| `status` | Show service status |
| `logs [service]` | View logs for specific service |
| `build [--no-cache]` | Build Docker images |
| `update` | Update services with backup |
| `backup` | Create full backup |
| `restore <backup>` | Restore from backup |
| `monitor` | Real-time service monitoring |
| `dev` | Setup development environment |
| `cleanup [--all]` | Clean up containers/images |
| `health` | Run health checks |

### Examples

```bash
# Check service status
./scripts/deploy.sh status

# View API logs
./scripts/deploy.sh logs api

# Create backup before update
./scripts/deploy.sh backup
./scripts/deploy.sh update

# Monitor resource usage
./scripts/deploy.sh monitor

# Development with hot reload
./scripts/deploy.sh dev
```

## 🔧 Advanced Configuration

### 1. Custom Docker Compose

Create `docker-compose.override.yml` for customizations:

```yaml
version: '3.8'

services:
  api:
    environment:
      - CUSTOM_VAR=value
    ports:
      - "8001:8000"  # Different port

  worker:
    deploy:
      replicas: 4    # More workers
      resources:
        limits:
          memory: 8G
```

### 2. SSL/HTTPS Configuration

```bash
# Generate SSL certificates
mkdir -p nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/key.pem \
  -out nginx/ssl/cert.pem

# Enable HTTPS in nginx.conf
# Uncomment HTTPS server block
```

### 3. External Database

```yaml
# Use external PostgreSQL
services:
  api:
    environment:
      - DATABASE_URL=postgresql://user:pass@external-host:5432/db

  # Remove postgres service
  # postgres: ...
```

### 4. Horizontal Scaling

```yaml
# Scale specific services
services:
  worker:
    deploy:
      replicas: 4

  api:
    deploy:
      replicas: 2
```

## 📊 Monitoring & Observability

### Built-in Monitoring Stack

- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **NGINX**: Access logs and metrics
- **Application**: Custom metrics and health checks

### Access Monitoring

```bash
# Grafana dashboard
open http://localhost:3000
# Login: admin / admin123

# Prometheus metrics
open http://localhost:9090

# Health check endpoint
curl http://localhost/health
```

### Custom Metrics

```python
# Add custom metrics in your application
from prometheus_client import Counter, Histogram

TASK_COUNTER = Counter('ai_tasks_total', 'Total AI tasks processed', ['task_type'])
PROCESSING_TIME = Histogram('ai_processing_seconds', 'Time spent processing')

# Use in your code
TASK_COUNTER.labels(task_type='upscale').inc()
with PROCESSING_TIME.time():
    process_video()
```

## 🔍 Troubleshooting

### Common Issues

#### 1. GPU Not Detected

```bash
# Check GPU setup
./scripts/setup-gpu.sh status

# Test GPU in container
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi

# Check Docker daemon configuration
cat /etc/docker/daemon.json
```

#### 2. Out of Memory

```bash
# Check memory usage
docker stats

# Reduce workers or tile size
# Edit .env:
MAX_WORKERS=2
TILE_SIZE=256
```

#### 3. Permission Issues

```bash
# Fix data directory permissions
sudo chown -R $USER:$USER data/ logs/

# Fix Docker socket permissions
sudo usermod -aG docker $USER
# Logout and login again
```

#### 4. Model Download Issues

```bash
# Pre-download models
docker-compose exec api python -c "
import whisper
whisper.load_model('base')
"

# Check internet connectivity in container
docker-compose exec api curl -I https://huggingface.co
```

### Debug Mode

```bash
# Enable debug logging
export DEBUG=True
export LOG_LEVEL=DEBUG

# Restart with debug
./scripts/deploy.sh restart

# View detailed logs
./scripts/deploy.sh logs api
```

### Health Checks

```bash
# Manual health check
curl -f http://localhost/health

# Check all services
./scripts/deploy.sh health

# Check individual components
curl http://localhost:9090/metrics  # Prometheus
curl http://localhost:6379/ping     # Redis (if exposed)
```

## 📈 Performance Optimization

### 1. GPU Memory Management

```bash
# Optimize CUDA memory allocation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Monitor GPU memory
watch -n 1 nvidia-smi
```

### 2. Docker Image Optimization

```dockerfile
# Multi-stage builds reduce image size
FROM base as dependencies
# Install dependencies

FROM dependencies as application
# Copy application

FROM application as production
# Final optimized image
```

### 3. Resource Limits

```yaml
services:
  worker:
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4.0'
        reservations:
          memory: 4G
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 4. Caching Strategy

```bash
# Enable BuildKit for better caching
export DOCKER_BUILDKIT=1

# Use cache mounts
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt
```

## 🔒 Security Considerations

### 1. Production Security

```bash
# Change default passwords
POSTGRES_PASSWORD=secure_random_password
GRAFANA_ADMIN_PASSWORD=secure_password

# Use secrets management
docker secret create postgres_password password.txt
```

### 2. Network Security

```yaml
# Restrict network access
networks:
  internal:
    internal: true  # No external access

  frontend:
    # Only nginx exposed
```

### 3. File Permissions

```bash
# Run as non-root user
USER app

# Secure file permissions
chmod 600 .env
chmod 700 data/
```

## 📦 Backup & Recovery

### Automated Backups

```bash
# Create backup
./scripts/deploy.sh backup

# Backup to external location
rsync -av backups/ user@backup-server:/backups/ai-content/
```

### Disaster Recovery

```bash
# Full system restore
./scripts/deploy.sh restore backup_20240101_120000

# Database only restore
docker-compose exec postgres psql -U ai_user -d ai_content_db < backup.sql
```

## 🚀 Deployment to Cloud

### AWS ECS

```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com

docker build -t ai-content-creator .
docker tag ai-content-creator:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/ai-content-creator:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/ai-content-creator:latest
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-content-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-content-api
  template:
    metadata:
      labels:
        app: ai-content-api
    spec:
      containers:
      - name: api
        image: ai-content-creator:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
```

### Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml ai-content-stack
```

## 📞 Support

For deployment issues:

1. **Check logs**: `./scripts/deploy.sh logs`
2. **Health check**: `./scripts/deploy.sh health`
3. **Resource monitoring**: `./scripts/deploy.sh monitor`
4. **GPU diagnostics**: `./scripts/setup-gpu.sh status`
5. **GitHub Issues**: [Report deployment issues](https://github.com/your-username/ai-content-creator/issues)

---

**Next Steps:**
- [API Documentation](API.md)
- [Usage Guide](USAGE_GUIDE.md)
- [Performance Optimization](PERFORMANCE.md)
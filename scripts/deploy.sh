#!/bin/bash

# AI Content Creator Deployment Script
# This script handles deployment of the AI Content Creator platform

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"
BACKUP_DIR="backups"
LOG_FILE="deploy.log"

# Functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi

    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi

    # Check if NVIDIA Docker is available (for GPU support)
    if command -v nvidia-docker &> /dev/null || docker info | grep -q nvidia; then
        log "NVIDIA Docker support detected"
    else
        warn "NVIDIA Docker not detected. GPU acceleration will not be available."
    fi

    # Check available disk space (need at least 10GB)
    available_space=$(df . | tail -1 | awk '{print $4}')
    if [ "$available_space" -lt 10485760 ]; then  # 10GB in KB
        warn "Less than 10GB disk space available. This might cause issues during deployment."
    fi

    log "Prerequisites check completed"
}

# Setup environment
setup_environment() {
    log "Setting up environment..."

    # Create necessary directories
    mkdir -p data/{input,output,temp} logs cache "$BACKUP_DIR"

    # Copy environment file if it doesn't exist
    if [ ! -f "$ENV_FILE" ]; then
        if [ -f ".env.docker" ]; then
            cp .env.docker "$ENV_FILE"
            log "Created $ENV_FILE from .env.docker template"
        else
            error "$ENV_FILE not found and no template available"
        fi
    fi

    # Set proper permissions
    chmod 755 data logs cache "$BACKUP_DIR"
    chmod 600 "$ENV_FILE"

    log "Environment setup completed"
}

# Build Docker images
build_images() {
    log "Building Docker images..."

    # Build with cache if possible
    if [ "$1" = "--no-cache" ]; then
        docker-compose -f "$COMPOSE_FILE" build --no-cache
    else
        docker-compose -f "$COMPOSE_FILE" build
    fi

    log "Docker images built successfully"
}

# Start services
start_services() {
    log "Starting services..."

    # Start infrastructure services first
    docker-compose -f "$COMPOSE_FILE" up -d redis postgres

    # Wait for database to be ready
    log "Waiting for database to be ready..."
    for i in {1..30}; do
        if docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_isready -U ai_user -d ai_content_db; then
            log "Database is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            error "Database failed to start within 30 seconds"
        fi
        sleep 1
    done

    # Start all services
    docker-compose -f "$COMPOSE_FILE" up -d

    log "All services started successfully"
}

# Health check
health_check() {
    log "Performing health checks..."

    # Wait for API to be ready
    for i in {1..60}; do
        if curl -f http://localhost:8000/health &> /dev/null; then
            log "API health check passed"
            break
        fi
        if [ $i -eq 60 ]; then
            error "API health check failed after 60 seconds"
        fi
        sleep 1
    done

    # Check all services
    services=("api" "worker" "redis" "postgres" "nginx")
    for service in "${services[@]}"; do
        if docker-compose -f "$COMPOSE_FILE" ps "$service" | grep -q "Up"; then
            log "Service $service is running"
        else
            error "Service $service is not running"
        fi
    done

    log "Health checks completed successfully"
}

# Backup data
backup_data() {
    log "Creating backup..."

    timestamp=$(date +%Y%m%d_%H%M%S)
    backup_name="backup_$timestamp"

    # Create backup directory
    mkdir -p "$BACKUP_DIR/$backup_name"

    # Backup database
    docker-compose -f "$COMPOSE_FILE" exec -T postgres pg_dump -U ai_user ai_content_db > "$BACKUP_DIR/$backup_name/database.sql"

    # Backup data directory
    if [ -d "data" ]; then
        tar -czf "$BACKUP_DIR/$backup_name/data.tar.gz" data/
    fi

    # Backup configuration
    cp "$ENV_FILE" "$BACKUP_DIR/$backup_name/"
    cp "$COMPOSE_FILE" "$BACKUP_DIR/$backup_name/"

    log "Backup created: $BACKUP_DIR/$backup_name"
}

# Restore from backup
restore_backup() {
    if [ -z "$1" ]; then
        error "Please specify backup name to restore from"
    fi

    backup_path="$BACKUP_DIR/$1"
    if [ ! -d "$backup_path" ]; then
        error "Backup not found: $backup_path"
    fi

    log "Restoring from backup: $backup_path"

    # Stop services
    docker-compose -f "$COMPOSE_FILE" down

    # Restore database
    if [ -f "$backup_path/database.sql" ]; then
        log "Restoring database..."
        docker-compose -f "$COMPOSE_FILE" up -d postgres
        sleep 10
        docker-compose -f "$COMPOSE_FILE" exec -T postgres psql -U ai_user -d ai_content_db < "$backup_path/database.sql"
    fi

    # Restore data
    if [ -f "$backup_path/data.tar.gz" ]; then
        log "Restoring data directory..."
        rm -rf data/
        tar -xzf "$backup_path/data.tar.gz"
    fi

    # Restore configuration
    if [ -f "$backup_path/.env" ]; then
        cp "$backup_path/.env" .
    fi

    log "Backup restored successfully"
}

# Update services
update_services() {
    log "Updating services..."

    # Create backup before update
    backup_data

    # Pull latest images
    docker-compose -f "$COMPOSE_FILE" pull

    # Rebuild local images
    build_images

    # Restart services with zero downtime
    docker-compose -f "$COMPOSE_FILE" up -d --remove-orphans

    # Clean up old images
    docker image prune -f

    log "Services updated successfully"
}

# Monitor services
monitor_services() {
    log "Monitoring services..."

    while true; do
        echo -e "\n${BLUE}=== Service Status ===${NC}"
        docker-compose -f "$COMPOSE_FILE" ps

        echo -e "\n${BLUE}=== Resource Usage ===${NC}"
        docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

        echo -e "\n${BLUE}=== Recent Logs ===${NC}"
        docker-compose -f "$COMPOSE_FILE" logs --tail=5 api

        echo -e "\nPress Ctrl+C to exit monitoring..."
        sleep 10
    done
}

# Show logs
show_logs() {
    service=${1:-"api"}
    lines=${2:-"100"}

    log "Showing logs for service: $service"
    docker-compose -f "$COMPOSE_FILE" logs --tail="$lines" -f "$service"
}

# Development setup
dev_setup() {
    log "Setting up development environment..."

    # Use development compose file
    COMPOSE_FILE="docker-compose.dev.yml"

    # Copy environment file
    if [ ! -f "$ENV_FILE" ]; then
        cp .env.docker "$ENV_FILE"
        # Set development defaults
        sed -i 's/DEBUG=False/DEBUG=True/' "$ENV_FILE"
        sed -i 's/LOG_LEVEL=INFO/LOG_LEVEL=DEBUG/' "$ENV_FILE"
    fi

    # Create development directories
    mkdir -p notebooks

    setup_environment
    build_images
    start_services
    health_check

    log "Development environment ready!"
    log "API: http://localhost:8000"
    log "Jupyter: http://localhost:8888"
    log "Grafana: http://localhost:3000 (admin/admin123)"
}

# Clean up everything
cleanup() {
    log "Cleaning up..."

    # Stop and remove containers
    docker-compose -f "$COMPOSE_FILE" down -v --remove-orphans

    # Remove images
    if [ "$1" = "--all" ]; then
        docker system prune -af --volumes
        warn "All Docker data has been removed"
    else
        docker image prune -f
        log "Unused images removed"
    fi

    log "Cleanup completed"
}

# Show usage
usage() {
    echo "AI Content Creator Deployment Script"
    echo
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo
    echo "Commands:"
    echo "  deploy              Full deployment (build, start, health check)"
    echo "  start               Start services"
    echo "  stop                Stop services"
    echo "  restart             Restart services"
    echo "  status              Show service status"
    echo "  logs [SERVICE]      Show logs for service (default: api)"
    echo "  build [--no-cache]  Build Docker images"
    echo "  update              Update services with backup"
    echo "  backup              Create backup"
    echo "  restore BACKUP      Restore from backup"
    echo "  monitor             Monitor services"
    echo "  dev                 Setup development environment"
    echo "  cleanup [--all]     Clean up containers and images"
    echo "  health              Run health checks"
    echo
    echo "Examples:"
    echo "  $0 deploy           # Full deployment"
    echo "  $0 dev              # Development setup"
    echo "  $0 logs api         # Show API logs"
    echo "  $0 backup           # Create backup"
    echo "  $0 restore backup_20240101_120000  # Restore backup"
}

# Main script logic
main() {
    case "${1:-}" in
        "deploy")
            check_prerequisites
            setup_environment
            build_images "${2:-}"
            start_services
            health_check
            log "Deployment completed successfully!"
            log "Access the application at: http://localhost"
            log "API documentation: http://localhost/docs"
            log "Monitoring: http://localhost:3000 (admin/admin123)"
            ;;
        "start")
            start_services
            ;;
        "stop")
            log "Stopping services..."
            docker-compose -f "$COMPOSE_FILE" down
            log "Services stopped"
            ;;
        "restart")
            log "Restarting services..."
            docker-compose -f "$COMPOSE_FILE" restart
            log "Services restarted"
            ;;
        "status")
            docker-compose -f "$COMPOSE_FILE" ps
            ;;
        "logs")
            show_logs "${2:-api}" "${3:-100}"
            ;;
        "build")
            build_images "${2:-}"
            ;;
        "update")
            update_services
            ;;
        "backup")
            backup_data
            ;;
        "restore")
            restore_backup "${2}"
            ;;
        "monitor")
            monitor_services
            ;;
        "dev")
            dev_setup
            ;;
        "cleanup")
            cleanup "${2:-}"
            ;;
        "health")
            health_check
            ;;
        "help"|"--help"|"-h")
            usage
            ;;
        *)
            error "Unknown command: ${1:-}. Use '$0 help' for usage information."
            ;;
    esac
}

# Run main function with all arguments
main "$@"
#!/bin/bash
# Docker run script - Simplifies the building and running of the Docker container

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "vLLM with cuSZp - Docker Script"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check for NVIDIA Container Toolkit
if ! docker info | grep -q "nvidia"; then
    echo "⚠️  Warning: NVIDIA Container Toolkit not detected. GPU support may not be available."
fi

# Enter docker directory
cd "$SCRIPT_DIR"

# Parse command line arguments
ACTION="${1:-build}"

case "$ACTION" in
    build)
        echo "🔨 Building Docker image..."
        docker build -t vllm-cuszp:latest -f Dockerfile "$PROJECT_ROOT"
        echo "✅ Image build completed!"
        echo ""
        echo "Run the following command to start the container:"
        echo "  ./run.sh run"
        ;;
    run)
        echo "🚀 Starting Docker container..."
        docker run --gpus all -it --rm \
            -v "$PROJECT_ROOT:/workspace" \
            -w /workspace \
            vllm-cuszp:latest
        ;;
    exec)
        echo "🔧 Entering running container..."
        CONTAINER_ID=$(docker ps -q -f ancestor=vllm-cuszp:latest)
        if [ -z "$CONTAINER_ID" ]; then
            echo "❌ Error: No running container found. Please run first: ./run.sh run"
            exit 1
        fi
        docker exec -it "$CONTAINER_ID" bash
        ;;
    test)
        echo "🧪 Running tests..."
        docker run --gpus all --rm \
            -v "$PROJECT_ROOT:/workspace" \
            -w /workspace \
            vllm-cuszp:latest \
            bash test.sh
        ;;
    compose-build)
        echo "🔨 Building with docker-compose..."
        docker-compose build
        ;;
    compose-up)
        echo "🚀 Starting with docker-compose..."
        docker-compose up -d
        echo "✅ Container is running in the background"
        echo "Enter container: docker-compose exec vllm-cuszp bash"
        ;;
    compose-down)
        echo "🛑 Stopping docker-compose container..."
        docker-compose down
        ;;
    *)
        echo "Usage: $0 {build|run|exec|test|compose-build|compose-up|compose-down}"
        echo ""
        echo "Commands:"
        echo "  build          - Build Docker image"
        echo "  run            - Run Docker container (interactive)"
        echo "  exec           - Enter running container"
        echo "  test           - Run test script"
        echo "  compose-build  - Build with docker-compose"
        echo "  compose-up     - Start with docker-compose (background)"
        echo "  compose-down   - Stop docker-compose container"
        exit 1
        ;;
esac


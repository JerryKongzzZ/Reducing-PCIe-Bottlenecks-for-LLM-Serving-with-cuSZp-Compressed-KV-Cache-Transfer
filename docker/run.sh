#!/bin/bash
# Docker运行脚本 - 简化Docker容器的构建和运行

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "vLLM with cuSZp - Docker运行脚本"
echo "=========================================="
echo ""

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "❌ 错误: Docker未安装。请先安装Docker。"
    exit 1
fi

# 检查NVIDIA Container Toolkit
if ! docker info | grep -q "nvidia"; then
    echo "⚠️  警告: 未检测到NVIDIA Container Toolkit。GPU支持可能不可用。"
fi

# 进入docker目录
cd "$SCRIPT_DIR"

# 解析命令行参数
ACTION="${1:-build}"

case "$ACTION" in
    build)
        echo "🔨 构建Docker镜像..."
        docker build -t vllm-cuszp:latest -f Dockerfile "$PROJECT_ROOT"
        echo "✅ 镜像构建完成！"
        echo ""
        echo "运行以下命令启动容器:"
        echo "  ./run.sh run"
        ;;
    run)
        echo "🚀 启动Docker容器..."
        docker run --gpus all -it --rm \
            -v "$PROJECT_ROOT:/workspace" \
            -w /workspace \
            vllm-cuszp:latest
        ;;
    exec)
        echo "🔧 进入运行中的容器..."
        CONTAINER_ID=$(docker ps -q -f ancestor=vllm-cuszp:latest)
        if [ -z "$CONTAINER_ID" ]; then
            echo "❌ 错误: 没有运行中的容器。请先运行: ./run.sh run"
            exit 1
        fi
        docker exec -it "$CONTAINER_ID" bash
        ;;
    test)
        echo "🧪 运行测试..."
        docker run --gpus all --rm \
            -v "$PROJECT_ROOT:/workspace" \
            -w /workspace \
            vllm-cuszp:latest \
            bash -c "python3 benchmarks/baseline_profiling.py && python3 benchmarks/compression_benchmark.py"
        ;;
    compose-build)
        echo "🔨 使用docker-compose构建..."
        docker-compose build
        ;;
    compose-up)
        echo "🚀 使用docker-compose启动..."
        docker-compose up -d
        echo "✅ 容器已在后台运行"
        echo "进入容器: docker-compose exec vllm-cuszp bash"
        ;;
    compose-down)
        echo "🛑 停止docker-compose容器..."
        docker-compose down
        ;;
    *)
        echo "用法: $0 {build|run|exec|test|compose-build|compose-up|compose-down}"
        echo ""
        echo "命令说明:"
        echo "  build          - 构建Docker镜像"
        echo "  run            - 运行Docker容器（交互式）"
        echo "  exec           - 进入运行中的容器"
        echo "  test           - 运行测试脚本"
        echo "  compose-build  - 使用docker-compose构建"
        echo "  compose-up     - 使用docker-compose启动（后台）"
        echo "  compose-down   - 停止docker-compose容器"
        exit 1
        ;;
esac


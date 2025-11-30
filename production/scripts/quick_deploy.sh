#!/bin/bash

echo "🚀 Self-Correcting RAG - Quick Production Deployment"
echo "===================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

command -v docker >/dev/null 2>&1 || { 
    echo -e "${RED}❌ Docker required but not installed.${NC}" 
    echo "Install from: https://docs.docker.com/get-docker/"
    exit 1
}

command -v docker-compose >/dev/null 2>&1 || { 
    echo -e "${RED}❌ Docker Compose required but not installed.${NC}"
    echo "Install from: https://docs.docker.com/compose/install/"
    exit 1
}

echo -e "${GREEN}✅ All prerequisites met${NC}\n"

# Navigate to docker directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DOCKER_DIR="$SCRIPT_DIR/../docker"
cd "$DOCKER_DIR"

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    cat > .env << EOF
LOG_LEVEL=INFO
ENABLE_API_AUTH=false
PRODUCTION_MODE=true
EOF
    echo -e "${GREEN}✅ .env file created${NC}"
fi

# Build and deploy
echo -e "\n${YELLOW}Building Docker image...${NC}"
docker-compose build

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Build failed${NC}"
    exit 1
fi

echo -e "\n${YELLOW}Starting services...${NC}"
docker-compose up -d

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to start services${NC}"
    exit 1
fi

# Wait for health check
echo -e "\n${YELLOW}Waiting for service to be ready...${NC}"
for i in {1..60}; do
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Service is healthy!${NC}"
        break
    fi
    if [ $i -eq 60 ]; then
        echo -e "${RED}❌ Service failed to start within 60 seconds${NC}"
        echo "Check logs: docker-compose logs"
        exit 1
    fi
    echo "Waiting... ($i/60)"
    sleep 2
done

# Display access information
echo ""
echo "======================================"
echo -e "${GREEN}✅ Deployment Complete!${NC}"
echo "======================================"
echo ""
echo "📍 API URL:    http://localhost:8000"
echo "📚 API Docs:   http://localhost:8000/docs"
echo "🔍 Health:     http://localhost:8000/health"
echo "📊 Metrics:    http://localhost:8000/metrics"
echo ""
echo "Test query:"
echo "curl -X POST http://localhost:8000/query \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"query\": \"What is machine learning?\"}'"
echo ""
echo "======================================"
echo "Useful commands:"
echo "  View logs:    docker-compose logs -f"
echo "  Stop:         docker-compose down"
echo "  Restart:      docker-compose restart"
echo "  Status:       docker-compose ps"
echo "======================================"

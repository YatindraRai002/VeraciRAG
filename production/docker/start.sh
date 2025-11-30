#!/bin/bash
set -e

echo "🚀 Starting Self-Correcting RAG System"
echo "======================================"

# Start Ollama service in background
echo "Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "✅ Ollama is ready!"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 2
done

# Pull required models
echo "Pulling required models..."
ollama pull mistral 2>&1 | grep -v "Already up to date" || true

# Check for custom models
if [ -d "/app/data/fine_tuning" ]; then
    echo "Loading custom models..."
    for modelfile in /app/data/fine_tuning/Modelfile*; do
        if [ -f "$modelfile" ]; then
            model_name=$(basename "$modelfile" | sed 's/Modelfile-//')
            echo "  - Creating model: $model_name"
            ollama create "$model_name" -f "$modelfile" 2>&1 | grep -v "already exists" || true
        fi
    done
fi

echo "======================================"
echo "✅ Ollama setup complete"
echo "======================================"

# Start FastAPI application
echo "Starting API server..."
exec uvicorn production.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level ${LOG_LEVEL:-info}

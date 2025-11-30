# Self-Correcting RAG - Quick Production Deployment (PowerShell)

Write-Host "`n🚀 Self-Correcting RAG - Quick Production Deployment" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan

# Check prerequisites
Write-Host "`nChecking prerequisites..." -ForegroundColor Yellow

try {
    docker --version | Out-Null
    Write-Host "✅ Docker found" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker required but not installed." -ForegroundColor Red
    Write-Host "Install from: https://docs.docker.com/get-docker/" -ForegroundColor Yellow
    exit 1
}

try {
    docker-compose --version | Out-Null
    Write-Host "✅ Docker Compose found" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker Compose required but not installed." -ForegroundColor Red
    Write-Host "Install from: https://docs.docker.com/compose/install/" -ForegroundColor Yellow
    exit 1
}

# Navigate to docker directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$dockerDir = Join-Path $scriptPath "..\docker"
Set-Location $dockerDir

# Check if .env file exists
if (-not (Test-Path .env)) {
    Write-Host "`nCreating .env file..." -ForegroundColor Yellow
    @"
LOG_LEVEL=INFO
ENABLE_API_AUTH=false
PRODUCTION_MODE=true
"@ | Out-File -FilePath .env -Encoding UTF8
    Write-Host "✅ .env file created" -ForegroundColor Green
}

# Build and deploy
Write-Host "`nBuilding Docker image..." -ForegroundColor Yellow
docker-compose build

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed" -ForegroundColor Red
    exit 1
}

Write-Host "`nStarting services..." -ForegroundColor Yellow
docker-compose up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to start services" -ForegroundColor Red
    exit 1
}

# Wait for health check
Write-Host "`nWaiting for service to be ready..." -ForegroundColor Yellow
$ready = $false
for ($i = 1; $i -le 60; $i++) {
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            Write-Host "✅ Service is healthy!" -ForegroundColor Green
            $ready = $true
            break
        }
    } catch {
        # Service not ready yet
    }
    
    if ($i -eq 60) {
        Write-Host "❌ Service failed to start within 60 seconds" -ForegroundColor Red
        Write-Host "Check logs: docker-compose logs" -ForegroundColor Yellow
        exit 1
    }
    
    Write-Host "Waiting... ($i/60)"
    Start-Sleep -Seconds 2
}

# Display access information
Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "✅ Deployment Complete!" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green
Write-Host ""
Write-Host "📍 API URL:    http://localhost:8000"
Write-Host "📚 API Docs:   http://localhost:8000/docs"
Write-Host "🔍 Health:     http://localhost:8000/health"
Write-Host "📊 Metrics:    http://localhost:8000/metrics"
Write-Host ""
Write-Host "Test query:" -ForegroundColor Cyan
Write-Host 'curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"query\": \"What is machine learning?\"}"'
Write-Host ""
Write-Host "Or using PowerShell:"
Write-Host '$body = @{query="What is machine learning?"} | ConvertTo-Json'
Write-Host 'Invoke-RestMethod -Uri http://localhost:8000/query -Method Post -Body $body -ContentType "application/json"'
Write-Host ""
Write-Host "======================================" -ForegroundColor Green
Write-Host "Useful commands:" -ForegroundColor Cyan
Write-Host "  View logs:    docker-compose logs -f"
Write-Host "  Stop:         docker-compose down"
Write-Host "  Restart:      docker-compose restart"
Write-Host "  Status:       docker-compose ps"
Write-Host "======================================" -ForegroundColor Green

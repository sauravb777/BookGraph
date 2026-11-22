$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

docker compose up -d
if ($LASTEXITCODE -ne 0) { throw "Docker Compose could not start Qdrant and Neo4j." }

$BackendPath = "$ProjectRoot\backend"
if (!(Test-Path "$BackendPath\.venv\Scripts\Activate.ps1")) {
    throw "Backend environment not found. Run .\setup.ps1 once first."
}
if (!(Test-Path "$ProjectRoot\frontend\node_modules")) {
    throw "Frontend dependencies not found. Run .\setup.ps1 once first."
}

Start-Process powershell -ArgumentList @("-NoExit", "-Command", "Set-Location '$BackendPath'; & '.\.venv\Scripts\Activate.ps1'; uvicorn app.main:app --host 0.0.0.0 --port 8000")
Set-Location "$ProjectRoot\frontend"
npm run dev

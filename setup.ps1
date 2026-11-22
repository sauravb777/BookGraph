$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "Starting the BookGraph databases..." -ForegroundColor Cyan
docker compose up -d
if ($LASTEXITCODE -ne 0) { throw "Docker Compose could not start Qdrant and Neo4j. Check Docker Desktop." }

Write-Host "Preparing the Python environment..." -ForegroundColor Cyan
Set-Location "$ProjectRoot\backend"
if (!(Test-Path ".venv")) { python -m venv .venv }
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Could not upgrade pip." }
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Python dependencies could not be installed." }
if (!(Test-Path ".env")) { Copy-Item ".env.example" ".env" }

Write-Host "Downloading and indexing all 10,000 books into Qdrant and Neo4j..." -ForegroundColor Cyan
& ".\.venv\Scripts\python.exe" scripts\prepare_and_index.py
if ($LASTEXITCODE -ne 0) { throw "The 10,000-book indexing process failed. See the error above." }
& ".\.venv\Scripts\python.exe" scripts\verify_index.py
if ($LASTEXITCODE -ne 0) { throw "The Qdrant / Neo4j index verification failed." }

Write-Host "Starting the BookGraph API..." -ForegroundColor Cyan
$BackendPath = "$ProjectRoot\backend"
Start-Process powershell -ArgumentList @("-NoExit", "-Command", "Set-Location '$BackendPath'; & '.\.venv\Scripts\Activate.ps1'; uvicorn app.main:app --host 0.0.0.0 --port 8000")

Set-Location "$ProjectRoot\frontend"
if (!(Test-Path ".env.local")) { Copy-Item ".env.example" ".env.local" }
Write-Host "Installing frontend dependencies..." -ForegroundColor Cyan
npm install
if ($LASTEXITCODE -ne 0) { throw "npm install failed." }
Write-Host "Starting BookGraph at http://localhost:3000 ..." -ForegroundColor Cyan
npm run dev

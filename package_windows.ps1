# ==============================================================================
# Design Matcher - Windows PowerShell Packaging & Code-Signing Script
# ==============================================================================
param (
    [string]$CertThumbprint = "",
    [switch]$SkipInstaller = $false
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Design Matcher - Windows Packaging Workflow" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Environment Activation
if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "[SETUP] Creating virtual environment '.venv'..." -ForegroundColor Yellow
    python -m venv .venv
}
Write-Host "[1/5] Activating virtual environment..." -ForegroundColor Green
. .venv\Scripts\Activate.ps1

# 2. Dependency Check & Installation
Write-Host "[2/5] Checking and installing dependencies from requirements.txt..." -ForegroundColor Green
python -m pip install --upgrade pip | Out-Null
python -m pip install -r requirements.txt
python -m pip install --upgrade pyinstaller pillow | Out-Null

if (-not (Test-Path "models\dinov2-small\config.json")) {
    Write-Host "[2.5/5] Pre-caching DINOv2 model weights for offline bundling..." -ForegroundColor Green
    python -c "from transformers import AutoImageProcessor, AutoModel; from pathlib import Path; p = Path('models/dinov2-small'); p.mkdir(parents=True, exist_ok=True); AutoImageProcessor.from_pretrained('facebook/dinov2-small').save_pretrained(p); AutoModel.from_pretrained('facebook/dinov2-small').save_pretrained(p); print('DINOv2 model cached.')"
}

# 3. Clean PyInstaller Build
Write-Host "[3/5] Compiling application with PyInstaller..." -ForegroundColor Green
pyinstaller --clean -y design_matcher.spec

if (-not (Test-Path "dist\DesignMatcher\DesignMatcher.exe")) {
    Write-Error "PyInstaller build failed: dist\DesignMatcher\DesignMatcher.exe not found!"
}
Write-Host "  -> Compiled standalone app in dist\DesignMatcher\" -ForegroundColor Green

# 4. Optional Code Signing of Binaries
if ($CertThumbprint -ne "") {
    Write-Host "[4/5] Signing binaries with Authenticode certificate..." -ForegroundColor Green
    $SignTool = (Get-ChildItem "C:\Program Files (x86)\Windows Kits\10\bin\*\x64\signtool.exe" -ErrorAction SilentlyContinue | Select-Object -Last 1).FullName
    if ($SignTool) {
        Write-Host "  Using signtool at: $SignTool"
        & $SignTool sign /tr http://timestamp.digicert.com /td sha256 /fd sha256 /sha1 $CertThumbprint dist\DesignMatcher\DesignMatcher.exe
        Get-ChildItem -Path dist\DesignMatcher\*.dll -Recurse | ForEach-Object {
            & $SignTool sign /tr http://timestamp.digicert.com /td sha256 /fd sha256 /sha1 $CertThumbprint $_.FullName
        }
    } else {
        Write-Warning "signtool.exe not found in Windows SDK. Skipping binary signing."
    }
} else {
    Write-Host "[4/5] Skipping code signing (no cert thumbprint provided)." -ForegroundColor Yellow
}

# 5. Inno Setup Compilation
if (-not $SkipInstaller) {
    Write-Host "[5/5] Compiling Inno Setup installer..." -ForegroundColor Green
    $IsccPaths = @(
        "C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
        "C:\Program Files\Inno Setup 6\ISCC.exe"
    )
    $Iscc = $IsccPaths | Where-Object { Test-Path $_ } | Select-Object -First 1

    if ($Iscc) {
        & $Iscc installer.iss
        $OutputFile = "Output\DesignMatcher_Setup_v1.0.1.exe"
        if (Test-Path $OutputFile) {
            if ($CertThumbprint -ne "" -and $SignTool) {
                Write-Host "  Signing final installer..." -ForegroundColor Green
                & $SignTool sign /tr http://timestamp.digicert.com /td sha256 /fd sha256 /sha1 $CertThumbprint $OutputFile
            }
            Write-Host "==========================================================" -ForegroundColor Green
            Write-Host " [SUCCESS] Installer generated at: $OutputFile" -ForegroundColor Green
            Write-Host "==========================================================" -ForegroundColor Green
        }
    } else {
        Write-Warning "Inno Setup 6 compiler (ISCC.exe) not found. Download it from https://jrsoftware.org/isdl.php"
        Write-Host "Standalone executable folder is ready in 'dist\DesignMatcher\'" -ForegroundColor Cyan
    }
}

@echo off
REM ==============================================================================
REM Design Matcher - Windows Build & Packaging Script
REM Builds standalone executable with PyInstaller and compiles Inno Setup installer.
REM ==============================================================================

setlocal enabledelayedexpansion

echo [1/4] Activating Virtual Environment...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo [ERROR] Virtual environment not found in .venv!
    echo Please create it first: python -m venv .venv
    exit /b 1
)

echo [2/4] Verifying and Installing Dependencies from requirements.txt...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install project requirements!
    exit /b 1
)
python -m pip install --upgrade pyinstaller pillow


echo [3/4] Running PyInstaller Build...
pyinstaller --clean -y design_matcher.spec
if errorlevel 1 (
    echo [ERROR] PyInstaller build failed!
    exit /b 1
)

echo [4/4] Compiling Inno Setup Installer...
set "ISCC_PATH=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if not exist "!ISCC_PATH!" (
    set "ISCC_PATH=C:\Program Files\Inno Setup 6\ISCC.exe"
)
if not exist "!ISCC_PATH!" (
    for /f "tokens=*" %%i in ('where iscc 2^>nul') do set "ISCC_PATH=%%i"
)

if exist "!ISCC_PATH!" (
    "!ISCC_PATH!" installer.iss
    if errorlevel 1 (
        echo [ERROR] Inno Setup compilation failed!
        exit /b 1
    )
    echo ==============================================================================
    echo [SUCCESS] Windows Installer created at: Output\DesignMatcher_Setup_v1.0.0.exe
    echo ==============================================================================
) else (
    echo [WARNING] Inno Setup 6 compiler (ISCC.exe) was not found at standard paths.
    echo Standalone executable folder is ready at: dist\DesignMatcher\
    echo To generate the final setup installer:
    echo 1. Download and install Inno Setup 6 from https://jrsoftware.org/isdl.php
    echo 2. Right-click installer.iss and select 'Compile'.
)

pause

@echo off
REM ==============================================================================
REM Design Matcher - Push Changes to GitHub
REM ==============================================================================
cd /d "%~dp0"

echo ==========================================================
echo  Pushing Design-Matcher-Windows-OS to GitHub...
echo  Remote: https://github.com/suhel499/Design-Matcher.git
echo ==========================================================
echo.
git push -u origin Design-Matcher-Windows-OS
echo.
if errorlevel 1 (
    echo [INFO] If a browser window opened, please click 'Sign in with your browser'.
) else (
    echo [SUCCESS] Git push completed successfully!
)
echo.
pause

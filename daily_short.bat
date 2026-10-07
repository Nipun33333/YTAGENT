@echo off
cd /d "%~dp0"

echo ========================================
echo       DAILY YOUTUBE PIPELINE
echo ========================================

set PIPELINE_TIMEZONE=Asia/Kolkata
python -m pipeline

echo.
echo ========================================
echo       PIPELINE FINISHED
echo ========================================

pause

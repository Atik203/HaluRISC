@echo off
REM Keep the HaluRISC FastAPI backend alive. Close this window to stop both.
pwsh -NoProfile -ExecutionPolicy Bypass -File "%~dp0serve_api.ps1" %*

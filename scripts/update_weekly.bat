@echo off
REM Weekly data refresh: pulls new start.gg events, rebuilds the models, commits + pushes them.
REM Scheduled with Windows Task Scheduler (see README, "Keeping the data up to date").
REM Output goes to logs\update_<date>.log (written by update.py) and logs\scheduler.log.

cd /d "%~dp0.."
if not exist logs mkdir logs
echo ==== %date% %time% ==== >> logs\scheduler.log
".venv\Scripts\python.exe" src\update.py --commit >> logs\scheduler.log 2>&1
echo exit code %errorlevel% >> logs\scheduler.log
exit /b %errorlevel%

@echo off
REM Script to prepare the ICT Forex Trading Bot for GitHub

echo ===============================================
echo Preparing ICT Forex Trading Bot for GitHub
echo ===============================================
echo.

REM Initialize Git repository if not already initialized
IF NOT EXIST ".git" (
    echo Initializing Git repository...
    git init
    echo.
)

REM Rename the GitHub README
echo Renaming GitHub README...
IF EXIST "README_GITHUB.md" (
    copy /Y README_GITHUB.md README.md
    del README_GITHUB.md
    echo README updated for GitHub.
    echo.
)

REM Check if .env file is in .gitignore
echo Checking .gitignore...
findstr /C:"config/.env" .gitignore > nul
IF %ERRORLEVEL% NEQ 0 (
    echo Adding sensitive files to .gitignore...
    echo config/.env >> .gitignore
    echo data/balance_history.json >> .gitignore
    echo.
)

REM Create data directory if it doesn't exist
IF NOT EXIST "data" (
    echo Creating data directory...
    mkdir data
    echo.
)

echo Your repository is now ready for GitHub!
echo.
echo Next steps:
echo 1. Create a new repository on GitHub
echo 2. Run the following commands:
echo    git add .
echo    git commit -m "Initial commit"
echo    git branch -M main
echo    git remote add origin https://github.com/yourusername/your-repo-name.git
echo    git push -u origin main
echo.
echo Remember to replace 'yourusername' and 'your-repo-name' with your actual GitHub username and repository name.
echo.

pause

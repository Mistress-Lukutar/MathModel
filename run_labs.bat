@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo ============================================
echo  Math Modeling Labs - Launcher
echo ============================================
echo.

REM === Check Python ===
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH
    echo Please install Python 3.8+ and add it to PATH
    pause
    exit /b 1
)

echo [OK] Python found:
python --version
echo.

REM === Create virtual environment ===
set "VENV_DIR=%~dp0.venv"
if not exist "%VENV_DIR%" (
    echo [1/4] Creating virtual environment...
    python -m venv "%VENV_DIR%"
    echo.
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
) else (
    echo [1/4] Virtual environment already exists
)

REM === Activate environment ===
echo [2/4] Activating environment...
call "%VENV_DIR%\Scripts\activate.bat"

REM === Upgrade pip ===
echo [3/4] Upgrading pip...
python -m pip install --upgrade pip

REM === Install dependencies ===
echo [4/4] Installing dependencies...
if exist "requirements.txt" (
    pip install -r requirements.txt
) else (
    pip install numpy scipy matplotlib networkx
)

if errorlevel 1 (
    echo [ERROR] Failed to install packages
    pause
    exit /b 1
)

echo.
echo [OK] All dependencies installed
echo.

REM === Lab Menu ===
:MENU
echo ============================================
echo  SELECT LAB WORK
echo ============================================
echo.
echo [1] - Lab 1: Continuous-time Markov Chains
echo [2] - Lab 2: [Placeholder]
echo [3] - Lab 3: [Placeholder]
echo [4] - Lab 4: [Placeholder]
echo.
echo [Q] - Quit
echo.

set /p choice="Enter lab number: "

if /i "%choice%"=="Q" goto :EXIT
if /i "%choice%"=="q" goto :EXIT
if "%choice%"=="1" goto :LAB1
if "%choice%"=="2" goto :LAB2
if "%choice%"=="3" goto :LAB3
if "%choice%"=="4" goto :LAB4

echo.
echo [ERROR] Invalid choice. Please try again.
echo.
goto :MENU

REM === Lab 1: Markov Chains ===
:LAB1
echo.
echo ============================================
echo  LAB 1: Continuous-time Markov Chains
echo ============================================
echo.

set "LAB_DIR=%~dp0L1"
if not exist "%LAB_DIR%" (
    echo [ERROR] Directory L1 not found
    pause
    goto :MENU
)

REM Create Output directory if not exists
set "OUTPUT_DIR=%~dp0Output"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

cd /d "%LAB_DIR%"

echo [Step 1] Building Markov chain graph...
python markov_graph.py
if exist "%~dp0Output\L1_markov_graph.png" (
    echo [OK] Graph saved: Output/L1_markov_graph.png
) else (
    echo [WARNING] Graph file not created
)

echo.
echo [Step 2] Solving ODEs and generating report...
python markov_solver.py

echo.
echo [Step 3] Analyzing absorption probabilities...
python stationary_check.py

echo.
echo ============================================
echo  LAB 1 COMPLETED
echo ============================================
echo.
echo Generated files in Output/:
echo.
if exist "%~dp0Output\L1_markov_graph.png" (
    echo   [PNG] L1_markov_graph.png - State transition diagram
)
if exist "%~dp0Output\L1_probabilities.png" (
    echo   [PNG] L1_probabilities.png - Probability evolution plot
)
if exist "%~dp0Output\L1_results.txt" (
    echo   [TXT] L1_results.txt - Full calculation report
)
echo.
echo [All results are ready for the report]

echo.
pause
cd /d "%~dp0"
goto :MENU

REM === Lab 2: Placeholder ===
:LAB2
echo.
echo ============================================
echo  LAB 2: [Placeholder]
echo ============================================
echo.
echo This lab is not implemented yet.
echo.
pause
goto :MENU

REM === Lab 3: Placeholder ===
:LAB3
echo.
echo ============================================
echo  LAB 3: [Placeholder]
echo ============================================
echo.
echo This lab is not implemented yet.
echo.
pause
goto :MENU

REM === Lab 4: Placeholder ===
:LAB4
echo.
echo ============================================
echo  LAB 4: [Placeholder]
echo ============================================
echo.
echo This lab is not implemented yet.
echo.
pause
goto :MENU

REM === Exit ===
:EXIT
echo.
echo Goodbye!
if exist "%VENV_DIR%\Scripts\deactivate.bat" (
    call "%VENV_DIR%\Scripts\deactivate.bat" >nul 2>&1
)
exit /b 0

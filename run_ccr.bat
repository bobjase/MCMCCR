@echo off
setlocal enabledelayedexpansion

if "%~1"=="" (
    echo Usage: %0 ^<input_file^>
    exit /b 1
)

pushd "%~dp1"
set INPUT=%~nx1
set BASE=%~n1
set MCM_EXE=%~dp0mcm.exe

echo Running MCM CCR pipeline on %INPUT%

echo Phase 1: Observer
"%MCM_EXE%" -observer "%INPUT%"

echo Phase 2: Segment
"%MCM_EXE%" -segment "%INPUT%"

echo Phase 3: Fingerprint
"%MCM_EXE%" -fingerprint "%INPUT%"

echo Phase 4: Oracle
"%MCM_EXE%" -oracle "%INPUT%"

echo Phase 5: Pathcover
"%MCM_EXE%" -pathcover "%INPUT%"

echo Compressing original file
if not exist "%BASE%.mcm" (
    "%MCM_EXE%" -x11 "%INPUT%" "%BASE%.mcm"
) else (
    echo Skipping original compression - %BASE%.mcm already exists
)

echo Compressing reordered file
"%MCM_EXE%" -x11 "%INPUT%.reordered" "%BASE%.reordered.mcm"

echo Getting file sizes...
for %%A in ("%BASE%.mcm") do set ORIG_SIZE=%%~zA
for %%A in ("%BASE%.reordered.mcm") do set REORDER_SIZE=%%~zA

echo Original compressed size: %ORIG_SIZE% bytes
echo Reordered compressed size: %REORDER_SIZE% bytes

if %ORIG_SIZE% gtr %REORDER_SIZE% (
    set /a SAVINGS=%ORIG_SIZE%-%REORDER_SIZE%
    set /a PERCENT=100*SAVINGS/ORIG_SIZE
    echo Reordering saved !SAVINGS! bytes ^(!PERCENT!%% savings^)
) else if %ORIG_SIZE% lss %REORDER_SIZE% (
    set /a INCREASE=%REORDER_SIZE%-%ORIG_SIZE%
    set /a PERCENT=100*INCREASE/ORIG_SIZE
    echo Reordering increased size by !INCREASE! bytes ^(!PERCENT!%% increase^)
) else (
    echo No change in size
)

echo Done.
popd
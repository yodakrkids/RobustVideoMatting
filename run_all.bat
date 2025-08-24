@echo off
echo ===============================================
echo STANDARD RVM BATCH PROCESSING
echo ===============================================
echo Total videos to process: 14
echo Model: MobileNetV3 (Standard quality)
echo Results will be saved to: output/standard/
echo ================================================

if not exist "output/standard" mkdir "output/standard"

echo.
echo Processing Video 1/14 - Sony Lens Test
call :ProcessVideo "VideoMatte/1/Sony_Lens_Test.mp4" "output_1_Sony_Lens_Test.mp4" "log_1_Sony_Lens_Test.txt" "Sony Lens Test"

echo.
echo Processing Video 2/14 - Sample Footage
call :ProcessVideo "VideoMatte/2/Sample_footage_.mp4" "output_2_Sample_footage.mp4" "log_2_Sample_footage.txt" "Sample Footage"

echo.
echo Processing Video 3/14 - Sony FR7
call :ProcessVideo "VideoMatte/3/Sony_FR7__Cinem.MP4" "output_3_Sony_FR7.mp4" "log_3_Sony_FR7.txt" "Sony FR7"

echo.
echo Processing Video 4/14 - ASO Japan 8K
call :ProcessVideo "VideoMatte/4/ASO_JAPAN_8K_-_.MP4" "output_4_ASO_JAPAN_8K.mp4" "log_4_ASO_JAPAN_8K.txt" "ASO Japan 8K"

echo.
echo Processing Video 5/14 - 4K Sample
call :ProcessVideo "VideoMatte/5/4K_Sample_Video.mp4" "output_5_4K_Sample.mp4" "log_5_4K_Sample.txt" "4K Sample"

echo.
echo Processing Video 6/14 - Sample Footage
call :ProcessVideo "VideoMatte/6/Sample_Footage_.MP4" "output_6_Sample_Footage.mp4" "log_6_Sample_Footage.txt" "Sample Footage"

echo.
echo Processing Video 7/14 - Sony HDC
call :ProcessVideo "VideoMatte/7/Sony___HDC-F550.mp4" "output_7_Sony_HDC.mp4" "log_7_Sony_HDC.txt" "Sony HDC"

echo.
echo Processing Video 8/14 - S7Edge
call :ProcessVideo "VideoMatte/8/S7Edge__Zhiyun_.MP4" "output_8_S7Edge.mp4" "log_8_S7Edge.txt" "S7Edge"

echo.
echo Processing Video 9/14 - Man Utd
call :ProcessVideo "VideoMatte/9/4K__Man_Utd._Vs.MP4" "output_9_Man_Utd.mp4" "log_9_Man_Utd.txt" "Man Utd"

echo.
echo Processing Video 10/14 - TWICE BDZ
call :ProcessVideo "VideoMatte/10/TWICE_-_BDZKore.MP4" "output_10_TWICE_BDZ.mp4" "log_10_TWICE_BDZ.txt" "TWICE BDZ"

echo.
echo Processing Video 11/14 - PEACE
call :ProcessVideo "VideoMatte/11/170903_PEACE_MA.MP4" "output_11a_PEACE.mp4" "log_11a_PEACE.txt" "PEACE"

echo.
echo Processing Video 12/14 - YOUTH
call :ProcessVideo "VideoMatte/11/A.YOUTH____Kiss.MP4" "output_11b_YOUTH.mp4" "log_11b_YOUTH.txt" "YOUTH"

echo.
echo Processing Video 13/14 - BTS
call :ProcessVideo "VideoMatte/12/_Beagles___BTS_.MP4" "output_12_BTS.mp4" "log_12_BTS.txt" "BTS"

echo.
echo Processing Video 14/14 - Gemini Man
call :ProcessVideo "VideoMatte/13/Gemini_Man_4K.mp4" "output_13_Gemini_Man.mp4" "log_13_Gemini_Man.txt" "Gemini Man"

echo.
echo ================================================
echo STANDARD BATCH PROCESSING COMPLETED!
echo ================================================
echo Check individual log files for details in output/standard/
echo ================================================
pause
exit /b

:ProcessVideo
set "input=%~1"
set "output=%~2"
set "logfile=%~3"
set "name=%~4"

echo === Processing %name% === > "output\standard\%logfile%"
echo Start Time: %date% %time% >> "output\standard\%logfile%"
echo. >> "output\standard\%logfile%"
echo Initial GPU Status: >> "output\standard\%logfile%"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits >> "output\standard\%logfile%" 2>&1
echo. >> "output\standard\%logfile%"
echo Input: %input% >> "output\standard\%logfile%"
echo Output: output\standard\%output% >> "output\standard\%logfile%"
echo. >> "output\standard\%logfile%"
echo Starting inference... >> "output\standard\%logfile%"
python inference.py --variant mobilenetv3 --checkpoint model/rvm_mobilenetv3.pth --device cuda --input-source "%input%" --output-type video --output-composition "output\standard\%output%" >> "output\standard\%logfile%" 2>&1
echo. >> "output\standard\%logfile%"
echo Final GPU Status: >> "output\standard\%logfile%"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits >> "output\standard\%logfile%" 2>&1
echo. >> "output\standard\%logfile%"
echo End Time: %date% %time% >> "output\standard\%logfile%"
exit /b

@echo off
echo ================================================
echo HIGH PERFORMANCE RVM BATCH PROCESSING
echo ================================================
echo Total videos to process: 15
echo Model: ResNet50 (High GPU utilization)
echo Settings: seq-chunk=8, downsample-ratio=0.25
echo Expected GPU Usage: 60-90%%
echo Results will be saved to: output/high_gpu/
echo ================================================

if not exist "output/high_gpu" mkdir "output/high_gpu"

echo.
echo Processing Video 1/15 - Sony Lens Test [HIGH GPU MODE]
call :ProcessVideo "VideoMatte/1/Sony_Lens_Test.mp4" "output_1_Sony_Lens_Test.mp4" "log_1_Sony_Lens_Test.txt" "Sony Lens Test" 8 0.25

echo.
echo Processing Video 2/15 - Sample Footage [HIGH GPU MODE]
call :ProcessVideo "VideoMatte/2/Sample_footage_.mp4" "output_2_Sample_footage.mp4" "log_2_Sample_footage.txt" "Sample Footage" 8 0.25

echo.
echo Processing Video 3/15 - Sony FR7 [HIGH GPU MODE]
call :ProcessVideo "VideoMatte/3/Sony_FR7__Cinem.MP4" "output_3_Sony_FR7.mp4" "log_3_Sony_FR7.txt" "Sony FR7" 8 0.25

echo.
echo Processing Video 4/15 - ASO Japan 8K [HIGH GPU MODE - 4K Processing]
call :ProcessVideo "VideoMatte/4/ASO_JAPAN_8K_-_.MP4" "output_4_ASO_JAPAN_8K.mp4" "log_4_ASO_JAPAN_8K.txt" "ASO Japan 8K" 6 0.125

echo.
echo Processing Video 5/15 - 4K Sample [HIGH GPU MODE - 4K Processing]
call :ProcessVideo "VideoMatte/5/4K_Sample_Video.mp4" "output_5_4K_Sample.mp4" "log_5_4K_Sample.txt" "4K Sample" 6 0.125

echo.
echo Processing Video 6/15 - Sample Footage [HIGH GPU MODE]
call :ProcessVideo "VideoMatte/6/Sample_Footage_.MP4" "output_6_Sample_Footage.mp4" "log_6_Sample_Footage.txt" "Sample Footage" 8 0.25

echo.
echo Processing Video 7/15 - Sony HDC [HIGH GPU MODE]
call :ProcessVideo "VideoMatte/7/Sony___HDC-F550.mp4" "output_7_Sony_HDC.mp4" "log_7_Sony_HDC.txt" "Sony HDC" 8 0.25

echo.
echo Processing Video 8/15 - S7Edge [HIGH GPU MODE]
call :ProcessVideo "VideoMatte/8/S7Edge__Zhiyun_.MP4" "output_8_S7Edge.mp4" "log_8_S7Edge.txt" "S7Edge" 8 0.25

echo.
echo Processing Video 9/15 - Man Utd [HIGH GPU MODE - 4K Processing]
call :ProcessVideo "VideoMatte/9/4K__Man_Utd._Vs.MP4" "output_9_Man_Utd.mp4" "log_9_Man_Utd.txt" "Man Utd" 6 0.125

echo.
echo Processing Video 10/15 - TWICE BDZ [HIGH GPU MODE]
call :ProcessVideo "VideoMatte/10/TWICE_-_BDZKore.MP4" "output_10_TWICE_BDZ.mp4" "log_10_TWICE_BDZ.txt" "TWICE BDZ" 8 0.25

echo.
echo Processing Video 11/15 - PEACE [HIGH GPU MODE]
call :ProcessVideo "VideoMatte/11/170903_PEACE_MA.MP4" "output_11a_PEACE.mp4" "log_11a_PEACE.txt" "PEACE" 8 0.25

echo.
echo Processing Video 12/15 - YOUTH [HIGH GPU MODE]
call :ProcessVideo "VideoMatte/11/A.YOUTH____Kiss.MP4" "output_11b_YOUTH.mp4" "log_11b_YOUTH.txt" "YOUTH" 8 0.25

echo.
echo Processing Video 13/15 - BTS [HIGH GPU MODE]
call :ProcessVideo "VideoMatte/12/_Beagles___BTS_.MP4" "output_12_BTS.mp4" "log_12_BTS.txt" "BTS" 8 0.25

echo.
echo Processing Video 14/15 - Gemini Man [HIGH GPU MODE - 4K Processing]
call :ProcessVideo "VideoMatte/13/Gemini_Man_4K.mp4" "output_13_Gemini_Man.mp4" "log_13_Gemini_Man.txt" "Gemini Man" 6 0.125

echo.
echo ================================================
echo HIGH PERFORMANCE BATCH PROCESSING COMPLETED!
echo ================================================
echo Performance Summary:
echo - Model: ResNet50 (High accuracy, High GPU usage)
echo - Batch Processing: 8 frames per chunk (6 for 4K)
echo - HD Processing: downsample-ratio 0.25 (better quality)
echo - 4K Processing: downsample-ratio 0.125 (maximum quality)
echo - Expected GPU Usage: 60-90%%
echo - Power Draw: 150-200W sustained
echo.
echo Check individual log files for details:
echo - output/high_gpu/log_1_Sony_Lens_Test.txt through output/high_gpu/log_13_Gemini_Man.txt
echo Output videos saved as:
echo - output/high_gpu/output_1_Sony_Lens_Test.mp4 through output/high_gpu/output_13_Gemini_Man.mp4
echo ================================================
pause
exit /b

:ProcessVideo
set "input=%~1"
set "output=%~2"
set "logfile=%~3"
set "name=%~4"
set "chunk=%~5"
set "ratio=%~6"

echo === Processing %name% === > "output\high_gpu\%logfile%"
echo Start Time: %date% %time% >> "output\high_gpu\%logfile%"
echo. >> "output\high_gpu\%logfile%"
echo Initial GPU Status: >> "output\high_gpu\%logfile%"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits >> "output\high_gpu\%logfile%" 2>&1
echo. >> "output\high_gpu\%logfile%"
echo Input: %input% >> "output\high_gpu\%logfile%"
echo Output: output\high_gpu\%output% >> "output\high_gpu\%logfile%"
echo Settings: seq-chunk=%chunk%, downsample-ratio=%ratio% >> "output\high_gpu\%logfile%"
echo. >> "output\high_gpu\%logfile%"
echo Starting inference... >> "output\high_gpu\%logfile%"
python inference.py --variant resnet50 --checkpoint model/rvm_resnet50.pth --device cuda --input-source "%input%" --output-type video --output-composition "output\high_gpu\%output%" --seq-chunk %chunk% --downsample-ratio %ratio% >> "output\high_gpu\%logfile%" 2>&1
echo. >> "output\high_gpu\%logfile%"
echo Final GPU Status: >> "output\high_gpu\%logfile%"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits >> "output\high_gpu\%logfile%" 2>&1
echo. >> "output\high_gpu\%logfile%"
echo End Time: %date% %time% >> "output\high_gpu\%logfile%"
exit /b
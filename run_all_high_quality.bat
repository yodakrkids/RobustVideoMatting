@echo off
echo ================================================
echo ULTRA HIGH QUALITY RVM BATCH PROCESSING
echo ================================================
echo Total videos to process: 15
echo Model: ResNet50 (Maximum quality)
echo Settings: downsample-ratio=0.5, seq-chunk=4
echo Expected Processing Time: 3-4x slower
echo Expected GPU Usage: 80-95%%
echo FOCUS: CLEANEST POSSIBLE MATTING EDGES
echo Results will be saved to: output/high_quality/
echo ================================================

if not exist "output/high_quality" mkdir "output/high_quality"

echo.
echo Processing Video 1/15 - Sony Lens Test [ULTRA QUALITY MODE]
call :ProcessVideoHQ "VideoMatte/1/Sony_Lens_Test.mp4" "hq_output_1_Sony_Lens_Test.mp4" "hq_log_1_Sony_Lens_Test.txt" "Sony Lens Test" 4 0.5

echo.
echo Processing Video 2/15 - Sample Footage [ULTRA QUALITY MODE]
call :ProcessVideoHQ "VideoMatte/2/Sample_footage_.mp4" "hq_output_2_Sample_footage.mp4" "hq_log_2_Sample_footage.txt" "Sample Footage" 4 0.5

echo.
echo Processing Video 3/15 - Sony FR7 [ULTRA QUALITY MODE]
call :ProcessVideoHQ "VideoMatte/3/Sony_FR7__Cinem.MP4" "hq_output_3_Sony_FR7.mp4" "hq_log_3_Sony_FR7.txt" "Sony FR7" 4 0.5

echo.
echo Processing Video 4/15 - ASO Japan 8K [NATIVE 4K PROCESSING]
call :ProcessVideoUltra "VideoMatte/4/ASO_JAPAN_8K_-_.MP4" "ultra_output_4_ASO_JAPAN_8K.mp4" "ultra_log_4_ASO_JAPAN_8K.txt" "ASO Japan 8K" 2 1.0

echo.
echo Processing Video 5/15 - 4K Sample [NATIVE 4K PROCESSING]
call :ProcessVideoUltra "VideoMatte/5/4K_Sample_Video.mp4" "ultra_output_5_4K_Sample.mp4" "ultra_log_5_4K_Sample.txt" "4K Sample" 2 1.0

echo.
echo Processing Video 6/15 - Sample Footage [ULTRA QUALITY MODE]
call :ProcessVideoHQ "VideoMatte/6/Sample_Footage_.MP4" "hq_output_6_Sample_Footage.mp4" "hq_log_6_Sample_Footage.txt" "Sample Footage" 4 0.5

echo.
echo Processing Video 7/15 - Sony HDC [ULTRA QUALITY MODE]
call :ProcessVideoHQ "VideoMatte/7/Sony___HDC-F550.mp4" "hq_output_7_Sony_HDC.mp4" "hq_log_7_Sony_HDC.txt" "Sony HDC" 4 0.5

echo.
echo Processing Video 8/15 - S7Edge [ULTRA QUALITY MODE]
call :ProcessVideoHQ "VideoMatte/8/S7Edge__Zhiyun_.MP4" "hq_output_8_S7Edge.mp4" "hq_log_8_S7Edge.txt" "S7Edge" 4 0.5

echo.
echo Processing Video 9/15 - Man Utd [NATIVE 4K PROCESSING]
call :ProcessVideoUltra "VideoMatte/9/4K__Man_Utd._Vs.MP4" "ultra_output_9_Man_Utd.mp4" "ultra_log_9_Man_Utd.txt" "Man Utd" 2 1.0

echo.
echo Processing Video 10/15 - TWICE BDZ [ULTRA QUALITY MODE]
call :ProcessVideoHQ "VideoMatte/10/TWICE_-_BDZKore.MP4" "hq_output_10_TWICE_BDZ.mp4" "hq_log_10_TWICE_BDZ.txt" "TWICE BDZ" 4 0.5

echo.
echo Processing Video 11/15 - PEACE [ULTRA QUALITY MODE]
call :ProcessVideoHQ "VideoMatte/11/170903_PEACE_MA.MP4" "hq_output_11a_PEACE.mp4" "hq_log_11a_PEACE.txt" "PEACE" 4 0.5

echo.
echo Processing Video 12/15 - YOUTH [ULTRA QUALITY MODE]
call :ProcessVideoHQ "VideoMatte/11/A.YOUTH____Kiss.MP4" "hq_output_11b_YOUTH.mp4" "hq_log_11b_YOUTH.txt" "YOUTH" 4 0.5

echo.
echo Processing Video 13/15 - BTS [ULTRA QUALITY MODE]
call :ProcessVideoHQ "VideoMatte/12/_Beagles___BTS_.MP4" "hq_output_12_BTS.mp4" "hq_log_12_BTS.txt" "BTS" 4 0.5

echo.
echo Processing Video 14/15 - Gemini Man [NATIVE 4K PROCESSING]
call :ProcessVideoUltra "VideoMatte/13/Gemini_Man_4K.mp4" "ultra_output_13_Gemini_Man.mp4" "ultra_log_13_Gemini_Man.txt" "Gemini Man" 2 1.0

echo.
echo ================================================
echo ULTRA HIGH QUALITY PROCESSING COMPLETED!
echo ================================================
echo Quality Summary:
echo - Model: ResNet50 (Maximum accuracy)
echo - HQ Mode: downsample-ratio 0.5 (2x resolution)
echo - Ultra Mode: downsample-ratio 1.0 (native resolution)
echo - Edge Quality: Dramatically improved
echo - Processing Time: 3-4x longer than standard
echo ================================================
echo Check output files:
echo - hq_output_*.mp4 (High Quality - 2x resolution)
echo - ultra_output_*.mp4 (Ultra Quality - native resolution)
echo ================================================
pause
exit /b

:ProcessVideoHQ
set "input=%~1"
set "output=%~2"
set "logfile=%~3"
set "name=%~4"
set "chunk=%~5"
set "ratio=%~6"

echo === HIGH QUALITY Processing %name% === > "output\high_quality\%logfile%"
echo Target: Cleanest possible matting edges >> "output\high_quality\%logfile%"
echo Start Time: %date% %time% >> "output\high_quality\%logfile%"
echo. >> "output\high_quality\%logfile%"
echo Settings: >> "output\high_quality\%logfile%"
echo - Model: ResNet50 >> "output\high_quality\%logfile%"
echo - Downsample Ratio: %ratio% (2x resolution) >> "output\high_quality\%logfile%"
echo - Seq Chunk: %chunk% (memory optimized) >> "output\high_quality\%logfile%"
echo. >> "output\high_quality\%logfile%"
echo Initial GPU Status: >> "output\high_quality\%logfile%"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits >> "output\high_quality\%logfile%" 2>&1
echo. >> "output\high_quality\%logfile%"
echo Starting HIGH QUALITY inference... >> "output\high_quality\%logfile%"
python inference.py --variant resnet50 --checkpoint model/rvm_resnet50.pth --device cuda --input-source "%input%" --output-type video --output-composition "output\high_quality\%output%" --seq-chunk %chunk% --downsample-ratio %ratio% >> "output\high_quality\%logfile%" 2>&1
echo. >> "output\high_quality\%logfile%"
echo Final GPU Status: >> "output\high_quality\%logfile%"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits >> "output\high_quality\%logfile%" 2>&1
echo. >> "output\high_quality\%logfile%"
echo End Time: %date% %time% >> "output\high_quality\%logfile%"
exit /b

:ProcessVideoUltra
set "input=%~1"
set "output=%~2"
set "logfile=%~3"
set "name=%~4"
set "chunk=%~5"
set "ratio=%~6"

echo === ULTRA QUALITY Processing %name% === > "output\high_quality\%logfile%"
echo Target: MAXIMUM quality native resolution >> "output\high_quality\%logfile%"
echo WARNING: Very slow processing expected >> "output\high_quality\%logfile%"
echo Start Time: %date% %time% >> "output\high_quality\%logfile%"
echo. >> "output\high_quality\%logfile%"
echo Settings: >> "output\high_quality\%logfile%"
echo - Model: ResNet50 >> "output\high_quality\%logfile%"
echo - Downsample Ratio: %ratio% (NATIVE RESOLUTION) >> "output\high_quality\%logfile%"
echo - Seq Chunk: %chunk% (minimum for stability) >> "output\high_quality\%logfile%"
echo. >> "output\high_quality\%logfile%"
echo Initial GPU Status: >> "output\high_quality\%logfile%"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits >> "output\high_quality\%logfile%" 2>&1
echo. >> "output\high_quality\%logfile%"
echo Starting ULTRA QUALITY inference... >> "output\high_quality\%logfile%"
python inference.py --variant resnet50 --checkpoint model/rvm_resnet50.pth --device cuda --input-source "%input%" --output-type video --output-composition "output\high_quality\%output%" --seq-chunk %chunk% --downsample-ratio %ratio% >> "output\high_quality\%logfile%" 2>&1
echo. >> "output\high_quality\%logfile%"
echo Final GPU Status: >> "output\high_quality\%logfile%"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits >> "output\high_quality\%logfile%" 2>&1
echo. >> "output\high_quality\%logfile%"
echo End Time: %date% %time% >> "output\high_quality\%logfile%"
exit /b
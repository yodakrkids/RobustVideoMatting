@echo off
echo Testing HIGH GPU usage during inference...

echo Starting GPU monitoring in background...
start /B powershell -Command "while($true) { nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | ForEach-Object { Write-Host \"[$(Get-Date -Format 'HH:mm:ss')] GPU: $_\" }; Start-Sleep -Seconds 1 }" > gpu_monitor.log 2>&1

echo.
echo === HIGH GPU USAGE TEST OPTIONS ===
echo 1. ResNet50 model (more GPU intensive)
echo 2. MobileNetV3 with high batch processing
echo 3. 4K processing (downsample-ratio 0.125)
echo.

echo Testing Option 1: ResNet50 model for maximum GPU usage...
python inference.py --variant resnet50 --checkpoint model/rvm_resnet50.pth --device cuda --input-source "VideoMatte/1/Sony_Lens_Test.mp4" --output-type video --output-composition "output/test_resnet50_gpu.mp4" --seq-chunk 8 --downsample-ratio 0.25

echo.
echo Testing Option 2: MobileNetV3 with high batch processing...
python inference.py --variant mobilenetv3 --checkpoint model/rvm_mobilenetv3.pth --device cuda --input-source "VideoMatte/1/Sony_Lens_Test.mp4" --output-type video --output-composition "output/test_mobilenet_batch.mp4" --seq-chunk 12 --downsample-ratio 0.25

echo.
echo Testing Option 3: 4K processing (if video supports it)...
python inference.py --variant resnet50 --checkpoint model/rvm_resnet50.pth --device cuda --input-source "VideoMatte/1/Sony_Lens_Test.mp4" --output-type video --output-composition "output/test_4k_processing.mp4" --seq-chunk 6 --downsample-ratio 0.125

echo.
echo === GPU STRESS TEST COMPLETED ===
echo Check GPU usage patterns above. You should see:
echo - ResNet50: 30-60%% GPU usage
echo - High batch: 20-40%% GPU usage  
echo - 4K processing: 40-80%% GPU usage
echo.
echo If still low usage, your video resolution might be limiting factor.

pause
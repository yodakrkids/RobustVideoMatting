@echo off
echo === GPU STRESS TEST WITH MONITORING ===

echo Starting intensive GPU monitoring (0.5 second intervals)...
start /B powershell -Command "while($true) { nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | ForEach-Object { Write-Host \"[$(Get-Date -Format 'HH:mm:ss.fff')] GPU: $_ W\" }; Start-Sleep -Milliseconds 500 }" > gpu_stress_monitor.log 2>&1

echo.
echo Running GPU stress test with multiple configurations...
echo Watch for HIGH GPU spikes during tensor operations!
echo.

python gpu_stress_test.py

echo.
echo Stress test completed. Check the monitoring output above.
echo If GPU usage was still under 50%%, the model might be I/O bound.

pause
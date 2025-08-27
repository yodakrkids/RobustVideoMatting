# Claude Settings

## Environment
```bash
conda activate rvm
```

## Setup
```bash
pip install -r requirements_inference.txt
pip install -r requirements_training.txt
```

## Models
```bash
# Download to model/ directory
wget https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth
wget https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50.pth
```

## Commands
```bash
python inference_speed_test.py
python webcam_demo.py
```

## Configuration
- HD: `downsample_ratio=0.25`
- 4K: `downsample_ratio=0.125`
- Batch size: 1
- Frame chunk: 1-12

## Claude Instructions
- 한국어로 대답하기
- Minimize natural language explanations
- Provide code and technical answers only
- Use concise responses
- Focus on implementation details
#!/usr/bin/env python3

import cv2
import torch
import numpy as np
import argparse
import sys
import os

# 다양한 모델 지원을 위한 클래스
class ModelWrapper:
    def __init__(self, model_type, device='cuda'):
        self.model_type = model_type
        self.device = device
        self.model = None
        self.rec = None
        
    def load_model(self, checkpoint_path=None):
        if self.model_type == 'rvm_mobilenetv3':
            from model import MattingNetwork
            self.model = MattingNetwork('mobilenetv3').eval().to(self.device)
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.rec = [None] * 4
            
        elif self.model_type == 'rvm_resnet50':
            from model import MattingNetwork
            self.model = MattingNetwork('resnet50').eval().to(self.device)
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.rec = [None] * 4
            
        elif self.model_type == 'rembg_u2net':
            try:
                from rembg import remove, new_session
                self.session = new_session('u2net')
                self.model = "rembg_loaded"
            except ImportError:
                raise ImportError("rembg not installed. Run: pip install rembg[gpu]")
                
        elif self.model_type == 'rembg_u2netp':
            try:
                from rembg import remove, new_session
                self.session = new_session('u2netp')
                self.model = "rembg_loaded"
            except ImportError:
                raise ImportError("rembg not installed. Run: pip install rembg[gpu]")
                
        elif self.model_type == 'rembg_human':
            try:
                from rembg import remove, new_session
                self.session = new_session('u2net_human_seg')
                self.model = "rembg_loaded"
            except ImportError:
                raise ImportError("rembg not installed. Run: pip install rembg[gpu]")
                
        elif self.model_type == 'rembg_isnet':
            try:
                from rembg import remove, new_session
                self.session = new_session('isnet-general-use')
                self.model = "rembg_loaded"
            except ImportError:
                raise ImportError("rembg not installed. Run: pip install rembg[gpu]")
                
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def process_frame(self, frame):
        """프레임 처리하여 alpha와 foreground 반환"""
        if self.model_type.startswith('rvm_'):
            return self._process_rvm(frame)
        elif self.model_type.startswith('rembg_'):
            return self._process_rembg(frame)
    
    def _process_rvm(self, frame):
        """RobustVideoMatting 모델 처리"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            fgr, pha, *self.rec = self.model(frame_tensor, *self.rec)
        
        pha = pha.squeeze().cpu().numpy()
        fgr = fgr.squeeze().permute(1, 2, 0).cpu().numpy()
        return pha, fgr
    
    def _process_rembg(self, frame):
        """REMBG 모델 처리"""
        from rembg import remove
        from PIL import Image
        
        # OpenCV BGR to PIL RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # 배경 제거
        output = remove(pil_img, session=self.session)
        output_np = np.array(output)
        
        if output_np.shape[2] == 4:  # RGBA
            # Alpha 채널 분리
            alpha = output_np[:, :, 3] / 255.0
            foreground = output_np[:, :, :3] / 255.0
        else:  # RGB (unlikely for rembg)
            alpha = np.ones((output_np.shape[0], output_np.shape[1]))
            foreground = output_np / 255.0
            
        return alpha, foreground

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default='rvm_mobilenetv3', 
                       choices=['rvm_mobilenetv3', 'rvm_resnet50', 'rembg_u2net', 
                               'rembg_u2netp', 'rembg_human', 'rembg_isnet'])
    parser.add_argument('--model-checkpoint', type=str, default='model/rvm_mobilenetv3.pth')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--webcam-id', type=int, default=0)
    parser.add_argument('--background-images', type=str, nargs='*', default=[])
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--resolution', type=str, choices=['480p', '720p', '1080p'])
    args = parser.parse_args()

    # 해상도 프리셋 처리
    if args.resolution:
        if args.resolution == '480p':
            args.width, args.height = 640, 480
        elif args.resolution == '720p':
            args.width, args.height = 1280, 720
        elif args.resolution == '1080p':
            args.width, args.height = 1920, 1080

    print(f"모델 타입: {args.model_type}")
    print(f"디바이스: {args.device}")
    print(f"해상도: {args.width}x{args.height}")

    # 모델 로드
    try:
        model_wrapper = ModelWrapper(args.model_type, args.device)
        if args.model_type.startswith('rvm_'):
            model_wrapper.load_model(args.model_checkpoint)
        else:
            model_wrapper.load_model()
        print("✓ 모델 로드 완료")
    except Exception as e:
        print(f"✗ 모델 로드 실패: {e}")
        return

    # 웹캠 초기화
    cap = cv2.VideoCapture(args.webcam_id)
    if not cap.isOpened():
        print(f"✗ 웹캠 {args.webcam_id}을 열 수 없습니다.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✓ 웹캠 초기화 완료 (실제 해상도: {actual_width}x{actual_height})")

    # 배경 이미지 로드
    background_images = []
    if args.background_images:
        for bg_path in args.background_images:
            try:
                bg_img = cv2.imread(bg_path)
                if bg_img is not None:
                    bg_img_resized = cv2.resize(bg_img, (actual_width, actual_height))
                    bg_img_rgb = cv2.cvtColor(bg_img_resized, cv2.COLOR_BGR2RGB) / 255.0
                    background_images.append(bg_img_rgb)
                    print(f"✓ 배경 이미지 로드: {bg_path}")
            except Exception as e:
                print(f"✗ 배경 이미지 오류: {e}")

    total_backgrounds = 3 + len(background_images)
    
    print(f"\n사용 가능한 모델:")
    print(f"  - RVM MobileNetV3: 빠름, 실시간")
    print(f"  - RVM ResNet50: 정확함, 고품질")
    print(f"  - REMBG U2Net: 범용")
    print(f"  - REMBG U2NetP: 가벼움")
    print(f"  - REMBG Human: 인간 전용")
    print(f"  - REMBG ISNet: 최신 범용")
    
    print(f"\n조작법:")
    print("- 'q' 키: 종료")
    print("- 'b' 키: 배경 변경")
    print("- 'f' 키: FPS 표시 토글")
    print("- 'm' 키: 모델 정보 표시")

    # 상태 변수
    background_mode = 0
    show_fps = True
    fps_count = 0
    fps_start_time = cv2.getTickCount()
    fps_display = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # FPS 계산
            fps_count += 1
            if fps_count % 10 == 0:
                fps_end_time = cv2.getTickCount()
                time_diff = (fps_end_time - fps_start_time) / cv2.getTickFrequency()
                fps_display = 10.0 / time_diff
                fps_start_time = fps_end_time

            # 모델 추론
            try:
                pha, fgr = model_wrapper.process_frame(frame)
            except Exception as e:
                print(f"추론 오류: {e}")
                continue

            # 배경 적용
            if background_mode == 0:
                # 투명 배경
                alpha_3ch = np.stack([pha] * 3, axis=2)
                result = fgr * alpha_3ch
            elif background_mode == 1:
                # 검정 배경
                result = fgr * np.stack([pha] * 3, axis=2)
            elif background_mode == 2:
                # 흰색 배경
                alpha_3ch = np.stack([pha] * 3, axis=2)
                white_bg = np.ones_like(fgr)
                result = fgr * alpha_3ch + white_bg * (1 - alpha_3ch)
            else:
                # 사용자 배경 이미지
                bg_index = background_mode - 3
                if bg_index < len(background_images):
                    alpha_3ch = np.stack([pha] * 3, axis=2)
                    bg_img = background_images[bg_index]
                    result = fgr * alpha_3ch + bg_img * (1 - alpha_3ch)
                else:
                    alpha_3ch = np.stack([pha] * 3, axis=2)
                    result = fgr * alpha_3ch

            # 화면에 표시
            result_bgr = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            original_bgr = frame
            
            # 원본과 결과를 나란히 표시
            combined = np.hstack([original_bgr, result_bgr])
            
            # 텍스트 추가
            cv2.putText(combined, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined, 'Result', (original_bgr.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 모델 정보 표시
            cv2.putText(combined, f'Model: {args.model_type}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # FPS 표시
            if show_fps:
                cv2.putText(combined, f'FPS: {fps_display:.1f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(combined, f'Device: {args.device.upper()}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # 배경 정보
            bg_names = ["Transparent", "Black", "White"] + [f"Image{i+1}" for i in range(len(background_images))]
            current_bg = bg_names[background_mode] if background_mode < len(bg_names) else "Unknown"
            cv2.putText(combined, f'Background: {current_bg}', (10, combined.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(f'Multi-Model Matting Demo - {args.model_type}', combined)

            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('b'):
                background_mode = (background_mode + 1) % total_backgrounds
                bg_names_ko = ["투명", "검정", "흰색"] + [f"이미지{i+1}" for i in range(len(background_images))]
                print(f"배경 변경: {bg_names_ko[background_mode]}")
            elif key == ord('f'):
                show_fps = not show_fps
                print(f"FPS 표시: {'켜짐' if show_fps else '꺼짐'}")
            elif key == ord('m'):
                print(f"현재 모델: {args.model_type}, FPS: {fps_display:.1f}, 배경: {current_bg}")

    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
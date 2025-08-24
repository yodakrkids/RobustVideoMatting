#!/usr/bin/env python3

import cv2
import torch
import numpy as np
from model import MattingNetwork
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-checkpoint', type=str, default='model/rvm_mobilenetv3.pth')
    parser.add_argument('--model-type', type=str, default='mobilenetv3', choices=['mobilenetv3', 'resnet50'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--webcam-id', type=int, default=0)
    parser.add_argument('--background-images', type=str, nargs='*', default=[], 
                       help='배경 이미지 파일 경로들 (예: bg1.jpg bg2.png)')
    parser.add_argument('--width', type=int, default=640, help='화면 너비 (기본값: 640)')
    parser.add_argument('--height', type=int, default=480, help='화면 높이 (기본값: 480)')
    parser.add_argument('--resolution', type=str, choices=['480p', '720p', '1080p'], 
                       help='미리 정의된 해상도 (480p=640x480, 720p=1280x720, 1080p=1920x1080)')
    args = parser.parse_args()

    # 해상도 프리셋 처리
    if args.resolution:
        if args.resolution == '480p':
            args.width, args.height = 640, 480
        elif args.resolution == '720p':
            args.width, args.height = 1280, 720
        elif args.resolution == '1080p':
            args.width, args.height = 1920, 1080

    print(f"디바이스: {args.device}")
    print(f"모델 타입: {args.model_type}")
    print(f"웹캠 ID: {args.webcam_id}")
    print(f"해상도: {args.width}x{args.height}")
    if args.background_images:
        print(f"배경 이미지: {len(args.background_images)}개 로드됨")

    # 모델 로드
    model = MattingNetwork(args.model_type).eval()
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=args.device))
    model = model.to(args.device)
    print("✓ 모델 로드 완료")

    # 웹캠 초기화
    cap = cv2.VideoCapture(args.webcam_id)
    if not cap.isOpened():
        print(f"✗ 웹캠 {args.webcam_id}을 열 수 없습니다.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # 실제 설정된 해상도 확인
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
                    # 현재 해상도에 맞게 리사이즈
                    bg_img_resized = cv2.resize(bg_img, (actual_width, actual_height))
                    # BGR to RGB 변환
                    bg_img_rgb = cv2.cvtColor(bg_img_resized, cv2.COLOR_BGR2RGB) / 255.0
                    background_images.append(bg_img_rgb)
                    print(f"✓ 배경 이미지 로드: {bg_path} ({actual_width}x{actual_height}으로 리사이즈)")
                else:
                    print(f"✗ 배경 이미지 로드 실패: {bg_path}")
            except Exception as e:
                print(f"✗ 배경 이미지 오류 ({bg_path}): {e}")
    
    total_backgrounds = 3 + len(background_images)  # 투명, 검정, 흰색 + 사용자 이미지들
    print(f"총 {total_backgrounds}개 배경 모드 사용 가능")

    print("\n조작법:")
    print("- 'q' 키: 종료")
    if background_images:
        print(f"- 'b' 키: 배경 변경 (투명 → 검정 → 흰색 → 이미지{len(background_images)}개)")
    else:
        print("- 'b' 키: 배경 변경 (투명 → 검정 → 흰색)")
    print("- 'f' 또는 'F' 키: FPS 정보 토글")
    print("- 'r' 키: 현재 상태 확인")
    print("- ESC: 종료")
    print("- 창이 활성화된 상태에서 키를 눌러주세요!")

    # 상태 변수
    rec = [None] * 4  # 순환 버퍼
    background_mode = 0  # 0: 투명, 1: 검정, 2: 흰색
    show_fps = True  # FPS 표시 여부
    
    # FPS 계산 변수
    fps_count = 0
    fps_start_time = cv2.getTickCount()
    fps_display = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("웹캠에서 프레임을 읽을 수 없습니다.")
                break

            # FPS 계산
            fps_count += 1
            if fps_count % 10 == 0:  # 10프레임마다 FPS 업데이트
                fps_end_time = cv2.getTickCount()
                time_diff = (fps_end_time - fps_start_time) / cv2.getTickFrequency()
                fps_display = 10.0 / time_diff
                fps_start_time = fps_end_time

            # 프레임 전처리
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().div(255).unsqueeze(0).to(args.device)

            # 추론
            with torch.no_grad():
                fgr, pha, *rec = model(frame_tensor, *rec)

            # 후처리
            pha = pha.squeeze().cpu().numpy()
            fgr = fgr.squeeze().permute(1, 2, 0).cpu().numpy()

            # 배경 설정
            if background_mode == 0:
                # 투명 배경 (알파 채널로 표시)
                alpha_3ch = np.stack([pha] * 3, axis=2)
                result = fgr * alpha_3ch
            elif background_mode == 1:
                # 검정 배경
                result = fgr * np.stack([pha] * 3, axis=2)
            elif background_mode == 2:
                # 흰색 배경
                alpha_3ch = np.stack([pha] * 3, axis=2)
                white_bg = np.ones_like(fgr)  # 흰색 배경 생성
                result = fgr * alpha_3ch + white_bg * (1 - alpha_3ch)
            else:
                # 사용자 배경 이미지
                bg_index = background_mode - 3
                if bg_index < len(background_images):
                    alpha_3ch = np.stack([pha] * 3, axis=2)
                    bg_img = background_images[bg_index]
                    result = fgr * alpha_3ch + bg_img * (1 - alpha_3ch)
                else:
                    # 인덱스 오류 시 투명 배경으로 대체
                    alpha_3ch = np.stack([pha] * 3, axis=2)
                    result = fgr * alpha_3ch

            # 화면에 표시
            result_bgr = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            original_bgr = cv2.cvtColor((frame_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # 원본과 결과를 나란히 표시
            combined = np.hstack([original_bgr, result_bgr])
            
            # 텍스트 추가
            cv2.putText(combined, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(combined, 'Matting Result', (original_bgr.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # FPS 표시 (토글 가능)
            if show_fps:
                cv2.putText(combined, f'FPS: {fps_display:.1f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                # GPU 정보와 해상도도 함께 표시
                cv2.putText(combined, f'Device: {args.device.upper()}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(combined, f'Resolution: {actual_width}x{actual_height}', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # 기타 정보 표시
            bg_names = ["Transparent", "Black", "White"] + [f"Image{i+1}" for i in range(len(background_images))]
            current_bg = bg_names[background_mode] if background_mode < len(bg_names) else "Unknown"
            cv2.putText(combined, f'Background: {current_bg}', (10, combined.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, 'Q:quit B:background F:fps R:status', (10, combined.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('RobustVideoMatting - Webcam Demo', combined)

            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            
            # 디버깅: 키 입력 확인
            if key != 255:  # 255는 키 입력이 없을 때의 값
                print(f"키 입력 감지: {key} (문자: '{chr(key) if 32 <= key <= 126 else 'N/A'}')")
            
            if key == ord('q') or key == 27:  # 'q' 또는 ESC
                break
            elif key == ord('b'):  # 'b' - 배경 변경
                background_mode = (background_mode + 1) % total_backgrounds
                bg_names = ["투명", "검정", "흰색"] + [f"이미지{i+1}" for i in range(len(background_images))]
                print(f"배경 모드 변경: {bg_names[background_mode]}")
            elif key == ord('f') or key == ord('F'):  # 'f' 또는 'F' - FPS 표시 토글
                show_fps = not show_fps
                print(f"FPS 표시: {'켜짐' if show_fps else '꺼짐'}")
            elif key == ord('r'):  # 'r' - 디버깅 정보
                bg_names = ["투명", "검정", "흰색"] + [f"이미지{i+1}" for i in range(len(background_images))]
                current_bg = bg_names[background_mode] if background_mode < len(bg_names) else "알 수 없음"
                print(f"현재 상태 - FPS 표시: {show_fps}, 배경 모드: {current_bg}, 현재 FPS: {fps_display:.1f}")

    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨")
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✓ 정리 완료")

if __name__ == '__main__':
    main()
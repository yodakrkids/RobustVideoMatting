#!/usr/bin/env python3

import cv2
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--webcam-id', type=int, default=0, help='웹캠 ID (기본값: 0)')
    parser.add_argument('--width', type=int, default=1280, help='설정할 너비 (기본값: 1280)')
    parser.add_argument('--height', type=int, default=720, help='설정할 높이 (기본값: 720)')
    parser.add_argument('--fps', type=int, default=60, help='설정할 FPS (기본값: 60)')
    args = parser.parse_args()
    
    print(f"웹캠 테스트 시작...")
    print(f"웹캠 ID: {args.webcam_id}")
    print(f"설정 시도 - 해상도: {args.width}x{args.height}, FPS: {args.fps}")
    
    # 웹캠 초기화
    cap = cv2.VideoCapture(args.webcam_id)
    if not cap.isOpened():
        print(f"✗ 웹캠 {args.webcam_id}을 열 수 없습니다.")
        return
    
    # 웹캠 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    
    # 실제 설정된 값들 확인
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 웹캠 추가 정보
    brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    contrast = cap.get(cv2.CAP_PROP_CONTRAST)
    saturation = cap.get(cv2.CAP_PROP_SATURATION)
    hue = cap.get(cv2.CAP_PROP_HUE)
    gain = cap.get(cv2.CAP_PROP_GAIN)
    exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
    
    # 백엔드 정보
    backend = cap.getBackendName()
    
    print(f"✓ 웹캠 초기화 완료")
    print(f"실제 해상도: {actual_width}x{actual_height}")
    print(f"실제 FPS: {actual_fps}")
    print(f"백엔드: {backend}")
    print(f"밝기: {brightness}, 대비: {contrast}")
    print(f"채도: {saturation}, 색조: {hue}")
    print(f"게인: {gain}, 노출: {exposure}")
    print(f"\n'q' 또는 ESC로 종료")
    
    # FPS 측정 변수
    frame_count = 0
    start_time = time.time()
    fps_display = 0.0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("웹캠에서 프레임을 읽을 수 없습니다.")
                break
            
            frame_count += 1
            
            # FPS 계산 (10프레임마다)
            if frame_count % 10 == 0:
                current_time = time.time()
                fps_display = 10.0 / (current_time - start_time)
                start_time = current_time
            
            # 웹캠 정보를 화면에 표시
            info_y = 30
            line_height = 25
            
            # 기본 정보
            cv2.putText(frame, f'Webcam ID: {args.webcam_id}', (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            info_y += line_height
            
            cv2.putText(frame, f'Backend: {backend}', (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            info_y += line_height
            
            # 해상도 정보
            cv2.putText(frame, f'Resolution: {actual_width}x{actual_height}', (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            info_y += line_height
            
            # FPS 정보
            cv2.putText(frame, f'Set FPS: {actual_fps} / Real FPS: {fps_display:.1f}', (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            info_y += line_height
            
            # 화질 설정
            cv2.putText(frame, f'Brightness: {brightness:.1f}', (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            info_y += line_height - 5
            
            cv2.putText(frame, f'Contrast: {contrast:.1f}', (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            info_y += line_height - 5
            
            cv2.putText(frame, f'Saturation: {saturation:.1f}', (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            info_y += line_height - 5
            
            cv2.putText(frame, f'Hue: {hue:.1f}', (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            info_y += line_height - 5
            
            cv2.putText(frame, f'Gain: {gain:.1f}', (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            info_y += line_height - 5
            
            cv2.putText(frame, f'Exposure: {exposure:.1f}', (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            info_y += line_height - 5
            
            # 프레임 카운터
            cv2.putText(frame, f'Frame: {frame_count}', (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 255, 128), 1)
            
            # 하단 정보
            cv2.putText(frame, 'Press Q or ESC to quit', (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 우상단에 현재 시간
            current_time_str = time.strftime("%H:%M:%S", time.localtime())
            cv2.putText(frame, current_time_str, (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 255), 2)
            
            cv2.imshow('Webcam Hardware Test', frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' 또는 ESC
                break
    
    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨")
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✓ 테스트 완료")

if __name__ == '__main__':
    main()
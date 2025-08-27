#!/usr/bin/env python3

import cv2
import torch
import numpy as np
from model import MattingNetwork
import argparse
import threading
import queue
import time

class FrameCaptureThread(threading.Thread):
    def __init__(self, cap, input_queue, stop_event):
        super().__init__()
        self.cap = cap
        self.input_queue = input_queue
        self.stop_event = stop_event
        self.daemon = True
        
    def run(self):
        frame_count = 0
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                print("웹캠에서 프레임을 읽을 수 없습니다.")
                break
            
            frame_count += 1
            try:
                # 큐가 가득 찬 경우 가장 오래된 프레임 제거
                if self.input_queue.full():
                    try:
                        self.input_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                # 타임스탬프와 함께 프레임 저장
                timestamp = time.time()
                self.input_queue.put((frame_count, frame, timestamp), timeout=0.01)
            except queue.Full:
                # 큐가 가득 찬 경우 프레임 드롭
                continue

class CompositeThread(threading.Thread):
    def __init__(self, matting_queue, output_queue, stop_event, background_images, display_mode_ref, background_mode_ref, use_opencv_gpu=False, use_torch_gpu=False, device='cuda'):
        super().__init__()
        self.matting_queue = matting_queue
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.background_images = background_images
        self.display_mode_ref = display_mode_ref  # 참조로 전달
        self.background_mode_ref = background_mode_ref  # 참조로 전달
        self.use_opencv_gpu = use_opencv_gpu
        self.use_torch_gpu = use_torch_gpu
        self.device = device
        self.daemon = True
        
    def run(self):
        while not self.stop_event.is_set():
            try:
                frame_count, original_frame, fgr, pha, input_timestamp, output_timestamp = self.matting_queue.get(timeout=0.1)
                
                display_mode = self.display_mode_ref[0]
                background_mode = self.background_mode_ref[0]
                
                # 배경 합성 처리
                if display_mode == 0:
                    # 원본과 결과를 나란히 표시 (비교 모드)
                    result = self._composite_background(fgr, pha, background_mode)
                    result_bgr = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    original_bgr = cv2.cvtColor((original_frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    combined = np.hstack([original_bgr, result_bgr])
                    title1, title2 = 'Original', 'Matting Result'
                elif display_mode == 1:
                    # 결과만 표시 모드
                    result = self._composite_background(fgr, pha, background_mode)
                    combined = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    title1, title2 = 'Matting Result', None
                else:
                    # matte(알파) 채널만 표시 모드
                    matte_display = (pha * 255).astype(np.uint8)
                    combined = cv2.cvtColor(matte_display, cv2.COLOR_GRAY2BGR)
                    title1, title2 = 'Alpha Matte', None
                
                # 출력 큐에 결과 전달
                composite_timestamp = time.time()
                if self.output_queue.full():
                    try:
                        self.output_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.output_queue.put((frame_count, combined, title1, title2, input_timestamp, output_timestamp, composite_timestamp), timeout=0.01)
                
            except queue.Empty:
                continue
            except queue.Full:
                continue
    
    def _composite_background(self, fgr, pha, background_mode):
        """ 배경 합성 처리 """
        if self.use_torch_gpu:
            return self._composite_torch_gpu(fgr, pha, background_mode)
        elif self.use_opencv_gpu:
            return self._composite_opencv_gpu(fgr, pha, background_mode)
        else:
            return self._composite_cpu(fgr, pha, background_mode)
    
    def _composite_cpu(self, fgr, pha, background_mode):
        """ CPU 기반 배경 합성 """
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
            if bg_index < len(self.background_images):
                alpha_3ch = np.stack([pha] * 3, axis=2)
                bg_img = self.background_images[bg_index]
                result = fgr * alpha_3ch + bg_img * (1 - alpha_3ch)
            else:
                # 인덱스 오류 시 투명 배경으로 대체
                alpha_3ch = np.stack([pha] * 3, axis=2)
                result = fgr * alpha_3ch
        return result
    
    def _composite_opencv_gpu(self, fgr, pha, background_mode):
        """ OpenCV GPU 기반 배경 합성 """
        try:
            # NumPy 배열을 GPU Mat으로 업로드
            gpu_fgr = cv2.cuda_GpuMat()
            gpu_fgr.upload(fgr)
            
            gpu_pha = cv2.cuda_GpuMat()
            gpu_pha.upload(pha)
            
            if background_mode == 0:
                # 투명 배경
                gpu_alpha_3ch = cv2.cuda.merge([gpu_pha, gpu_pha, gpu_pha])
                gpu_result = cv2.cuda.multiply(gpu_fgr, gpu_alpha_3ch)
            elif background_mode == 1:
                # 검정 배경
                gpu_alpha_3ch = cv2.cuda.merge([gpu_pha, gpu_pha, gpu_pha])
                gpu_result = cv2.cuda.multiply(gpu_fgr, gpu_alpha_3ch)
            elif background_mode == 2:
                # 흰색 배경
                gpu_alpha_3ch = cv2.cuda.merge([gpu_pha, gpu_pha, gpu_pha])
                gpu_white = cv2.cuda_GpuMat(fgr.shape, cv2.CV_32FC3)
                gpu_white.setTo((1.0, 1.0, 1.0))
                
                gpu_fg_part = cv2.cuda.multiply(gpu_fgr, gpu_alpha_3ch)
                gpu_ones = cv2.cuda_GpuMat(gpu_alpha_3ch.size(), cv2.CV_32FC3)
                gpu_ones.setTo((1.0, 1.0, 1.0))
                gpu_inv_alpha = cv2.cuda.subtract(gpu_ones, gpu_alpha_3ch)
                gpu_bg_part = cv2.cuda.multiply(gpu_white, gpu_inv_alpha)
                gpu_result = cv2.cuda.add(gpu_fg_part, gpu_bg_part)
            else:
                # 사용자 배경 이미지 (여전히 CPU에서 처리)
                bg_index = background_mode - 3
                if bg_index < len(self.background_images):
                    alpha_3ch = np.stack([pha] * 3, axis=2)
                    bg_img = self.background_images[bg_index]
                    result = fgr * alpha_3ch + bg_img * (1 - alpha_3ch)
                    return result
                else:
                    gpu_alpha_3ch = cv2.cuda.merge([gpu_pha, gpu_pha, gpu_pha])
                    gpu_result = cv2.cuda.multiply(gpu_fgr, gpu_alpha_3ch)
            
            # GPU에서 CPU로 다운로드
            result = gpu_result.download()
            return result
            
        except Exception as e:
            print(f"OpenCV GPU 오류, CPU 모드로 돌아감: {e}")
            return self._composite_cpu(fgr, pha, background_mode)
    
    def _composite_torch_gpu(self, fgr, pha, background_mode):
        """ PyTorch GPU 기반 배경 합성 """
        try:
            import torch
            
            # NumPy를 PyTorch 텐서로 변환 및 GPU로 이동
            device = torch.device(self.device)
            fgr_tensor = torch.from_numpy(fgr).to(device)
            pha_tensor = torch.from_numpy(pha).to(device)
            
            if background_mode == 0:
                # 투명 배경
                alpha_3ch = pha_tensor.unsqueeze(-1).expand(-1, -1, 3)
                result_tensor = fgr_tensor * alpha_3ch
            elif background_mode == 1:
                # 검정 배경
                alpha_3ch = pha_tensor.unsqueeze(-1).expand(-1, -1, 3)
                result_tensor = fgr_tensor * alpha_3ch
            elif background_mode == 2:
                # 흰색 배경
                alpha_3ch = pha_tensor.unsqueeze(-1).expand(-1, -1, 3)
                white_bg = torch.ones_like(fgr_tensor)
                result_tensor = fgr_tensor * alpha_3ch + white_bg * (1 - alpha_3ch)
            else:
                # 사용자 배경 이미지
                bg_index = background_mode - 3
                if bg_index < len(self.background_images):
                    alpha_3ch = pha_tensor.unsqueeze(-1).expand(-1, -1, 3)
                    bg_tensor = torch.from_numpy(self.background_images[bg_index]).to(device)
                    result_tensor = fgr_tensor * alpha_3ch + bg_tensor * (1 - alpha_3ch)
                else:
                    alpha_3ch = pha_tensor.unsqueeze(-1).expand(-1, -1, 3)
                    result_tensor = fgr_tensor * alpha_3ch
            
            # GPU에서 CPU로 이동 및 NumPy로 변환
            result = result_tensor.cpu().numpy()
            return result
            
        except Exception as e:
            print(f"PyTorch GPU 오류, CPU 모드로 돌아감: {e}")
            return self._composite_cpu(fgr, pha, background_mode)

class MattingThread(threading.Thread):
    def __init__(self, model, device, input_queue, matting_queue, stop_event):
        super().__init__()
        self.model = model
        self.device = device
        self.input_queue = input_queue
        self.matting_queue = matting_queue
        self.stop_event = stop_event
        self.daemon = True
        self.rec = [None] * 4  # 순환 버퍼
        
    def run(self):
        while not self.stop_event.is_set():
            try:
                frame_count, frame, input_timestamp = self.input_queue.get(timeout=0.1)
                
                # 프레임 전처리
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
                
                # 추론
                with torch.no_grad():
                    fgr, pha, *self.rec = self.model(frame_tensor, *self.rec)
                
                # 후처리
                pha = pha.squeeze().cpu().numpy()
                fgr = fgr.squeeze().permute(1, 2, 0).cpu().numpy()
                
                # matting_queue가 가득 찬 경우 가장 오래된 결과 제거
                if self.matting_queue.full():
                    try:
                        self.matting_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                # 처리 완료 타임스탬프와 함께 결과 저장
                output_timestamp = time.time()
                
                # matting_queue가 가득 찬 경우 가장 오래된 결과 제거
                if self.matting_queue.full():
                    try:
                        self.matting_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.matting_queue.put((frame_count, frame_rgb, fgr, pha, input_timestamp, output_timestamp), timeout=0.01)
                
            except queue.Empty:
                continue
            except queue.Full:
                continue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-checkpoint', type=str, default='model/rvm_mobilenetv3.pth')
    parser.add_argument('--model-type', type=str, default='mobilenetv3', choices=['mobilenetv3', 'resnet50'])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--webcam-id', type=int, default=0)
    parser.add_argument('--background-images', type=str, nargs='*', default=[], 
                       help='배경 이미지 파일 경로들 (예: bg1.jpg bg2.png)')
    parser.add_argument('--width', type=int, default=1280, help='화면 너비 (기본값: 1280)')
    parser.add_argument('--height', type=int, default=720, help='화면 높이 (기본값: 720)')
    parser.add_argument('--resolution', type=str, choices=['480p', '720p', '1080p'], 
                       help='미리 정의된 해상도 (480p=640x480, 720p=1280x720, 1080p=1920x1080)')
    parser.add_argument('--queue-size', type=int, default=2, help='큐 크기 (기본값: 2)')
    parser.add_argument('--target-fps', type=int, default=60, help='타겟 FPS (기본값: 60)')
    parser.add_argument('--use-opencv-gpu', action='store_true', help='OpenCV GPU 가속 사용')
    parser.add_argument('--use-torch-gpu', action='store_true', help='PyTorch GPU 알파블렌딩 사용')
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
    print(f"큐 크기: {args.queue_size}")
    print(f"타겟 FPS: {args.target_fps}")
    print(f"OpenCV GPU: {args.use_opencv_gpu}")
    print(f"PyTorch GPU 블렌딩: {args.use_torch_gpu}")
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
    print("- 's' 키: 통계 정보 표시")
    print("- 'o' 키: 화면 모드 순환 (비교 → 결과만 → matte만)")
    print("- ESC: 종료")
    print("- 창이 활성화된 상태에서 키를 눌러주세요!")

    # 큐 초기화 (3단계 파이프라인)
    input_queue = queue.Queue(maxsize=args.queue_size)    # 캡처 → 매팅
    matting_queue = queue.Queue(maxsize=args.queue_size)  # 매팅 → 합성
    output_queue = queue.Queue(maxsize=args.queue_size)   # 합성 → 화면표시
    stop_event = threading.Event()
    
    # 공유 변수 (참조 전달용)
    display_mode_ref = [0]  # 비교 모드
    background_mode_ref = [0]  # 투명 배경

    # 스레드 시작
    capture_thread = FrameCaptureThread(cap, input_queue, stop_event)
    matting_thread = MattingThread(model, args.device, input_queue, matting_queue, stop_event)
    composite_thread = CompositeThread(matting_queue, output_queue, stop_event, background_images, 
                                     display_mode_ref, background_mode_ref, 
                                     args.use_opencv_gpu, args.use_torch_gpu, args.device)
    
    capture_thread.start()
    matting_thread.start()
    composite_thread.start()
    print("✓ 멀티스레드 시작 완료 (4개 스레드)")

    # 상태 변수 (로컬 변수로 유지, 참조로 스레드에 전달)
    show_fps = True  # FPS 표시 여부
    show_stats = False  # 통계 정보 표시 여부
    
    # FPS 계산 변수
    fps_count = 0
    fps_start_time = time.time()
    fps_display = 0.0
    
    # 통계 변수
    processed_frames = 0
    dropped_frames = 0
    last_frame_count = 0
    
    # FPS 제한 변수
    target_frame_time = 1.0 / args.target_fps if args.target_fps > 0 else 0
    last_frame_time = time.time()
    
    # Latency 측정 변수
    latency_history = []
    avg_latency = 0.0
    max_latency = 0.0
    min_latency = float('inf')

    try:
        while True:
            try:
                # 합성 결과 가져오기
                frame_count, combined, title1, title2, input_timestamp, output_timestamp, composite_timestamp = output_queue.get(timeout=0.1)
                processed_frames += 1
                
                # Latency 계산 (전체 파이프라인)
                display_timestamp = time.time()
                total_latency = (display_timestamp - input_timestamp) * 1000  # ms 단위
                processing_latency = (output_timestamp - input_timestamp) * 1000  # AI 추론 시간
                composite_latency = (composite_timestamp - output_timestamp) * 1000  # 합성 시간
                
                # Latency 통계 업데이트
                latency_history.append(total_latency)
                if len(latency_history) > 30:  # 최근 30프레임만 유지
                    latency_history.pop(0)
                    
                avg_latency = sum(latency_history) / len(latency_history)
                max_latency = max(max_latency, total_latency)
                min_latency = min(min_latency, total_latency)
                
                # 프레임 드롭 계산
                if last_frame_count > 0:
                    frame_diff = frame_count - last_frame_count - 1
                    if frame_diff > 0:
                        dropped_frames += frame_diff
                last_frame_count = frame_count

                # FPS 계산
                fps_count += 1
                if fps_count % 10 == 0:  # 10프레임마다 FPS 업데이트
                    current_time = time.time()
                    time_diff = current_time - fps_start_time
                    fps_display = 10.0 / time_diff
                    fps_start_time = current_time

                
                # 텍스트 추가 (이미 CompositeThread에서 처리됨)
                if title1:
                    cv2.putText(combined, title1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if title2:
                    # 비교 모드에서 두 번째 제목
                    width_offset = combined.shape[1] // 2 + 10
                    cv2.putText(combined, title2, (width_offset, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # FPS 표시 (토글 가능)
                if show_fps:
                    cv2.putText(combined, f'FPS: {fps_display:.1f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    # GPU 정보와 해상도도 함께 표시
                    cv2.putText(combined, f'Device: {args.device.upper()}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(combined, f'Resolution: {actual_width}x{actual_height}', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(combined, f'Target FPS: {args.target_fps}', (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    # Latency 정보 표시
                    cv2.putText(combined, f'Latency: {avg_latency:.1f}ms (avg)', (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 2)
                    cv2.putText(combined, f'Min/Max: {min_latency:.1f}/{max_latency:.1f}ms', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 128), 1)
                    composite_method = 'PyTorch-GPU' if args.use_torch_gpu else ('OpenCV-GPU' if args.use_opencv_gpu else 'CPU')
                    cv2.putText(combined, f'AI: {processing_latency:.1f}ms, Composite({composite_method}): {composite_latency:.1f}ms', (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 128), 1)
                
                # 통계 정보 표시 (토글 가능)
                if show_stats:
                    cv2.putText(combined, f'Processed: {processed_frames}', (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
                    cv2.putText(combined, f'Dropped: {dropped_frames}', (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
                    cv2.putText(combined, f'Input Q: {input_queue.qsize()}/{args.queue_size}', (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
                    cv2.putText(combined, f'Matting Q: {matting_queue.qsize()}/{args.queue_size}', (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
                    cv2.putText(combined, f'Output Q: {output_queue.qsize()}/{args.queue_size}', (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 1)
                
                # 기타 정보 표시
                bg_names = ["Transparent", "Black", "White"] + [f"Image{i+1}" for i in range(len(background_images))]
                current_bg = bg_names[background_mode_ref[0]] if background_mode_ref[0] < len(bg_names) else "Unknown"
                cv2.putText(combined, f'Background: {current_bg}', (10, combined.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(combined, 'Q:quit B:background F:fps S:stats O:mode R:status', (10, combined.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow('RobustVideoMatting - Webcam Demo (MultiThread)', combined)
                
                # FPS 제한 적용
                if target_frame_time > 0:
                    current_time = time.time()
                    elapsed = current_time - last_frame_time
                    sleep_time = target_frame_time - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    last_frame_time = time.time()

            except queue.Empty:
                # 결과가 없으면 잠시 대기
                if target_frame_time > 0:
                    time.sleep(min(0.01, target_frame_time))
                else:
                    time.sleep(0.01)
                
            # 키 입력 처리 (FPS 제한과 무관하게 빠른 응답)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' 또는 ESC
                break
            elif key == ord('b'):  # 'b' - 배경 변경
                background_mode_ref[0] = (background_mode_ref[0] + 1) % total_backgrounds
                bg_names = ["투명", "검정", "흰색"] + [f"이미지{i+1}" for i in range(len(background_images))]
                print(f"배경 모드 변경: {bg_names[background_mode_ref[0]]}")
            elif key == ord('f') or key == ord('F'):  # 'f' 또는 'F' - FPS 표시 토글
                show_fps = not show_fps
                print(f"FPS 표시: {'켜짐' if show_fps else '꺼짐'}")
            elif key == ord('s'):  # 's' - 통계 정보 토글
                show_stats = not show_stats
                print(f"통계 정보 표시: {'켜짐' if show_stats else '꺼짐'}")
            elif key == ord('o'):  # 'o' - 화면 모드 순환
                display_mode_ref[0] = (display_mode_ref[0] + 1) % 3
                mode_names = ["비교 모드", "결과만 모드", "matte 모드"]
                print(f"화면 모드: {mode_names[display_mode_ref[0]]}")
            elif key == ord('r'):  # 'r' - 디버깅 정보
                bg_names = ["투명", "검정", "흰색"] + [f"이미지{i+1}" for i in range(len(background_images))]
                current_bg = bg_names[background_mode_ref[0]] if background_mode_ref[0] < len(bg_names) else "알 수 없음"
                mode_names = ["비교", "결과만", "matte"]
                print(f"현재 상태 - FPS 표시: {show_fps}, 통계 표시: {show_stats}, 화면 모드: {mode_names[display_mode_ref[0]]}")
                print(f"타겟 FPS: {args.target_fps}, 현재 FPS: {fps_display:.1f}")
                composite_method = 'PyTorch-GPU' if args.use_torch_gpu else ('OpenCV-GPU' if args.use_opencv_gpu else 'CPU')
                print(f"배경 합성 방식: {composite_method}")
                print(f"Latency - 전체: {avg_latency:.1f}ms, AI: {processing_latency:.1f}ms, 합성: {composite_latency:.1f}ms")
                print(f"배경 모드: {current_bg}, 현재 FPS: {fps_display:.1f}")
                print(f"처리된 프레임: {processed_frames}, 드롭된 프레임: {dropped_frames}")
                print(f"큐 상태 - 입력: {input_queue.qsize()}/{args.queue_size}, 매팅: {matting_queue.qsize()}/{args.queue_size}, 출력: {output_queue.qsize()}/{args.queue_size}")

    except KeyboardInterrupt:
        print("\n사용자에 의해 중단됨")
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        # 스레드 종료
        print("스레드 종료 중...")
        stop_event.set()
        
        # 스레드 종료 대기 (타임아웃 포함)
        capture_thread.join(timeout=2)
        matting_thread.join(timeout=2)
        composite_thread.join(timeout=2)
        
        cap.release()
        cv2.destroyAllWindows()
        print("✓ 정리 완료")

if __name__ == '__main__':
    main()
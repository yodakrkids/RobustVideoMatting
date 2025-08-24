#!/usr/bin/env python3

import sys
import time
import traceback
import numpy as np
from PIL import Image

def test_rembg_installation():
    """REMBG 설치 및 기능 테스트"""
    
    print("=" * 60)
    print("🔍 REMBG 설치 테스트 시작")
    print("=" * 60)
    
    # 1. 기본 import 테스트
    print("\n1️⃣ 기본 import 테스트...")
    try:
        import rembg
        print(f"   ✓ rembg 버전: {rembg.__version__ if hasattr(rembg, '__version__') else 'Unknown'}")
    except ImportError as e:
        print(f"   ✗ rembg import 실패: {e}")
        print("   💡 해결방법: pip install rembg[gpu]")
        return False
    
    # 2. 핵심 함수 import 테스트
    print("\n2️⃣ 핵심 함수 import 테스트...")
    try:
        from rembg import remove, new_session
        print("   ✓ remove, new_session 함수 import 성공")
    except ImportError as e:
        print(f"   ✗ 핵심 함수 import 실패: {e}")
        return False
    
    # 3. 사용 가능한 모델 확인
    print("\n3️⃣ 사용 가능한 모델 확인...")
    available_models = [
        'u2net', 'u2netp', 'u2net_human_seg', 
        'silueta', 'u2net_cloth_seg', 'isnet-general-use', 
        'isnet-anime', 'sam'
    ]
    
    working_models = []
    for model_name in available_models:
        try:
            session = new_session(model_name)
            working_models.append(model_name)
            print(f"   ✓ {model_name}: 사용 가능")
        except Exception as e:
            print(f"   ✗ {model_name}: 오류 ({str(e)[:50]}...)")
    
    if not working_models:
        print("   ⚠️  사용 가능한 모델이 없습니다!")
        return False
    
    # 4. 테스트 이미지 생성
    print("\n4️⃣ 테스트 이미지 생성...")
    try:
        # 간단한 테스트 이미지 생성 (사람 형태)
        test_img = create_test_image()
        print("   ✓ 테스트 이미지 생성 완료")
    except Exception as e:
        print(f"   ✗ 테스트 이미지 생성 실패: {e}")
        return False
    
    # 5. 실제 배경 제거 테스트
    print("\n5️⃣ 배경 제거 기능 테스트...")
    
    # 가장 안정적인 모델로 테스트
    test_models = ['u2netp', 'u2net', 'isnet-general-use']
    success_count = 0
    
    for model_name in test_models:
        if model_name not in working_models:
            continue
            
        try:
            print(f"   🔄 {model_name} 모델 테스트 중...")
            start_time = time.time()
            
            session = new_session(model_name)
            result = remove(test_img, session=session)
            
            end_time = time.time()
            process_time = end_time - start_time
            
            # 결과 검증
            result_np = np.array(result)
            
            if len(result_np.shape) == 3 and result_np.shape[2] == 4:
                alpha_channel = result_np[:, :, 3]
                unique_values = len(np.unique(alpha_channel))
                
                print(f"      ✓ 처리 완료 ({process_time:.2f}초)")
                print(f"      ✓ 출력 형태: {result_np.shape}")
                print(f"      ✓ 알파 채널: {unique_values}개 고유값")
                
                # 결과 이미지 저장 (선택사항)
                try:
                    result.save(f'test_output_{model_name}.png')
                    print(f"      ✓ 결과 저장: test_output_{model_name}.png")
                except:
                    pass
                    
                success_count += 1
            else:
                print(f"      ⚠️  예상과 다른 출력 형태: {result_np.shape}")
                
        except Exception as e:
            print(f"      ✗ {model_name} 테스트 실패: {e}")
            # 상세 오류 출력 (선택사항)
            # traceback.print_exc()
    
    # 6. GPU 지원 확인
    print("\n6️⃣ GPU 지원 확인...")
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        
        gpu_providers = [p for p in providers if 'CUDA' in p or 'GPU' in p]
        if gpu_providers:
            print(f"   ✓ GPU 지원 제공자: {gpu_providers}")
        else:
            print("   ⚠️  GPU 제공자 없음 (CPU 모드)")
            print(f"   📋 사용 가능한 제공자: {providers}")
            
    except ImportError:
        print("   ⚠️  onnxruntime 정보 확인 불가")
    except Exception as e:
        print(f"   ⚠️  GPU 확인 중 오류: {e}")
    
    # 7. 메모리 사용량 확인
    print("\n7️⃣ 시스템 정보...")
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"   📊 사용 가능한 메모리: {memory.available / (1024**3):.1f} GB")
        print(f"   📊 총 메모리: {memory.total / (1024**3):.1f} GB")
    except ImportError:
        print("   ⚠️  psutil 미설치 (메모리 정보 확인 불가)")
    
    # 8. 결과 요약
    print("\n" + "=" * 60)
    print("📋 테스트 결과 요약")
    print("=" * 60)
    
    if success_count > 0:
        print(f"✅ REMBG 설치 성공!")
        print(f"✅ 작동하는 모델: {success_count}/{len(test_models)}개")
        print(f"✅ 사용 가능한 모델: {', '.join(working_models[:3])}...")
        
        print(f"\n🚀 추천 사용법:")
        print(f"   # 빠른 처리")
        print(f"   python -c \"from rembg import remove, new_session; print('u2netp 사용 준비됨')\"")
        print(f"   ")
        print(f"   # 명령줄 사용")
        print(f"   rembg i input.jpg output.png")
        
        return True
    else:
        print(f"❌ REMBG 기능 테스트 실패")
        print(f"💡 다음을 시도해보세요:")
        print(f"   1. pip uninstall rembg")
        print(f"   2. pip install rembg[gpu]")
        print(f"   3. 인터넷 연결 확인 (모델 다운로드 필요)")
        
        return False

def create_test_image():
    """간단한 테스트 이미지 생성"""
    # 640x480 RGB 이미지 생성
    width, height = 640, 480
    
    # 배경 (하늘색)
    img_array = np.full((height, width, 3), [135, 206, 235], dtype=np.uint8)
    
    # 간단한 사람 형태 그리기 (검은색)
    center_x, center_y = width // 2, height // 2
    
    # 머리 (원)
    head_radius = 50
    y, x = np.ogrid[:height, :width]
    head_mask = (x - center_x)**2 + (y - center_y + 80)**2 <= head_radius**2
    img_array[head_mask] = [50, 50, 50]
    
    # 몸통 (사각형)
    body_top = center_y - 20
    body_bottom = center_y + 100
    body_left = center_x - 40
    body_right = center_x + 40
    img_array[body_top:body_bottom, body_left:body_right] = [50, 50, 50]
    
    # 팔 (선)
    arm_y = center_y + 20
    img_array[arm_y-10:arm_y+10, body_left-60:body_left] = [50, 50, 50]
    img_array[arm_y-10:arm_y+10, body_right:body_right+60] = [50, 50, 50]
    
    return Image.fromarray(img_array)

def quick_test():
    """빠른 설치 확인"""
    try:
        from rembg import remove, new_session
        print("✅ REMBG 기본 설치 확인됨")
        return True
    except ImportError:
        print("❌ REMBG 설치되지 않음")
        return False

if __name__ == "__main__":
    print("REMBG 설치 테스트 유틸리티")
    print("사용법:")
    print("  python rembg_test.py          # 전체 테스트")
    print("  python rembg_test.py quick    # 빠른 확인")
    
    if len(sys.argv) > 1 and sys.argv[1] == 'quick':
        quick_test()
    else:
        test_rembg_installation()
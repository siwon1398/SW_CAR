import numpy as np
import cv2
import Function_Library as fl
import os
import time

class DriveConfig:
    STEER_LEFT = 70      
    STEER_CENTER = 90    
    STEER_RIGHT = 110    

def main():
    arduino = fl.libARDUINO()
    camera = fl.libCAMERA()

    ARDUINO_PORT = 'COM5'
    BAUD_RATE = 9600

    print("아두이노 연결 시도...")
    try:
        ser = arduino.init(ARDUINO_PORT, BAUD_RATE)
    except Exception as e:
        print(f"아두이노 연결 실패: {e}")
        ser = None

    print("카메라 설정 중...")
    # 🌟 capnum=2로 변경하여 두 대의 카메라 모두 로드 (포트는 기존 mission.py 기준 0, 1)
    cap0, cap1 = camera.initial_setting(cam0port=0, cam1port=1, capnum=2)

    if cap0 is None or not cap0.isOpened():
        print("첫 번째 카메라(frame0)를 열 수 없습니다.")
        return
    if cap1 is None or not cap1.isOpened():
        print("두 번째 카메라(frame1)를 열 수 없습니다.")
        return

    cap0.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 🌟 각각의 카메라 이미지를 저장할 폴더 분리 (정리하기 편하게)
    save_dir_cam0 = "./dataset_raw/cam0"
    save_dir_cam1 = "./dataset_raw/cam1"
    os.makedirs(save_dir_cam0, exist_ok=True)
    os.makedirs(save_dir_cam1, exist_ok=True)

    img_counter = 0
    current_steer = DriveConfig.STEER_CENTER

    print("=================================================================")
    print("🎮 듀얼 카메라 수동 제어 및 데이터 수집 프로그램 (학습용)")
    print("=================================================================")
    print(" [w] : ▲ 전진 개시      [a] : ⬅️ 핸들만 좌회전 (구동 X)")
    print(" [s] : ▼ 후진 개시      [d] : ➡️ 핸들만 우회전 (구동 X)")
    print(" [Spacebar] : 🛑 구동 모터 즉시 정지 (핸들 각도는 유지)")
    print("-----------------------------------------------------------------")
    print(" [c] : 📸 두 카메라 화면 동시 캡처 및 저장")
    print("       (dataset_raw/cam0, dataset_raw/cam1 폴더에 분리 저장됨)")
    print(" [q] : 프로그램 완전 안전 종료")
    print("=================================================================")

    while True:
        # 두 카메라 모두 읽어오기
        read_data = camera.camera_read(cap0, cap1)
        ret0, frame0 = read_data[0], read_data[1]
        ret1, frame1 = read_data[2], read_data[3]

        if not ret0 or not ret1:
            break

        # 화면에 글씨가 찍히기 전의 원본(Raw) 이미지 복사
        raw_frame0 = frame0.copy()
        raw_frame1 = frame1.copy()

        # 화면 출력용 텍스트 (frame0)
        cv2.putText(frame0, "MANUAL CONTROL MODE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame0, f"Current Steer: {current_steer}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame0, f"Captured: {img_counter} pairs", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 화면 출력용 텍스트 (frame1)
        cv2.putText(frame1, "TRAFFIC LIGHT CAMERA", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 두 화면 동시에 띄우기
        camera.image_show(frame0, frame1)

        key = cv2.waitKey(1) & 0xFF

        if ser is not None and ser.is_open:
            if key == ord('a'):
                current_steer = DriveConfig.STEER_LEFT
                ser.write(f"A{current_steer}\n".encode('utf-8'))
                print(f"⬅️ 핸들 좌회전 ({current_steer}도)")
                
            elif key == ord('d'):
                current_steer = DriveConfig.STEER_RIGHT
                ser.write(f"A{current_steer}\n".encode('utf-8'))
                print(f"➡️ 핸들 우회전 ({current_steer}도)")
                
            elif key == ord('w'):
                ser.write(b"F\n")
                print("▲ 전진 신호 전송!")

            elif key == ord('s'):
                ser.write(b"B\n")
                print("▼ 후진 신호 전송!")

            elif key == 32: # Spacebar
                ser.write(b"S\n")
                print("🛑 구동 모터 정지!")

        # 3. 📸 수동 스크린샷 캡처 (c 키)
        if key == ord('c'):
            timestamp = int(time.time())
            
            # 각각의 폴더에 알맞은 이름으로 저장
            img_name_0 = os.path.join(save_dir_cam0, f"cam0_{timestamp}_{img_counter}.jpg")
            img_name_1 = os.path.join(save_dir_cam1, f"cam1_{timestamp}_{img_counter}.jpg")
            
            cv2.imwrite(img_name_0, raw_frame0)
            cv2.imwrite(img_name_1, raw_frame1)
            
            img_counter += 1
            print(f"📸 [캡처 완료] 두 카메라 이미지 저장됨 (누적: {img_counter}쌍)")

        elif key == ord('q'):
            print("수동 조종 프로그램을 종료합니다.")
            break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()
    if ser is not None and ser.is_open:
        ser.write(b'S\n') 
        ser.close()

if __name__ == "__main__":
    main()

import Function_Library as fl

EPOCH = 500000
#
# if __name__ == "__main__":
#     # Exercise Environment Setting
#     env = fl.libCAMERA()#libCAMERA 클래스를 env에 불러옴
#
#     """ Exercise 1: RGB Color Value Extracting """
#     ############## YOU MUST EDIT ONLY HERE ##############
#     #example = env.file_read("./Example Image.jpg")
#     #R, G, B = env.extract_rgb(example, print_enable=True)
#     #quit()
#     #####################################################
#
#     # Camera Initial Setting
#     ch0, ch1 = env.initial_setting(capnum=2)#capnum = pc에 연결한 카메라 개수(항상2) ch0, 1은 카메라의 객체 정보 출력
#
#     # Camera Reading..
#     for i in range(EPOCH):
#         _, frame0, _, frame1 = env.camera_read(ch0, ch1)#카메라를 디지털 값으로 불러옴
#
#         """ Exercise 2: Webcam Real-time Reading """
#         ############## YOU MUST EDIT ONLY HERE ##############
#         #env.image_show(frame0, frame1)#포착한 프레임을 화면에 송출, 따라서 output은 따로 없음
#         #####################################################
#
#         """ Exercise 3: Object Detection (Traffic Light Circle) """
#         #################### YOU MUST EDIT ONLY HERE ####################
#         #color = env.object_detection(frame1, sample=16, print_enable=True)
#         # 카메라 데이터(frame0)입력해서 신호등 샘플 16개를 읽는다
#
#         #################################################################
#
#         """ Exercise 4: Specific Edge Detection (Traffic Line) """
#         #################### YOU MUST EDIT ONLY HERE ####################
#         #direction = env.edge_detection(frame1, width=500, height=120, gap=40, threshold=150, print_enable=True)
#         #
#         #################################################################
#
#         # Process Termination (If you input the 'q', camera scanning is ended.)
#         if env.loop_break():
#             break


import numpy as np
import cv2
import Function_Library as fl


# ==========================================
# 🛠️ [튜닝]
# ==========================================
class Config:
    # 1. 주행 중 중앙 유지
    TARGET_LINE = 'RIGHT'  # 시작 시 추종할 차선 ('RIGHT' 또는 'LEFT')
    OFFSET_FROM_LINE = 120  # 타겟 차선으로부터 떨어져서 유지할 픽셀 간격 (튜닝 필수!)

    # 2. 제어 민감도
    KP_ANGLE = 1.25  # 각도 변화에 얼마나 민감하게 핸들을 꺾을 것인가?
    KP_OFFSET = 0.15  # 위치가 벗어났을 때 얼마나 빨리 중앙으로 돌아올 것인가?

    # 3. 하드웨어 스펙 (사용자 측정값)
    STEER_MIN = 60  # 좌회전 최대 한계
    STEER_CENTER = 90  # 직진
    STEER_MAX = 120  # 우회전 최대 한계

    # 4. 카메라 시야 (관심 영역)
    ROI_RATIO = 0.3  # 0.5 = 화면의 위쪽 50%는 까맣게 무시함 (먼 배경 노이즈 제거)

    # 5. [NEW] 노이즈 및 주름 제거 튜닝
    BLUR_SIZE = 7  # 주름을 뭉개는 강도 (5, 7, 9 등 홀수. 클수록 많이 뭉갬)
    CANNY_LOW = 150  # 엣지 검출 최소 기준 (올릴수록 미세한 선 무시)
    CANNY_HIGH = 250  # 엣지 검출 최대 기준 (올릴수록 뚜렷한 선만 인정)

# ==========================================

def get_steering_angle(camera, img, config):
    img_h, img_w = img.shape[:2]
    screen_center_x = img_w // 2

    # --- 1. 관심 영역(ROI) 설정 ---
    # 멀리 있는 점선이나 노이즈를 안 보기 위해 화면 윗부분을 까맣게 지웁니다.
    roi_img = img.copy()
    roi_img[0:int(img_h * config.ROI_RATIO), :] = 0

    # --- 2. 영상 전처리 ---
    gray_scale = camera.gray_conversion(roi_img)
    hist = camera.histogram_equalization(gray_scale)
    dst = camera.morphology(hist, (2, 2), mode="opening")
    blurring = camera.gaussian_blurring(dst, (config.BLUR_SIZE, config.BLUR_SIZE))
    canny = camera.canny_edge(blurring, config.CANNY_LOW, config.CANNY_HIGH)

    # --- 3. 직선 검출 ---
    lines = camera.hough_transform(canny, 1, np.pi / 180, 50, 10, 20, mode="lineP")
                          # 여기서 끝에서 2번째 변수는 최소 선 길이(초기값 10픽셀)->노이즈 제거

    target_gradients = []
    target_x_list = []

    if lines is not None:
        for line in lines:
            xa, ya, xb, yb = line[0]

            # 길이 기반 필터링 (점선 제거)
            if np.abs(yb - ya) > 50 and np.abs(xb - xa) < 100:
            #점선을 거르기 위해 선의 길이가 50픽셀보다 커야함
                if yb != ya:  # 0으로 나누기 방지
                    grad = (xb - xa) / -(yb - ya)
                    x_bottom = int(xa + (img_h - ya) * (xb - xa) / (yb - ya))

                    # 현재 목표 차선(RIGHT or LEFT)에 맞는 선만 골라냅니다.
                    if config.TARGET_LINE == 'RIGHT' and x_bottom > screen_center_x:
                        target_gradients.append(grad)
                        target_x_list.append(x_bottom)
                        cv2.line(img, (xa, ya), (xb, yb), (0, 255, 0), 3)  # 타겟은 초록색

                    elif config.TARGET_LINE == 'LEFT' and x_bottom < screen_center_x:
                        target_gradients.append(grad)
                        target_x_list.append(x_bottom)
                        cv2.line(img, (xa, ya), (xb, yb), (0, 255, 0), 3)  # 타겟은 초록색

                    else:
                        # 타겟이 아닌 선(반대편 실선, 무시된 점선 등)은 빨간색으로 표시
                        cv2.line(img, (xa, ya), (xb, yb), (0, 0, 255), 1)

    # --- 4. 최종 조향각 계산 ---
    final_steering = config.STEER_CENTER

    if len(target_x_list) > 0:
        # 1) 각도 보정 (타겟 차선의 기울기만 사용!)
        avg_grad = np.mean(target_gradients)
        angle_offset = np.degrees(np.arctan(avg_grad)) * config.KP_ANGLE

        # 2) 위치 보정 (Offset)
        target_line_x = np.mean(target_x_list)

        if config.TARGET_LINE == 'RIGHT':
            # 오른쪽 차선 기준: 차선에서 왼쪽(-)으로 간격만큼 떨어진 곳이 중앙
            target_center_x = target_line_x - config.OFFSET_FROM_LINE
        else:
            # 왼쪽 차선 기준: 차선에서 오른쪽(+)으로 간격만큼 떨어진 곳이 중앙
            target_center_x = target_line_x + config.OFFSET_FROM_LINE

        offset_error = target_center_x - screen_center_x
        offset_correction = offset_error * config.KP_OFFSET

        # 3) 각도와 위치 보정 합산
        final_steering = config.STEER_CENTER + angle_offset + offset_correction

        # 디버깅용 선 긋기
        cv2.line(img, (screen_center_x, img_h), (screen_center_x, img_h - 50), (255, 0, 0), 3)  # 파란선: 화면 중앙 (차의 현재 위치)
        cv2.line(img, (int(target_center_x), img_h), (int(target_center_x), img_h - 50), (0, 255, 255),
                 3)  # 노란선: 내가 가야 할 목표 중앙

    # --- 5. 조향각 물리적 제한 (60 ~ 120도) ---
    final_steering = max(config.STEER_MIN, min(config.STEER_MAX, int(final_steering)))

    return final_steering
    return 90  # 차선이 안 보이면 기본 직진(90도)


def main():
    arduino = fl.libARDUINO()
    camera = fl.libCAMERA()

    ARDUINO_PORT = 'COM5'  # 설정하신 COM5 포트
    BAUD_RATE = 9600

    print("아두이노 연결 시도...")
    ser = arduino.init(ARDUINO_PORT, BAUD_RATE)

    print("카메라 설정 중...")
    cap0, _ = camera.initial_setting(cam0port=1, capnum=1)

    if cap0 is None or not cap0.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    # 🌟 [NEW] 프로그램 시작 시 무조건 '정지' 상태로 둡니다.
    is_driving = False
    print("=======================================")
    print("프로그램이 대기 모드로 켜졌습니다.")
    print("단축키 안내: [s] 출발 / 정지, [q] 완전 종료")
    print("=======================================")

    while True:
        read_data = camera.camera_read(cap0)
        ret, frame = read_data[0], read_data[1]

        if not ret:
            break

        # 차선 각도 계산 (정지 상태일 때도 화면 확인을 위해 계속 돌아갑니다)
        angle = get_steering_angle(camera, frame, Config)

        # 🌟 [NEW] 현재 상태를 화면 왼쪽 위에 예쁘게 띄웁니다.
        status_text = "DRIVING!!" if is_driving else "STOPPED"
        color = (0, 0, 255) if is_driving else (0, 255, 255)  # 빨강(주행), 노랑(정지)

        cv2.putText(frame, f"State: {status_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Steering Angle: {angle}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        camera.image_show(frame)

        # --- 🚨 모터 제어 명령 전송 ---
        if ser is not None and ser.is_open:
            if is_driving:
                # 주행 모드일 때만 각도를 전송해서 모터를 돌립니다.
                command = f"A{angle}\n"
                ser.write(command.encode('utf-8'))
            else:
                # 정지 모드일 때는 무조건 'S'를 보내서 모터를 묶어둡니다.
                ser.write(b"S\n")

        # --- ⌨️ 키보드 조작 (기존 camera.loop_break() 대체) ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("프로그램을 종료합니다.")
            break

        elif key == ord('s'):
            # 's'를 누를 때마다 주행 <-> 정지 상태가 뒤바뀝니다.
            is_driving = not is_driving
            if is_driving:
                print("==== 🚗 출발! 모터가 작동합니다! ====")
            else:
                print("==== 🛑 정지! 모터가 멈춥니다! ====")

    # --- 자원 해제 ---
    cap0.release()
    cv2.destroyAllWindows()
    if ser is not None and ser.is_open:
        ser.write(b'S\n')
        ser.close()


if __name__ == "__main__":
    main()


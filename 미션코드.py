import numpy as np
import cv2
import Function_Library as fl
from ultralytics import YOLO
import threading
import time

# ==========================================
# 🛠️ [최종 자율주행 튜닝 제어 변수]
# ==========================================
class Config:
    # 🌟 코랩에서 학습시켜 가져온 내 가중치 파일명
    MODEL_RIGHT_PATH = "best.pt"
    MODEL_LEFT_PATH = "best_left.pt"

    # 🌟 포트 설정
    ARDUINO_PORT = 'COM10'
    # USB 허브를 쓰더라도 윈도우 환경에서는 보통 아두이노와 라이다의 포트 번호가 다르게 할당됩니다. (예: COM10, COM11)
    # 만약 충돌(PermissionError)이 나면 여기를 수정해주세요.
    LIDAR_PORT = 'COM10' 
    BAUD_RATE = 9600

    # ------------------------------------------
    # 📐 조향 제어 (Steering Control) 가중치
    # ------------------------------------------
    OFFSET_FROM_LINE = 210
    LOOK_AHEAD = 80
    KP_ANGLE = 0.05
    KP_OFFSET = 0.2

    STEER_MIN = 70
    STEER_CENTER = 90
    STEER_MAX = 110

    RIGHT_CURVE_THRESHOLD = -0.05
    RIGHT_CURVE_EXTRA_OFFSET = 0

    # ------------------------------------------
    # 🗺️ 역투영(Top View) 사다리꼴 좌표
    # ------------------------------------------
    SRC_POINTS = np.float32([
        [92, 230],
        [548, 230],
        [640, 360],
        [0, 360]
    ])

    # ------------------------------------------
    # 🪟 슬라이딩 윈도우 변수
    # ------------------------------------------
    N_WINDOWS = 9
    MARGIN = 35
    MIN_PIXELS = 40

    # ------------------------------------------
    # 🔍 이전 프레임 기억 변수
    # ------------------------------------------
    SEARCH_RADIUS = 80
    PREV_RIGHT_X = None
    PREV_STEER = 90

    # ------------------------------------------
    # 🚦 라이다(LiDAR) 미션 변수
    # ------------------------------------------
    LIDAR_MIN_ANGLE = 0
    LIDAR_MAX_ANGLE = 20 # 0~20도, 340~360도 전방 주시
    LIDAR_MIN_DIST = 100
    LIDAR_MAX_DIST = 800 # 800mm 이하에 장애물이 있으면 회피

    # ------------------------------------------
    # 🚗 차선 변경 (Open-loop) 미션 변수
    # ------------------------------------------
    LANE_SWITCH_COOLDOWN = 3.0 # 차선 변경 후 3초간 다시 변경 안 함
    FORCED_STEER_DURATION = 1.0 # 1초간 강제 조향
    FORCED_STEER_ANGLE = 30 # 중심(90) 기준 얼만큼 꺾을지 (예: 90 - 30 = 60도)

# 전역 변수: 최신 라이다 스캔 데이터 저장
latest_lidar_scan = np.array([])
is_program_running = True

def lidar_thread_func(port):
    global latest_lidar_scan, is_program_running
    lidar = fl.libLIDAR(port)
    try:
        lidar.init()
        # 제너레이터에서 지속적으로 데이터 읽기
        for scan in lidar.scanning():
            if not is_program_running:
                break
            latest_lidar_scan = scan
    except Exception as e:
        print(f"LiDAR Thread Error: {e}")
    finally:
        lidar.stop()
        print("LiDAR Thread Stopped.")


# ==========================================
# 🧠 YOLOv8 차선 검출 및 조향 계산 핵심 함수
# ==========================================
def get_steering_angle(camera, img, model, config, lane_state="RIGHT"):
    img_h, img_w = img.shape[:2]
    screen_center_x = img_w // 2
    dst_points = np.float32([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]])

    clean_img = img.copy()

    pts = config.SRC_POINTS.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    results = model.predict(source=clean_img, imgsz=432, conf=0.25, verbose=False)

    yolo_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    if results[0].masks is not None:
        for mask in results[0].masks.xy:
            polygon_pts = np.array(mask, dtype=np.int32)
            cv2.fillPoly(yolo_mask, [polygon_pts], 255)

    yolo_pure_view = clean_img.copy()
    color_mask = np.zeros_like(clean_img)
    color_mask[yolo_mask == 255] = [0, 255, 0]
    yolo_pure_view = cv2.addWeighted(yolo_pure_view, 0.8, color_mask, 0.4, 0)
    cv2.imshow("YOLO Pure View", yolo_pure_view)

    img = cv2.addWeighted(img, 0.8, color_mask, 0.4, 0)

    matrix = cv2.getPerspectiveTransform(config.SRC_POINTS, dst_points)
    bev_img = cv2.warpPerspective(yolo_mask, matrix, (img_w, img_h))

    histogram = np.sum(bev_img[img_h // 2:, :], axis=0)
    midpoint = np.int64(histogram.shape[0] // 2)

    if config.PREV_RIGHT_X is not None:
        search_min = max(midpoint, config.PREV_RIGHT_X - config.SEARCH_RADIUS)
        search_max = min(img_w, config.PREV_RIGHT_X + config.SEARCH_RADIUS)
        right_search_area = histogram[search_min:search_max]

        if np.max(right_search_area) > 0:
            right_x_base = np.argmax(right_search_area) + search_min
        else:
            right_x_base = np.argmax(histogram[midpoint:]) + midpoint
    else:
        right_x_base = np.argmax(histogram[midpoint:]) + midpoint

    config.PREV_RIGHT_X = right_x_base

    window_height = np.int64(img_h // config.N_WINDOWS)
    nonzero = bev_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    right_x_current = right_x_base
    out_img = np.dstack((bev_img, bev_img, bev_img))
    right_lane_inds = []

    for window in range(config.N_WINDOWS):
        win_y_low = img_h - (window + 1) * window_height
        win_y_high = img_h - window * window_height
        win_xright_low = right_x_current - config.MARGIN
        win_xright_high = right_x_current + config.MARGIN

        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 0, 255), 2)

        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        right_lane_inds.append(good_right_inds)

        if len(good_right_inds) > config.MIN_PIXELS:
            right_x_current = np.int64(np.mean(nonzerox[good_right_inds]))

    right_lane_inds = np.concatenate(right_lane_inds)
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    y_eval = img_h - config.LOOK_AHEAD
    curve_factor_display = 0.0

    if len(rightx) > 100:
        right_fit = np.polyfit(righty, rightx, 2)
        right_lane_x = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]

        curve_factor = right_fit[1]
        curve_factor_display = curve_factor

        if curve_factor < config.RIGHT_CURVE_THRESHOLD:
            fixed_offset = config.OFFSET_FROM_LINE + config.RIGHT_CURVE_EXTRA_OFFSET
        else:
            fixed_offset = config.OFFSET_FROM_LINE

        # 🌟 LANE_STATE에 따라 타겟 좌표 부호 변경
        if lane_state == "LEFT":
            target_center_x = right_lane_x + fixed_offset
        else: # "RIGHT"
            target_center_x = right_lane_x - fixed_offset

        offset_error = target_center_x - screen_center_x

        final_steering = config.STEER_CENTER + (offset_error * config.KP_OFFSET) - (
                curve_factor * config.KP_ANGLE * 100)
        config.PREV_STEER = final_steering

        cv2.line(out_img, (screen_center_x, img_h), (screen_center_x, img_h - 50), (255, 0, 0), 3)
        cv2.circle(out_img, (int(target_center_x), int(y_eval)), 10, (0, 255, 255), -1)
    else:
        final_steering = config.PREV_STEER
        cv2.putText(out_img, "Lane Lost! Keep PREV_STEER", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    final_steering = max(config.STEER_MIN, min(config.STEER_MAX, int(final_steering)))

    cv2.putText(out_img, f"Curve Factor: {curve_factor_display:.4f}", (10, img_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Bird Eye View (Top View)", out_img)

    return final_steering, curve_factor_display


def main():
    global latest_lidar_scan, is_program_running

    arduino = fl.libARDUINO()
    camera = fl.libCAMERA()

    print("아두이노 연결 시도...")
    try:
        ser = arduino.init(Config.ARDUINO_PORT, Config.BAUD_RATE)
    except Exception as e:
        print(f"아두이노 연결 실패: {e}")
        ser = None

    print("LiDAR 스레드 시작...")
    lidar_thread = threading.Thread(target=lidar_thread_func, args=(Config.LIDAR_PORT,))
    lidar_thread.daemon = True
    lidar_thread.start()

    print("YOLOv8 차선 인식 모델 로드 중...")
    try:
        model_right = YOLO(Config.MODEL_RIGHT_PATH)
        model_left = YOLO(Config.MODEL_LEFT_PATH)
    except Exception as e:
        print(f"YOLO 모델 로드 실패: {e}")
        is_program_running = False
        return

    print("카메라 설정 중...")
    cap0, cap1 = camera.initial_setting(cam0port=0, cam1port=1, capnum=2)

    if cap0 is None or not cap0.isOpened():
        print("첫 번째 카메라(주행용)를 열 수 없습니다.")
        is_program_running = False
        return
        
    if cap1 is None or not cap1.isOpened():
        print("두 번째 카메라(신호등용)를 열 수 없습니다.")
        is_program_running = False
        return

    cap0.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    is_driving = False
    is_waiting_light = False
    
    # 미션 관련 상태 변수
    lane_state = "RIGHT" # "RIGHT" or "LEFT"
    last_switch_time = 0.0
    forced_steer_end_time = 0.0
    current_forced_steer = Config.STEER_CENTER

    calib_left_msg = "Calib Left: Wait..."
    calib_right_msg = "Calib Right: Wait..."
    realtime_pot_msg = "Pot: Wait..."

    print("=======================================")
    print("🚀 YOLOv8 오토 파일럿 프로그램이 대기 모드로 켜졌습니다.")
    print("단축키 안내: [s] 자율주행 출발 / 정지, [q] 완전 종료, [p] 사다리꼴 좌표 출력")
    print("=======================================")

    while True:
        read_data = camera.camera_read(cap0, cap1)
        ret0, frame0 = read_data[0], read_data[1]
        ret1, frame1 = read_data[2], read_data[3]

        if not ret0 or not ret1:
            break

        current_time = time.time()

        # ----------------------------------------------------
        # 🚦 1. 신호등 미션 처리 (신호등 전용 카메라: frame1)
        # ----------------------------------------------------
        if is_driving:
            # 신호등 전용 카메라이므로 굳이 상단만 크롭하지 않아도 될 수 있지만, 
            # 필요하다면 크롭 영역을 조절하세요.
            detected_color = camera.object_detection(frame1, sample=10, mode="circle", print_enable=False)
            
            if detected_color == "RED":
                is_waiting_light = True
            elif detected_color == "GREEN":
                is_waiting_light = False
        
        # ----------------------------------------------------
        # 🚗 2. 라이다 장애물 감지 및 차선 변경 미션 처리
        # ----------------------------------------------------
        is_obstacle_detected = False
        if len(latest_lidar_scan) > 0:
            angles = latest_lidar_scan[:, 0]
            distances = latest_lidar_scan[:, 1]
            
            # 전방 각도 필터링 (0~20도, 340~360도)
            front_condition = ((angles >= 0) & (angles <= Config.LIDAR_MAX_ANGLE)) | ((angles >= 360 - Config.LIDAR_MAX_ANGLE) & (angles <= 360))
            front_scan = latest_lidar_scan[front_condition]
            
            if len(front_scan) > 0:
                front_distances = front_scan[:, 1]
                # 거리 필터링
                valid_distances = front_distances[(front_distances > Config.LIDAR_MIN_DIST) & (front_distances < Config.LIDAR_MAX_DIST)]
                if len(valid_distances) > 0:
                    is_obstacle_detected = True

        # 장애물 감지 시 상태 전환 (쿨다운 확인)
        if is_obstacle_detected and (current_time - last_switch_time > Config.LANE_SWITCH_COOLDOWN):
            if lane_state == "RIGHT":
                lane_state = "LEFT"
                current_forced_steer = Config.STEER_CENTER - Config.FORCED_STEER_ANGLE # 왼쪽으로 조향
            else:
                lane_state = "RIGHT"
                current_forced_steer = Config.STEER_CENTER + Config.FORCED_STEER_ANGLE # 오른쪽으로 조향
            
            last_switch_time = current_time
            forced_steer_end_time = current_time + Config.FORCED_STEER_DURATION
            print(f"🚨 장애물 감지! 차선 변경: {lane_state} (Open-loop steering 시작)")

        # ----------------------------------------------------
        # 🧠 3. YOLO 추론 및 조향각 계산 (주행용 카메라: frame0)
        # ----------------------------------------------------
        active_model = model_left if lane_state == "LEFT" else model_right
        
        # 현재 강제 조향 모드인지 확인
        if current_time < forced_steer_end_time:
            angle = current_forced_steer
            curve_factor = 0.0
            # 화면 업데이트(디버그 뷰)를 위해 함수는 호출하지만 결과값 대신 강제 조향값 사용
            get_steering_angle(camera, frame0, active_model, Config, lane_state)
            cv2.putText(frame0, f"FORCED STEER: {angle}", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
        else:
            angle, curve_factor = get_steering_angle(camera, frame0, active_model, Config, lane_state)

        # ----------------------------------------------------
        # 아두이노 데이터 읽기
        # ----------------------------------------------------
        if ser is not None and ser.is_open:
            while ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if "Calib Left" in line:
                        calib_left_msg = line
                    elif "Calib Right" in line:
                        calib_right_msg = line
                    elif "Pot:" in line:
                        realtime_pot_msg = line
                except Exception:
                    pass

        # ----------------------------------------------------
        # 화면 상태 출력
        # ----------------------------------------------------
        status_text = "DRIVING" if is_driving else "STOPPED"
        color = (0, 0, 255) if is_driving else (0, 255, 255)

        if is_waiting_light:
            status_text = "WAITING LIGHT(RED)"
            color = (0, 0, 255)

        cv2.putText(frame0, f"State: {status_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame0, f"Lane: {lane_state}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(frame0, f"Steering Angle: {angle}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 타이머 안 켜져있을 때만 curve factor 출력
        if current_time >= forced_steer_end_time:
            cv2.putText(frame0, f"Curve Factor: {curve_factor:.4f}", (10, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame0, realtime_pot_msg, (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame0, calib_left_msg, (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame0, calib_right_msg, (10, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        camera.image_show(frame0, frame1)

        # ----------------------------------------------------
        # 아두이노에 제어 명령 전송
        # ----------------------------------------------------
        if ser is not None and ser.is_open:
            if is_driving and not is_waiting_light:
                command = f"A{angle}\n"
                ser.write(command.encode('utf-8'))
            else:
                ser.write(b"S\n")

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("프로그램을 종료합니다.")
            is_program_running = False
            break
        elif key == ord('s'):
            is_driving = not is_driving
        elif key == ord('p'):
            print("\n[현재 사다리꼴 좌표(SRC_POINTS)]")
            print("    SRC_POINTS = np.float32([")
            for pt in Config.SRC_POINTS:
                print(f"        [{int(pt[0])}, {int(pt[1])}],")
            print("    ])\n")

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()
    if ser is not None and ser.is_open:
        ser.write(b'S\n')
        ser.close()


if __name__ == "__main__":
    main()

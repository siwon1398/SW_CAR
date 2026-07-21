import numpy as np
import cv2
import Function_Library as fl
from ultralytics import YOLO
import threading
import time
import os

# ==========================================
# 🛠️ [최종 자율주행 튜닝 제어 변수]
# ==========================================
class Config:
    # 🌟 코랩에서 학습시켜 가져온 내 가중치 파일명
    MODEL_RIGHT_PATH = "best.pt"
    MODEL_LEFT_PATH = "best_left.pt"
    MODEL_MISSION_PATH = "best_object.pt" # 🆕 미션용 가중치 파일명

    # 🌟 미션 객체 인식 클래스 ID (best_object.pt 맞춤 설정)
    CLASS_CAR = 0
    CLASS_RED_LIGHT = 3
    CLASS_GREEN_LIGHT = 1 
    CLASS_LANE = 2         # 모델의 'lane' (정지선이나 차선 인식용)

    # 🌟 정지선 감지 ROI 설정 (y 좌표 기준: 화면의 어느 지점에 선이 와야 멈출지)
    # [조정됨] 늦게 멈추는 현상을 해결하기 위해 멈춤 기준선을 위로 올렸습니다. (숫자가 작을수록 일찍 멈춤)
    STOP_LINE_Y_MIN = 300
    STOP_LINE_Y_MAX = 380

    # 🌟 포트 설정
    ARDUINO_PORT = 'COM5'
    # USB 허브를 쓰더라도 윈도우 환경에서는 보통 아두이노와 라이다의 포트 번호가 다르게 할당됩니다. (예: COM10, COM11)
    # 만약 충돌(PermissionError)이 나면 여기를 수정해주세요.
    LIDAR_PORT = 'COM6' 
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
    RIGHT_CURVE_EXTRA_OFFSET = 30

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
    LIDAR_MIN_ANGLE = 165 # 정면(180도) 기준 왼쪽 한계
    LIDAR_MAX_ANGLE = 195 # 정면(180도) 기준 오른쪽 한계 (총 40도 범위)
    LIDAR_MIN_DIST = 500  # 너무 가까운 노이즈 무시
    LIDAR_MAX_DIST = 1500 # 1.3m 앞 장애물 회피 시작 (YOLO와 함께 감지될 때)
    LIDAR_EMERGENCY_DIST = 1300 # 0.8m 이내면 YOLO 무시하고 즉각 회피!

    # ------------------------------------------
    # 🚗 차선 변경 (비전 하이브리드) 미션 변수
    # ------------------------------------------
    LANE_SWITCH_COOLDOWN = 2.5
    LANE_SWITCH_MAX_DUR = 5.0          # 차선 변경 전체 최대 타임아웃
    
    # 🌟 [시간 기반(하드코딩) 렉 극복 옵션]
    # [수정] 이제 USB 렉이 완전히 해결되었으므로 멍청한 시간 기반 회피(블라인드 매크로)를 끄고, 
    # 비전 기반 정밀 동적 정렬(스마트 회피)을 활성화합니다!
    USE_TIME_BASED_LANE_CHANGE = True
    PHASE1_DUR = 1.2                   # Phase 1 (진입) 유지 시간 (초) - 너무 많이 가면 줄이세요
    PHASE2_DUR = 1.0                   # Phase 2 (카운터 스티어) 유지 시간 (초)

    FORCED_STEER_PHASE1_MIN_DUR = 0.5  # 1단계: 진입 최소 유지 시간 (노이즈 방지)
    FORCED_STEER_ANGLE = 40 
    PHASE1_EXIT_MARGIN = 60            # 🌟 [신규 파라미터] 카운터 스티어 진입 시점 결정 (기본 30). 값이 클수록 카운터 스티어(Phase 2)를 더 일찍 시작합니다.
    ALIGN_CURVE_THRESHOLD = 0.05       # 비전 기반 정렬 완료로 판단할 기울기

# 전역 변수: 최신 라이다 스캔 데이터 저장
latest_lidar_scan = np.array([])
is_program_running = True

def lidar_thread_func(port):
    global latest_lidar_scan, is_program_running
    
    while is_program_running:
        lidar = fl.libLIDAR(port)
        try:
            lidar.init()
            # 제너레이터에서 지속적으로 데이터 읽기
            for scan in lidar.scanning():
                if not is_program_running:
                    break
                latest_lidar_scan = scan
        except Exception as e:
            print(f"⚠️ LiDAR 통신 에러 발생 (진동/노이즈). 1초 후 자동 재시작합니다... Error: {e}")
            import time
            time.sleep(1)
        finally:
            try:
                lidar.stop()
            except:
                pass
            
    print("LiDAR Thread Permanently Stopped.")


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

    # 차선 변경 시 이전 프레임 위치 초기화
    if getattr(config, 'PREV_LANE_STATE', None) != lane_state:
        config.PREV_RIGHT_X = None
        config.PREV_LANE_STATE = lane_state

    # 탐색할 화면의 절반 영역 설정
    if lane_state == "LEFT":
        search_half = histogram[:midpoint]
        base_offset = 0
        limit_min = 0
        limit_max = midpoint
    else: # RIGHT
        search_half = histogram[midpoint:]
        base_offset = midpoint
        limit_min = midpoint
        limit_max = img_w

    right_x_base = None
    if config.PREV_RIGHT_X is not None:
        # main.py와 동일하게 탐색 영역을 화면 절반(limit_min, limit_max)으로 엄격히 제한하여 반대편 차선을 잡는(왔다갔다 하는) 버그 원천 차단
        search_min = max(limit_min, config.PREV_RIGHT_X - config.SEARCH_RADIUS)
        search_max = min(limit_max, config.PREV_RIGHT_X + config.SEARCH_RADIUS)
        search_area = histogram[search_min:search_max]

        # main.py와 동일하게 > 0 조건으로 복구 (기존 > 10 필터링이 옅은 차선을 놓치고 뚝뚝 끊기게 만드는 주범)
        if len(search_area) > 0 and np.max(search_area) > 0: 
            right_x_base = np.argmax(search_area) + search_min

    # 이전 위치 근처에서 못 찾았거나(차선 변경 중 시야 이탈) 처음인 경우 전체 절반 영역에서 탐색
    if right_x_base is None:
        if len(search_half) > 0 and np.max(search_half) > 0:
            right_x_base = np.argmax(search_half) + base_offset
        else:
            # [수정] main.py와 완벽히 동일하게: 아예 안 보이면 PREV_RIGHT_X를 유지하는 대신 '기본 중앙값'으로 리셋시켜 진동(왔다갔다)을 방지합니다.
            right_x_base = limit_max - config.OFFSET_FROM_LINE if lane_state == "RIGHT" else limit_min + config.OFFSET_FROM_LINE

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
        
        final_steering = max(config.STEER_MIN, min(config.STEER_MAX, int(final_steering)))
        
        cv2.putText(out_img, f"Curve Factor: {curve_factor_display:.4f}", (10, img_h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Lane Detection (BEV)", out_img)
        return final_steering, curve_factor_display, True, target_center_x
    else:
        final_steering = config.PREV_STEER
        cv2.putText(out_img, "Lane Lost! Keep PREV_STEER", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        final_steering = max(config.STEER_MIN, min(config.STEER_MAX, int(final_steering)))
        
        cv2.imshow("Lane Detection (BEV)", out_img)
        return final_steering, 0.0, False, screen_center_x


def main():
    global latest_lidar_scan, is_program_running

    import torch
    print("\n" + "🔥"*25)
    if torch.cuda.is_available():
        print(f"🚀 SUCCESS: GPU 가속 완벽 적용됨! ({torch.cuda.get_device_name(0)})")
    else:
        print("❌ WARNING: GPU 가속 실패... 여전히 CPU로 구동 중입니다.")
    print("🔥"*25 + "\n")

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
        print(f"차선 인식 YOLO 모델 로드 실패: {e}")
        is_program_running = False
        return

    print("YOLOv8 미션(신호등, 자동차, 정지선) 모델 로드 중...")
    try:
        mission_model = YOLO(Config.MODEL_MISSION_PATH)
    except Exception as e:
        print(f"⚠️ 미션용 모델({Config.MODEL_MISSION_PATH})을 찾을 수 없거나 로드에 실패했습니다. 미션 인식은 제외하고 실행합니다. Error: {e}")
        mission_model = None

    print("🚀 추론 엔진 초기화 및 웜업 중... (수 초가 소요될 수 있습니다)")
    # 빈 이미지를 생성하여 첫 번째 추론을 미리 수행합니다. (VRAM 할당 및 웜업)
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    model_right.predict(source=dummy_img, imgsz=432, verbose=False)
    print("✅ 추론 엔진 웜업 완료!")

    print("카메라 설정 중 (주행용 바닥 카메라만 먼저 켭니다)...")
    # [최강 하드웨어 최적화] USB 병목(대역폭 초과)으로 인한 하드웨어 프레임 드랍을 원천 차단하기 위해 
    # 신호등 카메라(cap1)는 주행 중에는 아예 연결조차 하지 않습니다. (main.py와 물리적으로 동일한 상태)
    cap0, _ = camera.initial_setting(cam0port=0, capnum=1)
    cap1 = None

    if cap0 is None or not cap0.isOpened():
        print("첫 번째 카메라(주행용)를 열 수 없습니다.")
        is_program_running = False
        return

    cap0.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    is_driving = False
    is_waiting_light = False
    
    lane_state = "RIGHT" # "RIGHT" or "LEFT"
    lane_change_phase = 0 # 0: Closed loop, 1: Phase 1 (Enter), 2: Phase 2 (Align)
    last_detected_light = "NONE" # 최근 감지된 신호등 색상
    last_switch_time = 0.0

    calib_left_msg = "Calib Left: Wait..."
    calib_right_msg = "Calib Right: Wait..."
    realtime_pot_msg = "Pot: Wait..."

    frame_count = 0 # [추가] 연산량 최적화를 위한 프레임 카운터
    last_results_f1_boxes = [] # [추가] 프레임 스킵용 박스 캐시
    last_results_f0_boxes = [] # [추가] 프레임 스킵용 박스 캐시
    
    # [추가] 미션 단계(State) 추적용 변수
    mission_completed_lane_changes = 0 # 완료한 차선 변경 횟수
    traffic_light_passed = False       # 신호등 통과 여부
    was_waiting_light = False          # 이전에 신호대기 중이었는지 여부

    print("=======================================")
    print("🚀 YOLOv8 오토 파일럿 프로그램이 대기 모드로 켜졌습니다.")
    print("단축키 안내: [s] 자율주행 출발 / 정지, [q] 완전 종료, [p] 사다리꼴 좌표 출력")
    print("=======================================")

    while True:
        # [최적화] 카메라 프레임 읽기 딜레이(병목) 제거
        # 미션 모델은 차선 변경 2회 완료 후부터만 켬
        # [초극강 최적화] 신호등을 통과(traffic_light_passed)하고 나면 미션 모델을 완전히 영구 종료시켜 
        # 결승선까지 CPU 100%를 오직 주행 조향에만 쏟아붓도록 만듭니다!
        run_mission_model = False
        if mission_completed_lane_changes >= 2 and not traffic_light_passed:
            run_mission_model = True

        # 바닥 카메라(cap0)는 주행을 위해 무조건 읽음
        ret0, frame0 = cap0.read()
        if not ret0:
            break
            
        if run_mission_model and is_waiting_light:
            if cap1 is None:
                print("🚦 [시스템] 정지선 도착! 신호등 카메라(cap1) 전원 ON (약 1~2초 소요)...")
                # 신호등 카메라는 보통 1번 포트(웹캠)
                cap1 = cv2.VideoCapture(cv2.CAP_DSHOW + 1)
                cap1.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # 멈춰있을 때만 윗쪽 카메라를 완벽히 읽음
            ret1, frame1 = cap1.read()
            if not ret1:
                break
            Config.LAST_FRAME1 = frame1
        else:
            # 주행 중에는 cap1이 꺼져있으므로 하드웨어 렉 0%
            frame1 = getattr(Config, 'LAST_FRAME1', np.zeros((480, 640, 3), dtype=np.uint8))

        current_time = time.time()

        # ----------------------------------------------------
        # 🆕 YOLO 기반 미션 객체 탐지 (신호등, 자동차, 정지선)
        # ----------------------------------------------------
        car_detected = False
        stop_line_in_roi = False
        
        frame_count += 1
        
        if mission_model is not None:

            if run_mission_model:
                if not is_waiting_light:
                    # [최강 최적화 4] 미션 모델(정지선 찾기)은 5프레임에 1번만 실행하고, 
                    # 실행하는 프레임에서는 주행 모델(best.pt)을 완전히 건너뛰게 만들어(스킵) CPU 과부하를 원천 차단합니다!
                    # 직진 여부와 상관없이 주기적으로 무조건 실행하므로 차선 이탈 중에도 정지선을 절대 놓치지 않습니다.
                    if frame_count % 5 == 0:
                        results_f0 = mission_model.predict(source=frame0, imgsz=320, conf=0.3, verbose=False)
                        last_results_f0_boxes = results_f0[0].boxes if results_f0[0].boxes is not None else []
                        last_results_f1_boxes = [] # 신호등 카메라는 안 봄
                        Config.SKIP_MAIN_STEERING = True
                    else:
                        Config.SKIP_MAIN_STEERING = False
                else:
                    # 정지선에 멈춘 상태: 신호등 카메라만 검사!
                    results_f1 = mission_model.predict(source=frame1, imgsz=320, conf=0.3, verbose=False)
                    last_results_f1_boxes = results_f1[0].boxes if results_f1[0].boxes is not None else []
                    # last_results_f0_boxes는 지우지 않고 그대로 둡니다. (사용자가 정지선 박스를 볼 수 있도록)

            # 1. frame1 (위쪽 카메라) - 캐시된 박스 사용
            detected_color = None
            if len(last_results_f1_boxes) > 0:
                for box in last_results_f1_boxes:
                    cls_id = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame1, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    
                    if cls_id == Config.CLASS_RED_LIGHT:
                        detected_color = "RED"
                        cv2.putText(frame1, "RED_LIGHT", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    elif cls_id == Config.CLASS_GREEN_LIGHT:
                        detected_color = "GREEN"
                        cv2.putText(frame1, "GREEN_LIGHT", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if detected_color is not None:
                last_detected_light = detected_color
            else:
                last_detected_light = "NONE"

            # 2. frame0 (바닥 카메라) - 논리 판단만 (그리기는 조향 계산 후 맨 뒤에서 처리)
            if len(last_results_f0_boxes) > 0:
                for box in last_results_f0_boxes:
                    cls_id = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    if cls_id == Config.CLASS_CAR:
                        car_detected = True
                    elif cls_id == Config.CLASS_LANE:
                        if y2 >= Config.STOP_LINE_Y_MAX:
                            stop_line_in_roi = True

        # ----------------------------------------------------
        # 🚦 1. 신호등 및 정지선 연계 미션 처리 (State Machine)
        # ----------------------------------------------------
        if is_driving:
            if stop_line_in_roi and not traffic_light_passed and not is_waiting_light:
                is_waiting_light = True
                was_waiting_light = True
                print("🛑 정지선 감지! 무조건 정지하고 신호등을 확인합니다.")
                
            if is_waiting_light:
                # [수정] 카메라가 방금 켜져서 아무것도 안보이는 NONE 상태일때 바로 출발하는 버그 방지를 위해 GREEN일때만 출발!
                if last_detected_light == "GREEN":
                    is_waiting_light = False
                    traffic_light_passed = True
                    print("✅ [MISSION COMPLETE] 초록불 확인! 출발합니다. YOLO 미션 모델 완전 종료!")
                    was_waiting_light = False
                    
                    if cap1 is not None:
                        print("🔌 [시스템] 신호등 카메라(cap1) 전원 OFF. 주행에 100% 전력을 쏟습니다.")
                        cap1.release()
                        cap1 = None

        # ----------------------------------------------------
        # 🚗 2. 라이다 & 비전(Sensor Fusion) 장애물 감지 및 차선 변경
        # ----------------------------------------------------
        is_obstacle_detected = False
        is_emergency_obstacle = False
        
        # [최강 최적화 3] 라이다 레이더 그리기(파이썬 for문 연산)는 CPU를 엄청나게 갉아먹어 메인 주행(YOLO)에 렉을 유발합니다.
        # 시각화용 레이더 창은 3프레임에 1번씩만 그려서 인간의 눈에는 부드럽게 보이되 CPU 부담을 3분의 1로 줄입니다.
        if frame_count % 3 == 0:
            radar_size = 400
            radar_max_dist = 2000 # 2미터 반경 표시
            radar_img = np.zeros((radar_size, radar_size, 3), dtype=np.uint8)
            cx, cy = radar_size // 2, radar_size // 2
            
            import math
            cv2.circle(radar_img, (cx, cy), int(Config.LIDAR_MAX_DIST / radar_max_dist * (radar_size/2)), (50, 50, 100), 1) # 감지 최대 범위
            cv2.putText(radar_img, f"{Config.LIDAR_MAX_DIST}mm", (cx+5, cy - int(Config.LIDAR_MAX_DIST / radar_max_dist * (radar_size/2))), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 200), 1)
            
            # 전방 감지 영역(부채꼴) 선 그리기 (180도를 위쪽으로 표시)
            rad_left = math.radians(Config.LIDAR_MIN_ANGLE - 180)
            rad_right = math.radians(Config.LIDAR_MAX_ANGLE - 180)
            len_line = radar_size // 2
            cv2.line(radar_img, (cx, cy), (int(cx + len_line * math.sin(rad_left)), int(cy - len_line * math.cos(rad_left))), (100, 50, 50), 1)
            cv2.line(radar_img, (cx, cy), (int(cx + len_line * math.sin(rad_right)), int(cy - len_line * math.cos(rad_right))), (100, 50, 50), 1)

            if len(latest_lidar_scan) > 0:
                angles = latest_lidar_scan[:, 0]
                distances = latest_lidar_scan[:, 1]
                
                # 레이더에 점 찍기
                for angle, dist in zip(angles, distances):
                    if 0 < dist < radar_max_dist:
                        vis_rad = math.radians(angle - 180)
                        x = int(cx + (dist / radar_max_dist) * (radar_size/2) * math.sin(vis_rad))
                        y = int(cy - (dist / radar_max_dist) * (radar_size/2) * math.cos(vis_rad))
                        
                        if (Config.LIDAR_MIN_ANGLE <= angle <= Config.LIDAR_MAX_ANGLE) and (Config.LIDAR_MIN_DIST < dist < Config.LIDAR_MAX_DIST):
                            cv2.circle(radar_img, (x, y), 3, (0, 0, 255), -1)
                        else:
                            cv2.circle(radar_img, (x, y), 2, (0, 255, 0), -1)
                            
            cv2.imshow("LiDAR Radar", radar_img)

        # 라이다 장애물 판단 (수학 연산 없이 numpy 배열 필터링만 하므로 매 프레임 실행해도 렉 0%)
        if len(latest_lidar_scan) > 0:
            angles = latest_lidar_scan[:, 0]
            # 전방 각도 필터링 (정면이 180도이므로 MIN ~ MAX)
            front_condition = (angles >= Config.LIDAR_MIN_ANGLE) & (angles <= Config.LIDAR_MAX_ANGLE)
            front_scan = latest_lidar_scan[front_condition]
            
            if len(front_scan) > 0:
                front_distances = front_scan[:, 1]
                # 거리 필터링
                valid_distances = front_distances[(front_distances > Config.LIDAR_MIN_DIST) & (front_distances < Config.LIDAR_MAX_DIST)]
                if len(valid_distances) > 0:
                    is_obstacle_detected = True
                    
                # 비상 회피 (YOLO 무시)
                emergency_distances = front_distances[(front_distances > Config.LIDAR_MIN_DIST) & (front_distances < Config.LIDAR_EMERGENCY_DIST)]
                if len(emergency_distances) > 0:
                    is_emergency_obstacle = True
                else:
                    is_emergency_obstacle = False
            else:
                is_emergency_obstacle = False
                    
        # [Sensor Fusion] 라이다 감지 + 카메라 감지 OR 라이다 초근접 비상상황
        fusion_obstacle_condition = False
        
        if mission_completed_lane_changes < 2:
            # 장애물 회피 단계(차선변경 2회 미만)에서는 무조건 LiDAR만 믿고 회피 (YOLO 무시)
            # 설정한 회피 거리(LIDAR_MAX_DIST) 안에 들어오거나 비상 상황이면 바로 회피!
            fusion_obstacle_condition = is_obstacle_detected or is_emergency_obstacle
        else:
            # 2회 변경 완료(신호등 진입) 이후에는 어떠한 경우에도 추가 차선 변경(3회째)을 완벽 차단!
            fusion_obstacle_condition = False

        if fusion_obstacle_condition and (current_time - last_switch_time > Config.LANE_SWITCH_COOLDOWN):
            lane_state = "LEFT" if lane_state == "RIGHT" else "RIGHT"
            last_switch_time = current_time
            lane_change_phase = 1
            reason = "긴급 회피(초근접)!" if is_emergency_obstacle else "안전 회피(센서 퓨전)"
            print(f"🚨 장애물 감지 [{reason}] 차선 변경: {lane_state} (Phase 1 진입 시작)")

        # ----------------------------------------------------
        # 🧠 3. YOLO 추론 및 조향각 계산 (주행용 카메라: frame0)
        # ----------------------------------------------------
        active_model = model_left if lane_state == "LEFT" else model_right
        
        time_since_switch = current_time - last_switch_time
        screen_center_x = frame0.shape[1] // 2
        
        if lane_change_phase > 0:
            # 타임아웃 안전장치
            if time_since_switch > Config.LANE_SWITCH_MAX_DUR:
                lane_change_phase = 0
                mission_completed_lane_changes += 1
                print(f"⚠️ 차선 변경 최대 시간 초과, 강제 복귀 (누적 변경: {mission_completed_lane_changes}회)")
            else:
                closed_loop_angle, curve_factor, is_lane_detected, target_center_x = get_steering_angle(camera, frame0, active_model, Config, lane_state)
                
                if lane_change_phase == 1:
                    # Phase 1: 진입 (Hard Steer)
                    angle = Config.STEER_CENTER - Config.FORCED_STEER_ANGLE if lane_state == "LEFT" else Config.STEER_CENTER + Config.FORCED_STEER_ANGLE
                    angle = max(0, min(180, int(angle)))
                    
                    # 목표 차선을 제대로 밟았는지 확인 (진입 조기 종료)
                    if getattr(Config, 'USE_TIME_BASED_LANE_CHANGE', False):
                        if time_since_switch > Config.PHASE1_DUR:
                            lane_change_phase = 2
                            print(f"✅ [시간 기반] Phase 1 완료 ({Config.PHASE1_DUR}초 경과), Phase 2 카운터 스티어 시작!")
                    else:
                        if is_lane_detected and time_since_switch > Config.FORCED_STEER_PHASE1_MIN_DUR:
                            # [버그 수정] 부등호 방향이 반대였습니다! 
                            # 왼쪽으로 차선을 바꿀 때는 타겟이 화면 좌측(작은 값)에서 중앙(큰 값)으로 오므로 > 가 되어야 합니다.
                            if (lane_state == "LEFT" and target_center_x > screen_center_x - Config.PHASE1_EXIT_MARGIN) or \
                               (lane_state == "RIGHT" and target_center_x < screen_center_x + Config.PHASE1_EXIT_MARGIN):
                                lane_change_phase = 2
                                print(f"✅ Phase 1 완료 (타겟 인식: {target_center_x}), Phase 2 카운터 스티어 시작!")
                            
                    cv2.putText(frame0, f"PHASE 1 (ENTER): {angle}", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

                elif lane_change_phase == 2:
                    # Phase 2: 정렬 (Counter Steer)
                    angle = Config.STEER_CENTER + Config.FORCED_STEER_ANGLE if lane_state == "LEFT" else Config.STEER_CENTER - Config.FORCED_STEER_ANGLE
                    angle = max(0, min(180, int(angle)))
                    
                    # 차체가 똑바로 섰는지 확인 (정렬 조기 종료)
                    if getattr(Config, 'USE_TIME_BASED_LANE_CHANGE', False):
                        if time_since_switch > Config.PHASE1_DUR + Config.PHASE2_DUR:
                            lane_change_phase = 0
                            mission_completed_lane_changes += 1
                            angle = closed_loop_angle
                            print(f"✅ [시간 기반] Phase 2 완료 ({Config.PHASE2_DUR}초 경과)! 정상 복귀 (누적 변경: {mission_completed_lane_changes}회)")
                    else:
                        if is_lane_detected:
                            if (lane_state == "LEFT" and curve_factor >= -Config.ALIGN_CURVE_THRESHOLD) or \
                               (lane_state == "RIGHT" and curve_factor <= Config.ALIGN_CURVE_THRESHOLD):
                                lane_change_phase = 0
                                mission_completed_lane_changes += 1
                                angle = closed_loop_angle
                                print(f"✅ Phase 2 정렬 완료 (기울기: {curve_factor:.4f})! 정상 복귀 (누적 변경: {mission_completed_lane_changes}회)")
                    
                    if lane_change_phase == 2:
                        cv2.putText(frame0, f"PHASE 2 (ALIGN): {angle}", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)

        if lane_change_phase == 0:
            if getattr(Config, 'SKIP_MAIN_STEERING', False):
                # [최적화] 미션 모델이 동작한 프레임이므로 주행 연산(YOLO)을 건너뛰고 이전 각도를 그대로 유지 (더블 YOLO 충돌 방지)
                angle = getattr(Config, 'PREV_STEER', 90)
                curve_factor = 0.0
                is_lane_detected = True
            else:
                # 정상 Closed-loop 조향
                angle, curve_factor, is_lane_detected, _ = get_steering_angle(camera, frame0, active_model, Config, lane_state)

        # ----------------------------------------------------
        # 🎨 미션 객체 바운딩 박스 그리기 (반드시 get_steering_angle 이후에 그려야 차선 모델에 노이즈를 주지 않음)
        # ----------------------------------------------------
        if len(last_results_f0_boxes) > 0:
            for box in last_results_f0_boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if cls_id == Config.CLASS_CAR:
                    cv2.rectangle(frame0, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(frame0, "CAR", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                elif cls_id == Config.CLASS_LANE:
                    cv2.rectangle(frame0, (x1, y1), (x2, y2), (255, 255, 0), 2)
                    cv2.putText(frame0, "LANE (STOP)", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
        # 정지선 표시기(ROI)를 frame0에 그리기 (시각화용)
        cv2.line(frame0, (0, Config.STOP_LINE_Y_MIN), (frame0.shape[1], Config.STOP_LINE_Y_MIN), (0, 100, 255), 1)
        cv2.line(frame0, (0, Config.STOP_LINE_Y_MAX), (frame0.shape[1], Config.STOP_LINE_Y_MAX), (0, 100, 255), 1)

        # ----------------------------------------------------
        # 아두이노 데이터 읽기 (main.py와 동일하게 if문으로 1줄만 읽어 딜레이 방지)
        # ----------------------------------------------------
        if ser is not None and ser.is_open:
            if ser.in_waiting > 0:
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
                    
        # [추가] 최근 감지된 신호등 색상 표시 UI
        light_color = (0, 0, 255) if last_detected_light == "RED" else (0, 255, 0) if last_detected_light == "GREEN" else (255, 255, 255)
        cv2.putText(frame0, f"Light: {last_detected_light}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, light_color, 2)
                    
        # [추가] 차선 변경 횟수 및 미션 상태 표시 UI
        mission_text = f"Lane Changes: {mission_completed_lane_changes}/2"
        if traffic_light_passed:
            mission_text = "MISSION COMPLETE (YOLO OFF)"
        elif mission_completed_lane_changes >= 2:
            mission_text = "WAITING FOR TRAFFIC LIGHT"
            
        cv2.putText(frame0, mission_text, (10, 310),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 텍스트가 겹치지 않도록 아래 변수들의 출력 y좌표를 조금씩 내림
        cv2.putText(frame0, f"Lane: {lane_state}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(frame0, f"Steering Angle: {angle}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 강제 조향 중이 아닐 때만 curve factor 출력
        if lane_change_phase == 0:
            cv2.putText(frame0, f"Curve Factor: {curve_factor:.4f}", (10, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(frame0, realtime_pot_msg, (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame0, calib_left_msg, (10, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame0, calib_right_msg, (10, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 본넷 카메라(frame0) 우측 상단에 신호등 화면(frame1)을 작게 띄우기 (PIP)
        try:
            pip_w, pip_h = int(frame0.shape[1] * 0.3), int(frame0.shape[0] * 0.3) # 원본 대비 30% 크기
            frame1_pip = cv2.resize(frame1, (pip_w, pip_h))
            frame0[10:10+pip_h, frame0.shape[1]-pip_w-10:frame0.shape[1]-10] = frame1_pip
            cv2.rectangle(frame0, (frame0.shape[1]-pip_w-10, 10), (frame0.shape[1]-10, 10+pip_h), (0, 255, 255), 2)
        except Exception:
            pass

        camera.image_show(frame0, frame1)

        # ----------------------------------------------------
        # 아두이노에 제어 명령 전송 (main.py와 동일하게 매 프레임 즉각 전송)
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

    if cap0 is not None:
        cap0.release()
    if cap1 is not None:
        cap1.release()
    cv2.destroyAllWindows()
    if ser is not None and ser.is_open:
        ser.write(b'S\n')
        ser.close()


if __name__ == "__main__":
    main()

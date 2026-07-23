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
    STOP_LINE_Y_MIN = 240
    STOP_LINE_Y_MAX = 320

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
    LIDAR_MAX_DIST = 1700 # 1.3m 앞 장애물 회피 시작 (YOLO와 함께 감지될 때)
    LIDAR_EMERGENCY_DIST = 1600 # 0.8m 이내면 YOLO 무시하고 즉각 회피!

    # ------------------------------------------
    # 🚗 차선 변경 (비전 하이브리드) 미션 변수
    # ------------------------------------------
    LANE_SWITCH_COOLDOWN = 2.5
    LANE_SWITCH_MAX_DUR = 5.0          # 차선 변경 전체 최대 타임아웃
    
    # 🌟 1차 회피 시간 (우 -> 좌)
    PHASE1_DUR_1ST = 1.4                   # Phase 1 (진입) 유지 시간 (초) - 너무 많이 가면 줄이세요
    PHASE2_DUR_1ST = 1.2                   # Phase 2 (카운터 스티어) 유지 시간 (초)

    # 🌟 2차 복귀 시간 (좌 -> 우) - 첫 번째보다 0.2초씩 줄여서 설정
    PHASE1_DUR_2ND = 1.2
    PHASE2_DUR_2ND = 1.2
    FORCED_STEER_ANGLE = 40 
    
    # 🌟 초기 조향 보정 계수 (1도 쏠림당 가감할 시간) ex)40*0.005 = 0.2초
    DYNAMIC_TIME_COEF = 0.004

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
# 📷 [최강 최적화 5] 윈도우 카메라 버퍼 비우기 (딜레이 완벽 제거)
# ==========================================
class CameraReader:
    def __init__(self, cap):
        self.cap = cap
        self.ret = False
        self.frame = None
        self.running = True
        self.new_frame_event = threading.Event()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        # 백그라운드에서 프레임을 읽어와 항상 가장 최신(0초 딜레이) 프레임만 유지합니다.
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.ret = ret
                    self.frame = frame
                    self.new_frame_event.set() # 새 프레임 도착 알림

    def read(self):
        # 파이썬 메인 루프가 너무 빨라 통신 포트를 터뜨리지 않도록, 새 프레임이 올 때까지(최대 33ms) 대기합니다.
        self.new_frame_event.wait(timeout=0.1)
        self.new_frame_event.clear()
        return self.ret, self.frame

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()

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
    # cv2.imshow("YOLO Pure View", yolo_pure_view)

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
        
        # cv2.imshow("Lane Detection (BEV)", out_img)
        return final_steering, curve_factor_display, True, target_center_x
    else:
        final_steering = config.PREV_STEER
        cv2.putText(out_img, "Lane Lost! Keep PREV_STEER", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        final_steering = max(config.STEER_MIN, min(config.STEER_MAX, int(final_steering)))
        
        # cv2.imshow("Lane Detection (BEV)", out_img)
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
    model_left.predict(source=dummy_img, imgsz=432, verbose=False)
    if mission_model is not None:
        mission_model.predict(source=dummy_img, imgsz=320, verbose=False)
        
    print(f"✅ 추론 엔진 웜업 완료! (현재 구동 장치: {model_right.device.type.upper()})")
    if model_right.device.type != 'cuda':
        print("❌ 경고: GPU가 아닌 CPU로 구동 중입니다! 속도가 느릴 수 있습니다.")
    else:
        print("🚀 완벽합니다! RTX 5060 GPU 가속이 100% 정상 작동 중입니다.")

    print("카메라 설정 중 (주행용 바닥 카메라만 먼저 켭니다)...")
    cap0, _ = camera.initial_setting(cam0port=0, capnum=1)
    
    # [롤백] USB 대역폭 및 동기화 렉 방지를 위해 주행 중 신호등 카메라 연결 차단
    cap1 = None

    if cap0 is None or not cap0.isOpened():
        print("첫 번째 카메라(주행용)를 열 수 없습니다.")
        exit()

    # 윈도우 기본 버퍼사이즈 1 설정 (가끔 무시됨)
    cap0.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # [수정] 윈도우 카메라 버퍼 밀림 방지 백그라운드 리더 장착!
    cap0 = CameraReader(cap0)
    time.sleep(1) # 카메라 리더 워밍업

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
    angle = Config.STEER_CENTER        # 초기 각도 설정 (UnboundLocalError 방지)
    
    last_1st_p1_dur = 0.0
    last_1st_p2_dur = 0.0
    last_2nd_p1_dur = 0.0
    last_2nd_p2_dur = 0.0

    print("=======================================")
    print("🚀 YOLOv8 오토 파일럿 프로그램이 대기 모드로 켜졌습니다.")
    print("단축키 안내: [s] 자율주행 출발 / 정지, [q] 완전 종료, [p] 사다리꼴 좌표 출력")
    print("=======================================")

    try:
        while is_program_running:
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

                # 멈춰있을 때만 윗쪽 카메라를 계속 읽음
                ret1, frame1 = cap1.read()
                if not ret1:
                    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
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
                        # [진짜 원인 해결] 미션 모델을 돌릴 때 주행 모델을 강제로 스킵(끄기)했던 로직을 삭제합니다.
                        # RTX 5060은 충분히 빠르기 때문에 둘 다 돌려도 1프레임(33ms) 안에 연산을 마칠 수 있습니다.
                        # 억지로 주행 모델을 끄면 차가 1/5 확률로 장님이 되어 커브에서 이탈하게 됩니다.
                        if frame_count % 5 == 0:
                            results_f0 = mission_model.predict(source=frame0, imgsz=320, conf=0.3, verbose=False)
                            last_results_f0_boxes = results_f0[0].boxes if results_f0[0].boxes is not None else []
                            last_results_f1_boxes = [] # 신호등 카메라는 안 봄
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
                                # [핵심 로직] 박스의 세로 두께(y2 - y1)를 측정하여 진짜 횡단보도 정지선인지 구별
                                line_thickness = y2 - y1
                                if line_thickness > 20: # 픽셀 두께 임계값
                                    stop_line_in_roi = True
                                else:
                                    cv2.putText(frame0, f"IGNORE FAKE LINE ({line_thickness}px)", (x1, y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

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
                            print("🔌 [시스템] 신호등 카메라(cap1) 전원 OFF. (백그라운드 해제 중...)")
                            # [버그 수정] 윈도우 카메라 해제 시 발생하는 프리징(정지)을 막기 위해 백그라운드 스레드에서 해제
                            def release_cam(cam):
                                try:
                                    cam.release()
                                except Exception:
                                    pass
                            threading.Thread(target=release_cam, args=(cap1,)).start()
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
                    for lidar_ang, dist in zip(angles, distances):
                        if 0 < dist < radar_max_dist:
                            vis_rad = math.radians(lidar_ang - 180)
                            x = int(cx + (dist / radar_max_dist) * (radar_size/2) * math.sin(vis_rad))
                            y = int(cy - (dist / radar_max_dist) * (radar_size/2) * math.cos(vis_rad))
                        
                            if (Config.LIDAR_MIN_ANGLE <= lidar_ang <= Config.LIDAR_MAX_ANGLE) and (Config.LIDAR_MIN_DIST < dist < Config.LIDAR_MAX_DIST):
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
                
                # [동적 시간 보상] 진입 순간의 차량 각도(이전 프레임의 조향각)를 바탕으로 시간을 보정합니다.
                # angle - 90: 양수면 우회전(오른쪽 쏠림) 중, 음수면 좌회전(왼쪽 쏠림) 중
                angle_diff = angle - Config.STEER_CENTER
                
                # 보정 계수 (1도 쏠림당 DYNAMIC_TIME_COEF초 보정)
                compensation = angle_diff * Config.DYNAMIC_TIME_COEF
                
                if lane_state == "LEFT":
                    # 왼쪽으로 가야 하는데 오른쪽(양수)으로 쏠려있으면 시간이 더 필요함 -> 더함
                    Config.DYNAMIC_TIME_OFFSET = compensation
                else:
                    # 오른쪽으로 가야 하는데 왼쪽(음수)으로 쏠려있으면 시간이 더 필요함 -> 뺌
                    Config.DYNAMIC_TIME_OFFSET = -compensation
                    
                reason = "긴급 회피(초근접)!" if is_emergency_obstacle else "안전 회피(센서 퓨전)"
                print(f"🚨 장애물 감지 [{reason}] 차선 변경: {lane_state} (초기 앵글 보정: {Config.DYNAMIC_TIME_OFFSET:+.3f}초)")

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
                
                    p1_dur = Config.PHASE1_DUR_1ST if mission_completed_lane_changes == 0 else Config.PHASE1_DUR_2ND
                    p1_dur += getattr(Config, 'DYNAMIC_TIME_OFFSET', 0.0) # 초기 각도에 따른 보정치 추가
                    p1_dur = max(0.3, p1_dur) # [안전장치] 보정치가 너무 커서 Phase 1이 통째로 스킵되는 것을 방지
                    
                    p2_dur = Config.PHASE2_DUR_1ST if mission_completed_lane_changes == 0 else Config.PHASE2_DUR_2ND

                    if lane_change_phase == 1:
                        # Phase 1: 진입 (Hard Steer)
                        angle = Config.STEER_CENTER - Config.FORCED_STEER_ANGLE if lane_state == "LEFT" else Config.STEER_CENTER + Config.FORCED_STEER_ANGLE
                        angle = max(0, min(180, int(angle)))
                    
                        # 목표 차선을 제대로 밟았는지 확인 (진입 조기 종료)
                        if time_since_switch > p1_dur:
                            lane_change_phase = 2
                            if mission_completed_lane_changes == 0:
                                last_1st_p1_dur = p1_dur
                            else:
                                last_2nd_p1_dur = p1_dur
                            print(f"✅ Phase 1 완료 ({p1_dur}초 경과), Phase 2 카운터 스티어 시작!")
                            
                        cv2.putText(frame0, f"PHASE 1 (ENTER): {angle}", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

                    elif lane_change_phase == 2:
                        # Phase 2: 정렬 (Counter Steer)
                        angle = Config.STEER_CENTER + Config.FORCED_STEER_ANGLE if lane_state == "LEFT" else Config.STEER_CENTER - Config.FORCED_STEER_ANGLE
                        angle = max(0, min(180, int(angle)))
                    
                        # 차체가 똑바로 섰는지 확인 (정렬 조기 종료)
                        if time_since_switch > p1_dur + p2_dur:
                            lane_change_phase = 0
                            mission_completed_lane_changes += 1
                            if mission_completed_lane_changes == 1:
                                last_1st_p2_dur = p2_dur
                            else:
                                last_2nd_p2_dur = p2_dur
                            angle = closed_loop_angle
                            print(f"✅ Phase 2 완료 ({p2_dur}초 경과)! 정상 복귀 (누적 변경: {mission_completed_lane_changes}회)")
                    
                        if lane_change_phase == 2:
                            cv2.putText(frame0, f"PHASE 2 (ALIGN): {angle}", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)

            if lane_change_phase == 0:
                # 정상 Closed-loop 조향 (매 프레임 무조건 실행하여 조향 안정성 100% 확보)
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
            # [추가] 회피 단계 소요 시간 표시 UI (기존 State/Light 대체)
            cv2.putText(frame0, f"1st Evade - P1: {last_1st_p1_dur:.2f}s, P2: {last_1st_p2_dur:.2f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame0, f"2nd Evade - P1: {last_2nd_p1_dur:.2f}s, P2: {last_2nd_p2_dur:.2f}s", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
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

    finally:
        # 오류가 나거나 Ctrl+C로 강제 종료되어도 무조건 안전하게 모터를 정지시킵니다.
        print("🔌 프로그램 종료 시퀀스 가동: 차량을 안전하게 정지합니다.")
        is_program_running = False
        
        if ser is not None and ser.is_open:
            ser.write(b'S\n')
            time.sleep(0.1) # 아두이노가 마지막 명령을 받을 시간을 줌
            ser.close()
            
        if cap0 is not None:
            cap0.release()
        if cap1 is not None:
            cap1.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

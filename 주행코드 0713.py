import numpy as np
import cv2
import Function_Library as fl
from ultralytics import YOLO  # 🌟 YOLOv8 엔진 임포트


# ==========================================
# 🛠️ [최종 자율주행 튜닝 제어 변수]
# ==========================================
class Config:
    # 🌟 코랩에서 학습시켜 가져온 내 가중치 파일명
    MODEL_PATH = "best.pt"

    # ------------------------------------------
    # 📐 조향 제어 (Steering Control) 가중치
    # ------------------------------------------
    OFFSET_FROM_LINE = 210
    LOOK_AHEAD = 80
    KP_ANGLE = 0.2
    KP_OFFSET = 0.2

    STEER_MIN = 70
    STEER_CENTER = 90
    STEER_MAX = 110

    # ------------------------------------------
    # 🗺️ 역투영(Top View) 사다리꼴 좌표 (버드아이뷰 변환용)
    # ------------------------------------------
    SRC_POINTS = np.float32([
        [92, 230],
        [548, 230],
        [640, 360],
        [0, 360]
    ])

    # ------------------------------------------
    # 🪟 슬라이딩 윈도우 (Sliding Window) 알고리즘 변수
    # ------------------------------------------
    N_WINDOWS = 9
    MARGIN = 35
    MIN_PIXELS = 40

    # ------------------------------------------
    # 🔍 이전 프레임 기억(Tracking) 변수
    # ------------------------------------------
    SEARCH_RADIUS = 80
    PREV_RIGHT_X = None
    PREV_STEER = 90


# ==========================================
# 🧠 YOLOv8 차선 검출 및 조향 계산 핵심 함수
# ==========================================
def get_steering_angle(camera, img, model, config):
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

    if len(rightx) > 100:
        right_fit = np.polyfit(righty, rightx, 2)
        right_lane_x = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]

        curve_factor = right_fit[1]
        fixed_offset = config.OFFSET_FROM_LINE

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
    cv2.imshow("Bird Eye View (Top View)", out_img)

    return final_steering


def main():
    arduino = fl.libARDUINO()
    camera = fl.libCAMERA()

    ARDUINO_PORT = 'COM5'
    BAUD_RATE = 9600

    print("아두이노 연결 시도...")
    ser = arduino.init(ARDUINO_PORT, BAUD_RATE)

    print("YOLOv8 차선 인식 모델 로드 중...")
    model = YOLO(Config.MODEL_PATH)

    print("카메라 설정 중...")
    cap0, _ = camera.initial_setting(cam0port=1, capnum=1)

    if cap0 is None or not cap0.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    cap0.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow('Trackbars')
    cv2.resizeWindow('Trackbars', 400, 200)

    default_top_w = int(Config.SRC_POINTS[1][0] - Config.SRC_POINTS[0][0])
    default_bot_w = int(Config.SRC_POINTS[2][0] - Config.SRC_POINTS[3][0])
    default_top_y = int(Config.SRC_POINTS[0][1])
    default_bot_y = int(Config.SRC_POINTS[2][1])

    cv2.createTrackbar('TOP_WIDTH', 'Trackbars', default_top_w, 640, lambda x: None)
    cv2.createTrackbar('BOTTOM_WIDTH', 'Trackbars', default_bot_w, 640, lambda x: None)
    cv2.createTrackbar('TOP_Y', 'Trackbars', default_top_y, 480, lambda x: None)
    cv2.createTrackbar('BOTTOM_Y', 'Trackbars', default_bot_y, 480, lambda x: None)

    is_driving = False

    # ==========================================
    # 💡 [추가] 화면에 띄울 아두이노 상태 문자열 변수 초기화
    # ==========================================
    calib_left_msg = "Calib Left: Wait..."
    calib_right_msg = "Calib Right: Wait..."
    realtime_pot_msg = "Pot: Wait..."

    print("=======================================")
    print("🚀 YOLOv8 오토 파일럿 프로그램이 대기 모드로 켜졌습니다.")
    print("단축키 안내: [s] 자율주행 출발 / 정지, [q] 완전 종료, [p] 사다리꼴 좌표 출력")
    print("=======================================")

    while True:
        top_w = cv2.getTrackbarPos('TOP_WIDTH', 'Trackbars')
        bot_w = cv2.getTrackbarPos('BOTTOM_WIDTH', 'Trackbars')
        top_y = cv2.getTrackbarPos('TOP_Y', 'Trackbars')
        bot_y = cv2.getTrackbarPos('BOTTOM_Y', 'Trackbars')

        center_x = 320
        tl_x = center_x - (top_w // 2)
        tr_x = center_x + (top_w // 2)
        bl_x = center_x - (bot_w // 2)
        br_x = center_x + (bot_w // 2)

        Config.SRC_POINTS = np.float32([
            [tl_x, top_y],
            [tr_x, top_y],
            [br_x, bot_y],
            [bl_x, bot_y]
        ])

        read_data = camera.camera_read(cap0)
        ret, frame = read_data[0], read_data[1]

        if not ret:
            break

        # ==========================================
        # 💡 [추가] 아두이노에서 보내는 시리얼 데이터 수신 및 파싱
        # ==========================================
        if ser is not None and ser.is_open:
            while ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    # 터미널 프린트 대신 어떤 메시지인지 확인 후 각자의 변수에 저장합니다.
                    if "Calib Left" in line:
                        calib_left_msg = line
                    elif "Calib Right" in line:
                        calib_right_msg = line
                    elif "Pot:" in line:
                        realtime_pot_msg = line
                except Exception:
                    pass

        # 🧠 [변경] 기존 OpenCV 이미지 처리 대신 YOLOv8 추론 함수 호출
        angle = get_steering_angle(camera, frame, model, Config)

        status_text = "DRIVING!!" if is_driving else "STOPPED"
        color = (0, 0, 255) if is_driving else (0, 255, 255)

        cv2.putText(frame, f"State: {status_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Steering Angle: {angle}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # ==========================================
        # 💡 [추가] 아두이노 값 화면(OpenCV)에 텍스트로 표시
        # ==========================================
        # 1. 실시간 가변저항값 (하늘색)
        cv2.putText(frame, realtime_pot_msg, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        # 2. 왼쪽 캘리브레이션 값 (노란색)
        cv2.putText(frame, calib_left_msg, (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        # 3. 오른쪽 캘리브레이션 값 (노란색)
        cv2.putText(frame, calib_right_msg, (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        camera.image_show(frame)

        if ser is not None and ser.is_open:
            if is_driving:
                command = f"A{angle}\n"
                ser.write(command.encode('utf-8'))
            else:
                ser.write(b"S\n")

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("프로그램을 종료합니다.")
            break
        elif key == ord('s'):
            is_driving = not is_driving
        elif key == ord('p'):
            print("\n[현재 사다리꼴 좌표(SRC_POINTS)]")
            print("    SRC_POINTS = np.float32([")
            print(f"        [{tl_x}, {top_y}],")
            print(f"        [{tr_x}, {top_y}],")
            print(f"        [{br_x}, {bot_y}],")
            print(f"        [{bl_x}, {bot_y}]")
            print("    ])\n")

    cap0.release()
    cv2.destroyAllWindows()
    if ser is not None and ser.is_open:
        ser.write(b'S\n')
        ser.close()


if __name__ == "__main__":
    main()

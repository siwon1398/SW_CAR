import numpy as np
import cv2
import Function_Library as fl


# ==========================================
# 🛠️ [튜닝] 단일 차선(실선) 추종 + BEV 완전체
# ==========================================
# ==========================================
# 🛠️ [튜닝] 흑백 대비 필터 + 가변 오프셋 (코너링 강화)
# ==========================================
class Config:
    TARGET_LINE = 'RIGHT'
    OFFSET_FROM_LINE = 180

    # 🌟 [NEW] 전방 주시 및 코너링 방어 세팅
    LOOK_AHEAD = 80  # 시선을 내 앞 범퍼에서 얼마나 멀리 둘 것인가? (픽셀 단위)
    CURVE_PUSH_GAIN = 120  # 코너가 심할 때 차를 바깥으로 얼마나 강하게 밀어낼 것인가? (가변 오프셋 배수)

    KP_ANGLE = 0.5
    KP_OFFSET = 0.6
    STEER_MIN = 70
    STEER_CENTER = 90
    STEER_MAX = 110

    # (HSV 필터링 변수 삭제됨)

    SRC_POINTS = np.float32([
        [70, 230],
        [570, 230],
        [640, 360],
        [0, 360]
    ])

    N_WINDOWS = 9
    MARGIN = 35
    MIN_PIXELS = 40


# ==========================================

def get_steering_angle(camera, img, config):
    img_h, img_w = img.shape[:2]
    screen_center_x = img_w // 2

    dst_points = np.float32([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]])

    # 디버깅용 빨간 박스
    pts = config.SRC_POINTS.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    # --- 🌟 [NEW] 1. 흑백 명암 대비(Contrast) 필터 ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Adaptive Threshold: 반경 51픽셀 주변 평균보다 15 이상 더 밝은 녀석만 흰색으로 뽑아냄!
    # (연두색 테이프의 얕은 빛 반사는 여기서 거의 다 죽습니다)
    mask_contrast = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, -15)

    # 얇은 찌꺼기 노이즈(1~2픽셀 두께) 암살
    thickness_kernel = np.ones((5, 5), np.uint8)
    mask_thick_only = cv2.morphologyEx(mask_contrast, cv2.MORPH_OPEN, thickness_kernel)

    # 가로 정지선 지우기
    vertical_kernel = np.ones((20, 2), np.uint8)
    mask_no_horizontal = cv2.morphologyEx(mask_thick_only, cv2.MORPH_OPEN, vertical_kernel)

    # 점선 잇기 (Closing)
    kernel = np.ones((30, 5), np.uint8)
    closed_mask = cv2.morphologyEx(mask_no_horizontal, cv2.MORPH_CLOSE, kernel)

    # 원본 이미지에 덮어씌우는 대신, 이 깨끗한 마스크 자체를 바로 Canny에 넘겨도 됩니다!
    color_filtered = cv2.bitwise_and(img, img, mask=closed_mask)

    # --- 2. Canny Edge ---
    gray_scale = camera.gray_conversion(color_filtered)
    blurring = camera.gaussian_blurring(gray_scale, (7, 7))
    edges = camera.canny_edge(blurring, 100, 200)

    # --- 3. Bird's Eye View ---
    matrix = cv2.getPerspectiveTransform(config.SRC_POINTS, dst_points)
    bev_img = cv2.warpPerspective(edges, matrix, (img_w, img_h))

    # --- 4. Histogram ---
    histogram = np.sum(bev_img[img_h // 2:, :], axis=0)
    midpoint = np.int64(histogram.shape[0] // 2)

    left_x_base = np.argmax(histogram[:midpoint])
    right_x_base = np.argmax(histogram[midpoint:]) + midpoint

    # --- 5. Sliding Window ---
    window_height = np.int64(img_h // config.N_WINDOWS)
    nonzero = bev_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_x_current = left_x_base
    right_x_current = right_x_base

    out_img = np.dstack((bev_img, bev_img, bev_img))
    left_lane_inds, right_lane_inds = [], []

    for window in range(config.N_WINDOWS):
        win_y_low = img_h - (window + 1) * window_height
        win_y_high = img_h - window * window_height

        # 왼쪽 윈도우
        win_xleft_low = left_x_current - config.MARGIN
        win_xleft_high = left_x_current + config.MARGIN
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        if len(good_left_inds) > config.MIN_PIXELS:
            left_x_current = np.int64(np.mean(nonzerox[good_left_inds]))

        # 오른쪽 윈도우
        win_xright_low = right_x_current - config.MARGIN
        win_xright_high = right_x_current + config.MARGIN
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 0, 255), 2)
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        right_lane_inds.append(good_right_inds)
        if len(good_right_inds) > config.MIN_PIXELS:
            right_x_current = np.int64(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    leftx, lefty = nonzerox[left_lane_inds], nonzeroy[left_lane_inds]
    rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

    # --- 6. 🌟 단일 차선 조향 계산 ---
    final_steering = config.STEER_CENTER

    # 🌟 [수정] 시선을 차 바로 앞이 아니라, 설정한 거리만큼 멀리 둡니다! (선 밟기 방지)
    y_eval = img_h - config.LOOK_AHEAD

    if config.TARGET_LINE == 'RIGHT' and len(rightx) > 100:
        right_fit = np.polyfit(righty, rightx, 2)
        right_lane_x = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]

        curve_factor = right_fit[1]

        # 🌟 [수정] 가변 오프셋: 곡선이 심할수록(abs(curve_factor) 증가) 오프셋을 더 크게 늘려서 안쪽 선을 밟지 않게 밀어냅니다.
        dynamic_offset = config.OFFSET_FROM_LINE + (abs(curve_factor) * config.CURVE_PUSH_GAIN)

        target_center_x = right_lane_x - dynamic_offset
        offset_error = target_center_x - screen_center_x

        final_steering = config.STEER_CENTER + (offset_error * config.KP_OFFSET) - (
                    curve_factor * config.KP_ANGLE * 100)

        # 디버깅선: (파란선: 차체 중앙, 노란선: 내가 가야할 동적 목표점, y_eval 높이에 표시)
        cv2.line(out_img, (screen_center_x, img_h), (screen_center_x, img_h - 50), (255, 0, 0), 3)
        cv2.circle(out_img, (int(target_center_x), int(y_eval)), 10, (0, 255, 255), -1)  # 시선 위치에 동그라미!

    # (LEFT 코드 생략 - RIGHT만 사용하신다고 가정)

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

    print("카메라 설정 중...")
    cap0, _ = camera.initial_setting(cam0port=1, capnum=1)

    if cap0 is None or not cap0.isOpened():
        print("카메라를 열 수 없습니다.")
        return

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

        angle = get_steering_angle(camera, frame, Config)

        status_text = "DRIVING!!" if is_driving else "STOPPED"
        color = (0, 0, 255) if is_driving else (0, 255, 255)

        cv2.putText(frame, f"State: {status_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"Steering Angle: {angle}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 메인 카메라 화면 띄우기
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

    cap0.release()
    cv2.destroyAllWindows()
    if ser is not None and ser.is_open:
        ser.write(b'S\n')
        ser.close()


if __name__ == "__main__":
    main()

import numpy as np
import cv2
import Function_Library as fl

import numpy as np
import cv2
import Function_Library as fl


# ==========================================
# 🛠️ [튜닝] 색상+대비 융합 및 추적 메모리 적용
# ==========================================
class Config:
    TARGET_LINE = 'RIGHT'
    OFFSET_FROM_LINE = 260
    LOOK_AHEAD = 80
    CURVE_PUSH_GAIN = 125

    KP_ANGLE = 0.5
    KP_OFFSET = 0.6
    STEER_MIN = 70
    STEER_CENTER = 90
    STEER_MAX = 110

    # 1. 색상 방어: HLS 색공간 적용 (빛 반사 및 검은 트랙-흰 차선 대비 강조)
    # HLS에서 흰색은 L(Lightness, 밝기)이 높고 S(Saturation, 채도)가 낮은 특징을 가집니다.
    # 그늘진 곳도 잡기 위해 밝기 하한을 200에서 150으로 낮춤
    LOWER_WHITE_HLS = np.array([0, 150, 0])
    UPPER_WHITE_HLS = np.array([179, 255, 80])

    # 🌟 [NEW] 주변 문맥 방어: 트랙(아스팔트) 색상 임계값
    # 노이즈나 조명에 의해 트랙 채도가 살짝 올라도 인식되도록 S 상한을 50에서 100으로 완화
    LOWER_TRACK = np.array([0, 0, 0])
    UPPER_TRACK = np.array([179, 150, 100])

    SRC_POINTS = np.float32([
        [70, 230],
        [570, 230],
        [640, 360],
        [0, 360]
    ])

    N_WINDOWS = 9
    MARGIN = 35
    MIN_PIXELS = 40

    # 🌟 [NEW] 거리 방어: 노이즈 무시 및 이전 위치 기억
    SEARCH_RADIUS = 80  # 횡단보도나 엉뚱한 노이즈로 튀는 것을 막기 위해 탐색 반경 축소
    PREV_RIGHT_X = None  # 이전 프레임의 차선 시작점(X)을 저장하는 메모리 변수
    PREV_STEER = 90      # 차선을 잃었을 때 유지할 직전 조향각

    # 🌟 노이즈 필터링 커널 크기 (테스트 코드에서 조정 가능하도록 분리)
    MORPH_THICKNESS = (5, 5)
    # 코너를 돌 때 대각선/가로로 눕는 차선이 세로 필터에 의해 통째로 지워지는 문제 방지
    MORPH_VERTICAL = (5, 5)


# ==========================================

def get_steering_angle(camera, img, config):
    img_h, img_w = img.shape[:2]
    screen_center_x = img_w // 2
    dst_points = np.float32([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]])

    # 디버깅 박스
    pts = config.SRC_POINTS.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    # --- 1. 색상 방어 (HLS 필터 및 CLAHE 조명 보정) ---
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    
    # 🌟 [NEW] CLAHE (적응형 히스토그램 평활화) 적용: 창가/실내 조명 편차 극복
    h, l, s = cv2.split(hls)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    hls_clahe = cv2.merge((h, l_clahe, s))
    
    mask_white = cv2.inRange(hls_clahe, config.LOWER_WHITE_HLS, config.UPPER_WHITE_HLS)

    # 🌟 [NEW] 주변 문맥 방어: 검은색 트랙 주변의 흰색만 인정
    mask_track = cv2.inRange(hls_clahe, config.LOWER_TRACK, config.UPPER_TRACK)
    # 트랙 영역을 25픽셀만큼 팽창시켜 바로 옆의 하얀 차선까지만 덮도록 함
    dilate_kernel = np.ones((25, 25), np.uint8)
    mask_track_dilated = cv2.dilate(mask_track, dilate_kernel, iterations=1)
    
    # 두 마스크의 교집합 연산: 트랙 근처에 있는 흰색만 최종 흰색으로 인정!
    # (연두색 테이프 근처의 하얀 빛 반사는 여기서 지워짐)
    mask_white = cv2.bitwise_and(mask_white, mask_track_dilated)

    thickness_kernel = np.ones(config.MORPH_THICKNESS, np.uint8)
    mask_thick_only = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, thickness_kernel)

    vertical_kernel = np.ones(config.MORPH_VERTICAL, np.uint8)
    mask_no_horizontal = cv2.morphologyEx(mask_thick_only, cv2.MORPH_OPEN, vertical_kernel)

    kernel = np.ones((30, 5), np.uint8)
    closed_mask = cv2.morphologyEx(mask_no_horizontal, cv2.MORPH_CLOSE, kernel)
    color_filtered = cv2.bitwise_and(img, img, mask=closed_mask)

    # --- 2. 대비 방어 (Canny Edge) ---
    # 회색 트랙과 흰색 차선의 명확한 경계만 추출합니다.
    gray_scale = camera.gray_conversion(color_filtered)
    blurring = camera.gaussian_blurring(gray_scale, (7, 7))
    edges = camera.canny_edge(blurring, 70, 140)  # 회색 트랙과의 경계를 잘 잡도록 임계값을 약간 조정

    matrix = cv2.getPerspectiveTransform(config.SRC_POINTS, dst_points)
    bev_img = cv2.warpPerspective(edges, matrix, (img_w, img_h))

    # --- 3. 🌟 거리 방어 (Tracking Memory) ---
    histogram = np.sum(bev_img[img_h // 2:, :], axis=0)
    midpoint = np.int64(histogram.shape[0] // 2)

    # 이전 프레임에서 찾은 차선이 있다면, 그 근처만 탐색합니다.
    if config.PREV_RIGHT_X is not None:
        search_min = max(midpoint, config.PREV_RIGHT_X - config.SEARCH_RADIUS)
        search_max = min(img_w, config.PREV_RIGHT_X + config.SEARCH_RADIUS)

        right_search_area = histogram[search_min:search_max]

        # 지정된 범위 안에 하얀색 픽셀(차선)이 존재할 때만 위치 갱신
        if np.max(right_search_area) > 0:
            right_x_base = np.argmax(right_search_area) + search_min
        else:
            # 근처에 선이 없으면 다시 오른쪽 절반 전체를 탐색 (초기화)
            right_x_base = np.argmax(histogram[midpoint:]) + midpoint
    else:
        # 첫 프레임이거나 정보가 없을 때는 전체 탐색
        right_x_base = np.argmax(histogram[midpoint:]) + midpoint

    # 다음 프레임을 위해 현재 위치를 메모리에 저장
    config.PREV_RIGHT_X = right_x_base

    # --- 4. Sliding Window 추적 ---
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

    # --- 5. 조향 계산 ---
    y_eval = img_h - config.LOOK_AHEAD

    if len(rightx) > 100:
        right_fit = np.polyfit(righty, rightx, 2)
        right_lane_x = right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[2]

        curve_factor = right_fit[1]
        # 🌟 다이나믹 오프셋 제거: 차선을 잃거나 역주행하는 것을 막기 위해 항상 '정중앙(Fixed Offset)' 유지
        fixed_offset = config.OFFSET_FROM_LINE 

        target_center_x = right_lane_x - fixed_offset
        offset_error = target_center_x - screen_center_x

        final_steering = config.STEER_CENTER + (offset_error * config.KP_OFFSET) - (
                    curve_factor * config.KP_ANGLE * 100)

        # 성공적으로 계산된 경우, 다음 프레임을 위해 각도 기억
        config.PREV_STEER = final_steering

        cv2.line(out_img, (screen_center_x, img_h), (screen_center_x, img_h - 50), (255, 0, 0), 3)
        cv2.circle(out_img, (int(target_center_x), int(y_eval)), 10, (0, 255, 255), -1)
    else:
        # 🌟 차선을 놓쳤을 때 예외 처리: 중앙으로 푸는 대신 직전 각도 유지
        final_steering = config.PREV_STEER
        cv2.putText(out_img, "Lane Lost! Keep Prev Steer", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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

    # 🌟 최신 프레임만 받아오도록 버퍼 크기를 1로 강제 제한 (지연 방지)
    cap0.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # 🌟 대칭 사다리꼴 조절을 위한 트랙바 (윗변 길이, 아랫변 길이, 윗변 높이, 아랫변 높이)
    cv2.namedWindow('Trackbars')
    cv2.resizeWindow('Trackbars', 400, 200)
    
    # 기본값 계산 (화면 중앙 320 기준)
    default_top_w = int(Config.SRC_POINTS[1][0] - Config.SRC_POINTS[0][0])
    default_bot_w = int(Config.SRC_POINTS[2][0] - Config.SRC_POINTS[3][0])
    default_top_y = int(Config.SRC_POINTS[0][1])
    default_bot_y = int(Config.SRC_POINTS[2][1])

    cv2.createTrackbar('TOP_WIDTH', 'Trackbars', default_top_w, 640, lambda x: None)
    cv2.createTrackbar('BOTTOM_WIDTH', 'Trackbars', default_bot_w, 640, lambda x: None)
    cv2.createTrackbar('TOP_Y', 'Trackbars', default_top_y, 480, lambda x: None)
    cv2.createTrackbar('BOTTOM_Y', 'Trackbars', default_bot_y, 480, lambda x: None)

    # 🌟 동적 조명 제어 트랙바 추가
    cv2.createTrackbar('L_LOWER', 'Trackbars', int(Config.LOWER_WHITE_HLS[1]), 255, lambda x: None)

    is_driving = False
    print("=======================================")
    print("프로그램이 대기 모드로 켜졌습니다.")
    print("단축키 안내: [s] 출발 / 정지, [q] 완전 종료, [p] 현재 사다리꼴 좌표 출력")
    print("=======================================")

    while True:
        # 트랙바 값 읽어오기 및 실시간 반영
        top_w = cv2.getTrackbarPos('TOP_WIDTH', 'Trackbars')
        bot_w = cv2.getTrackbarPos('BOTTOM_WIDTH', 'Trackbars')
        top_y = cv2.getTrackbarPos('TOP_Y', 'Trackbars')
        bot_y = cv2.getTrackbarPos('BOTTOM_Y', 'Trackbars')
        l_lower = cv2.getTrackbarPos('L_LOWER', 'Trackbars')

        # 화면 중앙(320)을 기준으로 좌우 대칭 좌표 계산
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
        
        # 실시간 조명 하한값 반영
        Config.LOWER_WHITE_HLS[1] = l_lower

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

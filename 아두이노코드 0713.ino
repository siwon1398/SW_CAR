#include "Car_Library.h"
// --- 1. 모터 핀 설정 ---
int left_front = 4;
int left_back = 3;
int right_front = 11;
int right_back = 12;
int steer_right = 7;
int steer_left = 8;
int pot_pin = A2;

// --- 2. 속도 제어 변수 ---
int DRIVE_SPEED = 255;
int STEER_SPEED = 150;   // 캘리브레이션(끝까지 이동) 시 사용

int target_angle = 90;
int current_angle = 90;
int error_margin = 1;

// [추가] 좌우 끝값 캘리브레이션 변수
int pot_left_end = 0;    // 왼쪽 끝까지 돌렸을 때의 pot 값
int pot_right_end = 0;   // 오른쪽 끝까지 돌렸을 때의 pot 값
const unsigned long CALIB_MOVE_TIME = 3000; // 끝까지 이동에 걸리는 시간 (ms, 실측 후 조정 필요)

void setup() {
  Serial.begin(9600);

  pinMode(left_front, OUTPUT);
  pinMode(left_back, OUTPUT);
  pinMode(right_front, OUTPUT);
  pinMode(right_back, OUTPUT);
  pinMode(steer_right, OUTPUT);
  pinMode(steer_left, OUTPUT);

  stopDrive();
  analogWrite(steer_right, 0);
  analogWrite(steer_left, 0);

  // [추가] 1. 왼쪽 끝까지 강제 이동 후 값 측정
  analogWrite(steer_left, STEER_SPEED);
  delay(CALIB_MOVE_TIME);
  analogWrite(steer_left, 0);
  delay(100); // 관성 정지 대기 (임의값)
  pot_left_end = analogRead(pot_pin);

  // [추가] 2. 오른쪽 끝까지 강제 이동 후 값 측정
  analogWrite(steer_right, STEER_SPEED);
  delay(CALIB_MOVE_TIME);
  analogWrite(steer_right, 0);
  delay(100);
  pot_right_end = analogRead(pot_pin);

  // [추가] 3. 확인용 출력 (OpenCV 화면 출력을 위해 영문으로 변경)
  Serial.print("Calib Left: ");
  Serial.println(pot_left_end);
  Serial.print("Calib Right: ");
  Serial.println(pot_right_end);

  // [추가] 4. 캘리브레이션 후 중앙으로 복귀 시도 (선택 사항)
  int pot_center_target = (pot_left_end + pot_right_end) / 2;
  // 중앙 근처로 이동 (간단화된 방식, 정밀 제어는 loop() 로직과 별개)
  while (abs(analogRead(pot_pin) - pot_center_target) > 5) {
    if (analogRead(pot_pin) > pot_center_target) {
      analogWrite(steer_right, STEER_SPEED);
      analogWrite(steer_left, 0);
    } else {
      analogWrite(steer_right, 0);
      analogWrite(steer_left, STEER_SPEED);
    }
  }
  analogWrite(steer_right, 0);
  analogWrite(steer_left, 0);
}

void loop() {
  // --- 1. 파이썬 명령 수신 ---
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');

    if (data.startsWith("A")) {
      target_angle = data.substring(1).toInt();
      goForward();
    }
    else if (data.startsWith("S")) {
      stopDrive();
    }
  }

  // --- 2. 현재 조향각 읽기 (좌우 끝값 기반 매핑) ---
  int pot_value = analogRead(pot_pin);
  // pot_left_end → 70도, pot_right_end → 110도로 매핑
  current_angle = map(pot_value, pot_left_end, pot_right_end, 70, 110);

  // ==========================================
  // 💡 [추가] 실시간 가변저항값 시리얼 통신 송신 (500ms 간격 제한)
  // ==========================================
  static unsigned long last_pot_print = 0;
  if (millis() - last_pot_print > 500) {
    Serial.print("Pot: ");
    Serial.print(pot_value);
    Serial.print(" | Cur Ang: ");
    Serial.println(current_angle);
    last_pot_print = millis();
  }

  // --- 오차 기반 동적 속도 계산 ---
  int angle_error = abs(target_angle - current_angle);
  int dynamic_steer_speed = map(angle_error, 0, 30, 60, 130);
  dynamic_steer_speed = constrain(dynamic_steer_speed, 50, 130);

  // --- 조향 모터 작동 로직 ---
  if (current_angle < target_angle - error_margin) {
    analogWrite(steer_right, dynamic_steer_speed);
    analogWrite(steer_left, 0);
  }
  else if (current_angle > target_angle + error_margin) {
    analogWrite(steer_right, 0);
    analogWrite(steer_left, dynamic_steer_speed);
  }
  else {
    analogWrite(steer_right, 0);
    analogWrite(steer_left, 0);
  }
}

// --- 3. 하위 구동 함수 ---
void goForward() {
  analogWrite(left_front, DRIVE_SPEED);
  analogWrite(left_back, 0);
  analogWrite(right_front, DRIVE_SPEED);
  analogWrite(right_back, 0);
}

void stopDrive() {
  analogWrite(left_front, 0);
  analogWrite(left_back, 0);
  analogWrite(right_front, 0);
  analogWrite(right_back, 0);
}

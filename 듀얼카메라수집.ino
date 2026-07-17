#include "Car_Library.h"

// --- 1. 모터 핀 설정 ---
int left_front = 4;   // 왼쪽 전진
int left_back = 3;    // 왼쪽 후진
int right_front = 11; // 오른쪽 전진
int right_back = 12;  // 오른쪽 후진

int steer_right = 7;  // 조향 우회전
int steer_left = 8;   // 조향 좌회전
int pot_pin = A2;     // 가변 저항 핀

// --- 2. 속도 및 조향 변수 ---
int DRIVE_SPEED = 110; // 주행 속도 (전진/후진 동일)
int STEER_SPEED = 150; // 캘리브레이션(끝까지 이동) 시 사용하는 조향 모터 속도

int target_angle = 90;    
int current_angle = 90; 
int error_margin = 1;

// 🌟 [추가] 좌우 끝값 캘리브레이션 변수
int pot_left_end = 0;    // 왼쪽 끝까지 돌렸을 때의 pot 값
int pot_right_end = 0;   // 오른쪽 끝까지 돌렸을 때의 pot 값
const unsigned long CALIB_MOVE_TIME = 3000; // 끝까지 이동에 걸리는 시간 (ms)

void setup() {
  Serial.begin(9600);
  
  pinMode(left_front, OUTPUT);
  pinMode(left_back, OUTPUT);
  pinMode(right_front, OUTPUT);
  pinMode(right_back, OUTPUT);
  
  pinMode(steer_right, OUTPUT);
  pinMode(steer_left, OUTPUT);
  
  // 초기 상태 완전 정지
  stopDrive();
  analogWrite(steer_right, 0);
  analogWrite(steer_left, 0);

  // ==========================================
  // 💡 시작 시 조향 끝값 자동 캘리브레이션
  // ==========================================
  // 1. 왼쪽 끝까지 강제 이동 후 값 측정
  analogWrite(steer_left, STEER_SPEED);
  delay(CALIB_MOVE_TIME);
  analogWrite(steer_left, 0);
  delay(100); // 관성 정지 대기
  pot_left_end = analogRead(pot_pin);

  // 2. 오른쪽 끝까지 강제 이동 후 값 측정
  analogWrite(steer_right, STEER_SPEED);
  delay(CALIB_MOVE_TIME);
  analogWrite(steer_right, 0);
  delay(100);
  pot_right_end = analogRead(pot_pin);

  // 3. 파이썬 화면 출력을 위해 시리얼 전송
  Serial.print("Calib Left: ");
  Serial.println(pot_left_end);
  Serial.print("Calib Right: ");
  Serial.println(pot_right_end);

  // 4. 캘리브레이션 후 조향 중앙으로 복귀
  int pot_center_target = (pot_left_end + pot_right_end) / 2;
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
  // --- 1. 파이썬 명령 수신 (독립 제어 구조로 개편) ---
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    
    if (data.startsWith("A")) {
      // 조향 명령: 전진을 호출하지 않고 각도만 바꿈
      target_angle = data.substring(1).toInt();
    } 
    else if (data.startsWith("F")) {
      // 전진 명령 수신
      goForward();
    }
    else if (data.startsWith("B")) {
      // 후진 명령 수신
      goBackward();
    }
    else if (data.startsWith("S")) {
      // 정지 명령 수신
      stopDrive();
    }
  }
  
  // --- 2. 현재 조향각 읽기 및 P제어 ---
  int pot_value = analogRead(pot_pin);
  
  // 🌟 고정값(555, 460) 대신 setup()에서 방금 측정한 캘리브레이션 변수 사용
  current_angle = map(pot_value, pot_left_end, pot_right_end, 70, 110); 

  // ==========================================
  // 💡 파이썬 모니터링을 위한 실시간 데이터 송신 (0.5초 간격)
  // ==========================================
  static unsigned long last_pot_print = 0;
  if (millis() - last_pot_print > 500) {
    Serial.print("Pot: ");
    Serial.print(pot_value);
    Serial.print(" | Cur Ang: ");
    Serial.println(current_angle);
    last_pot_print = millis();
  }

  // --- 조향 모터 작동 로직 ---
  int angle_error = abs(target_angle - current_angle);
  int dynamic_steer_speed = map(angle_error, 0, 30, 50, 130);
  dynamic_steer_speed = constrain(dynamic_steer_speed, 50, 130);

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

void goBackward() {
  analogWrite(left_front, 0);
  analogWrite(left_back, DRIVE_SPEED);   // 왼쪽 후진 활성화
  analogWrite(right_front, 0);
  analogWrite(right_back, DRIVE_SPEED);  // 오른쪽 후진 활성화
}

void stopDrive() {
  analogWrite(left_front, 0);
  analogWrite(left_back, 0);
  analogWrite(right_front, 0);
  analogWrite(right_back, 0);
}

#include "Car_Library.h"

// --- 1. 모터 핀 설정 (기존 핀 번호 그대로 유지) ---
int left_front = 4;   // 왼쪽 전진
int left_back = 3;    // 왼쪽 후진
int right_front = 11; // 오른쪽 전진
int right_back = 12;  // 오른쪽 후진

int steer_right = 7;  // 조향 우회전
int steer_left = 8;   // 조향 좌회전

int pot_pin = A2;     // 가변 저항 핀 -> 확인필

// --- 2. 속도 제어 변수 (★튜닝 구역: 0 ~ 255 사이로 조절★) ---
int DRIVE_SPEED = 150; // 🚗 뒷바퀴 주행 속도 (초기 테스트용으로 느리게 설정)
int STEER_SPEED = 130; // ⚙️ 앞바퀴 조향 속도 (부드럽게 조향되도록 설정)

int target_angle = 90;    
int current_angle = 90; 
int error_margin = 1; 

void setup() {
  Serial.begin(9600);
  
  pinMode(left_front, OUTPUT);
  pinMode(left_back, OUTPUT);
  pinMode(right_front, OUTPUT);
  pinMode(right_back, OUTPUT);
  
  pinMode(steer_right, OUTPUT);
  pinMode(steer_left, OUTPUT);
  
  // 🌟 초기 상태는 안전하게 완전 정지
  stopDrive();
  analogWrite(steer_right, 0);
  analogWrite(steer_left, 0);
}

void loop() {
  // --- 1. 파이썬 명령 수신 ---
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n'); 
    
    if (data.startsWith("A")) {
      target_angle = data.substring(1).toInt(); 
      // 주행 신호(A)가 오면 설정한 속도(DRIVE_SPEED)로 전진
      goForward(); 
    } 
    else if (data.startsWith("S")) {
      // 정지 신호(S)가 오면 모터 즉시 끄기
      stopDrive(); 
    }
  }
  
  // --- 2. 현재 조향각 읽기 ---
  int pot_value = analogRead(pot_pin); 
  current_angle = map(pot_value, 555, 460, 70, 110); 

  // 🌟 [핵심 보완] 목표 각도와 현재 각도의 '오차(거리)' 계산
  int angle_error = abs(target_angle - current_angle);

  // 🌟 [핵심 보완] 오차에 비례하여 모터 속도를 유동적으로 결정
  // 오차가 30도 이상이면 최대 속도(130), 오차가 0에 가까워지면 최소 속도(50)로 줄임
  int dynamic_steer_speed = map(angle_error, 0, 30, 50, 130);
  
  // 속도가 50~130 범위를 벗어나지 않도록 안전장치
  dynamic_steer_speed = constrain(dynamic_steer_speed, 50, 130); 
  // (참고: 50은 모터가 바닥 마찰력을 이기고 움직일 수 있는 최소한의 힘. 테스트하며 조절 필요)

  // --- 조향 모터 작동 로직 ---
  if (current_angle < target_angle - error_margin) {
    // 고정된 STEER_SPEED 대신, 계산된 동적 속도(dynamic_steer_speed)를 넣습니다.
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


// --- 3. 하위 구동 함수 (analogWrite 적용) ---
void goForward() {
  // 설정한 DRIVE_SPEED 만큼만 전압을 주어 부드럽게 주행
  analogWrite(left_front, DRIVE_SPEED);
  analogWrite(left_back, 0);
  analogWrite(right_front, DRIVE_SPEED);
  analogWrite(right_back, 0);
}

void stopDrive() {
  // 완전히 멈출 때는 모두 0을 할당
  analogWrite(left_front, 0);
  analogWrite(left_back, 0);
  analogWrite(right_front, 0);
  analogWrite(right_back, 0);
}

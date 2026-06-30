#include "Car_Library.h"

// --- 1. 모터 핀 설정 (기존 핀 번호 그대로 유지) ---
int left_front = 4;   // 왼쪽 전진
int left_back = 3;    // 왼쪽 후진
int right_front = 11; // 오른쪽 전진
int right_back = 12;  // 오른쪽 후진

int steer_right = 7;  // 조향 우회전
int steer_left = 8;   // 조향 좌회전

int pot_pin = A5;     // 가변 저항 핀 -> 확인필

// --- 2. 속도 제어 변수 (★튜닝 구역: 0 ~ 255 사이로 조절★) ---
int DRIVE_SPEED = 50; // 🚗 뒷바퀴 주행 속도 (초기 테스트용으로 느리게 설정)
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

  // --- 2. 현재 조향각 읽기 및 앞바퀴 아날로그 제어 ---
  int pot_value = analogRead(pot_pin); 
  //current_angle = map(pot_value, 0, 55, 60, 120); 

  if (current_angle < target_angle - error_margin) {
    // 목표보다 왼쪽으로 치우침 -> 오른쪽으로 부드럽게 꺾기
    analogWrite(steer_right, STEER_SPEED);
    analogWrite(steer_left, 0);
  } 
  else if (current_angle > target_angle + error_margin) {
    // 목표보다 오른쪽으로 치우침 -> 왼쪽으로 부드럽게 꺾기
    analogWrite(steer_right, 0);
    analogWrite(steer_left, STEER_SPEED);
  } 
  else {
    // 목표 각도 도달 -> 조향 모터 부드럽게 정지
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

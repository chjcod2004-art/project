<div align="center">

# 🚨 의료·돌봄·에너지·통신 4-Way 통합 위기가구 조기 경보 시스템
**팀명: 복세편살** | 팀원: **이경규 · 정지훈 · 이강현**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://project-dsrtwkwamo499hp9lqw6hm.streamlit.app/)
[![YouTube Demo](https://img.shields.io/badge/YouTube-시연영상_보기-FF0000?style=flat&logo=youtube&logoColor=white)](https://youtu.be/YkY16A9mUWY)
[![Notion Plan](https://img.shields.io/badge/Notion-프로젝트_계획서-000000?style=flat&logo=notion&logoColor=white)](https://app.notion.com/p/40748740823e806fa9c4c2b08fe422a3?source=copy_link)

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost_%2B_SHAP-189FDD?style=for-the-badge)

</div>

---

## 🔗 주요 링크 및 자료
* **웹 시연 (Streamlit Cloud):** [바로가기](https://project-dsrtwkwamo499hp9lqw6hm.streamlit.app/)
* **시연 영상 (YouTube):** [바로가기](https://youtu.be/YkY16A9mUWY)
* **프로젝트 계획서 (Notion):** [바로가기](https://app.notion.com/p/40748740823e806fa9c4c2b08fe422a3?source=copy_link)
* **발표 PPT:** 저장소 내 `빅데이터 발표 자료.pptx` 파일 참고

---

## 📌 1. 프로젝트 개요 및 주제 선택 이유

### 프로젝트 설명
행정·의료 데이터(**진료 정보, 만성질환+급여현황**)와 일상 생활 데이터(**전력·가스 사용량, 통신 데이터**)를 결합하여 위기 징후가 나타나는 가구를 조기에 찾아내고, 사전에 예방할 수 있도록 만든 통합 조기 경보 시스템입니다.

### 주제 선택 이유
* **사후 발굴의 시차 극복:** 현재 보건복지부에서 운영 중인 시스템은 체납·연체가 발생한 이후 수개월이 지나서야 위험 가구를 포착하고 있습니다. 고독사 예방과 취약계층 보호를 위해서는 위기 징후 데이터를 최대한 빠르게 포착하는 것이 중요하다고 판단했습니다.
* **행정 효율화 및 사회적 가치 실현:** 데이터 기반 선제적 탐지를 통해 복지 담당 공무원들의 업무 과부하를 줄이고 고독사를 예방하고자 합니다.

---

## 📊 2. 데이터 분석 및 AI 활용 방안

### 데이터 분석 파이프라인
| 구분 | 활용 데이터 | 분석 방식 및 역할 |
| :--- | :--- | :--- |
| **정적 / 주기 데이터** | 진료과목별 진료정보 + 만성질환 + 급여현황 | 기본 위기 기초점수 산출 |
| **동적 시계열 데이터** | 통신, 전력, 가스 사용량 | 1주 단위의 활동 감소량 및 계절 대비 패턴 이상 분석 → 기본 위기점수에 가산 점수 부여 |

### AI 적용 및 목표 (`XGBoost + SHAP`)
* **현재 구현:** 실측 결합 데이터 확보의 제약으로 대시보드 내 프로토타입 시뮬레이터로 구현했습니다.
* **실제 도입 목표:** 4개 도메인 간의 비선형 상호작용(예: 여름철 냉방비 절감 vs 실제 위기 급감 구분)을 정교하게 학습하기 위해 **`XGBoost + SHAP`** 모델을 적용하여 AI가 이상치를 탐지하고 판단 근거를 설명할 수 있도록 하는 것이 목표입니다.

---

## 🗂️ 3. 사용 데이터 출처

| 데이터명 | 제공기관 / 출처 | 링크 |
| :--- | :--- | :--- |
| **진료과목별 진료정보** | 공공데이터포털 | [data.go.kr](https://www.data.go.kr) |
| **만성질환+ 급여현황** | 국민건강보험포털 | [바로가기](https://www.nhis.or.kr/nhis/together/wbhaec06900m01.do) |
| **전력 사용량** | 전력데이터 개방 포털 시스템 | [바로가기](https://bigdata.kepco.co.kr/cmsmain.do?scode=S01&pcode=000171&pstate=L&redirect=Y) |
| **가스 사용량** | 서울 열린데이터 광장 | [바로가기](https://data.seoul.go.kr/dataList/DT201004C090004/S/2/datasetView.do) |
| **통신데이터** | 공공데이터포털 | [data.go.kr](https://www.data.go.kr) |

---

## 🚀 4. 데이터 분석 개선 및 현장 도입 방향
1. **1단계 (현재):** 공공 개방 집계 데이터를 활용하여 자치구별 고위험 군집을 우선 선별합니다.
2. **2단계 (현장 도입):** **사회보장급여법** 근거 또는 **고위험 독거가구 사전 동의 기반(지자체 스마트돌봄 시범사업)**을 마련하고, **가명정보 결합전문기관**을 통해 단계적으로 연동하여 의료·전력·통신 데이터를 실제 가구 단위로 통합할 계획입니다.

---

## 👥 5. 팀 소개 (팀 복세편살)
* **이경규**
* **정지훈**
* **이강현**

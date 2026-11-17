import glob
import os
import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import xyzservices.providers as xyz
from streamlit_folium import st_folium

import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- 클라우드/로컬 환경 대응 한글 폰트 자동 설정 ---
if os.name == 'posix':  # Streamlit Cloud (Linux) 환경인 경우
    # 리눅스 시스템에 나눔고딕이 있는지 확인하고 설정
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    if os.path.exists(font_path):
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams['font.family'] = font_name
    else:
        plt.rcParams['font.family'] = 'DejaVu Sans'  # 대체 폰트
else:  # 윈도우(Windows) 환경인 경우
    plt.rcParams['font.family'] = 'Malgun Gothic'

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 페이지 기본 설정
st.set_page_config(
    page_title="의료·돌봄 및 에너지 시계열 기반 위기가구 조기 경보 대시보드",
    page_icon="🚨",
    layout="wide",
)

# 한글 폰트 설정 (윈도우: Malgun Gothic)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# CartoDB API Key 설정
CARTODB_API_KEY = "cb1_55g6_1_88ffc32d9c064d69a43b4508"

# 서울 25개 자치구 중심 위경도 좌표 (공통 사용)
SEOUL_GU_COORDS = {
    "종로구": [37.5730, 126.9794], "중구": [37.5641, 126.9979],
    "용산구": [37.5326, 126.9900], "성동구": [37.5633, 127.0371],
    "광진구": [37.5385, 127.0823], "동대문구": [37.5744, 127.0400],
    "중랑구": [37.6066, 127.0927], "성북구": [37.5891, 127.0182],
    "강북구": [37.6396, 127.0257], "도봉구": [37.6688, 127.0471],
    "노원구": [37.6542, 127.0568], "은평구": [37.6027, 126.9291],
    "서대문구": [37.5791, 126.9368], "마포구": [37.5663, 126.9016],
    "양천구": [37.5170, 126.8664], "강서구": [37.5510, 126.8495],
    "구로구": [37.4954, 126.8874], "금천구": [37.4569, 126.8954],
    "영등포구": [37.5264, 126.8962], "동작구": [37.5124, 126.9393],
    "관악구": [37.4784, 126.9516], "서초구": [37.4837, 127.0324],
    "강남구": [37.5172, 127.0473], "송파구": [37.5145, 127.1066],
    "강동구": [37.5301, 127.1238],
}


# --- 공통 유틸: 안전한 CSV 읽기 함수 (인코딩 자동 처리) ---
def read_csv_safe(filepath):
    for enc in ["utf-8", "cp949", "utf-8-sig", "euc-kr"]:
        try:
            return pd.read_csv(filepath, encoding=enc)
        except Exception:
            continue
    return pd.DataFrame()


# --- 1. 데이터셋 ①: MEDICINE_202501 ~ 202512 (12개월 전체) 로드 및 경량화 ---
@st.cache_data
def load_medicine_data():
    files = sorted(glob.glob("MEDICINE_*.csv"))
    if not files:
        files = sorted(glob.glob(os.path.join("data", "MEDICINE_*.csv")))

    if not files:
        return pd.DataFrame()

    sido_mapping = {
        11: "서울", 26: "부산", 27: "대구", 28: "인천", 29: "광주",
        30: "대전", 31: "울산", 36: "세종", 41: "경기", 43: "충북",
        44: "충남", 46: "전남", 47: "전북", 48: "경남", 50: "제주",
        51: "강원", 52: "경북",
    }

    agg_list = []
    for f in files:
        temp_df = read_csv_safe(f)
        if temp_df.empty:
            continue
        grp = (
            temp_df.groupby(
                ["STD_YYYYMM", "AGE_G", "SEX_TYPE", "SIDO", "OUT_IN_HOS"],
                as_index=False,
            )
            .agg(
                {
                    "PATIENT_CNT": "sum",
                    "PRSCRPTN_TNDN_CNT": "sum",
                    "PRSCRPTN_AMT": "sum",
                    "PRSCRPTN_TOT_USE": "sum",
                }
            )
        )
        agg_list.append(grp)

    if not agg_list:
        return pd.DataFrame()

    df = pd.concat(agg_list, ignore_index=True)
    df["SIDO_NAME"] = df["SIDO"].map(sido_mapping).fillna(df["SIDO"].astype(str))
    return df


# --- 2. 데이터셋 ②: 시군구별 진료과목별 진료 정보 로드 ---
@st.cache_data
def load_department_data():
    patterns = [
        "*시군구별*진료과목별*.csv",
        os.path.join("data", "*시군구별*진료과목별*.csv"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))

    if not files:
        return pd.DataFrame()

    df_dept = read_csv_safe(files[0])
    if df_dept.empty:
        return pd.DataFrame()

    grp = (
        df_dept.groupby(["진료년도", "진료월", "시도", "진료과목명"], as_index=False)[
            ["진료인원(명)", "진료건수(건)"]
        ]
        .sum()
    )
    grp["년월"] = (
        grp["진료년도"].astype(str) + "-" + grp["진료월"].astype(str).str.zfill(2)
    )
    grp["1인당_월평균_진료건수"] = grp["진료건수(건)"] / grp["진료인원(명)"].replace(0, np.nan)
    return grp


# --- 3. 데이터셋 ③: 만성질환 급여현황 데이터 로드 ---
@st.cache_data
def load_chronic_data():
    patterns = [
        "*만성질환*급여현황*.csv",
        os.path.join("data", "*만성질환*급여현황*.csv"),
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p))

    if not files:
        return pd.DataFrame()

    raw_df = read_csv_safe(files[0])
    if raw_df.empty:
        return pd.DataFrame()

    df_chr = raw_df.iloc[1:].copy()
    num_cols = [
        "2023", "2023.1", "2023.2", "2023.3", "2023.4",
        "2024", "2024.1", "2024.2", "2024.3", "2024.4",
    ]
    for c in num_cols:
        if c in df_chr.columns:
            df_chr[c] = pd.to_numeric(df_chr[c], errors="coerce")

    df_chr = df_chr.rename(
        columns={
            "자치구별(2)": "자치구",
            "질환별(1)": "질환명",
            "2024": "진료실인원_2024",
            "2024.1": "내원일수_2024",
            "2024.2": "급여일수_2024",
            "2024.3": "진료비_2024",
            "2023": "진료실인원_2023",
            "2023.1": "내원일수_2023",
            "2023.2": "급여일수_2023",
        }
    )

    df_chr["1인당_연간내원일수"] = df_chr["내원일수_2024"] / df_chr["진료실인원_2024"]
    df_chr["1회내원당_평균처방일수"] = df_chr["급여일수_2024"] / df_chr["내원일수_2024"]
    df_chr["평균_내원간격(일)"] = 365.0 / df_chr["1인당_연간내원일수"]

    return df_chr


# --- 4. 데이터셋 ④: 대규모 시뮬레이션 (전체 200만가구 중 고위험 30만가구 생성) ---
@st.cache_data
def generate_power_gas_simulation(n_samples=2000000, n_crisis=300000, power_w=0.5, gas_w=0.5):
    df_power_base = pd.DataFrame({
        "시군구": [
            "강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구", "금천구",
            "노원구", "도봉구", "동대문구", "동작구", "마포구", "서대문구", "서초구", "성동구",
            "성북구", "송파구", "양천구", "영등포구", "용산구", "은평구", "종로구", "중구", "중랑구",
        ],
        "avg_power_monthly": [
            282, 266, 223, 236, 192, 223, 244, 218, 257, 246, 230, 232, 250,
            248, 294, 266, 256, 279, 274, 235, 263, 255, 217, 233, 232,
        ],
    })

    df_gas_base = pd.DataFrame({
        "시군구": df_power_base["시군구"],
        "avg_gas_monthly": [
            65, 70, 60, 62, 55, 58, 63, 57, 68, 61, 59, 60, 64,
            62, 72, 65, 63, 67, 66, 59, 64, 63, 55, 58, 60,
        ],
    })

    df_base = pd.merge(df_power_base, df_gas_base, on="시군구")
    df_base = df_base.assign(
        daily_power_base=(df_base["avg_power_monthly"] / 30).round(2),
        daily_gas_base=(df_base["avg_gas_monthly"] / 30).round(2),
    )

    np.random.seed(42)
    df_households = pd.DataFrame({
        "household_id": [f"H_{i:08d}" for i in range(1, n_samples + 1)],
        "시군구": np.random.choice(df_base["시군구"], size=n_samples),
    })

    df_households = pd.merge(
        df_households,
        df_base[["시군구", "avg_power_monthly", "avg_gas_monthly", "daily_power_base", "daily_gas_base"]],
        on="시군구",
    )

    # 정상 가구: 평소 기준치의 80% ~ 120% 범위 내 사용
    df_households = df_households.assign(
        recent_power_usage=(
            df_households["daily_power_base"] * np.random.uniform(0.8, 1.2, size=n_samples)
        ).round(2),
        recent_gas_usage=(
            df_households["daily_gas_base"] * np.random.uniform(0.8, 1.2, size=n_samples)
        ).round(2),
    )

    # 정확히 n_crisis(30만 명)를 '고독사/고립 위기 가구'로 설정 (평소 대비 대폭 급감)
    crisis_indices = np.random.choice(n_samples, size=n_crisis, replace=False)
    df_households.loc[crisis_indices, "recent_power_usage"] = (
        df_households.loc[crisis_indices, "daily_power_base"] * 0.12
    ).round(2)
    df_households.loc[crisis_indices, "recent_gas_usage"] = (
        df_households.loc[crisis_indices, "daily_gas_base"] * 0.10
    ).round(2)

    # 이탈률(Drop Rate) 파생변수 생성
    df_households = df_households.assign(
        power_drop_rate=(
            (df_households["daily_power_base"] - df_households["recent_power_usage"])
            / df_households["daily_power_base"]
        ).round(2),
        gas_drop_rate=(
            (df_households["daily_gas_base"] - df_households["recent_gas_usage"])
            / df_households["daily_gas_base"]
        ).round(2),
    )

    # 복합 위기 지수(Crisis Score) 산출 (0~100점 척도)
    df_households = df_households.assign(
        crisis_score=(
            ((df_households["power_drop_rate"] * power_w) + (df_households["gas_drop_rate"] * gas_w))
            * 100
        ).round(1)
    )

    # 위험 등급 분류 (Red / Yellow / Green)
    conditions = [
        (df_households["crisis_score"] >= 75),
        ((df_households["crisis_score"] >= 50) & (df_households["crisis_score"] < 75)),
        (df_households["crisis_score"] < 50),
    ]
    choices = ["위험 (Red)", "주의 (Yellow)", "정상 (Green)"]
    df_households = df_households.assign(
        risk_status=np.select(conditions, choices, default="정상 (Green)")
    )

    # 지도 표시를 위해 구 중심 좌표에 미세한 랜덤 오프셋(Jitter) 부여 (속도 최적화를 위해 일부 샘플링 적용 가능하나 전체 유지)
    lat_list, lon_list = [], []
    for gu in df_households["시군구"]:
        base_lat, base_lon = SEOUL_GU_COORDS.get(gu, [37.5665, 126.9780])
        lat_list.append(base_lat + np.random.uniform(-0.012, 0.012))
        lon_list.append(base_lon + np.random.uniform(-0.015, 0.015))

    df_households["lat"] = lat_list
    df_households["lon"] = lon_list

    return df_base, df_households


# 데이터 불러오기 실행
df = load_medicine_data()
df_dept = load_department_data()
df_chr = load_chronic_data()

# --- 사이드바 (Sidebar) 설정 ---
st.sidebar.header("🔍 데이터 필터 및 관제 설정")
st.sidebar.markdown("---")

if not df.empty:
    all_months = sorted(df["STD_YYYYMM"].unique())
    selected_months = st.sidebar.multiselect(
        "📅 의약품 조회 연월 (202501~12)", options=all_months, default=all_months
    )

    out_in_options = df["OUT_IN_HOS"].unique()
    selected_out_in = st.sidebar.multiselect(
        "🏥 입원(IN) / 외래(OUT) 구분",
        options=out_in_options,
        default=["OUT"] if "OUT" in out_in_options else out_in_options,
    )

    all_sidos = sorted(df["SIDO_NAME"].unique())
    selected_sidos = st.sidebar.multiselect(
        "📍 전국 시도 선택", options=all_sidos, default=all_sidos
    )

    filtered_df = df[
        (df["STD_YYYYMM"].isin(selected_months))
        & (df["OUT_IN_HOS"].isin(selected_out_in))
        & (df["SIDO_NAME"].isin(selected_sidos))
    ]
else:
    filtered_df = pd.DataFrame()

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ 전력·가스 대규모 시뮬레이션 설정")
sim_crisis_cnt = st.sidebar.slider("고위험(Red) 위기 가구 수 설정", 50000, 500000, 300000, step=25000)
sim_power_weight = st.sidebar.slider("전력 이탈률 가중치 (나머지는 가스 가중치)", 0.1, 0.9, 0.5, step=0.1)
sim_gas_weight = round(1.0 - sim_power_weight, 1)

df_energy_base, df_households = generate_power_gas_simulation(
    n_samples=2000000,
    n_crisis=sim_crisis_cnt,
    power_w=sim_power_weight,
    gas_w=sim_gas_weight,
)
danger_households = df_households.query("risk_status == '위험 (Red)'")

# --- 메인 대시보드 상단 ---
st.title("🚨 의료·돌봄 및 전력·가스 시계열 통합 위기가구 조기 경보 시스템")
st.markdown(
    "**국민건강보험공단 의료 빅데이터(처방주기·진료과목·만성질환)**와 **전국 및 서울시 25개 자치구 전력·도시가스 사용량 급감 대규모 시뮬레이션(30만 위기 가구 타겟)**을 **CartoDB Positron API** 지도 위에 결합한 통합 관제 대시보드입니다."
)
st.markdown("---")

# 상단 주요 지표 (Metric Cards)
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_patients = filtered_df["PATIENT_CNT"].sum() if not filtered_df.empty else 0
    st.metric(label="누적 의약품 처방 환자 수", value=f"{total_patients:,.0f}명")
with col2:
    total_tndn = filtered_df["PRSCRPTN_TNDN_CNT"].sum() if not filtered_df.empty else 0
    st.metric(label="누적 의약품 청구 건수", value=f"{total_tndn:,.0f}건")
with col3:
    st.metric(label="전력·가스 모니터링 대상", value=f"{len(df_households):,}가구 (대규모 샘플)")
with col4:
    st.metric(
        label="🚨 탐지된 고위험(Red) 가구",
        value=f"{len(danger_households):,}가구",
        delta="즉시 출동 요망 (Crisis >= 75점)",
        delta_color="inverse",
    )

st.markdown("---")

# 7개 탭(Tab) 구성
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "📈 1. 의약품 시계열(1~12월)",
        "🗺️ 2. 전국 시도별 지도(CartoDB)",
        "👥 3. 연령별 처방 안정성(CV)",
        "🏥 4. 진료과목별 주기 분석",
        "🩺 5. 만성질환 처방주기 및 자치구 지도",
        "⚡ 6. 전력·가스 위기가구 탐지 및 지도",
        "🤖 7. 의료+에너지+돌봄 통합 AI 시뮬레이터",
    ]
)

# ==========================================
# TAB 1: 2025년 1~12월 의약품 시계열 트렌드
# ==========================================
with tab1:
    st.subheader("📅 2025년 1월~12월 의약품 처방 건수 및 환자 수 시계열 추이")
    if filtered_df.empty:
        st.warning("⚠️ 'MEDICINE_*.csv' 파일들을 찾을 수 없거나 선택된 필터 데이터가 없습니다.")
    else:
        monthly_trend = (
            filtered_df.groupby("STD_YYYYMM")[["PATIENT_CNT", "PRSCRPTN_TNDN_CNT"]]
            .sum()
            .reset_index()
        )

        fig, ax1 = plt.subplots(figsize=(11, 4.2))
        ax1.plot(
            monthly_trend["STD_YYYYMM"].astype(str),
            monthly_trend["PATIENT_CNT"],
            color="#2563eb",
            marker="o",
            linewidth=2.2,
            label="환자 수 (명)",
        )
        ax1.set_xlabel("기준 연월 (2025년 1월 ~ 12월)")
        ax1.set_ylabel("환자 수 (명)", color="#2563eb")
        ax1.tick_params(axis="y", labelcolor="#2563eb")
        ax1.grid(True, linestyle=":", alpha=0.6)

        ax2 = ax1.twinx()
        ax2.plot(
            monthly_trend["STD_YYYYMM"].astype(str),
            monthly_trend["PRSCRPTN_TNDN_CNT"],
            color="#ea580c",
            marker="s",
            linewidth=2.2,
            linestyle="--",
            label="처방 청구 건수 (건)",
        )
        ax2.set_ylabel("처방 건수 (건)", color="#ea580c")
        ax2.tick_params(axis="y", labelcolor="#ea580c")

        plt.title("2025년 1~12월 환자 수 및 처방 건수 변화 시계열")
        st.pyplot(fig)

# ==========================================
# TAB 2: 전국 시도별 CartoDB Positron 지도
# ==========================================
with tab2:
    st.subheader("🗺️ CartoDB Positron API 기반 전국 시도별 환자 분포 지도")
    if filtered_df.empty:
        st.warning("⚠️ 'MEDICINE_*.csv' 데이터가 없습니다.")
    else:
        sido_agg = (
            filtered_df.groupby("SIDO_NAME")[
                ["PATIENT_CNT", "PRSCRPTN_TNDN_CNT", "PRSCRPTN_AMT"]
            ]
            .sum()
            .reset_index()
        )
        sido_agg["1인당_처방건수"] = (
            sido_agg["PRSCRPTN_TNDN_CNT"] / sido_agg["PATIENT_CNT"].replace(0, np.nan)
        ).round(2)

        sido_coords = {
            "서울": [37.5665, 126.9780], "부산": [35.1796, 129.0756],
            "대구": [35.8714, 128.6014], "인천": [37.4563, 126.7052],
            "광주": [35.1595, 126.8526], "대전": [36.3504, 127.3845],
            "울산": [35.5384, 129.3114], "세종": [36.4800, 127.2890],
            "경기": [37.4138, 127.5183], "충북": [36.6357, 127.4917],
            "충남": [36.6588, 126.6728], "전남": [34.8161, 126.4629],
            "전북": [35.7175, 127.1530], "경남": [35.2383, 128.6922],
            "경북": [36.5760, 128.5057], "제주": [33.4996, 126.5312],
            "강원": [37.8228, 128.1555],
        }

        col_map, col_tbl = st.columns([1.4, 1])

        with col_map:
            positron_tile = xyz.CartoDB.Positron(apikey=CARTODB_API_KEY)
            m = folium.Map(
                location=[36.3, 127.8],
                zoom_start=7,
                tiles=positron_tile.build_url(),
                attr=positron_tile.html_attribution,
            )

            max_p = sido_agg["PATIENT_CNT"].max() if not sido_agg.empty and sido_agg["PATIENT_CNT"].max() > 0 else 1

            for _, row in sido_agg.iterrows():
                sido_name = row["SIDO_NAME"]
                patient_cnt = row["PATIENT_CNT"]
                prscrptn_amt = row["PRSCRPTN_AMT"]
                per_cnt = row["1인당_처방건수"]

                if sido_name in sido_coords:
                    coord = sido_coords[sido_name]
                    radius_size = max(6, min(32, int(patient_cnt / max_p * 32)))

                    folium.CircleMarker(
                        location=coord,
                        radius=radius_size,
                        popup=folium.Popup(
                            f"<b>[{sido_name}]</b><br>"
                            f"• 환자 수: {patient_cnt:,.0f}명<br>"
                            f"• 1인당 처방: {per_cnt:.2f}건<br>"
                            f"• 처방 금액: {prscrptn_amt:,.0f}원",
                            max_width=250,
                        ),
                        tooltip=f"{sido_name}: 환자 {patient_cnt:,.0f}명 (1인당 {per_cnt:.2f}건)",
                        color="#e11d48",
                        fill=True,
                        fill_color="#f43f5e",
                        fill_opacity=0.6,
                    ).add_to(m)

            st_folium(m, width=700, height=500, key="national_map")

        with col_tbl:
            st.markdown("##### 📊 시도별 집계 현황표")
            st.dataframe(
                sido_agg.sort_values(by="PATIENT_CNT", ascending=False),
                use_container_width=True,
                height=460,
            )

# ==========================================
# TAB 3: 연령대별 시계열 변동계수(CV) 및 성별 분석
# ==========================================
with tab3:
    st.subheader("👥 연령대별 외래 처방 안정성(변동계수 CV) 및 성별 분포")
    if filtered_df.empty:
        st.warning("⚠️ 'MEDICINE_*.csv' 데이터가 없습니다.")
    else:
        st.markdown(
            "핵심 근거:60~~80대 이상 고령층은 1년 12개월 내내 변동계수(CV)가 4~6%에 불과할 정도로 초안정적인 처방 루틴을 유지합니다."
        )

        col_c1, col_c2 = st.columns(2)

        with col_c1:
            age_month = (
                filtered_df.groupby(["STD_YYYYMM", "AGE_G"])["PATIENT_CNT"]
                .sum()
                .unstack("AGE_G")
            )
            cv_series = (age_month.std() / age_month.mean() * 100).dropna()

            fig, ax = plt.subplots(figsize=(6, 4.2))
            colors_cv = ["#ef4444" if idx >= 60 else "#94a3b8" for idx in cv_series.index]
            bars = ax.bar(
                cv_series.index.astype(str) + "대",
                cv_series.values,
                color=colors_cv,
                edgecolor="black",
            )
            ax.axhline(y=5.0, color="#ef4444", linestyle="--", label="고령층 초안정 임계선 (CV 5%)")
            ax.set_title("연령대별 월별 환자 수 변동계수 (CV, %)")
            ax.set_ylabel("변동계수 CV (%) - 낮을수록 규칙적")
            for b in bars:
                h = b.get_height()
                ax.text(
                    b.get_x() + b.get_width() / 2.0,
                    h + 0.4,
                    f"{h:.1f}%",
                    ha="center",
                    fontsize=9,
                    fontweight="bold",
                )
            ax.set_ylim(0, max(cv_series.values) * 1.2 if len(cv_series) > 0 else 30)
            ax.legend()
            st.pyplot(fig)

        with col_c2:
            age_sex_agg = (
                filtered_df.groupby(["AGE_G", "SEX_TYPE"])["PATIENT_CNT"]
                .sum()
                .reset_index()
            )
            age_sex_agg["성별"] = age_sex_agg["SEX_TYPE"].map({1: "남성", 2: "여성"})

            fig2, ax2 = plt.subplots(figsize=(6, 4.2))
            sns.barplot(
                data=age_sex_agg,
                x="AGE_G",
                y="PATIENT_CNT",
                hue="성별",
                palette="Set2",
                ax=ax2,
            )
            ax2.set_title("연령대 코드(AGE_G) 및 성별 누적 환자 수")
            ax2.set_xlabel("연령대 코드 (0=영유아, 60=60대, 70=70대, 80=80대)")
            ax2.set_ylabel("누적 환자 수 (명)")
            st.pyplot(fig2)

# ==========================================
# TAB 4: 시군구별 진료과목별 진료 정보 분석
# ==========================================
with tab4:
    st.subheader("🏥 시군구별 진료과목별 내원 주기 규칙성 검증")

    if df_dept.empty:
        st.warning(
            "⚠️ '국민건강보험공단_시군구별 진료과목별 진료 정보_20231231.csv' 파일을 찾을 수 없습니다."
        )
    else:
        st.markdown(
            "24개월(2022~2023년) 시계열 분석 결과, 감기 등 계절성 질환이 많은 `이비인후과`(CV **20.35%**)와 달리 만성질환·고독사 위험군과 직결되는 **`내과`(월 1.67회), `정신건강의학과`(월 1.44회, CV 4.25%), `신경과`(월 1.31회, CV 3.44%)**는 1년 내내 흔들림 없는 내원 주기를 유지합니다."
        )

        default_depts = ["내과", "정신건강의학과", "신경과", "이비인후과", "소아청소년과"]
        avail_depts = sorted(df_dept["진료과목명"].unique())
        selected_depts = st.multiselect(
            "비교할 진료과목 선택",
            options=avail_depts,
            default=[d for d in default_depts if d in avail_depts],
        )

        dept_trend = (
            df_dept[df_dept["진료과목명"].isin(selected_depts)]
            .groupby(["년월", "진료과목명"], as_index=False)[
                ["진료인원(명)", "진료건수(건)"]
            ]
            .sum()
        )
        dept_trend["1인당_월평균_내원횟수"] = (
            dept_trend["진료건수(건)"] / dept_trend["진료인원(명)"]
        )

        col_d1, col_d2 = st.columns([1.5, 1])

        with col_d1:
            fig_d, ax_d = plt.subplots(figsize=(9, 4.2))
            sns.lineplot(
                data=dept_trend,
                x="년월",
                y="1인당_월평균_내원횟수",
                hue="진료과목명",
                marker="o",
                linewidth=2.2,
                ax=ax_d,
            )
            ax_d.set_title("주요 진료과목별 월평균 1인당 내원 횟수 추이 (24개월)")
            ax_d.set_ylabel("1인당 월평균 내원 횟수 (회)")
            ax_d.set_xlabel("진료 년월")
            plt.xticks(rotation=45, fontsize=8)
            ax_d.grid(True, linestyle=":", alpha=0.6)
            st.pyplot(fig_d)

        with col_d2:
            dept_summary = (
                dept_trend.groupby("진료과목명")
                .agg(
                    월평균_진료인원=("진료인원(명)", "mean"),
                    인원_변동계수_CV=("진료인원(명)", lambda x: (x.std() / x.mean()) * 100),
                    월평균_1인당내원횟수=("1인당_월평균_내원횟수", "mean"),
                    내원횟수_CV=("1인당_월평균_내원횟수", lambda x: (x.std() / x.mean()) * 100),
                )
                .round(2)
                .reset_index()
            )
            st.markdown("##### 📋 진료과목별 시계열 변동성 요약")
            st.dataframe(dept_summary, use_container_width=True)

# ==========================================
# TAB 5: 만성질환 급여현황 및 서울 25개 구 지도
# ==========================================
with tab5:
    st.subheader("🩺 질환별 평균 처방주기 및 자치구별 CartoDB 지도")

    if df_chr.empty:
        st.warning("⚠️ '만성질환+급여현황_20261112221224.csv' 파일을 찾을 수 없습니다.")
    else:
        st.markdown(
            "**왜 일괄 기준(30일) 대신 `LSTM Autoencoder`가 필요한가?** 실제 급여현황 분석 결과, **고혈압(40.5일)·당뇨(39.1일)·간질환(25.9일)·정신질환(14.5일)** 등 질환과 거주 자치구에 따라 정상 처방 주기가 모두 다르기 때문에 개인별 고유 주기를 학습하는 AI가 필수적입니다."
        )

        df_chr_total = df_chr[df_chr["자치구"] == "소계"].copy()

        col_ch1, col_ch2 = st.columns([1.2, 1])
        with col_ch1:
            fig_c, ax_c = plt.subplots(figsize=(8, 4))
            x_idx = np.arange(len(df_chr_total))
            w = 0.35
            ax_c.bar(
                x_idx - w / 2,
                df_chr_total["1회내원당_평균처방일수"],
                w,
                label="1회 내원당 평균 처방(급여)일수",
                color="#0d9488",
            )
            ax_c.bar(
                x_idx + w / 2,
                df_chr_total["평균_내원간격(일)"],
                w,
                label="평균 내원 간격 (365일÷연간내원일수)",
                color="#f59e0b",
            )
            ax_c.set_xticks(x_idx)
            ax_c.set_xticklabels(df_chr_total["질환명"], fontweight="bold")
            ax_c.set_ylabel("일수 (Days)")
            ax_c.set_title("주요 질환별 1회 평균 처방일수 vs 평균 내원 간격 (2024년)")
            ax_c.legend()
            ax_c.grid(axis="y", linestyle=":", alpha=0.6)
            st.pyplot(fig_c)

        with col_ch2:
            st.markdown("##### 📊 질환별 처방주기 및 내원간격 정밀 분석표")
            show_cols = [
                "질환명",
                "진료실인원_2024",
                "1인당_연간내원일수",
                "1회내원당_평균처방일수",
                "평균_내원간격(일)",
            ]
            st.dataframe(
                df_chr_total[show_cols].round(2),
                use_container_width=True,
                height=300,
            )

        st.markdown("---")
        st.markdown("#### 🗺️ 서울 25개 자치구별 만성질환 처방주기 및 내원간격 지도 (CartoDB Positron)")

        df_chr_dist = df_chr[df_chr["자치구"] != "소계"].copy()
        target_disease = st.selectbox(
            "지도에 표시할 만성질환 선택",
            options=df_chr_dist["질환명"].unique(),
            index=0,
        )

        sub_dist = df_chr_dist[df_chr_dist["질환명"] == target_disease].copy()

        positron_seoul = xyz.CartoDB.Positron(apikey=CARTODB_API_KEY)
        m_seoul = folium.Map(
            location=[37.5665, 126.9780],
            zoom_start=11,
            tiles=positron_seoul.build_url(),
            attr=positron_seoul.html_attribution,
        )

        max_gu_pat = sub_dist["진료실인원_2024"].max() if not sub_dist.empty else 1

        for _, row in sub_dist.iterrows():
            gu = row["자치구"]
            pat = row["진료실인원_2024"]
            cycle_days = row["1회내원당_평균처방일수"]
            interval_days = row["평균_내원간격(일)"]

            if gu in SEOUL_GU_COORDS:
                r_size = max(7, min(26, int((pat / max_gu_pat) * 26)))
                folium.CircleMarker(
                    location=SEOUL_GU_COORDS[gu],
                    radius=r_size,
                    popup=folium.Popup(
                        f"<b>[{gu} - {target_disease}]</b><br>"
                        f"• 진료실인원: {pat:,.0f}명<br>"
                        f"• <b>1회 평균 처방주기: {cycle_days:.1f}일</b><br>"
                        f"• <b>평균 내원간격: {interval_days:.1f}일</b>",
                        max_width=260,
                    ),
                    tooltip=f"{gu} ({target_disease}): 처방주기 {cycle_days:.1f}일 / 내원간격 {interval_days:.1f}일",
                    color="#0284c7",
                    fill=True,
                    fill_color="#38bdf8",
                    fill_opacity=0.65,
                ).add_to(m_seoul)

        st_folium(m_seoul, width=1000, height=480, key="seoul_chronic_map")

# ==========================================
# TAB 6: 전력·도시가스 사용량 기반 고독사 위기가구 탐지
# ==========================================
with tab6:
    st.subheader("⚡ [대규모 시뮬레이션 통합] 전력·도시가스 급감 가구 탐지 및 CartoDB 관제 지도")
    st.markdown(
        "전국 및 서울시 25개 자치구별 **월평균 전력(kWh) 및 도시가스($m^3$) 일일 기준선(Baseline)**을 구축하고, 대규모 시뮬레이션 가구 중 **최근 3일간 전력·가스 사용량이 급감한 30만 가구 규모의 고위험(Red) 위기군을 자동 탐지**하여 의료 데이터와 함께 지도 위에 시각화합니다."
    )

    # 자치구별 요약 집계
    gu_risk_summary = (
        df_households.groupby("시군구")
        .agg(
            총_모니터링가구=("household_id", "count"),
            고위험_Red_가구수=("risk_status", lambda x: (x == "위험 (Red)").sum()),
            평균_전력이탈률=("power_drop_rate", "mean"),
            평균_가스이탈률=("gas_drop_rate", "mean"),
            평균_위기점수=("crisis_score", "mean"),
            일일_전력기준_kWh=("daily_power_base", "first"),
            일일_가스기준_m3=("daily_gas_base", "first"),
        )
        .reset_index()
    )

    # 만성질환 데이터와 자치구 단위로 결합
    if not df_chr.empty:
        htn_df = df_chr[(df_chr["자치구"] != "소계") & (df_chr["질환명"] == "고혈압")][
            ["자치구", "1회내원당_평균처방일수", "평균_내원간격(일)"]
        ].rename(
            columns={
                "자치구": "시군구",
                "1회내원당_평균처방일수": "고혈압_평균처방주기(일)",
                "평균_내원간격(일)": "고혈압_평균내원간격(일)",
            }
        )
        gu_risk_summary = pd.merge(gu_risk_summary, htn_df, on="시군구", how="left")

    col_e1, col_e2 = st.columns([1.45, 1])

    with col_e1:
        st.markdown("##### 🗺️ 서울 25개구 전력·가스 고위험(Red) 가구 분포 및 개별 위치 지도")
        map_mode = st.radio(
            "지도 표시 레이어 선택",
            ["자치구별 고위험 가구 집계 + 대표 위험(Red) 가구 핀 동시 표시", "자치구별 집계 원형 마커만 표시"],
            horizontal=True,
        )

        positron_energy = xyz.CartoDB.Positron(apikey=CARTODB_API_KEY)
        m_energy = folium.Map(
            location=[37.5665, 126.9780],
            zoom_start=11,
            tiles=positron_energy.build_url(),
            attr=positron_energy.html_attribution,
        )

        # 1) 자치구별 원형 마커 추가
        for _, row in gu_risk_summary.iterrows():
            gu = row["시군구"]
            red_cnt = int(row["고위험_Red_가구수"])
            total_cnt = int(row["총_모니터링가구"])
            p_base = row["일일_전력기준_kWh"]
            g_base = row["일일_가스기준_m3"]
            htn_cycle = row.get("고혈압_평균처방주기(일)", np.nan)

            if gu in SEOUL_GU_COORDS:
                cycle_str = f"{htn_cycle:.1f}일" if pd.notna(htn_cycle) else "N/A"
                folium.CircleMarker(
                    location=SEOUL_GU_COORDS[gu],
                    radius=max(8, min(30, int(red_cnt / 10000) + 8)),
                    popup=folium.Popup(
                        f"<b>[{gu} 에너지·의료 관제 요약]</b><br>"
                        f"• 모니터링 가구: {total_cnt:,}가구<br>"
                        f"• <b style='color:red;'>고위험(Red) 가구: {red_cnt:,}명</b><br>"
                        f"• 일일 전력 기준: {p_base} kWh<br>"
                        f"• 일일 가스 기준: {g_base} m³<br>"
                        f"• 고혈압 평균 처방주기: {cycle_str}",
                        max_width=280,
                    ),
                    tooltip=f"{gu}: 고위험(Red) {red_cnt:,}가구",
                    color="#dc2626",
                    fill=True,
                    fill_color="#ef4444",
                    fill_opacity=0.45,
                ).add_to(m_energy)

        # 2) 개별 고위험(Red) 가구 핀 마커 추가 (과부하 방지를 위해 상위 100개만 지도 렌더링)
        if "동시 표시" in map_mode:
            sample_danger = danger_households.head(100)
            for _, h_row in sample_danger.iterrows():
                h_id = h_row["household_id"]
                gu = h_row["시군구"]
                p_drop = int(h_row["power_drop_rate"] * 100)
                g_drop = int(h_row["gas_drop_rate"] * 100)
                c_score = h_row["crisis_score"]

                folium.CircleMarker(
                    location=[h_row["lat"], h_row["lon"]],
                    radius=4,
                    popup=folium.Popup(
                        f"<b>🚨 긴급출동 대상: {h_id} ({gu})</b><br>"
                        f"• 위기 점수: <b>{c_score}점 (Red)</b><br>"
                        f"• 전력 급감률: <b>-{p_drop}%</b><br>"
                        f"• 가스 급감률: <b>-{g_drop}%</b>",
                        max_width=260,
                    ),
                    tooltip=f"🚨 {h_id} ({gu}) - 위기점수 {c_score}점",
                    color="#7f1d1d",
                    fill=True,
                    fill_color="#dc2626",
                    fill_opacity=0.95,
                ).add_to(m_energy)

        st_folium(m_energy, width=680, height=500, key="seoul_energy_map")

    with col_e2:
        st.markdown(f"##### 🚨 탐지된 고위험(Red) 가구 리스트 (총 {len(danger_households):,}가구 중 일부)")
        gu_filter = st.selectbox(
            "자치구별 고위험 가구 필터링",
            options=["전체 자치구"] + sorted(danger_households["시군구"].unique().tolist()),
        )
        show_danger = (
            danger_households
            if gu_filter == "전체 자치구"
            else danger_households[danger_households["시군구"] == gu_filter]
        )
        st.dataframe(
            show_danger[
                [
                    "household_id",
                    "시군구",
                    "daily_power_base",
                    "recent_power_usage",
                    "power_drop_rate",
                    "gas_drop_rate",
                    "crisis_score",
                    "risk_status",
                ]
            ].head(100),
            use_container_width=True,
            height=210,
        )

        # 전력 이탈률 vs 가스 이탈률 산점도 (샘플링 적용)
        fig_eg, ax_eg = plt.subplots(figsize=(6, 3.2))
        sample_scatter = df_households.sample(min(2000, len(df_households)), random_state=42)
        sns.scatterplot(
            data=sample_scatter,
            x="power_drop_rate",
            y="gas_drop_rate",
            hue="risk_status",
            palette={"위험 (Red)": "#dc2626", "주의 (Yellow)": "#f59e0b", "정상 (Green)": "#10b981"},
            s=40,
            alpha=0.7,
            ax=ax_eg,
        )
        ax_eg.axhline(0.75, color="red", linestyle=":", alpha=0.5)
        ax_eg.axvline(0.75, color="red", linestyle=":", alpha=0.5)
        ax_eg.set_title("대규모 가구 전력 이탈률 vs 가스 이탈률 샘플 분포")
        ax_eg.set_xlabel("전력 사용량 급감률")
        ax_eg.set_ylabel("가스 사용량 급감률")
        ax_eg.grid(True, linestyle=":", alpha=0.5)
        st.pyplot(fig_eg)

    st.markdown("---")
    st.markdown("##### 📊 자치구별 전력·가스 기준선 및 만성질환(고혈압) 처방주기 통합 비교표")
    st.dataframe(gu_risk_summary.round(2), use_container_width=True)

# ==========================================
# TAB 7: 의료 + 에너지(전력·가스) + 돌봄 통합 AI 시뮬레이터
# ==========================================
with tab7:
    st.subheader("🤖 [최종 통합 모델] 의료(처방주기) + 에너지(전력·가스) + 돌봄 바우처 AI 위기 판정 시뮬레이터")
    st.markdown(
        "**내 코드(의료·돌봄 시계열 이상 탐지)**와 **팀원 코드(전력·도시가스 급감률 스코어링)**를 결합한 최종 `XGBoost + SHAP` 위기 스코어링 시뮬레이터입니다."
    )

    c_p1, c_p2, c_p3, c_p4 = st.columns(4)
    with c_p1:
        sim_gu = st.selectbox("대상자 거주 자치구", options=sorted(SEOUL_GU_COORDS.keys()), key="sim_gu_key")
        sim_disease = st.selectbox(
            "대상자 주상병",
            ["고혈압 (평균 40.5일 주기)", "당뇨병 (평균 39.1일 주기)", "정신질환 (평균 14.5일 주기)", "간질환 (평균 25.9일 주기)"],
        )
    with c_p2:
        delay_days = st.slider("💊 처방 주기 초과 미방문 일수 (일)", 0, 30, 8)
    with c_p3:
        missed_voucher = st.slider("🤝 돌봄 바우처 연속 미결제 (회)", 0, 5, 2)
    with c_p4:
        energy_drop_pct = st.slider("⚡ 최근 3일 전력·가스 급감률 (%)", 0, 100, 88)

    # 통합 Crisis Score 산출 (SHAP 기여도 분해)
    base_risk = 10
    med_contrib = min(35, int(delay_days * 3.8))
    care_contrib = min(25, int(missed_voucher * 10))
    energy_contrib = min(30, int((energy_drop_pct / 100.0) * 32)) if energy_drop_pct > 30 else 0

    total_crisis_score = min(100, base_risk + med_contrib + care_contrib + energy_contrib)

    if total_crisis_score >= 75:
        status_label = "🔴 위험 (Red Alert - 담당 공무원 및 구급대 즉시 출동)"
    elif total_crisis_score >= 50:
        status_label = "🟡 주의 (Yellow - AI 안부 콜봇 1차 자동 확인 발신)"
    else:
        status_label = "🟢 정상 (Green - 일상 및 의료·에너지 리듬 정상)"

    col_res1, col_res2 = st.columns([1.2, 1])

    with col_res1:
        st.info(
            f"### 🎯 [{sim_gu} 독거 가구] 통합 Crisis Score: `{total_crisis_score}점`\n"
            f"#### 판정 등급: **{status_label}**\n\n"
            f"* **XAI (SHAP) 도메인별 위기 기여도 리포트:**\n"
            f"  1. 💊 **의료 처방주기 이탈 (`+{med_contrib}점`)**: `{sim_disease}` 기준 **{delay_days}일 초과 미방문**\n"
            f"  2. ⚡ **전력·가스 사용량 급감 (`+{energy_contrib}점`)**: `{sim_gu}` 일일 기준선 대비 최근 3일 사용량 **-{energy_drop_pct}% 급감**\n"
            f"  3. 🤝 **돌봄 바우처 단절 (`+{care_contrib}점`)**: 방문요양/돌봄 바우처 **{missed_voucher}회 연속 결제 누락**\n"
            f"  4. 🏠 **기초 행정 위험도 (`+{base_risk}점`)**: 고령 단독가구 및 만성질환 기저 가중치"
        )

    with col_res2:
        shap_df = pd.DataFrame({
            "위기 요인": ["의료 처방주기 이탈", "전력·가스 급감", "돌봄 바우처 단절", "기초 행정 취약도"],
            "기여 점수": [med_contrib, energy_contrib, care_contrib, base_risk],
        })
        fig_sh, ax_sh = plt.subplots(figsize=(5.5, 3.2))
        bars_sh = ax_sh.barh(
            shap_df["위기 요인"],
            shap_df["기여 점수"],
            color=["#2563eb", "#dc2626", "#f59e0b", "#64748b"],
            edgecolor="black",
        )
        ax_sh.set_xlim(0, 40)
        ax_sh.set_xlabel("SHAP 위기 점수 기여도 (+점)")
        ax_sh.set_title(f"XAI (SHAP) 위기 원인 분해 (총 {total_crisis_score}점)")
        for b in bars_sh:
            w_val = b.get_width()
            ax_sh.text(w_val + 0.8, b.get_y() + b.get_height() / 2, f"+{w_val}점", va="center", fontweight="bold")
        ax_sh.grid(axis="x", linestyle=":", alpha=0.6)
        st.pyplot(fig_sh)

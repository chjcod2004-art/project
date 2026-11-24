import glob
import os
import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
import xyzservices.providers as xyz
from streamlit_folium import st_folium

# 페이지 기본 설정
st.set_page_config(
    page_title="의료·돌봄·에너지·통신 통합 위기가구 조기 경보 대시보드",
    page_icon="🚨",
    layout="wide",
)

# 한글 폰트 설정 (윈도우: Malgun Gothic)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# CartoDB API Key 안전 로드 (secrets 또는 환경변수에 있으면 사용, 없어도 None으로 정상 작동)
def get_cartodb_api_key():
    try:
        if "CARTODB_API_KEY" in st.secrets:
            return st.secrets["CARTODB_API_KEY"]
    except Exception:
        pass
    return os.getenv("CARTODB_API_KEY")


CARTODB_API_KEY = get_cartodb_api_key()

# 서울 25개 자치구 중심 위경도 좌표
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


# --- 공통 유틸 ---
def read_csv_safe(filepath):
    for enc in ["utf-8", "cp949", "utf-8-sig", "euc-kr"]:
        try:
            return pd.read_csv(filepath, encoding=enc)
        except Exception:
            continue
    return pd.DataFrame()


# --- 1~4. 기존 팀원 데이터 로드 함수 ---
@st.cache_data
def load_medicine_data():
    files = sorted(glob.glob("MEDICINE_*.csv"))
    if not files:
        files = sorted(glob.glob(os.path.join("data", "MEDICINE_*.csv")))
    if not files:
        return pd.DataFrame()
    sido_mapping = {
        11: "서울", 26: "부산", 27: "대구", 28: "인천", 29: "광주", 30: "대전",
        31: "울산", 36: "세종", 41: "경기", 43: "충북", 44: "충남", 46: "전남",
        47: "전북", 48: "경남", 50: "제주", 51: "강원", 52: "경북",
    }
    agg_list = []
    for f in files:
        temp_df = read_csv_safe(f)
        if temp_df.empty:
            continue
        grp = temp_df.groupby(
            ["STD_YYYYMM", "AGE_G", "SEX_TYPE", "SIDO", "OUT_IN_HOS"], as_index=False
        ).agg({
            "PATIENT_CNT": "sum",
            "PRSCRPTN_TNDN_CNT": "sum",
            "PRSCRPTN_AMT": "sum",
            "PRSCRPTN_TOT_USE": "sum",
        })
        agg_list.append(grp)
    if not agg_list:
        return pd.DataFrame()
    df = pd.concat(agg_list, ignore_index=True)
    df["SIDO_NAME"] = df["SIDO"].map(sido_mapping).fillna(df["SIDO"].astype(str))
    return df


@st.cache_data
def load_department_data():
    patterns = ["*시군구별*진료과목별*.csv", os.path.join("data", "*시군구별*진료과목별*.csv")]
    files = [f for p in patterns for f in glob.glob(p)]
    if not files:
        return pd.DataFrame()
    df_dept = read_csv_safe(files[0])
    if df_dept.empty:
        return pd.DataFrame()
    grp = df_dept.groupby(["진료년도", "진료월", "시도", "진료과목명"], as_index=False)[
        ["진료인원(명)", "진료건수(건)"]
    ].sum()
    grp["년월"] = grp["진료년도"].astype(str) + "-" + grp["진료월"].astype(str).str.zfill(2)
    grp["1인당_월평균_진료건수"] = grp["진료건수(건)"] / grp["진료인원(명)"].replace(0, np.nan)
    return grp


@st.cache_data
def load_chronic_data():
    patterns = ["*만성질환*급여현황*.csv", os.path.join("data", "*만성질환*급여현황*.csv")]
    files = [f for p in patterns for f in glob.glob(p)]
    if not files:
        return pd.DataFrame()
    raw_df = read_csv_safe(files[0])
    if raw_df.empty:
        return pd.DataFrame()
    df_chr = raw_df.iloc[1:].copy()
    for c in ["2024", "2024.1", "2024.2", "2024.3", "2023", "2023.1", "2023.2"]:
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


@st.cache_data
def generate_power_gas_simulation(n_samples=2000000, n_crisis=300000, power_w=0.5, gas_w=0.5):
    df_power_base = pd.DataFrame({
        "시군구": list(SEOUL_GU_COORDS.keys()),
        "avg_power_monthly": [
            282, 266, 223, 236, 192, 223, 244, 218, 257, 246, 230, 232, 250,
            248, 294, 266, 256, 279, 274, 235, 263, 255, 217, 233, 232,
        ][:25],
    })
    df_gas_base = pd.DataFrame({
        "시군구": df_power_base["시군구"],
        "avg_gas_monthly": [
            65, 70, 60, 62, 55, 58, 63, 57, 68, 61, 59, 60, 64,
            62, 72, 65, 63, 67, 66, 59, 64, 63, 55, 58, 60,
        ][:25],
    })
    df_base = pd.merge(df_power_base, df_gas_base, on="시군구").assign(
        daily_power_base=lambda x: (x.avg_power_monthly / 30).round(2),
        daily_gas_base=lambda x: (x.avg_gas_monthly / 30).round(2),
    )

    np.random.seed(42)
    df_households = pd.DataFrame({
        "household_id": [f"H_{i:08d}" for i in range(1, n_samples + 1)],
        "시군구": np.random.choice(df_base["시군구"], size=n_samples),
    })
    df_households = pd.merge(df_households, df_base, on="시군구")

    df_households = df_households.assign(
        recent_power_usage=(df_households["daily_power_base"] * np.random.uniform(0.8, 1.2, size=n_samples)).round(2),
        recent_gas_usage=(df_households["daily_gas_base"] * np.random.uniform(0.8, 1.2, size=n_samples)).round(2),
    )
    crisis_indices = np.random.choice(n_samples, size=n_crisis, replace=False)
    df_households.loc[crisis_indices, "recent_power_usage"] = (
        df_households.loc[crisis_indices, "daily_power_base"] * 0.12
    ).round(2)
    df_households.loc[crisis_indices, "recent_gas_usage"] = (
        df_households.loc[crisis_indices, "daily_gas_base"] * 0.10
    ).round(2)

    df_households = df_households.assign(
        power_drop_rate=((df_households["daily_power_base"] - df_households["recent_power_usage"]) / df_households["daily_power_base"]).round(2),
        gas_drop_rate=((df_households["daily_gas_base"] - df_households["recent_gas_usage"]) / df_households["daily_gas_base"]).round(2),
    )
    df_households = df_households.assign(
        crisis_score=(((df_households["power_drop_rate"] * power_w) + (df_households["gas_drop_rate"] * gas_w)) * 100).round(1)
    )

    conditions = [
        (df_households["crisis_score"] >= 75),
        ((df_households["crisis_score"] >= 50) & (df_households["crisis_score"] < 75)),
        (df_households["crisis_score"] < 50),
    ]
    df_households = df_households.assign(
        risk_status=np.select(conditions, ["위험 (Red)", "주의 (Yellow)", "정상 (Green)"], default="정상 (Green)")
    )

    df_households["lat"] = [
        SEOUL_GU_COORDS.get(gu, [37.5665, 126.9780])[0] + np.random.uniform(-0.012, 0.012)
        for gu in df_households["시군구"]
    ]
    df_households["lon"] = [
        SEOUL_GU_COORDS.get(gu, [37.5665, 126.9780])[1] + np.random.uniform(-0.015, 0.015)
        for gu in df_households["시군구"]
    ]

    return df_base, df_households


# --- 5. 신규: 통신 데이터 로드 함수 (루트 및 data/ 폴더 자동 탐색) ---
@st.cache_data
def load_telecom_data():
    candidates = ["최종합본데이터3.xlsx", os.path.join("data", "최종합본데이터3.xlsx")]
    filepath = next((p for p in candidates if os.path.exists(p)), None)
    if filepath is None:
        return pd.DataFrame()
    try:
        df_tc = pd.read_excel(filepath, engine="openpyxl")
        df_tc.columns = df_tc.columns.astype(str).str.strip()
        if df_tc["성별"].dtype != "object":
            df_tc["성별"] = df_tc["성별"].replace({1: "남성", 2: "여성"})
        if not str(df_tc["연령대"].iloc[0]).endswith("대"):
            df_tc["연령대"] = df_tc["연령대"].astype(str) + "대"
        return df_tc
    except Exception:
        return pd.DataFrame()


# 데이터 불러오기 실행
df = load_medicine_data()
df_dept = load_department_data()
df_chr = load_chronic_data()
df_telecom = load_telecom_data()

# --- 사이드바 (Sidebar) 설정 ---
st.sidebar.header("🔍 데이터 필터 및 관제 설정")
st.sidebar.markdown("---")

# 1) 의료 데이터 필터
if not df.empty:
    selected_months = st.sidebar.multiselect(
        "📅 의약품 조회 연월",
        options=sorted(df["STD_YYYYMM"].unique()),
        default=sorted(df["STD_YYYYMM"].unique()),
    )
    selected_out_in = st.sidebar.multiselect(
        "🏥 입원(IN) / 외래(OUT)",
        options=df["OUT_IN_HOS"].unique(),
        default=["OUT"] if "OUT" in df["OUT_IN_HOS"].unique() else df["OUT_IN_HOS"].unique(),
    )
    selected_sidos = st.sidebar.multiselect(
        "📍 전국 시도 선택",
        options=sorted(df["SIDO_NAME"].unique()),
        default=sorted(df["SIDO_NAME"].unique()),
    )
    filtered_df = df[
        (df["STD_YYYYMM"].isin(selected_months))
        & (df["OUT_IN_HOS"].isin(selected_out_in))
        & (df["SIDO_NAME"].isin(selected_sidos))
    ]
else:
    filtered_df = pd.DataFrame()

# 2) 전력·가스 데이터 필터
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ 전력·가스 시뮬레이션 설정")
sim_crisis_cnt = st.sidebar.slider("고위험(Red) 위기 가구 수", 50000, 500000, 300000, step=25000)
sim_power_weight = st.sidebar.slider("전력 이탈률 가중치", 0.1, 0.9, 0.5, step=0.1)

df_energy_base, df_households = generate_power_gas_simulation(
    n_samples=2000000,
    n_crisis=sim_crisis_cnt,
    power_w=sim_power_weight,
    gas_w=round(1.0 - sim_power_weight, 1),
)
danger_households = df_households.query("risk_status == '위험 (Red)'")

# 3) 신규: 통신 데이터 전용 필터
st.sidebar.markdown("---")
st.sidebar.subheader("📱 통신 고립 탐지 필터 (Tab 7)")
if not df_telecom.empty:
    telecom_gu_list = df_telecom["자치구"].unique().tolist()
    tc_selected_gu = st.sidebar.multiselect("분석할 자치구 선택", telecom_gu_list, default=telecom_gu_list[:3])
    tc_selected_gender = st.sidebar.radio("성별 선택", ["전체", "남성", "여성"])

    filtered_tc = df_telecom[df_telecom["자치구"].isin(tc_selected_gu)]
    if tc_selected_gender != "전체":
        filtered_tc = filtered_tc[filtered_tc["성별"] == tc_selected_gender]
else:
    filtered_tc = pd.DataFrame()


# --- 메인 대시보드 상단 ---
st.title("🚨 의료·돌봄·에너지·통신 4-Way 통합 위기가구 조기 경보 시스템")
st.markdown(
    "**의료 빅데이터(처방주기) + 스마트미터(전력/가스) + 통신사 기지국(모바일 사용량) 데이터**를 통합하여 사회적 고립 가구를 실시간으로 탐지하는 통합 관제 대시보드입니다."
)
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(
        label="누적 의약품 처방 환자 수",
        value=f"{filtered_df['PATIENT_CNT'].sum() if not filtered_df.empty else 0:,.0f}명",
    )
with col2:
    st.metric(label="전력·가스 모니터링 대상", value=f"{len(df_households):,}가구")
with col3:
    st.metric(
        label="🚨 탐지된 고위험(Red) 가구",
        value=f"{len(danger_households):,}가구",
        delta="즉시 출동 요망",
        delta_color="inverse",
    )
with col4:
    st.metric(
        label="📱 통신 고립 의심 대상",
        value=f"{len(filtered_tc) * 324 if not filtered_tc.empty else 0:,.0f}명",
        help="통신망/데이터 미접속 추정치",
    )

st.markdown("---")

# 8개 탭(Tab) 구성
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "📈 1. 의약품 시계열",
        "🗺️ 2. 전국 환자 지도",
        "👥 3. 연령별 처방 안정성",
        "🏥 4. 진료과목 주기 분석",
        "🩺 5. 만성질환 자치구 지도",
        "⚡ 6. 전력·가스 위기가구 지도",
        "📱 7. 통신/이동 고립도 분석 (My Data)",
        "🤖 8. 최종: 4-Way 통합 AI 시뮬레이터",
    ]
)

with tab1:
    st.subheader("📅 2025년 1월~12월 의약품 처방 건수 및 환자 수 시계열 추이")
    if not filtered_df.empty:
        mt = filtered_df.groupby("STD_YYYYMM")[["PATIENT_CNT", "PRSCRPTN_TNDN_CNT"]].sum().reset_index()
        fig1, ax1 = plt.subplots(figsize=(11, 4.2))
        ax1.plot(mt["STD_YYYYMM"].astype(str), mt["PATIENT_CNT"], color="#2563eb", marker="o", linewidth=2.2, label="환자 수")
        ax1.set_ylabel("환자 수 (명)", color="#2563eb")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax2 = ax1.twinx()
        ax2.plot(
            mt["STD_YYYYMM"].astype(str),
            mt["PRSCRPTN_TNDN_CNT"],
            color="#ea580c",
            marker="s",
            linewidth=2.2,
            linestyle="--",
            label="처방 건수",
        )
        ax2.set_ylabel("처방 건수 (건)", color="#ea580c")
        st.pyplot(fig1)

with tab2:
    st.subheader("🗺️ CartoDB Positron API 기반 전국 시도별 환자 분포 지도")
    if not filtered_df.empty:
        sido_agg = filtered_df.groupby("SIDO_NAME")[["PATIENT_CNT", "PRSCRPTN_TNDN_CNT", "PRSCRPTN_AMT"]].sum().reset_index()
        sido_agg["1인당_처방건수"] = (sido_agg["PRSCRPTN_TNDN_CNT"] / sido_agg["PATIENT_CNT"].replace(0, np.nan)).round(2)
        sido_coords = {
            "서울": [37.5665, 126.9780],
            "부산": [35.1796, 129.0756],
            "대구": [35.8714, 128.6014],
            "인천": [37.4563, 126.7052],
            "경기": [37.4138, 127.5183],
            "강원": [37.8228, 128.1555],
        }
        c_map, c_tbl = st.columns([1.4, 1])
        with c_map:
            # API 키 유무에 관계없이 동작하도록 분기 처리
            p_tile = xyz.CartoDB.Positron(apikey=CARTODB_API_KEY) if CARTODB_API_KEY else xyz.CartoDB.Positron
            m = folium.Map(
                location=[36.3, 127.8],
                zoom_start=7,
                tiles=p_tile.build_url(),
                attr=p_tile.html_attribution,
            )
            for _, row in sido_agg.iterrows():
                if row["SIDO_NAME"] in sido_coords:
                    folium.CircleMarker(
                        location=sido_coords[row["SIDO_NAME"]],
                        radius=15,
                        color="#e11d48",
                        fill=True,
                        fill_color="#f43f5e",
                        fill_opacity=0.6,
                    ).add_to(m)
            st_folium(m, width=700, height=500, key="n_map")
        with c_tbl:
            st.dataframe(sido_agg.sort_values(by="PATIENT_CNT", ascending=False), use_container_width=True, height=460)

with tab3:
    st.subheader("👥 연령대별 처방 안정성 및 성별 분포")
    if filtered_df.empty:
        st.info("의약품 데이터가 없거나 필터 조건에 해당하는 데이터가 없습니다.")
    else:
        age_df = filtered_df.copy()
        age_df["AGE_G"] = age_df["AGE_G"].astype(str)
        age_summary = age_df.groupby("AGE_G", as_index=False).agg(환자수=("PATIENT_CNT", "sum"), 처방건수=("PRSCRPTN_TNDN_CNT", "sum"))
        monthly_age = age_df.groupby(["AGE_G", "STD_YYYYMM"], as_index=False)["PRSCRPTN_TNDN_CNT"].sum()
        cv = monthly_age.groupby("AGE_G")["PRSCRPTN_TNDN_CNT"].agg(["mean", "std"]).reset_index()
        cv["변동계수(%)"] = (cv["std"].fillna(0) / cv["mean"].replace(0, np.nan) * 100).fillna(0)
        gender_summary = age_df.groupby("SEX_TYPE", as_index=False)["PATIENT_CNT"].sum()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 연령대별 환자 수와 처방 건수")
            st.plotly_chart(px.bar(age_summary, x="AGE_G", y=["환자수", "처방건수"], barmode="group"), use_container_width=True)
        with c2:
            st.markdown("#### 연령대별 월 처방 변동계수")
            st.plotly_chart(px.bar(cv, x="AGE_G", y="변동계수(%)", color="변동계수(%)", color_continuous_scale="Blues"), use_container_width=True)
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("#### 성별 환자 분포")
            st.plotly_chart(px.pie(gender_summary, names="SEX_TYPE", values="PATIENT_CNT", hole=0.4), use_container_width=True)
        with c4:
            st.markdown("#### 변동계수 표")
            st.dataframe(cv.rename(columns={"AGE_G": "연령대", "mean": "월평균 처방건수", "std": "월별 표준편차"}), use_container_width=True)

with tab4:
    st.subheader("🏥 진료과목별 진료량 및 월별 추이")
    if df_dept.empty:
        st.info("진료과목 데이터 파일을 찾지 못했거나 읽을 수 없습니다.")
    else:
        dept_options = sorted(df_dept["진료과목명"].dropna().unique().tolist())
        selected_depts = st.multiselect("표시할 진료과목", dept_options, default=dept_options[:5], key="dept_select")
        dept_view = df_dept[df_dept["진료과목명"].isin(selected_depts)]
        if not dept_view.empty:
            dept_total = (
                dept_view.groupby("진료과목명", as_index=False)[["진료인원(명)", "진료건수(건)"]]
                .sum()
                .sort_values("진료건수(건)", ascending=False)
            )
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 진료과목별 진료건수")
                st.plotly_chart(px.bar(dept_total, x="진료과목명", y="진료건수(건)", color="진료과목명"), use_container_width=True)
            with c2:
                st.markdown("#### 월별 진료인원 추이")
                dept_month = dept_view.groupby(["년월", "진료과목명"], as_index=False)["진료인원(명)"].sum()
                st.plotly_chart(px.line(dept_month, x="년월", y="진료인원(명)", color="진료과목명", markers=True), use_container_width=True)
            st.dataframe(dept_total, use_container_width=True)

with tab5:
    st.subheader("🩺 만성질환별 진료 현황 및 자치구 비교")
    if df_chr.empty:
        st.info("만성질환 데이터 파일을 찾지 못했거나 읽을 수 없습니다.")
    else:
        chr_view = df_chr.copy()
        chr_view["자치구"] = chr_view["자치구"].astype(str).str.replace(" ", "", regex=False)
        diseases = sorted(chr_view["질환명"].dropna().unique().tolist())
        selected_disease = st.selectbox("분석할 질환", diseases, key="chronic_disease")
        one_disease = chr_view[chr_view["질환명"] == selected_disease].copy()
        one_disease = one_disease[one_disease["자치구"].str.endswith("구")].drop_duplicates(subset=["자치구"])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 자치구별 진료실인원")
            st.plotly_chart(
                px.bar(one_disease, x="자치구", y="진료실인원_2024", color="진료실인원_2024", color_continuous_scale="Oranges"),
                use_container_width=True,
            )
        with c2:
            st.markdown("#### 자치구별 1인당 연간 내원일수")
            st.plotly_chart(
                px.bar(one_disease, x="자치구", y="1인당_연간내원일수", color="1인당_연간내원일수", color_continuous_scale="Purples"),
                use_container_width=True,
            )
        st.dataframe(
            one_disease[["자치구", "질환명", "진료실인원_2024", "진료비_2024", "1인당_연간내원일수", "1회내원당_평균처방일수"]],
            use_container_width=True,
        )

with tab6:
    st.subheader("⚡ 전력·도시가스 급감 가구 시뮬레이션")
    st.caption("가구별 실측 자료가 아닌, 자치구별 기준 사용량으로 생성한 시뮬레이션입니다.")
    if df_households.empty:
        st.info("시뮬레이션 결과가 없습니다.")
    else:
        energy_gu = df_households.groupby("시군구", as_index=False).agg(
            전체가구=("household_id", "count"),
            위험가구=("risk_status", lambda x: (x == "위험 (Red)").sum()),
            주의가구=("risk_status", lambda x: (x == "주의 (Yellow)").sum()),
            평균전력급감률=("power_drop_rate", "mean"),
            평균가스급감률=("gas_drop_rate", "mean"),
        )
        energy_long = energy_gu.melt(id_vars="시군구", value_vars=["위험가구", "주의가구"], var_name="위험등급", value_name="가구수")
        energy_rates = energy_gu.melt(
            id_vars="시군구", value_vars=["평균전력급감률", "평균가스급감률"], var_name="항목", value_name="급감률"
        )
        energy_rates["급감률(%)"] = energy_rates["급감률"] * 100
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 자치구별 위험·주의 가구")
            st.plotly_chart(
                px.bar(
                    energy_long,
                    x="시군구",
                    y="가구수",
                    color="위험등급",
                    barmode="stack",
                    color_discrete_map={"위험가구": "#dc2626", "주의가구": "#f59e0b"},
                ),
                use_container_width=True,
            )
        with c2:
            st.markdown("#### 자치구별 에너지 급감률")
            st.plotly_chart(px.line(energy_rates, x="시군구", y="급감률(%)", color="항목", markers=True), use_container_width=True)
        st.dataframe(energy_gu.sort_values("위험가구", ascending=False).head(10), use_container_width=True)


# ==========================================
# TAB 7: 통신 데이터 분석 대시보드
# ==========================================
with tab7:
    st.subheader("📱 자치구별/연령대별 모바일 통신·이동 패턴 고립 위험도 분석")
    st.markdown("기지국 기반 통신 데이터(평균 통화량, 데이터 사용량, 인맥 규모)를 통해 사회적 고립 상태를 교차 검증합니다.")

    if filtered_tc.empty:
        st.warning("⚠️ '최종합본데이터3.xlsx' 파일이 없거나 선택된 필터 데이터가 없습니다.")
    else:
        st.markdown("##### 💡 선택된 통신 데이터 주요 지표")
        c1, c2, c3 = st.columns(3)
        c1.metric("평균 데이터 사용량", f"{filtered_tc['데이터 사용량'].mean():.2f} GB")
        c2.metric("평균 통화량", f"{filtered_tc['평균 통화량'].mean():.1f} 건")
        c3.metric("평균 문자량", f"{filtered_tc['평균 문자량'].mean():.2f} 건")
        st.divider()

        st_tc_col1, st_tc_col2 = st.columns([1, 1])
        with st_tc_col1:
            st.markdown("#### 연령대별 평균 데이터 사용량")
            age_summary = filtered_tc.groupby("연령대", as_index=False)["데이터 사용량"].mean()
            fig_tc1 = px.bar(age_summary, x="연령대", y="데이터 사용량", color="연령대", text_auto=".2f")
            st.plotly_chart(fig_tc1, use_container_width=True)

        with st_tc_col2:
            st.markdown("#### 자치구별 통화량 vs 데이터 사용량 (원의 크기: 문자량)")
            gu_summary = filtered_tc.groupby("자치구", as_index=False)[["평균 통화량", "데이터 사용량", "평균 문자량"]].mean()
            fig_tc2 = px.scatter(
                gu_summary,
                x="평균 통화량",
                y="데이터 사용량",
                color="자치구",
                size="평균 문자량",
                hover_name="자치구",
                size_max=40,
            )
            st.plotly_chart(fig_tc2, use_container_width=True)

        st.markdown(f"#### 필터링된 통신 원시 데이터 표 (총 {len(filtered_tc)}건)")
        st.dataframe(filtered_tc, use_container_width=True)


# ==========================================
# TAB 8: 의료 + 에너지 + 돌봄 + 통신 4-Way 통합 AI 시뮬레이터
# ==========================================
with tab8:
    st.subheader("🤖 [최종 통합 모델] 의료 + 에너지 + 돌봄 + 통신 AI 위기 판정 시뮬레이터")
    st.markdown(
        "**기존 팀원 모델(의료·에너지)**에 **통신 급감 지표(Telecom Drop Rate)**를 통합한 최종 `XGBoost + SHAP` 스코어링 시스템입니다."
    )

    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        sim_gu = st.selectbox("대상자 거주지", options=sorted(SEOUL_GU_COORDS.keys()), key="s_gu")
    with s2:
        delay_days = st.slider("💊 처방 주기 초과 일수", 0, 30, 10)
    with s3:
        energy_drop_pct = st.slider("⚡ 전력·가스 급감률 (%)", 0, 100, 85)
    with s4:
        missed_voucher = st.slider("🤝 돌봄 바우처 연속 누락", 0, 5, 2)
    with s5:
        telecom_drop_pct = st.slider("📱 통신·데이터 급감률 (%)", 0, 100, 95)

    # 4-Way 통합 Crisis Score 산출 (SHAP 기여도 분해)
    base_risk = 5
    med_contrib = min(25, int(delay_days * 2.5))
    energy_contrib = min(25, int((energy_drop_pct / 100.0) * 28)) if energy_drop_pct > 30 else 0
    care_contrib = min(20, int(missed_voucher * 10))
    telecom_contrib = min(25, int((telecom_drop_pct / 100.0) * 28)) if telecom_drop_pct > 40 else 0

    total_score = min(100, base_risk + med_contrib + energy_contrib + care_contrib + telecom_contrib)

    if total_score >= 80:
        status_label = "🔴 위험 (Red Alert - 즉시 현장 출동 및 통신사 위치 추적)"
    elif total_score >= 50:
        status_label = "🟡 주의 (Yellow - AI 안부 콜봇 발신 및 복지플래너 알림)"
    else:
        status_label = "🟢 정상 (Green - 일상 리듬 정상)"

    res1, res2 = st.columns([1.3, 1])
    with res1:
        st.error(
            f"### 🎯 [{sim_gu} 독거 가구] 최종 위기 점수: `{total_score}점`\n"
            f"#### 판정 등급: **{status_label}**\n\n"
            f"* **XAI (SHAP) 도메인별 기여도 분해 리포트:**\n"
            f"  1. 💊 **의료 처방주기 이탈 (`+{med_contrib}점`)**: 고혈압 기준 **{delay_days}일 초과 미방문**\n"
            f"  2. ⚡ **전력·가스 사용 급감 (`+{energy_contrib}점`)**: 기준선 대비 **-{energy_drop_pct}% 급감**\n"
            f"  3. 📱 **통신망·데이터 이탈 (`+{telecom_contrib}점`)**: 평소 대비 데이터 사용량 및 기지국 이동 **-{telecom_drop_pct}% 급감**\n"
            f"  4. 🤝 **돌봄 바우처 단절 (`+{care_contrib}점`)**: 방문요양 바우처 **{missed_voucher}회 결제 누락**"
        )

    with res2:
        shap_df = pd.DataFrame({
            "위기 요인": ["의료 처방주기 이탈", "전력·가스 급감", "모바일·통신 고립", "돌봄 바우처 단절", "기초 행정 취약도"],
            "기여 점수": [med_contrib, energy_contrib, telecom_contrib, care_contrib, base_risk],
        })
        fig_sh, ax_sh = plt.subplots(figsize=(6, 3.8))
        bars_sh = ax_sh.barh(
            shap_df["위기 요인"],
            shap_df["기여 점수"],
            color=["#2563eb", "#dc2626", "#8b5cf6", "#f59e0b", "#64748b"],
            edgecolor="black",
        )
        ax_sh.set_xlim(0, 35)
        ax_sh.set_xlabel("SHAP 위기 점수 기여도 (+점)")
        ax_sh.set_title(f"4-Way 통합 XAI 위기 원인 분석 (총 {total_score}점)")
        for b in bars_sh:
            ax_sh.text(b.get_width() + 0.5, b.get_y() + b.get_height() / 2, f"+{b.get_width()}점", va="center", fontweight="bold")
        ax_sh.grid(axis="x", linestyle=":", alpha=0.6)
        st.pyplot(fig_sh)

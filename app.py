import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup, Comment
from datetime import datetime, timedelta, date
from scipy import stats
from statsmodels.tsa.seasonal import STL
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import floor
import re
from requests.adapters import HTTPAdapter, Retry
import warnings
import pytz

KST = pytz.timezone('Asia/Seoul')

def get_korea_today():
    return datetime.now(KST).date()

st.set_page_config(page_title="영산강보 지하수측정망 이상치 검색", layout="wide")
st.title("영산강보 지하수측정망 이상치 검색 서비스")

# 세션 상태 초기화
for key, default in [
    ('analysis_complete', False),
    ('df2_anomal', {}),
    ('df_results_filtered', pd.DataFrame()),
    ('anomal_day', 7),
    ('recent_cut', get_korea_today() - timedelta(days=7)),
    ('use_decomposition', True),
    ('station_list', [])
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Sidebar
st.sidebar.header("설정")
max_anoms_frac = st.sidebar.slider("최대 이상치 비율", 0.005, 0.1, 0.01, step=0.005)
alpha = st.sidebar.slider("유의수준 α", 0.001, 0.10, 0.05, step=0.001)
anomal_day = st.sidebar.slider("최근 몇 일 검색(일)", 1, 30, 7, step=1)
max_workers = st.sidebar.slider("동시 요청 수", 4, 32, 8, step=1)
use_decomposition = st.sidebar.checkbox("시계열 분해 사용", value=True)
run_button = st.sidebar.button("이상값 검출 시작 🔍")

# GESD
def generalized_esd(arr, max_anoms_count, alpha):
    x = arr.copy().astype(float)
    n = len(x)
    if n == 0 or max_anoms_count < 1:
        return []
    
    R, lam, removed_idx = [], [], []
    x_work = x.copy()
    idx_map = list(range(n))
    
    for r in range(1, max_anoms_count + 1):
        mu, sigma = np.mean(x_work), np.std(x_work, ddof=1)
        if sigma == 0 or np.isnan(sigma):
            break
        abs_dev = np.abs(x_work - mu)
        max_idx_local = int(np.nanargmax(abs_dev))
        R.append(abs_dev[max_idx_local] / sigma)
        
        p = 1 - alpha / (2 * (n - r + 1))
        df = n - r - 1
        if df <= 0:
            lam.append(np.inf)
        else:
            t_dist = stats.t.ppf(p, df)
            lam.append((n - r) * t_dist / np.sqrt((df + t_dist**2) * (n - r + 1)))
        
        removed_idx.append(idx_map.pop(max_idx_local))
        x_work = np.delete(x_work, max_idx_local)
        if len(x_work) < 3:
            break
    
    k = sum(1 for i in range(len(R)) if R[i] > lam[i])
    return removed_idx[:k]

# 시계열 분해
def time_decompose(df, value_col, freq=7):
    try:
        if len(df) < freq * 2:
            window = min(7, max(3, len(df)//2))
            df['trend'] = df[value_col].rolling(window=window, center=True, min_periods=1).mean()
            df['seasonal'] = 0
            df['remainder'] = df[value_col] - df['trend']
            return df
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            stl = STL(df[value_col].fillna(method='ffill').fillna(method='bfill').to_numpy(),
                     seasonal=freq, trend=None)
            result = stl.fit()
        
        df = df.reset_index(drop=True)
        df['trend'], df['seasonal'], df['remainder'] = result.trend, result.seasonal, result.resid
        return df
    except:
        window = min(7, max(3, len(df)//2))
        df['trend'] = df[value_col].rolling(window=window, center=True, min_periods=1).mean()
        df['seasonal'] = 0
        df['remainder'] = df[value_col] - df['trend']
        return df

def time_recompose(df, anomaly_indices):
    df = df.reset_index(drop=True)
    df['anomaly'] = 'No'
    if len(anomaly_indices) > 0:
        df.loc[anomaly_indices, 'anomaly'] = 'Yes'
    
    remainder_std = df['remainder'].std()
    for i in range(1, 4):
        df[f'recomposed_l{i}'] = df['trend'] + df['seasonal'] - (4-i) * remainder_std
        df[f'recomposed_u{i}'] = df['trend'] + df['seasonal'] + (4-i) * remainder_std
    df['observed'] = df['trend'] + df['seasonal'] + df['remainder']
    return df

# 데이터 로드
@st.cache_data(show_spinner=False)
def load_local_df2(path="./input/df2.csv"):
    df = pd.read_csv(path)
    if "V1" in df.columns:
        df = df.drop(columns=["V1"])
    df.columns = df.columns.str.strip()
    return df

def detect_date_col(df):
    candidates = [c for c in df.columns if any(x in c.lower() for x in ["date", "time"])]
    if candidates:
        return candidates[0]
    for c in df.columns:
        try:
            if pd.to_datetime(df[c], errors="coerce").notna().sum() > 0.5 * len(df):
                return c
        except:
            pass
    return None

def detect_level_col(df):
    for cand in ["gw_level_daily", "gw_level", "datavalue", "value", "level"]:
        for col in df.columns:
            if col.lower() == cand:
                return col
    for col in df.columns:
        if any(x in col.lower() for x in ["gw", "level", "water", "수위"]):
            return col
    return None

def detect_gennum_col(df):
    for cand in ["gennum", "resultid", "station", "gennm", "site", "code"]:
        for col in df.columns:
            if col.lower() == cand:
                return col
    return None

# 크롤링
def create_retry_session(retries=3, backoff_factor=0.5):
    session = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff_factor, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def crawl_station_list(url_weir):
    pattern = re.compile(r"(?:SCM|SCMR|SCMA|SCC|JSM|JSMR|JSMA)-\d{3}")
    sess = create_retry_session()
    r = sess.get(url_weir, timeout=20)
    soup = BeautifulSoup(r.text, "html.parser")
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    ids = set()
    for c in comments:
        ids.update(pattern.findall(c))
    if not ids:
        ids.update(pattern.findall(soup.get_text(" ", strip=True)))
    return sorted(ids)

def fetch_station_json(gennum, from_date, to_date, session=None):
    url = "http://www.gims.go.kr/odmUndergroundChartJson"
    params = {"resultId": gennum, "fromDate": from_date, "toDate": to_date}
    sess = session or create_retry_session()
    try:
        resp = sess.get(url, params=params, timeout=20)
        resp.raise_for_status()
        js = resp.json()
        if not isinstance(js, dict) or "list" not in js or not js["list"]:
            return None
        df_temp = pd.DataFrame(js["list"])
        date_col = "valuedatetimech" if "valuedatetimech" in df_temp.columns else "valuedatetime"
        df_temp["valuedatetimech"] = pd.to_datetime(df_temp[date_col], errors="coerce")
        df_temp["gennum"] = gennum
        return df_temp
    except:
        return None

# 메인 실행
if run_button:
    st.session_state.analysis_complete = False
    
    try:
        df2 = load_local_df2()
        date_col = detect_date_col(df2)
        level_col = detect_level_col(df2)
        gennum_col = detect_gennum_col(df2)
        
        df2 = df2.rename(columns={date_col: "valuedatetimech", gennum_col: "gennum", level_col: "gw_level_raw"})
        df2["valuedatetimech"] = pd.to_datetime(df2["valuedatetimech"], errors="coerce").dt.date
        df2 = df2.dropna(subset=["valuedatetimech"])
        df2["gw_level_raw"] = pd.to_numeric(df2["gw_level_raw"], errors="coerce")
        df2 = df2.groupby(["gennum", "valuedatetimech"], as_index=False)["gw_level_raw"].mean()
        df2 = df2.rename(columns={"gw_level_raw": "gw_level_daily"})
        df2 = df2[df2["valuedatetimech"] != date(2024, 7, 31)]
        
        st.info("관측소 목록 크롤링 중...")
        url_weir = "http://www.gims.go.kr/odmUnderground?resultId=JSM-008&fromDate=2023-04-01&toDate=2023-04-03"
        station_list = crawl_station_list(url_weir)
        st.session_state.station_list = station_list
        st.write(f"크롤된 관측소: {len(station_list)}개")
        
        last_local_date = pd.to_datetime(df2["valuedatetimech"]).dt.date.max()
        from_date, to_date = last_local_date.isoformat(), get_korea_today().isoformat()
        
        st.info(f"새 관측자료 수집 중... ({from_date} ~ {to_date})")
        progress_bar = st.progress(0)
        fetched_frames = []
        
        with requests.Session() as session:
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                futures = {exe.submit(fetch_station_json, s, from_date, to_date, session): s for s in station_list}
                for i, fut in enumerate(as_completed(futures), 1):
                    progress_bar.progress(i / len(station_list))
                    try:
                        df_temp = fut.result()
                        if df_temp is not None and not df_temp.empty:
                            fetched_frames.append(df_temp)
                    except:
                        pass
        
        st.write(f"수집 완료: {len(fetched_frames)}개")
        
        if fetched_frames:
            df_new = pd.concat(fetched_frames, ignore_index=True)
            df_new_pivot = df_new.pivot_table(index=["gennum", "valuedatetimech"], 
                                              columns="datatype", values="datavalue", aggfunc="mean").reset_index()
            rename_map = {"01": "gw_level"}
            df_new_pivot = df_new_pivot.rename(columns=rename_map)
            df_new_pivot["valuedatetimech"] = pd.to_datetime(df_new_pivot["valuedatetimech"], errors="coerce").dt.date
            
            if "gw_level" in df_new_pivot.columns:
                df_url3 = df_new_pivot.groupby(["gennum", "valuedatetimech"], as_index=False)["gw_level"].mean()
                df_url3 = df_url3.rename(columns={"gw_level": "gw_level_daily"})
                df_url3 = df_url3[df_url3["valuedatetimech"] > last_local_date]
                df_url3["gw_level_daily"] = pd.to_numeric(df_url3["gw_level_daily"], errors="coerce")
                combined = pd.concat([df2, df_url3], ignore_index=True)
            else:
                combined = df2.copy()
        else:
            combined = df2.copy()
        
        # 이상치 탐지
        st.info("이상치 검색 중...")
        df2_anomal, results = {}, []
        recent_cut = get_korea_today() - timedelta(days=anomal_day)
        st.session_state.update({'recent_cut': recent_cut, 'anomal_day': anomal_day, 'use_decomposition': use_decomposition})
        
        pbar = st.progress(0)
        for idx, site in enumerate(combined["gennum"].unique(), 1):
            pbar.progress(idx / len(combined["gennum"].unique()))
            
            df_temp = combined[combined["gennum"] == site].dropna(subset=["gw_level_daily"])
            df_temp = df_temp.sort_values("valuedatetimech").reset_index(drop=True).copy()
            
            if len(df_temp) < 6:
                continue
            
            if use_decomposition:
                df_temp = time_decompose(df_temp, 'gw_level_daily', freq=7)
                arr = df_temp['remainder'].fillna(0).to_numpy()
            else:
                arr = df_temp['gw_level_daily'].to_numpy()
            
            max_anoms_count = max(1, floor(max_anoms_frac * len(arr)))
            anomalous_idx = generalized_esd(arr, max_anoms_count, alpha)
            
            if use_decomposition:
                df_temp = time_recompose(df_temp, anomalous_idx)
            else:
                df_temp['anomaly'] = 'No'
                if anomalous_idx:
                    df_temp.loc[anomalous_idx, 'anomaly'] = 'Yes'
                mean_val, std_val = np.mean(arr), np.std(arr, ddof=1)
                df_temp['observed'] = df_temp['gw_level_daily']
                for i in range(1, 4):
                    df_temp[f'recomposed_l{i}'] = mean_val - (4-i) * std_val
                    df_temp[f'recomposed_u{i}'] = mean_val + (4-i) * std_val
            
            recent_data = df_temp[df_temp['valuedatetimech'] >= recent_cut]
            if (recent_data['anomaly'] == 'Yes').any():
                df2_anomal[site] = df_temp
                results.append({
                    "관측소명": site,
                    "이상상황": "수위자료확인필요",
                    "해발수위": df_temp['gw_level_daily'].iloc[-1],
                    "평균수위": df_temp['gw_level_daily'].mean(),
                    "표준편차": df_temp['gw_level_daily'].std(),
                    "이상치개수": (df_temp['anomaly'] == 'Yes').sum(),
                    "최근이상치개수": (recent_data['anomaly'] == 'Yes').sum(),
                    "anomaly_dates": ", ".join(str(d) for d in df_temp[df_temp['anomaly'] == 'Yes']['valuedatetimech'].tolist()),
                    "recent_anomaly_flag": True
                })
        
        st.success("이상치 탐지 완료!")
        
        # 미수신 확인
        st.info("미수신 관측소 확인 중...")
        recent_data = combined[combined["valuedatetimech"] >= recent_cut]
        
        for site in station_list:
            site_data = recent_data[recent_data["gennum"] == site]
            valid_days = site_data.dropna(subset=["gw_level_daily"])["valuedatetimech"].nunique()
            
            if valid_days < anomal_day and not any(r["관측소명"] == site for r in results):
                results.append({
                    "관측소명": site,
                    "이상상황": "미수신",
                    "해발수위": np.nan,
                    "평균수위": np.nan,
                    "표준편차": np.nan,
                    "이상치개수": 0,
                    "최근이상치개수": 0,
                    "anomaly_dates": "",
                    "recent_anomaly_flag": False
                })
        
        # **수정된 부분: 빈 DataFrame 처리**
        if not results:
            df_results_filtered = pd.DataFrame(columns=[
                "관측소명", "이상상황", "해발수위", "평균수위", "표준편차", 
                "이상치개수", "최근이상치개수", "anomaly_dates", "recent_anomaly_flag"
            ])
        else:
            df_results = pd.DataFrame(results).drop_duplicates(subset=["관측소명"])
            df_results_filtered = df_results[
                (df_results["이상상황"] == "미수신") | (df_results["recent_anomaly_flag"])
            ].sort_values("관측소명")
        
        st.session_state.update({
            'df2_anomal': df2_anomal,
            'df_results_filtered': df_results_filtered,
            'analysis_complete': True
        })
        
    except Exception as e:
        st.error(f"오류: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

# 결과 표시
if st.session_state.analysis_complete:
    df_results_filtered = st.session_state.df_results_filtered
    df2_anomal = st.session_state.df2_anomal
    anomal_day = st.session_state.anomal_day
    recent_cut = st.session_state.recent_cut
    use_decomposition = st.session_state.use_decomposition
    station_list = st.session_state.station_list
    
    st.subheader(f"🔎 최근 {anomal_day}일 내 이상치 / 미수신 관측소 요약")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("총 관측소 수", len(station_list))
    with col2:
        anomaly_cnt = 0 if df_results_filtered.empty else (df_results_filtered["이상상황"] == "수위자료확인필요").sum()
        st.metric("이상치 관측소", anomaly_cnt)
    with col3:
        missing_cnt = 0 if df_results_filtered.empty else (df_results_filtered["이상상황"] == "미수신").sum()
        st.metric("미수신 관측소", missing_cnt)
    
    if df_results_filtered.empty:
        st.success(f"✅ 최근 {anomal_day}일 내 이상치 또는 미수신 관측소가 없습니다.")
    else:
        display_cols = ["관측소명", "이상상황", "해발수위", "평균수위", "표준편차", "이상치개수", "최근이상치개수", "anomaly_dates"]
        st.dataframe(
            df_results_filtered[display_cols].reset_index(drop=True).style.format({
                "해발수위": "{:.2f}",
                "평균수위": "{:.2f}",
                "표준편차": "{:.2f}"
            }, na_rep="-"),
            use_container_width=True
        )
        
        # 시각화
        st.subheader(f"📊 최근 {anomal_day}일 내 이상치 관측소 시각화")
        
        sites_with_anomalies = df_results_filtered[
            (df_results_filtered["recent_anomaly_flag"]) & 
            (df_results_filtered["이상상황"] == "수위자료확인필요")
        ]["관측소명"].tolist()
        
        if not sites_with_anomalies:
            st.info("시각화할 데이터가 없습니다.")
        else:
            selected_station = st.selectbox("관측소 선택", sites_with_anomalies, key="station_selector")
            
            if selected_station in df2_anomal:
                plot_data = df2_anomal[selected_station].copy()
                
                fig = go.Figure()
                
                # 범위 음영
                for i, alpha_val, name in [(3, 0.1, '±3σ'), (2, 0.1, '±2σ')]:
                    fig.add_trace(go.Scatter(x=plot_data['valuedatetimech'], y=plot_data[f'recomposed_u{i}'],
                                            mode='lines', line=dict(color='rgba(200,200,200,0)'), showlegend=False, hoverinfo='skip'))
                    fig.add_trace(go.Scatter(x=plot_data['valuedatetimech'], y=plot_data[f'recomposed_l{i}'],
                                            mode='lines', line=dict(color='rgba(200,200,200,0)'), fill='tonexty',
                                            fillcolor=f'rgba(150,150,150,{alpha_val})', name=f'정상 범위 ({name})', hoverinfo='skip'))
                
                # 정상 관측값
                normal = plot_data[plot_data['anomaly'] == 'No']
                fig.add_trace(go.Scatter(x=normal['valuedatetimech'], y=normal['observed'],
                                        mode='lines+markers', line=dict(color='steelblue', width=2),
                                        marker=dict(size=5), name='관측값 (정상)'))
                
                # 이상치
                anomaly_past = plot_data[(plot_data['anomaly'] == 'Yes') & (plot_data['valuedatetimech'] < recent_cut)]
                if not anomaly_past.empty:
                    fig.add_trace(go.Scatter(x=anomaly_past['valuedatetimech'], y=anomaly_past['observed'],
                                            mode='markers', marker=dict(color='orange', size=12, symbol='x', line=dict(width=2)),
                                            name='과거 이상치'))
                
                anomaly_recent = plot_data[(plot_data['anomaly'] == 'Yes') & (plot_data['valuedatetimech'] >= recent_cut)]
                if not anomaly_recent.empty:
                    fig.add_trace(go.Scatter(x=anomaly_recent['valuedatetimech'], y=anomaly_recent['observed'],
                                            mode='markers', marker=dict(color='red', size=15, symbol='diamond', line=dict(width=2, color='darkred')),
                                            name=f'최근 {anomal_day}일 이상치'))
                
                # 추세선
                if use_decomposition and 'trend' in plot_data.columns:
                    fig.add_trace(go.Scatter(x=plot_data['valuedatetimech'], y=plot_data['trend'] + plot_data['seasonal'],
                                            mode='lines', line=dict(color='green', dash='dash', width=2), name='추세+계절성'))
                
                # 최근 구간 강조
                fig.add_vrect(x0=recent_cut, x1=get_korea_today(), fillcolor="rgba(255,0,0,0.05)", 
                            layer="below", line_width=0, annotation_text=f"최근 {anomal_day}일", annotation_position="top left")
                
                fig.update_layout(
                    title=f"🌊 {selected_station} 지하수위 시계열",
                    xaxis_title="날짜", yaxis_title="지하수위 (m)", height=600, hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 통계 정보
                info = df_results_filtered[df_results_filtered["관측소명"] == selected_station].iloc[0]
                st.subheader("📊 통계 정보")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("평균 수위", f"{info['평균수위']:.2f} m")
                c2.metric("표준편차", f"{info['표준편차']:.2f} m")
                c3.metric("전체 이상치", f"{info['이상치개수']}개")
                c4.metric("최근 이상치", f"{info['최근이상치개수']}개")
                st.write(f"**이상치 발생 날짜:** {info['anomaly_dates']}")

elif not run_button:
    st.info("👈 좌측 설정창에서 **이상값 검출 시작** 버튼을 클릭하세요")
    st.markdown("""
    ### 사용 방법
    1. **최대 이상치 비율**: 전체 데이터 중 이상치로 간주할 최대 비율
    2. **유의수준 α**: 통계적 검정의 유의수준 (낮을수록 엄격)
    3. **검색 일수**: 최근 며칠 동안의 이상치 확인
    4. **시계열 분해**: R의 anomalize 패키지 방식 적용
    
    ### 주요 기능
    - ✅ GESD 알고리즘 기반 이상치 탐지
    - ✅ 시계열 분해 (추세/계절성/잔차)
    - ✅ 실시간 관측소 데이터 수집
    - ✅ 미수신 관측소 자동 감지
    """)
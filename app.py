import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup, Comment
from datetime import datetime, timedelta, date
from scipy import stats
from statsmodels.tsa.seasonal import STL
import plotly.express as px
import plotly.graph_objects as go
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import floor
import re
from requests.adapters import HTTPAdapter, Retry
import warnings

st.set_page_config(page_title="영산강보 지하수측정망 이상치 검색", layout="wide")

st.title("영산강보 지하수측정망 이상치 검색 서비스")

# --------------------------------------------------------------
# Sidebar: 사용자 입력
# --------------------------------------------------------------
st.sidebar.header("설정")
max_anoms_frac = st.sidebar.slider("최대 이상치 비율 (max_anoms, fraction)", 0.005, 0.1, 0.01, step=0.005)
alpha = st.sidebar.slider("유의수준 α (GESD에서 사용하는 통계적 유의수준)", 0.001, 0.10, 0.05, step=0.001)
anomal_day = st.sidebar.slider("최근 몇 일 안의 이상을 검색할지(일)", 1, 30, 7, step=1)
max_workers = st.sidebar.slider("동시 요청(스레드) 수", 4, 32, 8, step=1)
use_decomposition = st.sidebar.checkbox("시계열 분해 사용 (anomalize 방식)", value=True)
run_button = st.sidebar.button("이상값 검출 시작 🔍")


# --------------------------------------------------------------
# GESD implementation
# --------------------------------------------------------------
def generalized_esd(arr, max_anoms_count, alpha):
    """Generalized ESD Test for Outliers"""
    x = arr.copy().astype(float)
    n = len(x)
    if n == 0 or max_anoms_count < 1:
        return []

    R = []
    lam = []
    removed_idx = []
    x_work = x.copy()
    idx_map = list(range(n))

    for r in range(1, max_anoms_count + 1):
        mu = np.mean(x_work)
        sigma = np.std(x_work, ddof=1)
        if sigma == 0 or np.isnan(sigma):
            break
        abs_dev = np.abs(x_work - mu)
        max_idx_local = int(np.nanargmax(abs_dev))
        Ri = abs_dev[max_idx_local] / sigma
        R.append(Ri)

        p = 1 - alpha / (2 * (n - r + 1))
        df = n - r - 1
        if df <= 0:
            lam.append(np.inf)
        else:
            t_dist = stats.t.ppf(p, df)
            numerator = (n - r) * t_dist
            denominator = np.sqrt((df + t_dist**2) * (n - r + 1))
            lambda_r = numerator / denominator
            lam.append(lambda_r)

        removed_idx.append(idx_map.pop(max_idx_local))
        x_work = np.delete(x_work, max_idx_local)
        if len(x_work) < 3:
            break

    k = 0
    for i in range(len(R)):
        if R[i] > lam[i]:
            k = i + 1
    return removed_idx[:k]


# --------------------------------------------------------------
# Time Series Decomposition (anomalize 방식) - 수정된 버전
# --------------------------------------------------------------
def time_decompose(df, value_col, freq=7):
    """
    시계열 분해: 관측값 = 추세(trend) + 계절성(seasonal) + 잔차(remainder)
    R의 anomalize::time_decompose()와 유사한 방식
    - STL 호출 시 나오는 경고/메시지를 UI에 노출하지 않고 조용히 처리합니다.
    """
    try:
        # 데이터가 충분한지 확인
        if len(df) < freq * 2:
            # 데이터가 부족하면 간단한 이동평균으로 추세 추출
            window = min(7, max(3, len(df)//2))
            df['trend'] = df[value_col].rolling(window=window, center=True, min_periods=1).mean()
            df['seasonal'] = 0
            df['remainder'] = df[value_col] - df['trend']
            return df

        # STL 분해: statsmodels의 STL은 period/seasonal 인자를 받음. 
        # STL/fit 과정에서 발생하는 경고를 UI로 노출하지 않도록 warnings를 컨트롤합니다.
        with warnings.catch_warnings():
            # STL이나 내부에서 발생할 수 있는 PeriodWarning 등 경고를 억제
            warnings.filterwarnings("ignore")
            stl = STL(df[value_col].fillna(method='ffill').fillna(method='bfill').to_numpy(),
                      seasonal=freq, trend=None)
            result = stl.fit()

        # result.trend/seasonal/resid은 numpy 배열이므로 DataFrame에 다시 넣기
        df = df.reset_index(drop=True)
        df['trend'] = result.trend
        df['seasonal'] = result.seasonal
        df['remainder'] = result.resid

        return df

    except Exception:
        # 예외 발생 시 조용히 단순 분해 방식으로 대체 (사용자에게 경고 메시지 노출 안 함)
        window = min(7, max(3, len(df)//2))
        df['trend'] = df[value_col].rolling(window=window, center=True, min_periods=1).mean()
        df['seasonal'] = 0
        df['remainder'] = df[value_col] - df['trend']
        return df

def time_recompose(df, anomaly_indices):
    """
    시계열 재구성 및 이상치 경계 계산
    R의 anomalize::time_recompose()와 유사
    """
    # DataFrame을 reset_index하여 정수 인덱스 사용
    df = df.reset_index(drop=True)
    
    df['anomaly'] = 'No'
    # 정수 위치(iloc)를 사용하여 이상치 표시
    if len(anomaly_indices) > 0:
        df.loc[anomaly_indices, 'anomaly'] = 'Yes'
    
    # 정상 범위 계산 (추세 + 계절성 ± 3*잔차의 표준편차)
    remainder_std = df['remainder'].std()
    df['recomposed_l1'] = df['trend'] + df['seasonal'] - 3 * remainder_std
    df['recomposed_l2'] = df['trend'] + df['seasonal'] - 2 * remainder_std
    df['recomposed_l3'] = df['trend'] + df['seasonal'] - 1 * remainder_std
    df['observed'] = df['trend'] + df['seasonal'] + df['remainder']
    df['recomposed_u1'] = df['trend'] + df['seasonal'] + 1 * remainder_std
    df['recomposed_u2'] = df['trend'] + df['seasonal'] + 2 * remainder_std
    df['recomposed_u3'] = df['trend'] + df['seasonal'] + 3 * remainder_std
    
    return df


# --------------------------------------------------------------
# 데이터 로드 및 유틸 함수
# --------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_local_df2(path="./input/df2.csv"):
    df = pd.read_csv(path)
    if "V1" in df.columns:
        df = df.drop(columns=["V1"])
    df.columns = df.columns.str.strip()
    return df


def detect_date_col(df):
    date_candidates = [c for c in df.columns if ("date" in c.lower()) or ("time" in c.lower()) or ("valuedatetime" in c.lower())]
    if date_candidates:
        return date_candidates[0]
    for c in df.columns:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().sum() > 0.5 * len(parsed):
                return c
        except Exception:
            pass
    return None


def detect_level_col(df):
    candidates = ["gw_level_daily", "gw_level", "datavalue", "value", "level"]
    for cand in candidates:
        for col in df.columns:
            if col.lower() == cand:
                return col
    for col in df.columns:
        lname = col.lower()
        if "gw" in lname and "level" in lname:
            return col
        if "water" in lname or "수위" in lname or "level" in lname:
            return col
    return None


def detect_gennum_col(df):
    candidates = ["gennum", "resultid", "station", "gennm", "site", "code"]
    for cand in candidates:
        for col in df.columns:
            if col.lower() == cand:
                return col
    for col in df.columns:
        lname = col.lower()
        if "gen" in lname or "station" in lname or "site" in lname or "code" in lname:
            return col
    return None


# --------------------------------------------------------------
# 개선된 크롤링 함수
# --------------------------------------------------------------
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
        text = soup.get_text(" ", strip=True)
        ids.update(pattern.findall(text))
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
        if "valuedatetimech" in df_temp.columns:
            df_temp["valuedatetimech"] = pd.to_datetime(df_temp["valuedatetimech"], errors="coerce")
        elif "valuedatetime" in df_temp.columns:
            df_temp["valuedatetimech"] = pd.to_datetime(df_temp["valuedatetime"], errors="coerce")
        df_temp["gennum"] = gennum
        return df_temp
    except Exception:
        return None


# --------------------------------------------------------------
# 메인 실행
# --------------------------------------------------------------
if run_button:
    try:
        df2 = load_local_df2()
        date_col = detect_date_col(df2)
        level_col = detect_level_col(df2)
        gennum_col = detect_gennum_col(df2)

        df2 = df2.rename(columns={date_col: "valuedatetimech", gennum_col: "gennum", level_col: "gw_level_raw"})
        df2["valuedatetimech"] = pd.to_datetime(df2["valuedatetimech"], errors="coerce")
        df2 = df2.dropna(subset=["valuedatetimech"])
        df2["valuedatetimech"] = df2["valuedatetimech"].dt.date
        df2["gw_level_raw"] = pd.to_numeric(df2["gw_level_raw"], errors="coerce")
        df2_daily = df2.groupby(["gennum", "valuedatetimech"], as_index=False)["gw_level_raw"].mean().rename(columns={"gw_level_raw": "gw_level_daily"})
        df2 = df2_daily.copy()
        df2 = df2[df2["valuedatetimech"] != date(2024, 7, 31)]

        st.info("관측소 목록을 크롤링합니다...")
        url_weir = "http://www.gims.go.kr/odmUnderground?resultId=JSM-008&fromDate=2023-04-01&toDate=2023-04-03"
        try:
            station_list = crawl_station_list(url_weir)
        except Exception as e:
            st.error(f"관측소 목록 크롤링 실패: {e}")
            st.stop()

        st.write(f"크롤된 관측소 수: {len(station_list)}")

        last_local_date = pd.to_datetime(df2["valuedatetimech"]).dt.date.max()
        from_date = last_local_date.isoformat()
        to_date = date.today().isoformat()

        st.info(f"새 관측자료 수집 중... ({from_date} ~ {to_date})")
        progress_bar = st.progress(0)
        fetched_frames = []
        failures = []

        with requests.Session() as session:
            total = len(station_list)
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                futures = {exe.submit(fetch_station_json, s, from_date, to_date, session): s for s in station_list}
                done = 0
                for fut in as_completed(futures):
                    s = futures[fut]
                    done += 1
                    progress_bar.progress(done / total)
                    try:
                        df_temp = fut.result()
                        if df_temp is not None and not df_temp.empty:
                            fetched_frames.append(df_temp)
                    except Exception as e:
                        failures.append((s, str(e)))

        st.write(f"수집 완료: 성공 {len(fetched_frames)} / 실패 {len(failures)}")

        if len(fetched_frames) == 0:
            combined = df2.copy()
        else:
            df_new = pd.concat(fetched_frames, ignore_index=True)
            df_new_pivot = df_new.pivot_table(index=["gennum", "valuedatetimech"], columns="datatype", values="datavalue", aggfunc="mean").reset_index()
            rename_map = {"01": "gw_level", "02": "river_up", "03": "river_down", "04": "rain", "05": "temper"}
            df_new_pivot = df_new_pivot.rename(columns=rename_map)
            df_new_pivot["valuedatetimech"] = pd.to_datetime(df_new_pivot["valuedatetimech"], errors="coerce").dt.date
            if "gw_level" in df_new_pivot.columns:
                df_url3 = df_new_pivot.groupby(["gennum", "valuedatetimech"], as_index=False)["gw_level"].mean().rename(columns={"gw_level": "gw_level_daily"})
            else:
                df_url3 = pd.DataFrame(columns=["gennum", "valuedatetimech", "gw_level_daily"])

            df_url3 = df_url3[df_url3["valuedatetimech"] > last_local_date]
            df_url3["gw_level_daily"] = pd.to_numeric(df_url3["gw_level_daily"], errors="coerce")
            combined = pd.concat([df2, df_url3], ignore_index=True)

        # -------------------------------
        # 이상치 탐지 (anomalize 방식)
        # -------------------------------
        st.info("이상치를 검색중입니다...")
        df2_anomal = {}
        results = []
        unique_sites = combined["gennum"].unique()
        pbar = st.progress(0)
        tot = len(unique_sites)

        # 최근 N일 기준 날짜 계산
        recent_cut = date.today() - timedelta(days=anomal_day)

        for idx, site in enumerate(unique_sites):
            pbar.progress((idx + 1) / tot)
            
            # 관측소별 데이터 추출 및 정렬 (인덱스 리셋)
            df2_temp = combined[combined["gennum"] == site].dropna(subset=["gw_level_daily"]).sort_values("valuedatetimech").reset_index(drop=True).copy()
            
            if df2_temp.shape[0] < 6:
                continue
            
            # 시계열 분해 여부에 따라 분기
            if use_decomposition:
                # 1. time_decompose: 추세, 계절성, 잔차 분해
                df2_temp = time_decompose(df2_temp, 'gw_level_daily', freq=7)
                
                # 2. anomalize: 잔차(remainder)에 대해 GESD 적용
                remainder_arr = df2_temp['remainder'].fillna(0).to_numpy()
            else:
                # 분해 없이 원본 데이터에 직접 GESD 적용
                remainder_arr = df2_temp['gw_level_daily'].to_numpy()
            
            n = len(remainder_arr)
            max_anoms_count = max(1, floor(max_anoms_frac * n))
            
            # GESD 이상치 탐지
            anomalous_idx_list = generalized_esd(remainder_arr, max_anoms_count, alpha)
            
            if use_decomposition:
                # 3. time_recompose: 결과 재구성 및 경계 계산
                df2_temp = time_recompose(df2_temp, anomalous_idx_list)
            else:
                # 단순 방식: 평균 ± 3σ 사용
                df2_temp['anomaly'] = 'No'
                if len(anomalous_idx_list) > 0:
                    df2_temp.loc[anomalous_idx_list, 'anomaly'] = 'Yes'
                
                mean_val = np.mean(remainder_arr)
                std_val = np.std(remainder_arr, ddof=1)
                df2_temp['observed'] = df2_temp['gw_level_daily']
                df2_temp['recomposed_l3'] = mean_val - 3 * std_val
                df2_temp['recomposed_l2'] = mean_val - 2 * std_val
                df2_temp['recomposed_l1'] = mean_val - 1 * std_val
                df2_temp['recomposed_u1'] = mean_val + 1 * std_val
                df2_temp['recomposed_u2'] = mean_val + 2 * std_val
                df2_temp['recomposed_u3'] = mean_val + 3 * std_val
            
            # 최근 N일 내 이상치 확인
            recent_data = df2_temp[df2_temp['valuedatetimech'] >= recent_cut]
            has_recent_anomaly = (recent_data['anomaly'] == 'Yes').any()
            
            # 최근 N일 내 이상치가 있으면 저장
            if has_recent_anomaly:
                df2_anomal[site] = df2_temp
                
                # 통계 정보 계산
                last_val = df2_temp['gw_level_daily'].iloc[-1]
                mean_val = df2_temp['gw_level_daily'].mean()
                std_val = df2_temp['gw_level_daily'].std()
                anomaly_count = (df2_temp['anomaly'] == 'Yes').sum()
                recent_anomaly_count = (recent_data['anomaly'] == 'Yes').sum()
                anomaly_dates = df2_temp[df2_temp['anomaly'] == 'Yes']['valuedatetimech'].tolist()
                
                results.append({
                    "관측소명": site,
                    "이상상황": "수위자료확인필요",
                    "해발수위": last_val,
                    "평균수위": mean_val,
                    "표준편차": std_val,
                    "이상치개수": anomaly_count,
                    "최근이상치개수": recent_anomaly_count,
                    "anomaly_dates": ", ".join([str(d) for d in anomaly_dates]),
                    "recent_anomaly_flag": True
                })

        st.success("이상치 탐지 완료!")

        # -------------------------------
        # 미수신 관측소 확인
        # -------------------------------
        st.info("미수신 관측소 확인 중...")
        recent_data = combined[combined["valuedatetimech"] >= recent_cut]

        missing_sites = []
        for site in station_list:
            site_data = recent_data[recent_data["gennum"] == site]
            valid_days = site_data.dropna(subset=["gw_level_daily"])["valuedatetimech"].nunique()
            
            # 최근 N일 중 하루라도 누락된 경우 미수신으로 분류
            if valid_days < anomal_day:
                missing_sites.append(site)

        for ms in missing_sites:
            # 중복 방지: 이미 results에 있는 관측소는 제외
            if not any(r["관측소명"] == ms for r in results):
                results.append({
                    "관측소명": ms,
                    "이상상황": "미수신",
                    "해발수위": np.nan,
                    "평균수위": np.nan,
                    "표준편차": np.nan,
                    "이상치개수": 0,
                    "최근이상치개수": 0,
                    "anomaly_dates": "",
                    "recent_anomaly_flag": False
                })

        # -------------------------------
        # 결과 데이터프레임 생성 및 필터링
        # -------------------------------
        df_results = pd.DataFrame(results).drop_duplicates(subset=["관측소명"])

        # 최근 N일 내 이상치 OR 미수신 관측소만 필터링
        df_results_filtered = df_results[
            (df_results["이상상황"] == "미수신") | (df_results["recent_anomaly_flag"])
        ].sort_values("관측소명")

        # -------------------------------
        # 결과 요약 표시
        # -------------------------------
        st.subheader(f"🔎 최근 {anomal_day}일 내 이상치 / 미수신 관측소 요약")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 관측소 수", len(station_list))
        with col2:
            anomaly_count = df_results_filtered[df_results_filtered["이상상황"] == "수위자료확인필요"].shape[0]
            st.metric("이상치 관측소", anomaly_count)
        with col3:
            missing_count = df_results_filtered[df_results_filtered["이상상황"] == "미수신"].shape[0]
            st.metric("미수신 관측소", missing_count)

        if df_results_filtered.empty:
            st.success(f"✅ 최근 {anomal_day}일 내 이상치 또는 미수신 관측소가 없습니다.")
        else:
            # 결과 테이블 표시
            display_cols = ["관측소명", "이상상황", "해발수위", "평균수위", "표준편차", "이상치개수", "최근이상치개수", "anomaly_dates"]
            st.dataframe(
                df_results_filtered[display_cols].reset_index(drop=True).style.format({
                    "해발수위": "{:.2f}",
                    "평균수위": "{:.2f}",
                    "표준편차": "{:.2f}"
                }, na_rep="-"),
                use_container_width=True
            )
            
            # -------------------------------
            # 이상치 시각화 (anomalize 스타일)
            # -------------------------------
            st.subheader(f"📊 최근 {anomal_day}일 내 이상치 관측소 시각화")
            
            # 최근 이상치가 있는 관측소만 필터링
            sites_with_recent_anomalies = df_results_filtered[
                (df_results_filtered["recent_anomaly_flag"]) & 
                (df_results_filtered["이상상황"] == "수위자료확인필요")
            ]["관측소명"].tolist()
            
            if not sites_with_recent_anomalies:
                st.info("시각화할 최근 이상치 데이터가 없습니다.")
            else:
                # 관측소 선택 옵션 (세션 상태로 선택 유지)
                # sites_with_recent_anomalies 가 비었으면 위에서 이미 처리되므로 여기서는 비어있지 않음 가정
                if 'selected_station' not in st.session_state or st.session_state['selected_station'] not in sites_with_recent_anomalies:
                    # 세션 상태에 값이 없거나 현재 목록에 없는 값이면 첫 항목으로 초기화
                    st.session_state['selected_station'] = sites_with_recent_anomalies[0]

                selected_station = st.selectbox(
                    "관측소 선택",
                    options=sites_with_recent_anomalies,
                    key='selected_station'
                )
                
                if selected_station in df2_anomal:
                    plot_data = df2_anomal[selected_station].copy()
                    
                    # Plotly 그래프 생성 (anomalize 스타일)
                    fig = go.Figure()
                    
                    # 정상 범위 음영 (3σ)
                    fig.add_trace(go.Scatter(
                        x=plot_data['valuedatetimech'],
                        y=plot_data['recomposed_u3'],
                        mode='lines',
                        line=dict(color='rgba(200, 200, 200, 0)'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=plot_data['valuedatetimech'],
                        y=plot_data['recomposed_l3'],
                        mode='lines',
                        line=dict(color='rgba(200, 200, 200, 0)'),
                        fill='tonexty',
                        fillcolor='rgba(200, 200, 200, 0.1)',
                        name='정상 범위 (±3σ)',
                        hoverinfo='skip'
                    ))
                    
                    # 2σ 범위
                    fig.add_trace(go.Scatter(
                        x=plot_data['valuedatetimech'],
                        y=plot_data['recomposed_u2'],
                        mode='lines',
                        line=dict(color='rgba(150, 150, 150, 0)'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=plot_data['valuedatetimech'],
                        y=plot_data['recomposed_l2'],
                        mode='lines',
                        line=dict(color='rgba(150, 150, 150, 0)'),
                        fill='tonexty',
                        fillcolor='rgba(150, 150, 150, 0.1)',
                        name='경고 범위 (±2σ)',
                        hoverinfo='skip'
                    ))
                    
                    # 관측값 (정상)
                    normal_data = plot_data[plot_data['anomaly'] == 'No']
                    fig.add_trace(go.Scatter(
                        x=normal_data['valuedatetimech'],
                        y=normal_data['observed'],
                        mode='lines+markers',
                        line=dict(color='steelblue', width=2),
                        marker=dict(size=5),
                        name='관측값 (정상)'
                    ))
                    
                    # 이상치 (과거)
                    anomaly_data = plot_data[(plot_data['anomaly'] == 'Yes') & 
                                            (plot_data['valuedatetimech'] < recent_cut)]
                    if not anomaly_data.empty:
                        fig.add_trace(go.Scatter(
                            x=anomaly_data['valuedatetimech'],
                            y=anomaly_data['observed'],
                            mode='markers',
                            marker=dict(color='orange', size=12, symbol='x', line=dict(width=2)),
                            name='과거 이상치'
                        ))
                    
                    # 이상치 (최근)
                    recent_anomaly_data = plot_data[(plot_data['anomaly'] == 'Yes') & 
                                                   (plot_data['valuedatetimech'] >= recent_cut)]
                    if not recent_anomaly_data.empty:
                        fig.add_trace(go.Scatter(
                            x=recent_anomaly_data['valuedatetimech'],
                            y=recent_anomaly_data['observed'],
                            mode='markers',
                            marker=dict(color='red', size=15, symbol='diamond', line=dict(width=2, color='darkred')),
                            name=f'최근 {anomal_day}일 이상치'
                        ))
                    
                    # 추세선 (시계열 분해 사용 시)
                    if use_decomposition and 'trend' in plot_data.columns:
                        fig.add_trace(go.Scatter(
                            x=plot_data['valuedatetimech'],
                            y=plot_data['trend'] + plot_data['seasonal'],
                            mode='lines',
                            line=dict(color='green', dash='dash', width=2),
                            name='추세 + 계절성'
                        ))
                    
                    # 최근 N일 구간 강조
                    fig.add_vrect(
                        x0=recent_cut,
                        x1=date.today(),
                        fillcolor="rgba(255, 0, 0, 0.05)",
                        layer="below",
                        line_width=0,
                        annotation_text=f"최근 {anomal_day}일",
                        annotation_position="top left"
                    )
                    
                    fig.update_layout(
                        title=f"🌊 {selected_station} 지하수위 시계열 (anomalize 방식 이상치 탐지)",
                        xaxis_title="날짜",
                        yaxis_title="지하수위 (m)",
                        height=600,
                        hovermode="x unified",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 상세 통계 정보
                    info_row = df_results_filtered[df_results_filtered["관측소명"] == selected_station].iloc[0]
                    
                    st.subheader("📊 통계 정보")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("평균 수위", f"{info_row['평균수위']:.2f} m")
                    with col2:
                        st.metric("표준편차", f"{info_row['표준편차']:.2f} m")
                    with col3:
                        st.metric("전체 이상치", f"{info_row['이상치개수']}개")
                    with col4:
                        st.metric("최근 이상치", f"{info_row['최근이상치개수']}개")
                    
                    st.write(f"**이상치 발생 날짜:** {info_row['anomaly_dates']}")
                    
                    # 시계열 분해 결과 (사용 시)
                    if use_decomposition and all(col in plot_data.columns for col in ['trend', 'seasonal', 'remainder']):
                        with st.expander("📈 시계열 분해 결과 (Time Decomposition)"):
                            fig_decomp = go.Figure()
                            
                            # 서브플롯 생성
                            from plotly.subplots import make_subplots
                            
                            fig_decomp = make_subplots(
                                rows=4, cols=1,
                                subplot_titles=('원본 관측값 (Observed)', '추세 (Trend)', 
                                              '계절성 (Seasonal)', '잔차 (Remainder)'),
                                vertical_spacing=0.08
                            )
                            
                            # 원본
                            fig_decomp.add_trace(
                                go.Scatter(x=plot_data['valuedatetimech'], y=plot_data['observed'],
                                          mode='lines', name='Observed', line=dict(color='steelblue')),
                                row=1, col=1
                            )
                            
                            # 추세
                            fig_decomp.add_trace(
                                go.Scatter(x=plot_data['valuedatetimech'], y=plot_data['trend'],
                                          mode='lines', name='Trend', line=dict(color='green')),
                                row=2, col=1
                            )
                            
                            # 계절성
                            fig_decomp.add_trace(
                                go.Scatter(x=plot_data['valuedatetimech'], y=plot_data['seasonal'],
                                          mode='lines', name='Seasonal', line=dict(color='orange')),
                                row=3, col=1
                            )
                            
                            # 잔차 (이상치 표시)
                            remainder_normal = plot_data[plot_data['anomaly'] == 'No']
                            remainder_anomaly = plot_data[plot_data['anomaly'] == 'Yes']
                            
                            fig_decomp.add_trace(
                                go.Scatter(x=remainder_normal['valuedatetimech'], y=remainder_normal['remainder'],
                                          mode='lines', name='Remainder', line=dict(color='purple')),
                                row=4, col=1
                            )
                            
                            if not remainder_anomaly.empty:
                                fig_decomp.add_trace(
                                    go.Scatter(x=remainder_anomaly['valuedatetimech'], y=remainder_anomaly['remainder'],
                                              mode='markers', name='Anomaly', 
                                              marker=dict(color='red', size=10, symbol='diamond')),
                                    row=4, col=1
                                )
                            
                            fig_decomp.update_xaxes(title_text="날짜", row=4, col=1)
                            fig_decomp.update_yaxes(title_text="수위 (m)", row=1, col=1)
                            fig_decomp.update_yaxes(title_text="수위 (m)", row=2, col=1)
                            fig_decomp.update_yaxes(title_text="수위 (m)", row=3, col=1)
                            fig_decomp.update_yaxes(title_text="수위 (m)", row=4, col=1)
                            
                            fig_decomp.update_layout(
                                height=800,
                                showlegend=False,
                                title_text=f"{selected_station} 시계열 분해 분석"
                            )
                            
                            st.plotly_chart(fig_decomp, use_container_width=True)
                    
                    # 데이터 테이블
                    with st.expander("📄 전체 데이터 보기"):
                        display_columns = ['valuedatetimech', 'gw_level_daily', 'anomaly']
                        if use_decomposition:
                            display_columns.extend(['trend', 'seasonal', 'remainder', 'observed'])
                        
                        st.dataframe(
                            plot_data[display_columns].style.format({
                                'gw_level_daily': '{:.3f}',
                                'trend': '{:.3f}',
                                'seasonal': '{:.3f}',
                                'remainder': '{:.3f}',
                                'observed': '{:.3f}'
                            }, na_rep='-'),
                            use_container_width=True
                        )
                    
                    # 이상치 상세 정보
                    with st.expander("⚠️ 이상치 상세 정보"):
                        anomaly_details = plot_data[plot_data['anomaly'] == 'Yes'][
                            ['valuedatetimech', 'gw_level_daily', 'observed']
                        ].copy()
                        
                        if not anomaly_details.empty:
                            anomaly_details['최근여부'] = anomaly_details['valuedatetimech'].apply(
                                lambda x: '최근' if x >= recent_cut else '과거'
                            )
                            anomaly_details = anomaly_details.rename(columns={
                                'valuedatetimech': '날짜',
                                'gw_level_daily': '지하수위',
                                'observed': '관측값'
                            })
                            
                            st.dataframe(
                                anomaly_details.style.format({
                                    '지하수위': '{:.3f}',
                                    '관측값': '{:.3f}'
                                }),
                                use_container_width=True
                            )
                        else:
                            st.info("이상치가 없습니다.")
    
    except Exception as e:
        st.error(f"오류 발생: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.info("👈 왼쪽 사이드바에서 설정을 조정한 후 '이상값 검출 시작' 버튼을 클릭하세요.")
    st.markdown("""
    ### 사용 방법
    1. **최대 이상치 비율**: 전체 데이터 중 이상치로 간주할 최대 비율 설정
    2. **유의수준 α**: 통계적 검정의 유의수준 (낮을수록 엄격)
    3. **검색 일수**: 최근 며칠 동안의 이상치를 확인할지 설정
    4. **동시 요청 수**: 데이터 수집 시 동시에 처리할 스레드 수
    5. **시계열 분해 사용**: R의 anomalize 패키지와 동일한 방식 적용 여부
    
    ### 기능
    - ✅ **GESD** (Generalized Extreme Studentized Deviate) 알고리즘 기반 이상치 탐지
    - ✅ **시계열 분해** (Time Decomposition): 추세, 계절성, 잔차 분리
    - ✅ **anomalize 방식**: R의 anomalize 패키지와 동일한 워크플로우
      1. `time_decompose()`: STL 분해로 추세/계절성/잔차 추출
      2. `anomalize()`: 잔차에 GESD 적용하여 이상치 탐지
      3. `time_recompose()`: 결과 재구성 및 신뢰구간 계산
    - ✅ 실시간 관측소 데이터 수집
    - ✅ 미수신 관측소 자동 감지
    - ✅ 시각화를 통한 이상치 확인
    
    ### 시계열 분해 (Time Decomposition)
    - **추세 (Trend)**: 장기적인 상승/하락 경향
    - **계절성 (Seasonal)**: 주기적으로 반복되는 패턴 (기본 7일 주기)
    - **잔차 (Remainder)**: 추세와 계절성을 제외한 나머지 변동
    - 잔차에서 GESD를 적용하여 이상치를 더 정확하게 탐지합니다.
    """)
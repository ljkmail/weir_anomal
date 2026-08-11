"""input/df2.csv를 최신 관측자료로 갱신한다.

app.py는 매 실행마다 `df2.csv의 마지막 날짜 ~ 오늘` 구간을 크롤링한다.
따라서 CSV가 오래될수록 앱이 느려진다 (관측소당 1일치 ≈ 17KB, 53개소면 하루에 약 0.9MB씩 증가).
이 스크립트로 CSV를 주기적으로 갱신해 두면 앱의 수집 구간이 며칠로 줄어든다.

사용법:
    python tools/refresh_df2.py                     # 마지막 날짜 이후만 받아 이어붙임
    python tools/refresh_df2.py --from 2025-01-01   # 해당 날짜부터 다시 받아 덮어씀
    python tools/refresh_df2.py --workers 16        # 동시 요청 수 조정
    python tools/refresh_df2.py --dry-run           # 저장하지 않고 요약만 확인
"""

import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
from requests.adapters import HTTPAdapter, Retry

KST = ZoneInfo("Asia/Seoul")
CSV_PATH = Path(__file__).resolve().parents[1] / "input" / "df2.csv"

STATION_LIST_URL = (
    "http://www.gims.go.kr/odmUnderground"
    "?resultId=JSM-008&fromDate=2023-04-01&toDate=2023-04-03"
)
CHART_URL = "http://www.gims.go.kr/odmUndergroundChartJson"
STATION_PATTERN = re.compile(r"(?:SCM|SCMR|SCMA|SCC|JSM|JSMR|JSMA)-\d{3}")

GW_LEVEL_DATATYPE = "01"  # 02~05(하천수위/강우/수온)는 사용하지 않는다
CHUNK_DAYS = 180  # 한 요청의 최대 기간. 너무 길면 응답이 수십 MB로 커진다.
COLUMNS = ["valuedatetimech", "gennum", "gw_level_daily"]


def create_session(workers):
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
    )
    pool = max(workers, 10)
    adapter = HTTPAdapter(max_retries=retry, pool_connections=pool, pool_maxsize=pool)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def crawl_station_list(session):
    resp = session.get(STATION_LIST_URL, timeout=(5, 30))
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    ids = set()
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        ids.update(STATION_PATTERN.findall(comment))
    if not ids:
        ids.update(STATION_PATTERN.findall(soup.get_text(" ", strip=True)))
    return sorted(ids)


def date_chunks(from_date, to_date, days=CHUNK_DAYS):
    """긴 기간을 days 단위로 쪼갠다. 14년치를 한 번에 요청하면 응답이 지나치게 커진다."""
    start = from_date
    while start <= to_date:
        end = min(start + timedelta(days=days - 1), to_date)
        yield start, end
        start = end + timedelta(days=1)


def fetch_station_daily(gennum, from_date, to_date, session):
    """관측소 1개소의 일평균 지하수위를 반환한다. 집계를 워커에서 끝낸다."""
    parts = []
    for chunk_from, chunk_to in date_chunks(from_date, to_date):
        resp = session.get(
            CHART_URL,
            params={
                "resultId": gennum,
                "fromDate": chunk_from.isoformat(),
                "toDate": chunk_to.isoformat(),
            },
            timeout=(5, 120),
        )
        resp.raise_for_status()
        rows = resp.json().get("list")
        if rows:
            parts.append(pd.DataFrame(rows, columns=["datatype", "datavalue", "valuedatetimech"]))

    if not parts:
        return None

    df = pd.concat(parts, ignore_index=True)
    df = df[df["datatype"] == GW_LEVEL_DATATYPE]
    if df.empty:
        return None

    df["valuedatetimech"] = pd.to_datetime(
        df["valuedatetimech"], format="%Y-%m-%d-%H-%M", errors="coerce"
    ).dt.date
    df["datavalue"] = pd.to_numeric(df["datavalue"], errors="coerce")
    df = df.dropna(subset=["valuedatetimech", "datavalue"])
    if df.empty:
        return None

    daily = df.groupby("valuedatetimech", as_index=False)["datavalue"].mean()
    daily = daily.rename(columns={"datavalue": "gw_level_daily"})
    daily["gennum"] = gennum
    return daily[COLUMNS]


def load_existing():
    if not CSV_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(CSV_PATH)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed") or c == "V1"],
                 errors="ignore")
    df["valuedatetimech"] = pd.to_datetime(df["valuedatetimech"], errors="coerce").dt.date
    df["gw_level_daily"] = pd.to_numeric(df["gw_level_daily"], errors="coerce")
    return df.dropna(subset=["valuedatetimech"])[COLUMNS]


def main():
    ap = argparse.ArgumentParser(description="input/df2.csv 갱신")
    ap.add_argument("--from", dest="from_date", help="시작일 YYYY-MM-DD (기본: CSV 마지막 날짜)")
    ap.add_argument("--to", dest="to_date", help="종료일 YYYY-MM-DD (기본: 오늘, KST)")
    ap.add_argument("--workers", type=int, default=8, help="동시 요청 수 (기본 8)")
    ap.add_argument("--dry-run", action="store_true", help="저장하지 않고 요약만 출력")
    args = ap.parse_args()

    existing = load_existing()
    today = datetime.now(KST).date()

    if args.from_date:
        from_date = datetime.strptime(args.from_date, "%Y-%m-%d").date()
    elif existing.empty:
        sys.exit("기존 CSV가 없습니다. --from 으로 시작일을 지정하세요.")
    else:
        from_date = existing["valuedatetimech"].max()

    to_date = datetime.strptime(args.to_date, "%Y-%m-%d").date() if args.to_date else today

    if from_date > to_date:
        sys.exit(f"시작일({from_date})이 종료일({to_date})보다 늦습니다.")

    print(f"기존 자료 : {len(existing):,}행", end="")
    if not existing.empty:
        print(f" ({existing['valuedatetimech'].min()} ~ {existing['valuedatetimech'].max()})")
    else:
        print(" (없음)")
    print(f"수집 구간 : {from_date} ~ {to_date} ({(to_date - from_date).days + 1}일)")

    session = create_session(args.workers)
    with session:
        stations = crawl_station_list(session)
        print(f"관측소    : {len(stations)}개소, 동시요청 {args.workers}\n")

        frames, failed = [], []
        started = datetime.now(KST)
        with ThreadPoolExecutor(max_workers=args.workers) as exe:
            futures = {
                exe.submit(fetch_station_daily, s, from_date, to_date, session): s
                for s in stations
            }
            for done, fut in enumerate(as_completed(futures), 1):
                station = futures[fut]
                try:
                    daily = fut.result()
                    if daily is not None and not daily.empty:
                        frames.append(daily)
                        note = f"{len(daily):>4}일"
                    else:
                        note = "자료없음"
                except Exception as exc:
                    failed.append(station)
                    note = f"실패 ({type(exc).__name__})"
                print(f"  [{done:>3}/{len(stations)}] {station}  {note}")

    elapsed = (datetime.now(KST) - started).total_seconds()
    print(f"\n수집 완료: {len(frames)}개소 / {elapsed:.1f}초")
    if failed:
        print(f"실패 {len(failed)}개소: {', '.join(failed)}")

    if not frames:
        sys.exit("수집된 자료가 없어 중단합니다.")

    fetched = pd.concat(frames, ignore_index=True)

    # 겹치는 (관측소, 날짜)는 새로 받은 값을 채택한다.
    merged = pd.concat([existing, fetched], ignore_index=True)
    merged = merged.drop_duplicates(subset=["gennum", "valuedatetimech"], keep="last")
    merged = merged.sort_values(["gennum", "valuedatetimech"]).reset_index(drop=True)

    added = len(merged) - len(existing)
    print(f"\n신규/갱신 : {len(fetched):,}행 수집 → 전체 {len(merged):,}행 (순증 {added:+,})")
    print(f"기간      : {merged['valuedatetimech'].min()} ~ {merged['valuedatetimech'].max()}")

    if args.dry_run:
        print("\n--dry-run 이므로 저장하지 않았습니다.")
        return

    merged.to_csv(CSV_PATH, index=False, encoding="utf-8")
    size_mb = CSV_PATH.stat().st_size / 1024 / 1024
    print(f"\n저장 완료 : {CSV_PATH} ({size_mb:.1f}MB)")
    print("git add input/df2.csv && git commit && git push 로 반영하세요.")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
from datetime import date, datetime, timedelta
import json
import time
import urllib.parse

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

from utils.feature_engineering import create_environment_features


# ============================================================
# 1. 페이지 설정
# ============================================================
st.set_page_config(
    page_title="Pico W 실측 기반 문화유산 환경 취약도 예측",
    page_icon="📡",
    layout="wide",
)

st.title("📡 Pico W 실측 데이터 기반 문화유산 환경 취약도 예측")
st.caption(
    "Pico W 센서 실측값을 일별 통계로 변환하고, 센서에 없는 기상·대기환경 변수는 "
    "기상청 ASOS와 AirKorea 자료로 보완한 뒤 기존 학습 모델로 환경 취약도를 예측합니다."
)
st.info(
    "📌 센서값 우선 원칙: 기온·습도·PM10·PM2.5는 Pico W 실측값이 있으면 실측값을 사용하고, "
    "해당 날짜의 센서값이 없을 때만 공공데이터로 보완합니다. 강수량·풍속·일조·지면온도와 "
    "O₃·NO₂·CO·SO₂는 공공데이터를 사용합니다."
)


# ============================================================
# 2. 경로 / API / Firebase 설정
# ============================================================
MODEL_DIR = Path("models")
DATA_DIR = Path("data/processed")

MODEL_PATH = MODEL_DIR / "best_model.pkl"
FEATURE_COLS_PATH = MODEL_DIR / "feature_cols.pkl"
TRAIN_MEDIANS_PATH = MODEL_DIR / "train_medians.pkl"
BUNDLE_PATH = MODEL_DIR / "heritage_risk_bundle.pkl"
MODEL_META_PATH = MODEL_DIR / "model_metadata.json"

HERITAGE_CANDIDATES = [
    DATA_DIR / "yc_heritage_feature.csv",
    DATA_DIR / "yc_heritage_detail_enriched.csv",
    DATA_DIR / "yc_heritage_features.csv",
]

# 사용자가 제공한 실시간 모니터링 코드와 동일한 Firebase history
FIREBASE_HISTORY_URL = (
    "https://heritage-project-4a361-default-rtdb.asia-southeast1."
    "firebasedatabase.app/sensor/history.json"
)

ASOS_URL = (
    "https://apis.data.go.kr/"
    "1360000/AsosDalyInfoService/getWthrDataList"
)
AIR_URL = (
    "https://apis.data.go.kr/"
    "B552584/ArpltnStatsSvc/getMsrstnAcctoRDyrg"
)

STN_ID = "281"  # 영천 ASOS
ASOS_SERVICE_KEY = st.secrets.get(
    "ASOS_SERVICE_KEY",
    st.secrets.get("SERVICE_KEY", ""),
)
AIR_SERVICE_KEY = st.secrets.get(
    "AIR_SERVICE_KEY",
    st.secrets.get("SERVICE_KEY", ""),
)
AIR_STATION_NAME = st.secrets.get(
    "AIR_STATION_NAME",
    "영천",
)

GRADE_ORDER = ["안전", "주의", "위험"]
GRADE_COLOR = {
    "안전": "#2ECC71",
    "주의": "#F39C12",
    "위험": "#E74C3C",
}
MATERIAL_ORDER = ["석조", "목조", "금속", "회화", "기타"]
EXPOSURE_ORDER = ["실외", "반실외", "실내"]


# ============================================================
# 3. 공통 유틸리티
# ============================================================
def to_float(value):
    if value in ("", "-", None, "null", "None"):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def find_heritage_path() -> Path:
    for path in HERITAGE_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "문화유산 특성 데이터 파일을 찾을 수 없습니다. "
        "data/processed 폴더를 확인하세요."
    )


@st.cache_resource(show_spinner=False)
def load_model_assets():
    """Bundle 우선, 없으면 개별 pkl 파일을 사용한다."""
    if BUNDLE_PATH.exists():
        bundle = joblib.load(BUNDLE_PATH)
        if not isinstance(bundle, dict):
            raise ValueError("heritage_risk_bundle.pkl 형식이 올바르지 않습니다.")
        model = bundle.get("model")
        feature_cols = bundle.get("features")
        train_medians = bundle.get("train_medians", {})
        metadata = bundle.get("metadata", {})
        if model is None or not feature_cols:
            raise ValueError("Bundle에 model 또는 features가 없습니다.")
        return model, list(feature_cols), train_medians or {}, metadata or {}

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")
    if not FEATURE_COLS_PATH.exists():
        raise FileNotFoundError(f"Feature 파일이 없습니다: {FEATURE_COLS_PATH}")

    model = joblib.load(MODEL_PATH)
    feature_cols = list(joblib.load(FEATURE_COLS_PATH))

    train_medians = {}
    if TRAIN_MEDIANS_PATH.exists():
        loaded = joblib.load(TRAIN_MEDIANS_PATH)
        if isinstance(loaded, dict):
            train_medians = loaded

    metadata = {}
    if MODEL_META_PATH.exists():
        try:
            metadata = json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

    return model, feature_cols, train_medians, metadata


@st.cache_data(show_spinner=False)
def load_heritage_data(path_str: str) -> pd.DataFrame:
    try:
        heritage = pd.read_csv(path_str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        heritage = pd.read_csv(path_str, encoding="utf-8")

    heritage.columns = heritage.columns.astype(str).str.strip()

    rename_candidates = {
        "문화재명(국문)": "heritage_name",
        "문화재명": "heritage_name",
        "국가유산명": "heritage_name",
        "재질": "material",
        "재질분류": "material",
        "노출형태": "exposure",
        "노출환경": "exposure",
        "위도": "latitude",
        "경도": "longitude",
        "소재지": "address",
        "주소": "address",
        "종목": "heritage_type",
    }
    heritage = heritage.rename(
        columns={k: v for k, v in rename_candidates.items() if k in heritage.columns}
    )

    required = ["heritage_name", "material", "exposure"]
    missing = [c for c in required if c not in heritage.columns]
    if missing:
        raise ValueError(f"문화유산 데이터 필수 컬럼이 없습니다: {missing}")

    heritage["heritage_name"] = heritage["heritage_name"].astype(str).str.strip()
    heritage["material"] = (
        heritage["material"].astype(str).str.strip().replace(
            {"벽화": "회화", "그림": "회화", "회화류": "회화"}
        )
    )
    heritage["material"] = heritage["material"].where(
        heritage["material"].isin(MATERIAL_ORDER), "기타"
    )
    heritage["exposure"] = (
        heritage["exposure"].astype(str).str.strip().replace(
            {"옥외": "실외", "야외": "실외", "반옥외": "반실외", "옥내": "실내"}
        )
    )
    heritage["exposure"] = heritage["exposure"].where(
        heritage["exposure"].isin(EXPOSURE_ORDER), "실외"
    )

    for col in ["latitude", "longitude"]:
        if col in heritage.columns:
            heritage[col] = pd.to_numeric(heritage[col], errors="coerce")

    return heritage.drop_duplicates("heritage_name").reset_index(drop=True)


# ============================================================
# 4. Firebase Pico W 이력 데이터
# ============================================================
@st.cache_data(ttl=60, show_spinner=False)
def load_pico_history() -> pd.DataFrame:
    response = requests.get(
        FIREBASE_HISTORY_URL,
        params={"t": time.time()},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data).T.reset_index(drop=True)

    if "timestamp" not in df.columns:
        raise ValueError("Firebase history에 timestamp 컬럼이 없습니다.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    numeric_cols = [
        "temperature", "humidity", "pressure", "light",
        "pm1", "pm25", "pm10",
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "device" not in df.columns:
        df["device"] = "device_1"

    return (
        df.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def make_pico_daily(sensor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pico W 센서 시계열을 모델과 맞는 일자료로 집계한다.

    전처리 원칙
    - None, 빈 문자열, 숫자 변환 불가 값 -> NaN
    - 센서값 0 -> 측정 실패로 간주하여 NaN
    - 일별 mean/max/min/count 계산 시 NaN은 자동 제외
    - 해당 날짜에 유효 센서값이 하나도 없으면 일별 센서값은 NaN으로 남고,
      이후 기상청/AirKorea 공공데이터로 보완한다.
    """
    work = sensor_df.copy()
    work["date"] = work["timestamp"].dt.floor("D")

    sensor_cols = [
        "temperature",
        "humidity",
        "pressure",
        "light",
        "pm1",
        "pm25",
        "pm10",
    ]

    for col in sensor_cols:
        if col not in work.columns:
            work[col] = np.nan

        work[col] = pd.to_numeric(
            work[col],
            errors="coerce",
        )

        # 이 프로젝트의 Pico 센서에서는 0을 측정 실패값으로 처리한다.
        # 공공데이터의 강수량 0 mm 등에는 이 규칙을 적용하지 않는다.
        work.loc[work[col] == 0, col] = np.nan

    daily = (
        work.groupby("date", as_index=False)
        .agg(
            sensor_count=("timestamp", "count"),
            temp_valid_count=("temperature", "count"),
            temp_avg_sensor=("temperature", "mean"),
            temp_max_sensor=("temperature", "max"),
            temp_min_sensor=("temperature", "min"),
            humidity_valid_count=("humidity", "count"),
            humidity_sensor=("humidity", "mean"),
            pressure_valid_count=("pressure", "count"),
            pressure_sensor=("pressure", "mean"),
            light_valid_count=("light", "count"),
            light_sensor=("light", "mean"),
            pm1_valid_count=("pm1", "count"),
            pm1_sensor=("pm1", "mean"),
            pm25_valid_count=("pm25", "count"),
            pm25_sensor=("pm25", "mean"),
            pm10_valid_count=("pm10", "count"),
            pm10_sensor=("pm10", "mean"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    return daily


# ============================================================
# 5. 기상청 ASOS 수집
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weather_range(start_date: date, end_date: date) -> pd.DataFrame:
    if not ASOS_SERVICE_KEY:
        raise ValueError("Streamlit Secrets에 ASOS_SERVICE_KEY 또는 SERVICE_KEY가 없습니다.")

    params = {
        "serviceKey": ASOS_SERVICE_KEY,
        "numOfRows": "999",
        "pageNo": "1",
        "dataType": "JSON",
        "dataCd": "ASOS",
        "dateCd": "DAY",
        "startDt": start_date.strftime("%Y%m%d"),
        "endDt": end_date.strftime("%Y%m%d"),
        "stnIds": STN_ID,
    }
    response = requests.get(ASOS_URL, params=params, timeout=60)
    response.raise_for_status()
    result = response.json()
    items = (
        result.get("response", {})
        .get("body", {})
        .get("items", {})
        .get("item", [])
    )
    if not items:
        raise RuntimeError(f"ASOS 데이터가 없습니다: {start_date} ~ {end_date}")

    weather = pd.DataFrame(items)
    cols = ["tm", "avgTa", "maxTa", "minTa", "avgRhm", "sumRn", "avgWs", "sumSsHr", "avgTs"]
    missing = [c for c in cols if c not in weather.columns]
    if missing:
        raise ValueError(f"ASOS 응답 필수 컬럼 누락: {missing}")

    weather = weather[cols].copy()
    weather.columns = [
        "date", "temp_avg_public", "temp_max_public", "temp_min_public",
        "humidity_public", "rainfall", "wind_speed", "sunshine_hours", "ground_temp",
    ]
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce").dt.floor("D")
    for col in weather.columns.drop("date"):
        weather[col] = pd.to_numeric(weather[col], errors="coerce")
    weather["rainfall"] = weather["rainfall"].fillna(0)
    return weather.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")


# ============================================================
# 6. AirKorea 수집
# ============================================================
def _request_air_chunk(start_date: date, end_date: date, station_name: str) -> list[dict]:
    safe_key = urllib.parse.unquote(AIR_SERVICE_KEY)
    params = {
        "serviceKey": safe_key,
        "returnType": "json",
        "numOfRows": "200",
        "pageNo": "1",
        "inqBginDt": start_date.strftime("%Y%m%d"),
        "inqEndDt": end_date.strftime("%Y%m%d"),
        "msrstnName": station_name,
    }
    last_error = None
    for wait_seconds in [0, 3, 6, 10]:
        if wait_seconds:
            time.sleep(wait_seconds)
        try:
            response = requests.get(AIR_URL, params=params, timeout=60)
            response.raise_for_status()
            if not response.text.strip().startswith("{"):
                raise RuntimeError("AirKorea API가 JSON이 아닌 응답을 반환했습니다.")
            data = response.json()
            return (
                data.get("response", {})
                .get("body", {})
                .get("items", [])
            ) or []
        except Exception as e:
            last_error = e
    raise RuntimeError(f"AirKorea 수집 실패 {start_date}~{end_date}: {last_error}")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_air_range(start_date: date, end_date: date) -> tuple[pd.DataFrame, str]:
    if not AIR_SERVICE_KEY:
        raise ValueError("Streamlit Secrets에 AIR_SERVICE_KEY 또는 SERVICE_KEY가 없습니다.")

    candidates = []
    for name in [AIR_STATION_NAME, "영천시", "영천"]:
        if name not in candidates:
            candidates.append(name)

    for station in candidates:
        all_items = []
        chunk_start = start_date
        while chunk_start <= end_date:
            chunk_end = min(chunk_start + timedelta(days=6), end_date)
            all_items.extend(_request_air_chunk(chunk_start, chunk_end, station))
            chunk_start = chunk_end + timedelta(days=1)

        if not all_items:
            continue

        air = pd.DataFrame(all_items).rename(
            columns={
                "msurDt": "date",
                "pm10Value": "pm10_public",
                "pm25Value": "pm25_public",
                "o3Value": "o3",
                "no2Value": "no2",
                "coValue": "co",
                "so2Value": "so2",
            }
        )
        required = ["date", "pm10_public", "pm25_public", "o3", "no2", "co", "so2"]
        for col in required:
            if col not in air.columns:
                air[col] = np.nan
        air = air[required].copy()
        air["date"] = pd.to_datetime(air["date"], errors="coerce").dt.floor("D")
        for col in required[1:]:
            air[col] = air[col].replace(["-", "", "null", "None"], np.nan)
            air[col] = pd.to_numeric(air[col], errors="coerce")
        air = (
            air.dropna(subset=["date"])
            .groupby("date", as_index=False)
            .mean(numeric_only=True)
            .sort_values("date")
        )
        if not air.empty:
            return air, station

    raise RuntimeError("영천 대기환경 자료를 찾지 못했습니다. AIR_STATION_NAME을 확인하세요.")


# ============================================================
# 7. Pico + 공공데이터 병합
# ============================================================
def merge_sensor_public(
    pico_daily: pd.DataFrame,
    weather: pd.DataFrame,
    air: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """
    날짜축은 ASOS 일자료를 기준으로 만든다.
    센서 측정이 있는 날은 센서의 기온/습도/PM을 우선 사용한다.
    없는 값은 ASOS/AirKorea 값으로 채운다.
    """
    merged = weather.merge(air, on="date", how="left")
    merged = merged.merge(pico_daily, on="date", how="left")
    merged = merged.sort_values("date").reset_index(drop=True)

    # 센서 우선, 없으면 공공데이터
    merged["temp_avg"] = merged["temp_avg_sensor"].combine_first(merged["temp_avg_public"])
    merged["temp_max"] = merged["temp_max_sensor"].combine_first(merged["temp_max_public"])
    merged["temp_min"] = merged["temp_min_sensor"].combine_first(merged["temp_min_public"])
    merged["humidity"] = merged["humidity_sensor"].combine_first(merged["humidity_public"])
    merged["pm10"] = merged["pm10_sensor"].combine_first(merged["pm10_public"])
    merged["pm25"] = merged["pm25_sensor"].combine_first(merged["pm25_public"])

    # 데이터 출처 표시
    merged["temp_source"] = np.where(merged["temp_avg_sensor"].notna(), "Pico W", "기상청")
    merged["humidity_source"] = np.where(merged["humidity_sensor"].notna(), "Pico W", "기상청")
    merged["pm10_source"] = np.where(merged["pm10_sensor"].notna(), "Pico W", "AirKorea")
    merged["pm25_source"] = np.where(merged["pm25_sensor"].notna(), "Pico W", "AirKorea")

    # 기본 이상치 처리
    non_negative = [
        "rainfall", "wind_speed", "sunshine_hours", "humidity",
        "pm10", "pm25", "o3", "no2", "co", "so2",
    ]
    for col in non_negative:
        if col in merged.columns:
            merged.loc[merged[col] < 0, col] = np.nan

    merged.loc[(merged["humidity"] < 0) | (merged["humidity"] > 100), "humidity"] = np.nan
    merged.loc[merged["pm10"] > 1000, "pm10"] = np.nan
    merged.loc[merged["pm25"] > 500, "pm25"] = np.nan
    merged["rainfall"] = merged["rainfall"].fillna(0)

    model_base_cols = [
        "temp_avg", "temp_max", "temp_min", "humidity", "rainfall",
        "wind_speed", "sunshine_hours", "ground_temp",
        "pm10", "pm25", "o3", "no2", "co", "so2",
    ]
    for col in model_base_cols:
        if col not in merged.columns:
            merged[col] = np.nan

    # 미래값을 쓰지 않도록 과거값으로만 보완
    merged[model_base_cols] = merged[model_base_cols].ffill()

    before = len(merged)
    merged = merged.dropna(subset=model_base_cols).reset_index(drop=True)

    if len(merged) < 28:
        raise ValueError(
            f"병합 후 유효한 일자료가 {len(merged)}일입니다. 28일 파생변수 계산을 위해 최소 28일이 필요합니다."
        )

    # 날짜 연속성 검사
    expected = pd.date_range(merged["date"].min(), merged["date"].max(), freq="D")
    missing_dates = expected.difference(merged["date"])
    if len(missing_dates) > 0:
        raise ValueError(f"병합 데이터에 누락 날짜가 {len(missing_dates)}일 있습니다.")

    feature_df = create_environment_features(
        merged.copy(),
        fill_remaining_numeric=True,
    )

    quality = {
        "public_days": len(weather),
        "pico_days": int(pico_daily["date"].nunique()),
        "pico_rows": int(pico_daily["sensor_count"].sum()),
        "removed_initial_rows": before - len(merged),
        "feature_days": len(feature_df),
        "sensor_temp_days": int(merged["temp_avg_sensor"].notna().sum()),
        "sensor_humidity_days": int(merged["humidity_sensor"].notna().sum()),
        "sensor_pm10_days": int(merged["pm10_sensor"].notna().sum()),
        "sensor_pm25_days": int(merged["pm25_sensor"].notna().sum()),
    }

    return feature_df, quality


# ============================================================
# 8. 모델 입력 / 예측
# ============================================================
def select_target_environment(feature_df: pd.DataFrame, target_date: date) -> pd.DataFrame:
    target_ts = pd.Timestamp(target_date).floor("D")
    rows = feature_df.loc[feature_df["date"].dt.floor("D") == target_ts]
    if rows.empty:
        available = feature_df["date"].max()
        raise ValueError(
            f"{target_ts:%Y-%m-%d}의 최종 Feature가 없습니다. "
            f"현재 사용 가능한 최신 Feature는 {available:%Y-%m-%d}입니다."
        )
    return rows.tail(1).reset_index(drop=True)


def combine_environment_and_heritage(target_environment, heritage_df):
    env = target_environment.iloc[0].to_dict()
    rows = []
    for _, h in heritage_df.iterrows():
        row = dict(env)
        row.update(h.to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def build_inference_features(prediction_df, feature_cols, train_medians):
    work = prediction_df.copy()
    categorical_cols = [c for c in ["material", "exposure", "season"] if c in work.columns]
    work = pd.get_dummies(work, columns=categorical_cols, dtype=int)
    X = work.reindex(columns=feature_cols, fill_value=0)

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in X.columns[X.isna().any()].tolist():
        median = train_medians.get(col) if isinstance(train_medians, dict) else None
        if pd.notna(median):
            X[col] = X[col].fillna(median)

    remaining = X.columns[X.isna().any()].tolist()
    if remaining:
        raise ValueError(f"모델 입력 Feature에 결측값이 남아 있습니다: {remaining}")
    return X


def get_class_probability(model, matrix, class_name):
    classes = list(model.classes_)
    if class_name not in classes:
        return np.zeros(matrix.shape[0])
    return matrix[:, classes.index(class_name)]


def run_prediction(model, feature_cols, train_medians, prediction_df):
    X = build_inference_features(prediction_df, feature_cols, train_medians)
    result = prediction_df.copy()
    result["predicted_grade"] = model.predict(X)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        result["safe_probability"] = get_class_probability(model, proba, "안전") * 100
        result["caution_probability"] = get_class_probability(model, proba, "주의") * 100
        result["danger_probability"] = get_class_probability(model, proba, "위험") * 100
    else:
        result["safe_probability"] = np.where(result["predicted_grade"] == "안전", 100.0, 0.0)
        result["caution_probability"] = np.where(result["predicted_grade"] == "주의", 100.0, 0.0)
        result["danger_probability"] = np.where(result["predicted_grade"] == "위험", 100.0, 0.0)

    result["attention_probability"] = (
        result["caution_probability"] + result["danger_probability"]
    )
    result["risk_index"] = (
        result["caution_probability"] * 0.5 + result["danger_probability"]
    ).clip(0, 100)
    return result


def make_display_result(result_df):
    cols = [
        "heritage_name", "material", "exposure", "predicted_grade", "risk_index",
        "safe_probability", "caution_probability", "danger_probability",
        "attention_probability",
    ]
    for optional in ["heritage_type", "address", "latitude", "longitude"]:
        if optional in result_df.columns:
            cols.append(optional)
    return (
        result_df[cols]
        .copy()
        .sort_values(["risk_index", "danger_probability"], ascending=False)
        .reset_index(drop=True)
    )


# ============================================================
# 9. 모델 / 문화유산 데이터 준비
# ============================================================
try:
    model, feature_cols, train_medians, metadata = load_model_assets()
    heritage_path = find_heritage_path()
    heritage_df = load_heritage_data(str(heritage_path))
except Exception as e:
    st.error(f"모델 또는 문화유산 데이터 로드 실패: {e}")
    st.stop()

model_name = metadata.get("model_name", type(model).__name__) if isinstance(metadata, dict) else type(model).__name__

m1, m2, m3 = st.columns(3)
m1.metric("🤖 적용 모델", model_name)
m2.metric("🧩 모델 Feature", f"{len(feature_cols):,}개")
m3.metric("🏛️ 분석 문화유산", f"{len(heritage_df):,}개")


# ============================================================
# 10. Pico 데이터 불러오기 / 장치 선택
# ============================================================
st.markdown("---")
st.subheader("① Pico W 실측 데이터")

try:
    sensor_all = load_pico_history()
except Exception as e:
    st.error(f"Firebase 센서 이력 조회 실패: {e}")
    st.stop()

if sensor_all.empty:
    st.warning("Firebase sensor/history에 저장된 센서 데이터가 없습니다.")
    st.stop()

devices = sorted(sensor_all["device"].dropna().astype(str).unique().tolist())
default_index = devices.index("device_1") if "device_1" in devices else 0
selected_device = st.selectbox("예측에 사용할 센서 장치", devices, index=default_index)

sensor_df = sensor_all[sensor_all["device"].astype(str) == selected_device].copy()
pico_daily = make_pico_daily(sensor_df)

if pico_daily.empty:
    st.warning("선택한 장치의 일별 데이터가 없습니다.")
    st.stop()

latest_sensor_date = pico_daily["date"].max().date()
earliest_sensor_date = pico_daily["date"].min().date()

c1, c2, c3, c4 = st.columns(4)
c1.metric("📡 선택 장치", selected_device)
c2.metric("🧾 센서 측정 건수", f"{len(sensor_df):,}건")
c3.metric("📅 센서 측정 일수", f"{pico_daily['date'].nunique():,}일")
c4.metric("🕒 센서 최신일", f"{latest_sensor_date:%Y-%m-%d}")

st.caption(f"센서 수집 기간: {earliest_sensor_date:%Y-%m-%d} ~ {latest_sensor_date:%Y-%m-%d}")

with st.expander("📊 Pico W 일별 통계 보기", expanded=False):
    show_cols = [
        "date", "sensor_count", "temp_avg_sensor", "temp_max_sensor", "temp_min_sensor",
        "humidity_sensor", "pressure_sensor", "light_sensor", "pm1_sensor",
        "pm25_sensor", "pm10_sensor",
    ]

    # 화면 표시용 한글 컬럼명
    daily_display = (
        pico_daily[show_cols]
        .sort_values("date", ascending=False)
        .rename(
            columns={
                "date": "날짜",
                "sensor_count": "전체 측정 건수",
                "temp_avg_sensor": "평균 기온(℃)",
                "temp_max_sensor": "최고 기온(℃)",
                "temp_min_sensor": "최저 기온(℃)",
                "humidity_sensor": "평균 습도(%)",
                "pressure_sensor": "평균 기압(hPa)",
                "light_sensor": "평균 조도(lx)",
                "pm1_sensor": "평균 PM1(㎍/㎥)",
                "pm25_sensor": "평균 PM2.5(㎍/㎥)",
                "pm10_sensor": "평균 PM10(㎍/㎥)",
            }
        )
    )

    st.dataframe(
        daily_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "날짜": st.column_config.DateColumn("날짜", format="YYYY-MM-DD"),
            "전체 측정 건수": st.column_config.NumberColumn("전체 측정 건수", format="%d"),
            "평균 기온(℃)": st.column_config.NumberColumn("평균 기온(℃)", format="%.1f"),
            "최고 기온(℃)": st.column_config.NumberColumn("최고 기온(℃)", format="%.1f"),
            "최저 기온(℃)": st.column_config.NumberColumn("최저 기온(℃)", format="%.1f"),
            "평균 습도(%)": st.column_config.NumberColumn("평균 습도(%)", format="%.1f"),
            "평균 기압(hPa)": st.column_config.NumberColumn("평균 기압(hPa)", format="%.1f"),
            "평균 조도(lx)": st.column_config.NumberColumn("평균 조도(lx)", format="%.1f"),
            "평균 PM1(㎍/㎥)": st.column_config.NumberColumn("평균 PM1(㎍/㎥)", format="%.1f"),
            "평균 PM2.5(㎍/㎥)": st.column_config.NumberColumn("평균 PM2.5(㎍/㎥)", format="%.1f"),
            "평균 PM10(㎍/㎥)": st.column_config.NumberColumn("평균 PM10(㎍/㎥)", format="%.1f"),
        },
    )


# ============================================================
# 11. 예측 기준일
# ============================================================
st.markdown("---")
st.subheader("② 실측 + 공공데이터 결합")

# 오늘 기준 전일
yesterday = date.today() - timedelta(days=1)

target_date = st.date_input(
    "예측 기준일",
    value=yesterday,
    max_value=yesterday,
)

st.caption(
    "28일 rolling 파생변수를 안정적으로 만들기 위해 기준일 이전 40일의 공공데이터를 수집합니다. "
    "해당 기간에 Pico W 실측이 존재하는 날짜는 실측값을 우선 적용합니다."
)

run_clicked = st.button(
    "🚀 실측 데이터 + 공공데이터 결합 후 환경 취약도 예측",
    type="primary",
    use_container_width=True,
)


# ============================================================
# 12. 수집 / 병합 / 예측 실행
# ============================================================
if run_clicked:
    try:
        status = st.status("데이터를 준비하고 있습니다...", expanded=True)

        # 40일 window
        start_date = target_date - timedelta(days=39)

        status.update(label="🌦 기상청 ASOS 일자료 수집 중...", state="running")
        weather = fetch_weather_range(start_date, target_date)

        status.update(label="🌫 AirKorea 대기환경 일자료 수집 중...", state="running")
        air, used_station = fetch_air_range(start_date, target_date)

        # 센서는 해당 40일만 사용
        pico_window = pico_daily[
            (pico_daily["date"] >= pd.Timestamp(start_date))
            & (pico_daily["date"] <= pd.Timestamp(target_date))
        ].copy()

        status.update(label="🔗 Pico W 실측값과 공공데이터 병합 중...", state="running")
        feature_df, quality = merge_sensor_public(pico_window, weather, air)

        status.update(label="🧮 7일·28일 파생변수 생성 및 기준일 Feature 선택 중...", state="running")
        target_environment = select_target_environment(feature_df, target_date)

        status.update(label="🏛️ 문화유산 특성과 결합 중...", state="running")
        prediction_input = combine_environment_and_heritage(target_environment, heritage_df)

        status.update(label="🤖 best_model.pkl 기반 환경 취약도 예측 중...", state="running")
        result_df = run_prediction(model, feature_cols, train_medians, prediction_input)
        result_df = make_display_result(result_df)

        st.session_state["pico_prediction_result"] = result_df
        st.session_state["pico_feature_df"] = feature_df
        st.session_state["pico_target_environment"] = target_environment
        st.session_state["pico_quality"] = quality
        st.session_state["pico_used_station"] = used_station
        st.session_state["pico_prediction_date"] = target_date

        status.update(label="✅ 실측 기반 환경 취약도 예측 완료", state="complete", expanded=False)

    except Exception as e:
        st.error(f"예측 실행 실패: {e}")


# ============================================================
# 13. 결합 데이터 확인
# ============================================================
result_df = st.session_state.get("pico_prediction_result")
feature_df = st.session_state.get("pico_feature_df")
target_environment = st.session_state.get("pico_target_environment")
quality = st.session_state.get("pico_quality")
used_station = st.session_state.get("pico_used_station")
prediction_date = st.session_state.get("pico_prediction_date")

if result_df is None:
    st.stop()

st.markdown("---")
st.subheader("③ 데이터 결합 결과")

q1, q2, q3, q4 = st.columns(4)
q1.metric("Pico 실측 적용 일수", f"{quality['sensor_temp_days']}일")
q2.metric("PM10 실측 적용 일수", f"{quality['sensor_pm10_days']}일")
q3.metric("PM2.5 실측 적용 일수", f"{quality['sensor_pm25_days']}일")
q4.metric("대기환경 측정소", used_station)

with st.expander("🔎 일별 결합 데이터와 데이터 출처 확인", expanded=True):
    display_cols = [
        "date", "sensor_count",
        "temp_avg", "temp_source",
        "humidity", "humidity_source",
        "pm10", "pm10_source",
        "pm25", "pm25_source",
        "rainfall", "wind_speed", "sunshine_hours", "ground_temp",
        "o3", "no2", "co", "so2",
    ]
    display_cols = [c for c in display_cols if c in feature_df.columns]
    st.dataframe(
        feature_df[display_cols].sort_values("date", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

# 기준일 데이터 출처
row = target_environment.iloc[0]
st.markdown("#### 📌 예측 기준일에 실제 사용한 환경값")
source_cols = st.columns(4)
source_cols[0].metric(
    "평균 기온",
    f"{row['temp_avg']:.1f} ℃",
    help=f"출처: {row.get('temp_source', '-')}",
)
source_cols[1].metric(
    "평균 습도",
    f"{row['humidity']:.1f} %",
    help=f"출처: {row.get('humidity_source', '-')}",
)
source_cols[2].metric(
    "PM10",
    f"{row['pm10']:.1f} ㎍/㎥",
    help=f"출처: {row.get('pm10_source', '-')}",
)
source_cols[3].metric(
    "PM2.5",
    f"{row['pm25']:.1f} ㎍/㎥",
    help=f"출처: {row.get('pm25_source', '-')}",
)


# ============================================================
# 14. 예측 결과 요약
# ============================================================
st.markdown("---")
st.subheader(f"④ {pd.Timestamp(prediction_date):%Y-%m-%d} 문화유산 환경 취약도 예측 결과")

total_count = len(result_df)
safe_count = int((result_df["predicted_grade"] == "안전").sum())
caution_count = int((result_df["predicted_grade"] == "주의").sum())
danger_count = int((result_df["predicted_grade"] == "위험").sum())

r1, r2, r3, r4 = st.columns(4)
r1.metric("🏛️ 전체", f"{total_count}개")
r2.metric("🟢 안전", f"{safe_count}개")
r3.metric("🟠 주의", f"{caution_count}개")
r4.metric("🔴 위험", f"{danger_count}개")

chart1, chart2 = st.columns(2, gap="large")

with chart1:
    grade_counts = (
        result_df["predicted_grade"]
        .value_counts()
        .reindex(GRADE_ORDER, fill_value=0)
        .rename_axis("등급")
        .reset_index(name="문화유산 수")
    )
    fig = px.pie(
        grade_counts,
        names="등급",
        values="문화유산 수",
        color="등급",
        color_discrete_map=GRADE_COLOR,
        hole=0.48,
        title="안전·주의·위험 등급 분포",
    )
    fig.update_layout(height=430, legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True)

with chart2:
    material_grade = (
        result_df.groupby(["material", "predicted_grade"], observed=True)
        .size()
        .reset_index(name="문화유산 수")
    )
    fig2 = px.bar(
        material_grade,
        x="material",
        y="문화유산 수",
        color="predicted_grade",
        barmode="stack",
        color_discrete_map=GRADE_COLOR,
        category_orders={
            "material": MATERIAL_ORDER,
            "predicted_grade": GRADE_ORDER,
        },
        title="재질별 안전·주의·위험 분포",
        labels={"material": "재질", "predicted_grade": "예측 등급"},
    )
    fig2.update_layout(height=430, legend_orientation="h")
    st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# 15. 상세 결과 / 다운로드
# ============================================================
st.markdown("#### 🧾 문화유산별 상세 예측")
show_result = result_df.copy()
for col in ["risk_index", "safe_probability", "caution_probability", "danger_probability"]:
    if col in show_result.columns:
        show_result[col] = show_result[col].round(1)

st.dataframe(show_result, use_container_width=True, hide_index=True)

csv = result_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "📥 실측 기반 예측 결과 CSV 다운로드",
    data=csv,
    file_name=f"pico_heritage_prediction_{pd.Timestamp(prediction_date):%Y%m%d}.csv",
    mime="text/csv",
    use_container_width=True,
)

st.caption(
    "※ 이 결과의 '위험'은 실제 훼손 발생을 의미하지 않으며, 프로젝트에서 정의한 환경 취약도 Target을 "
    "학습한 모델의 분류 결과입니다. Pico W 실측값과 공공데이터는 측정 위치·장비·수집 주기가 서로 다르므로 "
    "연구 결과 해석 시 데이터 출처 차이를 함께 고려해야 합니다."
)

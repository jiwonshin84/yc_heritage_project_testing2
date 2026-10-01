from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import json
import time
import urllib.parse

import joblib
from github import Github
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from utils.feature_engineering import create_environment_features


# ============================================================
# 1. 페이지 설정
# ============================================================

st.set_page_config(
    page_title="문화유산 환경 취약도 예측",
    page_icon="🏛️",
    layout="wide",
)

st.title("🏛️ 영천 문화유산 환경 취약도 예측")
st.caption(
    "예측 버튼을 누르면 전일 기준 최근 40일 기상·대기환경 데이터를 자동 수집하고, "
    "파생변수 생성 후 학습된 분류모델로 문화유산별 환경 취약도를 "
    "안전·주의·위험으로 바로 예측합니다."
)

st.info(
    "📌 이 페이지의 '위험'은 실제 문화재 훼손 발생을 의미하지 않습니다. "
    "문헌 기반 환경조건과 프로젝트에서 정의한 상대 가중치를 이용해 만든 "
    "환경 취약도 등급을 분류한 결과입니다."
)


# ============================================================
# 2. 화면 디자인
# ============================================================

st.markdown(
    """
    <style>
    .prediction-hero {
        padding: 18px 22px;
        border-radius: 18px;
        border: 1px solid rgba(128,128,128,0.22);
        background: linear-gradient(
            135deg,
            rgba(52,152,219,0.08),
            rgba(155,89,182,0.07)
        );
        margin-bottom: 14px;
    }

    .prediction-hero h3 {
        margin: 0 0 6px 0;
        font-size: 1.25rem;
    }

    .prediction-hero p {
        margin: 0;
        opacity: 0.82;
    }

    .risk-safe {
        border-left: 6px solid #2ecc71;
        padding: 10px 14px;
        border-radius: 10px;
        background: rgba(46,204,113,0.08);
    }

    .risk-caution {
        border-left: 6px solid #f39c12;
        padding: 10px 14px;
        border-radius: 10px;
        background: rgba(243,156,18,0.08);
    }

    .risk-danger {
        border-left: 6px solid #e74c3c;
        padding: 10px 14px;
        border-radius: 10px;
        background: rgba(231,76,60,0.08);
    }
    </style>
    """,

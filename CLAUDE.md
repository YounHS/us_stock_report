# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

US Stock Report는 매일 아침 S&P 500 기반 미국 주식 시장 분석 리포트를 Slack으로 자동 발송하는 Python 시스템입니다. 추가로 장 시작 전 프리마켓 리포트도 제공합니다.

## Commands

```bash
# 환경 설정
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Slack 설정 필요 (또는 직접 .env 작성)

# 시스템 의존성 (PDF 변환용)
sudo apt-get install -y google-chrome-stable fonts-noto-cjk

# 실행
python main.py              # 전체 실행 (리포트 생성 + Slack 발송)
python main.py --dry-run    # 발송 없이 리포트만 생성
python main.py --test-slack # 테스트 Slack 메시지 발송

# 프리마켓 리포트 (장 시작 전)
python main_premarket.py              # 프리마켓 리포트 생성 + Slack 발송
python main_premarket.py --dry-run    # 발송 없이 리포트만 생성

# cron 설정
# 일일 리포트 (KST 07:00 = UTC 22:00)
0 22 * * 1-5 /path/to/venv/bin/python /path/to/main.py
# 프리마켓 리포트 (KST 21:00 = UTC 12:00)
0 12 * * 1-5 /path/to/venv/bin/python /path/to/main_premarket.py
```

### GitHub Actions 스케줄링

`.github/workflows/run_main.yml`로 자동 실행 (KST 06:55 = UTC 21:55, 월-금).
워크플로우에서 한글 폰트(`fonts-noto-cjk`) 설치 포함 (Chrome은 `ubuntu-latest`에 기본 포함).
민감 정보는 **GitHub Repository Secrets**에 등록하여 워크플로우 `env:` 블록으로 주입:
- `SLACK_BOT_TOKEN`, `SLACK_CHANNEL`
- (레거시) `SMTP_USER`, `SMTP_PASSWORD`, `EMAIL_FROM`, `EMAIL_RECIPIENTS`
- Settings → Secrets and variables → Actions → New repository secret

### 프리마켓 리포트 스케줄링

`.github/workflows/run_premarket.yml`로 자동 실행 (KST 21:00 = UTC 12:00, 월-금).
`actions/download-artifact@v4`로 일일 리포트에서 저장한 전일 추천 종목(`last_recommendations.json`)을 다운로드.
일일 리포트 워크플로우에서 `actions/upload-artifact@v4`로 추천 상태 업로드 (retention: 2일).

## Architecture

데이터 흐름 (일일): `main.py` → 데이터 수집 → 기술적 분석 → 섹터 분석 → **경기 사이클 분석** → 신호 감지 → 캘린더/뉴스 수집 → 뉴스 감성 분석 → 리포트 생성(HTML) → PDF 변환 → Slack 발송 → 추천 종목 저장

데이터 흐름 (프리마켓): `main_premarket.py` → 대상 종목 결정 → 전일 OHLCV 배치 수집 → 프리마켓 가격 개별 수집 → 전일 기술적 분석 → 뉴스/캘린더 수집 → 감성 분석 → S&P 500 PM gainer 배치 스캔(`scan_premarket_gainers`) → gainer 상세 분석 → 개장 급등 추천 → 프리마켓 HTML 리포트 → PDF → Slack 발송

### 핵심 모듈

- **config/settings.py**: Pydantic 기반 환경변수 관리. `settings` 싱글톤으로 전역 접근
- **config/sp500_tickers.py**: Wikipedia에서 S&P 500 종목 스크래핑 (User-Agent 필수), 11개 GICS 섹터 매핑
- **data/fetcher.py**: `yfinance`로 OHLCV 배치 다운로드. `StockDataFetcher.fetch_batch()` 사용
- **data/calendar.py**: 경제 캘린더(`EconomicCalendar`) 및 실적 발표 일정(`EarningsCalendar`) 수집
- **analysis/technical.py**: RSI, MACD, 볼린저밴드, ADX, ATR, 거래량, 상대강도, 52주 위치, 칼만 필터(단기/장기), OBV, Stochastic, TTM Squeeze 계산. 칼만 필터는 두 가지 메서드 제공: `calculate_kalman_filter()` (단기, 1-step 예측) / `calculate_kalman_filter_longterm()` (장기 전용, N-step 예측). 결과는 dataclass로 반환 (`RSIResult`, `MACDResult`, `BollingerResult`, `ADXResult`, `ATRResult`, `VolumeResult`, `RelativeStrengthResult`, `Week52Result`, `KalmanResult`, `OBVResult`, `StochasticResult`, `SqueezeResult`)
- **analysis/signals.py**: `SignalDetector`가 분석 결과에서 매수 신호 감지. 네 가지 추천 방식 제공:
  - `get_enhanced_recommendation()`: 가중치 점수 시스템 기반 단기 추천 (우선 사용)
  - `get_top_recommendation()`: 레거시 방식 (Enhanced 실패 시 폴백)
  - `get_kalman_recommendation()`: 칼만 예측가 > 현재가 필터 기반 추가 단기 추천 (기존 추천과 중복 제외)
  - `get_longterm_recommendations()`: 추세 추종 기반 장기 투자 추천 Top N
  - `get_opening_surge_recommendations()`: 프리마켓 모멘텀 + 뉴스 촉매 + 기술적 돌파 기반 개장 급등 추천 Top N
- **analysis/business_cycle.py**: 경기 순환 사이클 분석. `BusinessCycleAnalyzer`가 6개 복합 팩터(섹터 로테이션, 수익률 곡선, 시장 너비, VIX, 위험 선호도, 신용 스프레드)를 종합하여 4개 국면(회복기/확장기/과열기/수축기) 판별. 원형 가중 평균(circular mean)으로 0-360도 연속 위치 계산. 추가 데이터 소스: `^TNX`, `^IRX`, `^VIX`, `HYG`, `TLT` (별도 `fetch_batch` 호출). dataclass: `CycleFactorReading`, `BusinessCycleResult`. 실패 시 기존 섹터 테이블로 폴백
- **analysis/sentiment.py**: VADER(nltk) 기반 뉴스 헤드라인 감성 분석. `NewsSentimentAnalyzer` 클래스가 종목별 뉴스를 분석하여 5단계 라벨(매우 긍정/긍정/중립/부정/매우 부정) 및 0-100 게이지 점수 반환. VADER lazy 초기화 + lexicon 자동 다운로드 fallback. dataclass: `NewsItemSentiment`, `TickerSentiment`
- **data/premarket.py**: `yf.Ticker(symbol).info`로 프리마켓 가격 개별 조회. `PreMarketFetcher.fetch_batch()`, `get_significant_movers()` (±1% 변동 필터), `scan_premarket_gainers()` (`yf.download()` 배치 2회 호출로 S&P 500 전체를 수초 내 스캔하여 PM 상승 종목만 추출). dataclass: `PreMarketData`
- **data/premarket_tickers.py**: 프리마켓 대상 종목 관리. `save_recommendations()` / `load_previous_recommendations()` (JSON 파일 기반), `get_todays_earnings_tickers()`, `get_premarket_tickers()` (시장 ETF + 섹터 ETF + 전일 추천 + 당일 실적 합산)
- **news/fetcher.py**: 핫한 종목/섹터 뉴스 수집. `fetch_hot_stocks_news()`, `fetch_sector_highlights()`. yfinance 신규 응답 구조(`item["content"]["title"]`) 및 구형 구조 모두 호환
- **report/generator.py**: Jinja2로 `report/templates/daily_report.html` 렌더링. `BusinessCycleResult` → dict 변환 및 SVG 마커 좌표(cos/sin) 사전 계산
- **report/premarket_generator.py**: Jinja2로 `report/templates/premarket_report.html` 렌더링. 프리마켓 전용 컨텍스트 (시장/섹터 PM 데이터, 변동 종목, 전일 추천 현황 등)
- **notification/slack_sender.py**: Slack Bot Token 기반 메시지 + PDF 리포트 발송. 요약 메시지에 단기/칼만 추천 및 뉴스 감성 정보 포함. HTML→PDF 변환은 `google-chrome --headless --print-to-pdf` 사용

### 리포트 구성 (일일)

1. 시장 요약 (SPY, QQQ, DIA, IWM)
2. 주요 뉴스 (핫한 종목/섹터 뉴스)
3. 경제 캘린더 (FOMC, ISM PMI, 고용지표 등)
4. 실적 발표 캘린더 (주요 종목 2주간 일정)
5. 경기 사이클 인디케이터 (4국면 원형 SVG 다이어그램, 6개 팩터 판독, 국면 확률 바, 주도/부진 섹터)
6. 기술적 분석 신호 (1시그마 근접, RSI 과매도, MACD 골든크로스, 복합 신호)
7. Top 10 상승/하락
8. 오늘의 추천 종목 (RSI, ADX, MACD, BB Z-Score, 거래량, ATR%, SPY대비, 52주위치, 매도 목표가, 손절가, 보유 기간, 뉴스 감성 게이지)
9. 칼만 필터 추천 종목 (칼만 예측가 > 현재가 필터, 보라색 카드 레이아웃, 뉴스 감성 게이지)
10. 장기 투자 추천 Top 3 (추세 추종 기반, 녹색 카드 레이아웃, 축소 감성 게이지)

### 리포트 구성 (프리마켓)

1. 헤더: "미국 주식 프리마켓 리포트" (오렌지 그라데이션)
2. 시장 지수 프리마켓: SPY/QQQ/DIA/IWM — 전일종가, PM가격, PM변동%
3. 섹터 ETF 프리마켓: 11개 섹터 ETF 히트맵 (변동% 색상 강도)
4. 프리마켓 주요 변동: 상승/하락 테이블 (±1% 이상)
4-1. 개장 급등 추천 (Opening Surge): PM +1.5%이상 종목 중 점수 상위 3개, 빨강-오렌지 카드
5. 전일 추천 종목 현황: PM변동 + RSI/MACD/칼만예측가 오버레이
6. 경제 캘린더
7. 실적 발표 캘린더
8. 최신 뉴스
9. 푸터: 면책조항 + 생성 시각

### 기술적 분석 파라미터

모든 파라미터는 `.env`에서 설정 가능:
- RSI: 14일 기준, 30 미만 = 과매도
- MACD: 12/26/9 EMA
- 볼린저밴드: 20일 SMA ± 2σ, Z-score로 1시그마 근접 판별 (-1.2 ≤ z ≤ -0.8)
- 칼만 필터 (단기, `calculate_kalman_filter`): process_variance=1e-5, measurement_variance=1e-2, blend_alpha=0.5 (20일 Bollinger SMA와 블렌딩, 1-step 예측)
- 칼만 필터 (장기, `calculate_kalman_filter_longterm`): process_variance=1e-3 (단기 대비 100배), measurement_variance=1e-2, blend_alpha=0.7 (50일 SMA와 블렌딩, N-step 예측 N=`longterm_prediction_days` 기본 40거래일 ≈ 2개월). `longterm_kalman_sma_period`=50
- 개장 급등 (Opening Surge): `surge_top_n`=3, `surge_min_score`=30, `surge_min_pm_change_pct`=1.5%, `surge_atr_target_multiplier`=1.0, `surge_atr_stop_multiplier`=0.5 (ATR 기반 동적 목표가/손절가)
- OBV: 20일 SMA 기준, 매집(accumulation)/분산(distribution)/중립(neutral) 추세 판별
- Stochastic: %K=14일, %D=3일 SMA, 과매도(<20), 과매수(>80)
- TTM Squeeze: 볼린저밴드 20일/2σ, 켈트너채널 20일/1.5ATR. Squeeze ON = 변동성 수축
- 경기 사이클: 6개 팩터 가중 투표 (섹터 로테이션 30, 수익률 곡선 25, 시장 너비 15, VIX 15, 위험 선호도 10, 신용 스프레드 5). 수익률 곡선 임계값: 급경사 1.5%, 평탄 0.5%, 역전 -0.1%. VIX 임계값: 저 15, 고 25. 시장 너비 A/D ratio: 강 1.5, 약 0.8

### 추천 종목 선정 로직

**매도 목표가 계산 흐름**:
- 칼만 필터 예측가와 Bollinger SMA를 α:(1-α) 비율로 블렌딩 (기본 50:50)
- 블렌딩된 값이 ATR의 target_price로 사용됨
- 칼만 필터 실패 시 Bollinger SMA 단독 사용 (기존 로직 폴백)

**Enhanced 방식** (`get_enhanced_recommendation()`):
- **칼만 상승 여력 하드 필터**: 칼만 예측가 기준 종가 대비 10% 이상 상승 여력 있는 종목만 후보 (칼만 데이터 없는 종목도 제외)
- **점수 기준 점진적 하향**: 1차 시도 `min_recommendation_score`(기본 35) → 후보 없으면 5점씩 하향 → 하한 10점 → 하한 미달 시에도 칼만 상승 여력 최고 점수 종목 선정
- 가중치 점수 시스템으로 종목 순위 산정
- 70점 이상: High 신뢰도, 그 외: Medium 신뢰도
- ATR 기반 손절가, 칼만 블렌딩 목표가 계산
- R:R 필터: 양수이면서 2:1 미만일 때만 필터링. 음수 R:R(목표가 < 현재가)은 허용하되 warning으로 경고

**Legacy 방식** (`get_top_recommendation()`):
- Enhanced 실패 시 폴백으로 사용
- **칼만 상승 여력 하드 필터**: 각 신호 그룹 내 종목을 순회하며 칼만 예측가 기준 종가 대비 10% 이상 상승 여력인 첫 종목 선정
- 우선순위: 복합 신호 → RSI 과매도 → MACD 골든크로스 → 1시그마 근접
- 모든 그룹에서 칼만 유효 후보 없으면 `None` 반환
- 매도 목표가는 칼만 블렌딩 목표가 기준 (폴백: 20일 SMA)

**칼만 필터 방식** (`get_kalman_recommendation()`):
- 칼만 예측가가 현재가보다 높은 종목만 후보로 선정 (상승 여력 필터)
- 기존 Enhanced 점수 시스템 재사용 (동일 가중치, 동일 최소 점수)
- 기존 단기 추천 종목과 중복 제외 (`exclude_tickers` 파라미터)
- ATR 기반 손절가, 칼만 블렌딩 목표가 계산 (Enhanced와 동일)
- `recommendation_method`: "Kalman"
- 리포트에서 보라색 카드로 표시

**장기 추천 방식** (`get_longterm_recommendations()`):
- 추세 추종(Trend-Following) 전략 기반, 단기 추천과 별도 운영
- **장기 전용 칼만 필터** (`kalman_longterm`, `calculate_kalman_filter_longterm()`) 사용:
  - 단기 칼만과 완전 분리된 독자적 산식 (파라미터, 블렌딩 앵커, 예측 스텝 모두 다름)
  - process_variance=1e-3 (단기 1e-5 대비 100배 → 추세 변화에 빠르게 반응)
  - blend_alpha=0.7 (단기 0.5 대비 칼만 예측가 비중 높음), 50일 SMA 앵커 (단기는 20일 Bollinger SMA)
  - N-step 예측: `longterm_prediction_days`=40 거래일 후 예측가 (단기는 1-step)
  - `analyze()` 결과에서 `kalman_longterm` 키로 별도 저장 (단기는 `kalman` 키)
- 하드 필터: 장기 Kalman velocity > 0, ADX 방향 != bearish, RSI < 75, 거래량 비율 >= 0.7
- SPY, QQQ, DIA, IWM 제외
- 가중치 합계 100점: ADX(15) > MACD/상대강도(12) > RSI/볼린저/거래량/52주/칼만/OBV/Squeeze(8) > Stochastic(5)
- 최소 점수(`longterm_min_score`) 이상만 추천 대상, 점수 내림차순 Top N

**개장 급등 추천 방식** (`get_opening_surge_recommendations()`):
- 프리마켓 모멘텀 + 뉴스 촉매 + 기술적 돌파 중심의 인트라데이 전략
- **후보 스캔**: `scan_premarket_gainers()`로 S&P 500 전체를 `yf.download()` 배치 2회(전일종가 + PM)로 빠르게 스캔 → PM 상승 종목만 추출 → 해당 종목만 상세 PM + OHLCV + 기술적 분석 + 감성 분석 수행. 종목 유형 제한 없음 (개별 주식, ETF, 레버리지 모두 허용)
- 하드 필터: PM 데이터 필수, PM 변동 >= `surge_min_pm_change_pct`(기본 1.5%), ADX>30+bearish 제외, RSI>85 제외
- 가중치 합계 100점: PM모멘텀(25) > 뉴스촉매(20) > 거래량(15) > 갭설정(10) > Squeeze(10) > 상대강도(8) > ADX(7) > Stochastic(5)
- `surge_min_score`(기본 30) 이상만 추천 대상, 점수 내림차순 Top N(기본 3)
- 목표가: PM가격 + ATR × `surge_atr_target_multiplier`, 손절가: PM가격 - ATR × `surge_atr_stop_multiplier` (ATR 없을 시 2% 추정)
- 보유 기간: 30분~1시간
- 리포트에서 빨강-오렌지 그라데이션 카드로 표시

**추천에 포함되는 기술적 지표** (12개):
- RSI, ADX, MACD (골든크로스 여부), 볼린저밴드 Z-Score
- 거래량 비율, ATR%, SPY 대비 상대강도(20일), 52주 위치
- 칼만 필터 예측가, 추세 속도
- OBV 추세 (매집/분산), Stochastic %K, TTM Squeeze 상태
- 손절가, 리스크:리워드 비율

### 설정 구조

`config/settings.py`의 `Settings` 클래스가 5개 설정 그룹 관리:
- `settings.smtp`: SMTP 서버 정보 (레거시)
- `settings.email`: 수신자, 제목 등 (레거시)
- `settings.analysis`: 기술적 분석 파라미터
- `settings.general`: 타임존, 로그 레벨 등
- `settings.slack`: Slack Bot Token, 채널

### 리포트 템플릿 구조

`report/templates/daily_report.html`은 Jinja2 템플릿으로, 추천 종목 섹션에서 다음 필드들을 사용:

```python
recommendation = {
    "ticker", "close", "change_pct",           # 기본 정보
    "rsi", "adx", "volume_ratio",              # 기술적 지표
    "macd_signal", "bollinger_z_score",        # MACD/볼린저
    "atr_pct", "relative_strength_20d",        # ATR/상대강도
    "week52_position",                         # 52주 위치
    "kalman_predicted_price",                  # 칼만 예측가
    "kalman_trend_velocity",                   # 칼만 추세 속도
    "obv_trend",                               # OBV 추세 (accumulation/distribution/neutral)
    "stochastic_k",                            # Stochastic %K
    "squeeze_status", "squeeze_momentum",      # TTM Squeeze 상태 및 모멘텀
    "target_price", "target_return",           # 목표가 (칼만+볼린저 블렌딩)
    "stop_loss", "risk_reward_ratio",          # 리스크 관리 (Enhanced만)
    "score", "confidence", "score_breakdown",  # 점수 시스템 (Enhanced만)
    "bullish_factors", "warning_factors",      # 매수/주의 요인 (Enhanced만)
    "reasons", "holding_period", "source",     # 추천 근거
    "recommendation_method",                   # "Enhanced", "Legacy", 또는 "Kalman"
    "sentiment",                               # 뉴스 감성 분석 (선택, 아래 구조 참고)
}
```

**ScoreBreakdown (점수 구성 상세)**:
```python
score_breakdown = {
    "rsi_score", "volume_score", "adx_score",        # 최대 15/10/12점
    "macd_score", "bollinger_score",                 # 최대 12/12점
    "relative_strength_score", "week52_score",       # 최대 8/8점
    "obv_score", "stochastic_score", "squeeze_score" # 최대 8/8/7점 (총합 100점)
}
```

**Sentiment (뉴스 감성 분석, 선택)**:
```python
sentiment = {
    "avg_compound",       # 평균 compound score (-1.0 ~ 1.0)
    "gauge_score",        # 0-100 게이지 점수 (50=중립)
    "label",              # 매우 긍정 / 긍정 / 중립 / 부정 / 매우 부정
    "positive_count",     # 긍정 뉴스 건수
    "negative_count",     # 부정 뉴스 건수
    "neutral_count",      # 중립 뉴스 건수
    "total_count",        # 전체 뉴스 건수
    "news_items": [       # 개별 뉴스 감성 (최대 5건 표시)
        {"title", "link", "compound", "label"}  # label: positive/negative/neutral
    ]
}
```

**BusinessCycle (경기 사이클, 선택)**:
```python
business_cycle = {
    "current_phase",          # "early", "mid", "late", "contraction"
    "current_phase_label",    # "회복기", "확장기", "과열기", "수축기"
    "phase_position",         # 0-360도
    "marker_x", "marker_y",  # SVG 마커 좌표 (사전 계산)
    "phase_probabilities",    # {"early": 0.3, "mid": 0.4, "late": 0.2, "contraction": 0.1}
    "leading_sectors",        # 현재 국면 주도 섹터 리스트
    "lagging_sectors",        # 현재 국면 부진 섹터 리스트
    "summary",                # 요약 문장
    "factor_readings": [      # 6개 팩터 판독
        {"name", "display_value", "phase_signal", "phase_signal_label", "weight", "description"}
    ],
}
```

`business_cycle`이 `None`이면 기존 섹터 테이블로 폴백 표시 (Jinja2 `{% if business_cycle %}` 분기).

템플릿에서 `is not none` 체크로 None 값 처리. Enhanced 전용 필드는 Legacy 사용 시 main.py에서 기본값 설정.
`recommendation_method` 필드로 리포트/Slack에 사용된 추천 방식(Enhanced/Legacy/Kalman) 표시.
`sentiment` 필드는 감성 분석 성공 시에만 주입됨. 템플릿에서 `is defined` 체크로 없을 때 안전하게 생략. 감성 분석 실패해도 리포트 생성에 영향 없음 (try/except 감싸져 있음).

### 기술적 지표 설명 페이지 (GitHub Pages)

- **`docs/index.html`**: 12개 기술적 지표의 계산 방식, 파라미터, 값별 해석, 점수 배점을 설명하는 정적 HTML 페이지
- GitHub Pages로 배포: `https://younhs.github.io/us_stock_report/` (repo Settings → Pages → Source: `docs/`)
- `daily_report.html`의 단기/장기 추천 섹션에 "지표 설명 보기" 링크가 해당 URL로 연결됨
- 순수 HTML+CSS로 구성 (외부 의존성 없음, 반응형 지원)
- 지표 추가/배점 변경 시 `docs/index.html`도 함께 업데이트 필요

## 주의사항

- **Wikipedia 스크래핑**: `config/sp500_tickers.py`에서 User-Agent 헤더 필수 (403 방지)
- **yfinance 캐시**: `~/.cache/py-yfinance` 폴더 권한 문제 발생 시 무시 가능
- **yfinance 뉴스 구조**: `stock.news` 응답이 `item["content"]["title"]` 중첩 구조. `news/fetcher.py`에서 신규/구형 모두 호환 처리
- **VADER lexicon**: `nltk` 의 `vader_lexicon` 데이터 필요. `NewsSentimentAnalyzer`가 자동 다운로드 시도하지만, GitHub Actions에서는 워크플로우에 별도 다운로드 단계 포함됨
- **템플릿 호환성**: 추천 dict 변환 시 템플릿에서 사용하는 모든 필드 포함 필요. Enhanced/Legacy 모두 `macd_signal`, `bollinger_z_score`, `atr_pct`, `target_return`, `reasons`, `kalman_predicted_price`, `kalman_trend_velocity`, `recommendation_method` 필드가 있어야 함
- **PDF 변환**: `google-chrome --headless --print-to-pdf` 사용. 한글 출력을 위해 `fonts-noto-cjk` 필요
- **Slack 프리마켓**: `SlackSender.send_premarket()` 메서드로 프리마켓 요약 + PDF 발송. 파일명: `premarket_report_{date}.pdf`
- **Slack files_upload_v2**: 채널 이름이 아닌 **채널 ID**가 필요. `chat_postMessage` 응답에서 채널 ID를 추출하여 사용
- **GitHub Actions 환경변수**: `.env` 파일은 로컬 전용. GitHub Actions에서는 Repository Secrets → 워크플로우 `env:` 블록으로 주입
- **GitHub Actions Chrome**: `ubuntu-latest`에 Google Chrome 기본 포함. 한글 폰트(`fonts-noto-cjk`)만 추가 설치 필요
- **경기 사이클 데이터**: `^TNX`, `^IRX`, `^VIX`, `HYG`, `TLT`를 별도 `fetch_batch(include_spy=False)` 호출로 수집. S&P 500 `stock_data`와 분리하여 `SectorAnalyzer`/`TechnicalAnalyzer`가 순회 시 오염 방지. 분석 시 `{**stock_data, **cycle_data}` 병합하여 IWM/SPY도 사용
- **사이클 SVG**: 인라인 SVG 사용 (외부 이미지 없음). Chrome headless PDF 변환 호환. Jinja2에 cos/sin 없으므로 generator.py에서 마커 좌표 사전 계산

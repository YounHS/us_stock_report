# US Stock Report

S&P 500 기반 미국 주식 시장 일일 분석 리포트를 자동 생성하여 Slack으로 발송하는 Python 시스템입니다. 추가로 장 시작 전 프리마켓 리포트도 제공합니다.

## 주요 기능

- **시장 요약**: SPY, QQQ, DIA, IWM 지수 현황
- **기술적 분석**: RSI, MACD, 볼린저밴드, ADX, ATR, 거래량, 상대강도, 52주 위치, 칼만 필터(단기/장기 독립 산식), OBV, Stochastic, TTM Squeeze
- **매수 신호 감지**: RSI 과매도, MACD 골든크로스, 1시그마 근접, 복합 신호
- **단기 추천 종목**: 가중치 점수 시스템 기반 평균 회귀 추천 (칼만 예측가 기준 10% 이상 상승 여력 필터, 손절가/목표가 포함)
- **칼만 필터 추천 종목**: 칼만 예측가 > 현재가 필터 기반 추가 단기 추천 (기존 추천과 중복 제외)
- **장기 투자 추천**: 추세 추종(Trend-Following) 기반 장기 투자 Top 3 종목 추천 (장기 전용 칼만 필터 사용)
- **경기 사이클 인디케이터**: 6개 복합 팩터(섹터 로테이션, 수익률 곡선, 시장 너비, VIX, 위험 선호도, 신용 스프레드) 기반 4국면(회복기/확장기/과열기/수축기) 판별, 원형 SVG 다이어그램 시각화, Top 10 상승/하락 종목
- **뉴스 감성 분석**: 추천 종목별 뉴스 헤드라인 VADER 감성 분석 (긍정/부정/중립 게이지 시각화)
- **뉴스 및 캘린더**: 핫 종목 뉴스, 경제 캘린더, 실적 발표 일정
- **프리마켓 리포트**: 장 시작 전 시장 지수/섹터 ETF 프리마켓 가격, 주요 변동 종목, 전일 추천 종목 현황 분석
- **개장 급등 추천**: S&P 500 전체 배치 스캔으로 PM 상승 종목 추출 → 프리마켓 모멘텀 + 뉴스 촉매 + 기술적 돌파 기반 인트라데이 추천 Top 3 (ATR 기반 동적 목표가/손절가, 보유 30분~1시간, 종목 유형 제한 없음)
- **성과 추적 대시보드**: 추천 종목의 실제 성과(win/loss/expired)를 자동 평가하여 JSON으로 기록, GitHub Pages 대시보드에서 승률, 평균 수익률, 방식별 비교, 누적 수익률 차트 제공
- **파라미터 자동 튜닝**: 평가 완료된 추천 성과를 분석하여 각 방식(Enhanced/Kalman/Long-term/Opening Surge)의 scoring weights와 min_score threshold를 EMA 스무딩 기반으로 자동 조정. 다음 실행 시 튜닝된 가중치 적용 (closed-loop)

## 프로젝트 구조

```
us_stock_report/
├── main.py                      # 일일 리포트 진입점
├── main_premarket.py            # 프리마켓 리포트 진입점
├── config/
│   ├── settings.py              # Pydantic 기반 환경변수 관리
│   └── sp500_tickers.py         # S&P 500 종목 스크래핑
├── data/
│   ├── fetcher.py               # yfinance 주가 데이터 수집
│   ├── calendar.py              # 경제/실적 캘린더
│   ├── premarket.py             # 프리마켓 가격 수집
│   └── premarket_tickers.py     # 프리마켓 대상 종목 관리
├── analysis/
│   ├── technical.py             # 기술적 분석 (RSI, MACD, BB 등)
│   ├── signals.py               # 매수 신호 감지 및 추천
│   ├── sentiment.py             # VADER 기반 뉴스 감성 분석
│   ├── sector.py                # 섹터별 분석
│   └── business_cycle.py        # 경기 사이클 인디케이터 (4국면 판별)
├── news/
│   └── fetcher.py               # 뉴스 수집
├── report/
│   ├── generator.py             # 일일 리포트 생성 (Jinja2)
│   ├── premarket_generator.py   # 프리마켓 리포트 생성 (Jinja2)
│   └── templates/
│       ├── daily_report.html    # 일일 리포트 HTML 템플릿
│       └── premarket_report.html # 프리마켓 리포트 HTML 템플릿
├── notification/
│   └── slack_sender.py          # Slack 발송 (요약 메시지 + PDF 리포트)
├── tracking/
│   ├── recorder.py              # 추천 종목 기록 (JSON)
│   ├── evaluator.py             # 성과 평가 (yfinance 가격 조회)
│   └── summary.py               # 통계 요약 생성
├── tuning/
│   └── optimizer.py             # 파라미터 자동 튜닝 (가중치/임계값 조정)
├── tests/
│   └── test_optimizer.py        # 튜닝 모듈 단위 테스트
├── docs/
│   ├── index.html               # 기술적 지표 설명 페이지 (GitHub Pages)
│   ├── dashboard.html           # 성과 추적 대시보드 (GitHub Pages)
│   └── data/                    # 추적 데이터 (GitHub Actions 자동 커밋)
│       ├── recommendations.json # 추천 기록 + 평가 결과
│       ├── summary.json         # 통계 요약 (대시보드용)
│       └── tuned_params.json    # 자동 튜닝된 가중치/임계값
├── .github/workflows/
│   ├── run_main.yml             # 일일 리포트 GitHub Actions 스케줄링
│   └── run_premarket.yml        # 프리마켓 리포트 GitHub Actions 스케줄링
├── requirements.txt
└── .env                         # 환경변수 (git 미추적)
```

## 설치

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 시스템 의존성 (PDF 변환용)
sudo apt-get install -y google-chrome-stable fonts-noto-cjk
```

## 환경 설정

`.env` 파일을 프로젝트 루트에 생성합니다:

```env
# Slack
SLACK_BOT_TOKEN=xoxb-your-bot-token
SLACK_CHANNEL=your-channel-name
```

> Slack Bot Token은 [Slack API](https://api.slack.com/apps)에서 Bot을 생성하고 `chat:write`, `files:write` 권한을 부여한 뒤 발급받습니다.

기술적 분석 파라미터도 `.env`에서 커스터마이징 가능합니다 (`config/settings.py` 참고).

### 장기 추천 설정 (선택)

```env
LONGTERM_TOP_N=3                    # 장기 추천 종목 수 (기본: 3)
LONGTERM_MIN_SCORE=40               # 최소 추천 점수 (기본: 40)
LONGTERM_WEIGHT_ADX=15              # ADX 가중치 (기본: 15)
LONGTERM_WEIGHT_MACD=12             # MACD 가중치 (기본: 12)
LONGTERM_WEIGHT_RELATIVE_STRENGTH=12 # 상대강도 가중치 (기본: 12)
```

### 장기 전용 칼만 필터 설정 (선택)

장기 추천에는 단기와 완전 분리된 독자적 칼만 필터 산식을 사용합니다.

```env
LONGTERM_KALMAN_PROCESS_VARIANCE=1e-3   # 프로세스 노이즈 (단기 1e-5 대비 100배 → 추세 변화 빠르게 반응)
LONGTERM_KALMAN_MEASUREMENT_VARIANCE=1e-2 # 측정 노이즈
LONGTERM_KALMAN_BLEND_ALPHA=0.7         # 칼만 예측가 블렌딩 비율 (단기 0.5 대비 칼만 비중 높음)
LONGTERM_KALMAN_SMA_PERIOD=50           # 블렌딩 앵커 SMA 기간 (단기는 20일 Bollinger SMA)
LONGTERM_PREDICTION_DAYS=40             # N-step 예측 거래일 수 (약 2개월, 단기는 1-step)
```

### 개장 급등 추천 설정 (선택)

프리마켓 리포트에서 정규장 개장 직후 단기 급등 가능 종목을 추천합니다.

```env
SURGE_TOP_N=3                    # 추천 종목 수 (기본: 3)
SURGE_MIN_SCORE=30               # 최소 추천 점수 (기본: 30)
SURGE_MIN_PM_CHANGE_PCT=1.5      # 최소 PM 변동률% (기본: 1.5)
SURGE_ATR_TARGET_MULT=1.0        # 목표가 = PM가격 + ATR × 배수 (기본: 1.0)
SURGE_ATR_STOP_MULT=0.5          # 손절가 = PM가격 - ATR × 배수 (기본: 0.5)
SURGE_WEIGHT_PM_MOMENTUM=25      # PM 모멘텀 가중치 (기본: 25)
SURGE_WEIGHT_NEWS_CATALYST=20    # 뉴스 촉매 가중치 (기본: 20)
SURGE_WEIGHT_VOLUME=15           # 거래량 가중치 (기본: 15)
```

### 경기 사이클 인디케이터 설정 (선택)

일일 리포트에서 6개 복합 팩터 기반 경기 순환 국면을 시각화합니다.

```env
CYCLE_YC_STEEP=1.5               # 수익률 곡선 급경사 임계값 (기본: 1.5)
CYCLE_YC_FLAT=0.5                # 수익률 곡선 평탄 임계값 (기본: 0.5)
CYCLE_YC_INVERSION=-0.1          # 수익률 곡선 역전 임계값 (기본: -0.1)
CYCLE_VIX_LOW=15.0               # VIX 저변동성 임계값 (기본: 15.0)
CYCLE_VIX_HIGH=25.0              # VIX 고변동성 임계값 (기본: 25.0)
CYCLE_BREADTH_STRONG=1.5         # 시장 너비 강세 A/D ratio (기본: 1.5)
CYCLE_BREADTH_WEAK=0.8           # 시장 너비 약세 A/D ratio (기본: 0.8)
CYCLE_W_SECTOR=30                # 섹터 로테이션 가중치 (기본: 30)
CYCLE_W_YIELD=25                 # 수익률 곡선 가중치 (기본: 25)
CYCLE_W_BREADTH=15               # 시장 너비 가중치 (기본: 15)
CYCLE_W_VIX=15                   # VIX 가중치 (기본: 15)
CYCLE_W_RISK=10                  # 위험 선호도 가중치 (기본: 10)
CYCLE_W_CREDIT=5                 # 신용 스프레드 가중치 (기본: 5)
```

### 성과 추적 설정 (선택)

추천 종목의 실제 성과를 자동 추적하여 대시보드에 표시합니다.

```env
TRACKING_RETENTION_DAYS=90       # 추적 데이터 보관 기간 (기본: 90일)
TRACKING_LT_STOP_PCT=8.0        # 장기 추천 기본 손절 비율% (기본: 8.0)
```

### 파라미터 자동 튜닝 설정 (선택)

평가 완료된 추천 성과를 분석하여 scoring weights와 min_score를 자동 조정합니다.

```env
TUNING_ENABLED=true              # 튜닝 활성화 (기본: true)
TUNING_SMOOTHING_ALPHA=0.3       # EMA 스무딩 계수 (기본: 0.3, 0=변경없음, 1=즉시반영)
TUNING_WEIGHT_FLOOR=2.0          # 가중치 하한 (기본: 2.0)
TUNING_WEIGHT_CEILING=30.0       # 가중치 상한 (기본: 30.0)
TUNING_MIN_SAMPLES_SHORT=15      # Enhanced/Kalman 최소 샘플 수 (기본: 15)
TUNING_MIN_SAMPLES_LONG=5        # Long-term 최소 샘플 수 (기본: 5)
TUNING_MIN_SAMPLES_SURGE=10      # Opening Surge 최소 샘플 수 (기본: 10)
```

## 실행

```bash
# 일일 리포트 (리포트 생성 + Slack 발송)
python main.py

# 발송 없이 리포트만 생성
python main.py --dry-run

# 테스트 Slack 메시지 발송
python main.py --test-slack

# 프리마켓 리포트 (장 시작 전)
python main_premarket.py

# 발송 없이 프리마켓 리포트만 생성
python main_premarket.py --dry-run
```

## 자동 스케줄링

### GitHub Actions (권장)

두 개의 워크플로우가 월-금 자동 실행됩니다:

| 워크플로우 | 파일 | 실행 시간 (KST) | 설명 |
|---|---|---|---|
| 일일 리포트 | `run_main.yml` | 06:55 (UTC 21:55) | 시장 분석 + 추천 종목 |
| 프리마켓 리포트 | `run_premarket.yml` | 21:00 (UTC 12:00) | 장 시작 전 프리마켓 분석 |

일일 리포트 워크플로우에서 `actions/upload-artifact@v4`로 추천 종목 상태(`last_recommendations.json`)를 저장하고, 프리마켓 워크플로우에서 `actions/download-artifact@v4`로 다운로드하여 전일 추천 종목 현황을 표시합니다. 성과 추적 및 튜닝 데이터(`recommendations.json`, `summary.json`, `tuned_params.json`)를 `[skip ci]` 커밋으로 자동 푸시합니다. 프리마켓 워크플로우에서도 Opening Surge 추천 기록 후 `recommendations.json`을 자동 커밋합니다.

GitHub Repository에 Secrets를 등록해야 합니다:

> Settings > Secrets and variables > Actions > New repository secret

| Secret Name | 값 |
|---|---|
| `SLACK_BOT_TOKEN` | Slack Bot Token (`xoxb-...`) |
| `SLACK_CHANNEL` | Slack 채널명 |

### cron (로컬 서버)

```bash
# 일일 리포트: KST 07:00 = UTC 22:00, 월-금
0 22 * * 1-5 /path/to/venv/bin/python /path/to/main.py

# 프리마켓 리포트: KST 21:00 = UTC 12:00, 월-금
0 12 * * 1-5 /path/to/venv/bin/python /path/to/main_premarket.py
```

## 리포트 구성

### 일일 리포트

1. 시장 요약 (SPY, QQQ, DIA, IWM)
2. 주요 뉴스 (핫 종목/섹터 하이라이트)
3. 경제 캘린더 (FOMC, ISM PMI, 고용지표 등)
4. 실적 발표 캘린더 (주요 종목 2주간 일정)
5. 경기 사이클 인디케이터 (4국면 원형 SVG 다이어그램 + 6개 팩터 판독 + 국면 확률 바 + 주도/부진 섹터)
6. 기술적 분석 신호 (1시그마 근접, RSI 과매도, MACD 골든크로스, 복합 신호)
7. Top 10 상승/하락
8. 오늘의 추천 종목 — 단기 평균 회귀 전략 (보유 기간 1-4주, 뉴스 감성 게이지 포함)
9. 칼만 필터 추천 종목 — 칼만 예측가 상승 여력 기반 추천 (보유 기간 2-4주, 뉴스 감성 게이지 포함)
10. 장기 투자 추천 Top 3 — 추세 추종 전략 (보유 기간 1-3개월, 축소 감성 게이지)

### 프리마켓 리포트

1. 시장 지수 프리마켓 (SPY/QQQ/DIA/IWM — 전일종가, PM가격, PM변동%)
2. 섹터 ETF 프리마켓 (11개 섹터 ETF 히트맵, 변동% 색상 강도)
3. 프리마켓 주요 변동 (±1% 이상 상승/하락 종목)
4. 개장 급등 추천 (Opening Surge) — S&P 500 배치 스캔 → PM +1.5% 이상 종목 중 점수 상위 3개, ATR 기반 목표가/손절가
5. 전일 추천 종목 현황 (PM변동 + RSI/MACD/칼만예측가 오버레이)
6. 경제 캘린더
7. 실적 발표 캘린더
8. 최신 뉴스

프리마켓 기본 대상 종목은 시장 ETF + 11개 섹터 ETF + 전일 추천 종목 + 당일 실적 발표 종목으로 구성됩니다. 개장 급등 추천은 S&P 500 전체를 `yf.download()` 배치 스캔하여 PM 상승 종목을 빠르게 추출한 뒤, 해당 종목만 상세 분석합니다 (종목 유형 제한 없음).

### 추천 전략 비교

| | 단기 추천 | 칼만 필터 추천 | 장기 추천 | 개장 급등 (Opening Surge) |
|---|---|---|---|---|
| **전략** | 평균 회귀 (Mean Reversion) | 칼만 예측가 상승 여력 | 추세 추종 (Trend Following) | PM 모멘텀 + 뉴스 촉매 + 기술적 돌파 |
| **칼만 필터** | 단기 칼만 (1-step, 20일 SMA 블렌딩) | 단기 칼만 (1-step) | **장기 전용 칼만** (N-step, 50일 SMA 블렌딩) | 미사용 |
| **핵심 조건** | 칼만 예측가 ≥ 종가+10%, RSI 과매도 등 | 칼만 예측가 > 현재가 | 장기 Kalman velocity > 0, ADX bullish | S&P 500 배치 스캔 → PM ≥ +1.5%, 8개 항목 가중 점수 |
| **보유 기간** | 1-4주 | 2-4주 | 1-3개월 | 30분~1시간 |
| **종목 수** | 1개 (최적) | 1개 (단기 추천과 중복 제외) | 최대 3개 | 최대 3개 |
| **목표가/손절가** | 칼만+볼린저 블렌딩 / ATR 기반 | 칼만+볼린저 블렌딩 / ATR 기반 | — | ATR 기반 동적 (종목별 변동성 반영) |
| **하드 필터** | 칼만 수익률 ≥ 10%, falling knife 제외, 최소 점수 점진적 하향(35→10) | 칼만 예측가 > 종가, 최소 점수 이상 | Kalman velocity > 0, RSI < 75 | ADX>30+bearish 제외, RSI>85 제외 |
| **리포트** | 일일 리포트 | 일일 리포트 | 일일 리포트 | 프리마켓 리포트 |
| **자동 튜닝** | 지원 (min_samples=15) | 지원 (Enhanced 가중치 맵 재사용) | 지원 (min_samples=5) | 지원 (min_samples=10) |

## 기술적 지표 설명

리포트에 사용되는 12개 기술적 지표(RSI, MACD, 볼린저밴드, ADX, ATR, 거래량, 상대강도, 52주 위치, 칼만 필터, OBV, Stochastic, TTM Squeeze)의 계산 방식, 값별 해석, 점수 배점 상세는 GitHub Pages에서 확인할 수 있습니다:

**[기술적 지표 설명 보기](https://younhs.github.io/us_stock_report/)**

> GitHub Pages 배포: repo Settings → Pages → Source를 `docs/` 폴더로 설정

## 성과 추적 대시보드

추천 종목의 실제 성과를 자동으로 추적하고, 평가 결과를 기반으로 파라미터를 자동 튜닝합니다.

일일 리포트 (`main.py`) 실행 시:
1. **이전 추천 평가** — 보유 기간이 경과한 pending 종목(Enhanced/Kalman/Long-term/Opening Surge)의 실제 가격을 조회하여 win/loss/expired 판정
2. **오늘의 추천 기록** — 새 추천 종목을 진입가, 목표가, 손절가, `score_breakdown`과 함께 기록
3. **통계 요약 생성** — 승률, 평균 수익률, profit factor 등 통계 계산
4. **파라미터 자동 튜닝** — 평가 완료된 추천의 지표별 효과성(win-rate lift + return lift)을 분석하여 가중치/임계값 조정 → `tuned_params.json` 저장 → 다음 실행 시 적용

프리마켓 리포트 (`main_premarket.py`) 실행 시:
- **개장 급등 추천 기록** — Opening Surge 추천을 `recommendations.json`에 기록 (PM가격 기반, `holding_period_days=1`)

**[Performance Dashboard 보기](https://younhs.github.io/us_stock_report/dashboard.html)**

대시보드 기능:
- 전체 요약 카드 (총 추천, 승률, 평균 수익률, Profit Factor)
- 방식별 비교 카드 (Enhanced / Kalman / Long-term)
- 누적 수익률 라인 차트 (Chart.js)
- 필터링 가능한 추천 기록 테이블 (방식, 결과, 기간)

> 데이터는 GitHub Actions에서 `docs/data/` 폴더에 자동 커밋됩니다. 90일 보관 후 완료 항목은 자동 정리됩니다.

### 자동 튜닝 데이터 흐름

```
일일 (main.py, KST 06:55):
  평가 → 기록 → 요약 → 튜닝 → 자동 커밋
  (tuned_params.json → 다음 실행 시 signals.py가 로드)

프리마켓 (main_premarket.py, KST 21:00):
  리포트 생성 → Opening Surge 기록 → 자동 커밋
  (tuned_params.json 로드 → 튜닝된 가중치로 추천 생성)
```

## 기술 스택

- **Python 3.12**
- **yfinance** - 주가 데이터 수집
- **pandas / numpy** - 데이터 처리
- **Jinja2** - HTML 리포트 템플릿
- **pydantic-settings** - 환경변수 관리
- **beautifulsoup4** - S&P 500 종목 스크래핑
- **nltk (VADER)** - 뉴스 헤드라인 감성 분석
- **slack-sdk** - Slack Bot 메시지/파일 발송
- **Google Chrome (headless)** - HTML → PDF 변환
- **fonts-noto-cjk** - PDF 한글 렌더링
- **Chart.js** (CDN) - 성과 대시보드 누적 수익률 차트

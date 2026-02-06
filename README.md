# US Stock Report

S&P 500 기반 미국 주식 시장 일일 분석 리포트를 자동 생성하여 Slack으로 발송하는 Python 시스템입니다.

## 주요 기능

- **시장 요약**: SPY, QQQ, DIA, IWM 지수 현황
- **기술적 분석**: RSI, MACD, 볼린저밴드, ADX, ATR, 거래량, 상대강도, 52주 위치
- **매수 신호 감지**: RSI 과매도, MACD 골든크로스, 1시그마 근접, 복합 신호
- **단기 추천 종목**: 가중치 점수 시스템 기반 평균 회귀 추천 (손절가/목표가 포함)
- **장기 투자 추천**: 추세 추종(Trend-Following) 기반 장기 투자 Top 3 종목 추천
- **섹터별 트렌드**: 11개 GICS 섹터 분석, Top 10 상승/하락 종목
- **뉴스 및 캘린더**: 핫 종목 뉴스, 경제 캘린더, 실적 발표 일정

## 프로젝트 구조

```
us_stock_report/
├── main.py                      # 진입점
├── config/
│   ├── settings.py              # Pydantic 기반 환경변수 관리
│   └── sp500_tickers.py         # S&P 500 종목 스크래핑
├── data/
│   ├── fetcher.py               # yfinance 주가 데이터 수집
│   └── calendar.py              # 경제/실적 캘린더
├── analysis/
│   ├── technical.py             # 기술적 분석 (RSI, MACD, BB 등)
│   ├── signals.py               # 매수 신호 감지 및 추천
│   └── sector.py                # 섹터별 분석
├── news/
│   └── fetcher.py               # 뉴스 수집
├── report/
│   ├── generator.py             # Jinja2 리포트 생성
│   └── templates/
│       └── daily_report.html    # HTML 리포트 템플릿
├── notification/
│   └── slack_sender.py          # Slack 발송 (요약 메시지 + PDF 리포트)
├── .github/workflows/
│   └── run_main.yml             # GitHub Actions 스케줄링
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
LONGTERM_WEIGHT_ADX=20              # ADX 가중치 (기본: 20)
LONGTERM_WEIGHT_MACD=15             # MACD 가중치 (기본: 15)
LONGTERM_WEIGHT_RELATIVE_STRENGTH=15 # 상대강도 가중치 (기본: 15)
```

## 실행

```bash
# 전체 실행 (리포트 생성 + Slack 발송)
python main.py

# 발송 없이 리포트만 생성
python main.py --dry-run

# 테스트 Slack 메시지 발송
python main.py --test-slack
```

## 자동 스케줄링

### GitHub Actions (권장)

`.github/workflows/run_main.yml`로 월-금 KST 06:55에 자동 실행됩니다.
워크플로우에서 한글 폰트(`fonts-noto-cjk`)를 자동 설치합니다 (Chrome은 `ubuntu-latest`에 기본 포함).

GitHub Repository에 Secrets를 등록해야 합니다:

> Settings > Secrets and variables > Actions > New repository secret

| Secret Name | 값 |
|---|---|
| `SLACK_BOT_TOKEN` | Slack Bot Token (`xoxb-...`) |
| `SLACK_CHANNEL` | Slack 채널명 |

### cron (로컬 서버)

```bash
# KST 07:00 = UTC 22:00, 월-금
0 22 * * 1-5 /path/to/venv/bin/python /path/to/main.py
```

## 리포트 구성

1. 시장 요약 (SPY, QQQ, DIA, IWM)
2. 주요 뉴스 (핫 종목/섹터 하이라이트)
3. 경제 캘린더 (FOMC, ISM PMI, 고용지표 등)
4. 실적 발표 캘린더 (주요 종목 2주간 일정)
5. 섹터별 트렌드 (11개 GICS 섹터)
6. 기술적 분석 신호 (1시그마 근접, RSI 과매도, MACD 골든크로스, 복합 신호)
7. Top 10 상승/하락
8. 오늘의 추천 종목 — 단기 평균 회귀 전략 (보유 기간 1-4주)
9. 장기 투자 추천 Top 3 — 추세 추종 전략 (보유 기간 1-3개월)

### 추천 전략 비교

| | 단기 추천 | 장기 추천 |
|---|---|---|
| **전략** | 평균 회귀 (Mean Reversion) | 추세 추종 (Trend Following) |
| **핵심 조건** | RSI 과매도, 볼린저 하단 근접 | Kalman 상승 추세, ADX bullish |
| **보유 기간** | 1-4주 | 1-3개월 |
| **종목 수** | 1개 (최적) | 최대 3개 |
| **하드 필터** | R:R >= 2:1, falling knife 제외 | Kalman velocity > 0, RSI < 75 |

## 기술 스택

- **Python 3.12**
- **yfinance** - 주가 데이터 수집
- **pandas / numpy** - 데이터 처리
- **Jinja2** - HTML 리포트 템플릿
- **pydantic-settings** - 환경변수 관리
- **beautifulsoup4** - S&P 500 종목 스크래핑
- **slack-sdk** - Slack Bot 메시지/파일 발송
- **Google Chrome (headless)** - HTML → PDF 변환
- **fonts-noto-cjk** - PDF 한글 렌더링

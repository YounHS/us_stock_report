# US Stock Report

S&P 500 기반 미국 주식 시장 일일 분석 리포트를 자동 생성하여 이메일로 발송하는 Python 시스템입니다.

## 주요 기능

- **시장 요약**: SPY, QQQ, DIA, IWM 지수 현황
- **기술적 분석**: RSI, MACD, 볼린저밴드, ADX, ATR, 거래량, 상대강도, 52주 위치
- **매수 신호 감지**: RSI 과매도, MACD 골든크로스, 1시그마 근접, 복합 신호
- **추천 종목 선정**: 가중치 점수 시스템 기반 종목 추천 (손절가/목표가 포함)
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
│       └── daily_report.html    # 이메일 HTML 템플릿
├── notification/
│   └── email_sender.py          # SMTP 이메일 발송
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
```

## 환경 설정

`.env` 파일을 프로젝트 루트에 생성합니다:

```env
# SMTP
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password

# Email
EMAIL_FROM=your_email@gmail.com
EMAIL_RECIPIENTS=recipient1@gmail.com,recipient2@gmail.com
```

> Gmail 사용 시 일반 비밀번호가 아닌 **앱 비밀번호**가 필요합니다.
> Google 계정 > 보안 > 2단계 인증 > 앱 비밀번호에서 생성할 수 있습니다.

기술적 분석 파라미터도 `.env`에서 커스터마이징 가능합니다 (`config/settings.py` 참고).

## 실행

```bash
# 전체 실행 (리포트 생성 + 이메일 발송)
python main.py

# 이메일 발송 없이 리포트만 생성
python main.py --dry-run

# 테스트 이메일 발송
python main.py --test-email
```

## 자동 스케줄링

### GitHub Actions (권장)

`.github/workflows/run_main.yml`로 월-금 KST 06:55에 자동 실행됩니다.

GitHub Repository에 Secrets를 등록해야 합니다:

> Settings > Secrets and variables > Actions > New repository secret

| Secret Name | 값 |
|---|---|
| `SMTP_USER` | SMTP 로그인 이메일 |
| `SMTP_PASSWORD` | SMTP 앱 비밀번호 |
| `EMAIL_FROM` | 발신자 이메일 |
| `EMAIL_RECIPIENTS` | 수신자 이메일 (쉼표 구분) |

### cron (로컬 서버)

```bash
# KST 07:00 = UTC 22:00, 월-금
0 22 * * 1-5 /path/to/venv/bin/python /path/to/main.py
```

## 기술 스택

- **Python 3.12**
- **yfinance** - 주가 데이터 수집
- **pandas / numpy** - 데이터 처리
- **Jinja2** - HTML 리포트 템플릿
- **pydantic-settings** - 환경변수 관리
- **beautifulsoup4** - S&P 500 종목 스크래핑

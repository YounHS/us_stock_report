# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

US Stock Report는 매일 아침 S&P 500 기반 미국 주식 시장 분석 리포트를 이메일로 자동 발송하는 Python 시스템입니다.

## Commands

```bash
# 환경 설정
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # SMTP 설정 필요 (또는 직접 .env 작성)

# 실행
python main.py              # 전체 실행 (리포트 생성 + 이메일 발송)
python main.py --dry-run    # 이메일 발송 없이 리포트만 생성
python main.py --test-email # 테스트 이메일 발송

# cron 설정 (KST 07:00 = UTC 22:00)
0 22 * * 1-5 /path/to/venv/bin/python /path/to/main.py
```

### GitHub Actions 스케줄링

`.github/workflows/run_main.yml`로 자동 실행 (KST 06:55 = UTC 21:55, 월-금).
민감 정보는 **GitHub Repository Secrets**에 등록하여 워크플로우 `env:` 블록으로 주입:
- `SMTP_USER`, `SMTP_PASSWORD`, `EMAIL_FROM`, `EMAIL_RECIPIENTS`
- Settings → Secrets and variables → Actions → New repository secret

## Architecture

데이터 흐름: `main.py` → 데이터 수집 → 기술적 분석 → 신호 감지 → 캘린더/뉴스 수집 → 리포트 생성 → 이메일 발송

### 핵심 모듈

- **config/settings.py**: Pydantic 기반 환경변수 관리. `settings` 싱글톤으로 전역 접근
- **config/sp500_tickers.py**: Wikipedia에서 S&P 500 종목 스크래핑 (User-Agent 필수), 11개 GICS 섹터 매핑
- **data/fetcher.py**: `yfinance`로 OHLCV 배치 다운로드. `StockDataFetcher.fetch_batch()` 사용
- **data/calendar.py**: 경제 캘린더(`EconomicCalendar`) 및 실적 발표 일정(`EarningsCalendar`) 수집
- **analysis/technical.py**: RSI, MACD, 볼린저밴드, ADX, ATR, 거래량, 상대강도, 52주 위치 계산. 결과는 dataclass로 반환 (`RSIResult`, `MACDResult`, `BollingerResult`, `ADXResult`, `ATRResult`, `VolumeResult`, `RelativeStrengthResult`, `Week52Result`)
- **analysis/signals.py**: `SignalDetector`가 분석 결과에서 매수 신호 감지. 두 가지 추천 방식 제공:
  - `get_enhanced_recommendation()`: 가중치 점수 시스템 기반 (우선 사용)
  - `get_top_recommendation()`: 레거시 방식 (Enhanced 실패 시 폴백)
- **news/fetcher.py**: 핫한 종목/섹터 뉴스 수집. `fetch_hot_stocks_news()`, `fetch_sector_highlights()`
- **report/generator.py**: Jinja2로 `report/templates/daily_report.html` 렌더링

### 리포트 구성

1. 시장 요약 (SPY, QQQ, DIA, IWM)
2. 주요 뉴스 (핫한 종목/섹터 뉴스)
3. 경제 캘린더 (FOMC, ISM PMI, 고용지표 등)
4. 실적 발표 캘린더 (주요 종목 2주간 일정)
5. 섹터별 트렌드
6. 기술적 분석 신호 (1시그마 근접, RSI 과매도, MACD 골든크로스, 복합 신호)
7. Top 10 상승/하락
8. 오늘의 추천 종목 (RSI, ADX, MACD, BB Z-Score, 거래량, ATR%, SPY대비, 52주위치, 매도 목표가, 손절가, 보유 기간)

### 기술적 분석 파라미터

모든 파라미터는 `.env`에서 설정 가능:
- RSI: 14일 기준, 30 미만 = 과매도
- MACD: 12/26/9 EMA
- 볼린저밴드: 20일 SMA ± 2σ, Z-score로 1시그마 근접 판별 (-1.2 ≤ z ≤ -0.8)

### 추천 종목 선정 로직

**Enhanced 방식** (`get_enhanced_recommendation()`):
- 가중치 점수 시스템으로 종목 순위 산정
- 최소 점수(`min_recommendation_score`) 이상만 추천 대상
- 70점 이상: High 신뢰도, 그 외: Medium 신뢰도
- ATR 기반 손절가/목표가 계산

**Legacy 방식** (`get_top_recommendation()`):
- Enhanced 실패 시 폴백으로 사용
- 우선순위: 복합 신호 → RSI 과매도 → MACD 골든크로스 → 1시그마 근접
- 매도 목표가는 20일 SMA 기준

**추천에 포함되는 기술적 지표**:
- RSI, ADX, MACD (골든크로스 여부), 볼린저밴드 Z-Score
- 거래량 비율, ATR%, SPY 대비 상대강도(20일), 52주 위치
- 손절가, 리스크:리워드 비율

### 설정 구조

`config/settings.py`의 `Settings` 클래스가 4개 설정 그룹 관리:
- `settings.smtp`: SMTP 서버 정보
- `settings.email`: 수신자, 제목 등
- `settings.analysis`: 기술적 분석 파라미터
- `settings.general`: 타임존, 로그 레벨 등

### 리포트 템플릿 구조

`report/templates/daily_report.html`은 Jinja2 템플릿으로, 추천 종목 섹션에서 다음 필드들을 사용:

```python
recommendation = {
    "ticker", "close", "change_pct",           # 기본 정보
    "rsi", "adx", "volume_ratio",              # 기술적 지표
    "macd_signal", "bollinger_z_score",        # MACD/볼린저
    "atr_pct", "relative_strength_20d",        # ATR/상대강도
    "week52_position",                         # 52주 위치
    "target_price", "target_return",           # 목표가
    "stop_loss", "risk_reward_ratio",          # 리스크 관리 (Enhanced만)
    "score", "confidence", "score_breakdown",  # 점수 시스템 (Enhanced만)
    "bullish_factors", "warning_factors",      # 매수/주의 요인 (Enhanced만)
    "reasons", "holding_period", "source",     # 추천 근거
}
```

템플릿에서 `is not none` 체크로 None 값 처리. Enhanced 전용 필드는 Legacy 사용 시 main.py에서 기본값 설정.

## 주의사항

- **Wikipedia 스크래핑**: `config/sp500_tickers.py`에서 User-Agent 헤더 필수 (403 방지)
- **Gmail SMTP**: 앱 비밀번호 필요 (일반 비밀번호 사용 불가)
- **yfinance 캐시**: `~/.cache/py-yfinance` 폴더 권한 문제 발생 시 무시 가능
- **템플릿 호환성**: 추천 dict 변환 시 템플릿에서 사용하는 모든 필드 포함 필요. Enhanced/Legacy 모두 `macd_signal`, `bollinger_z_score`, `atr_pct`, `target_return`, `reasons` 필드가 있어야 함
- **GitHub Actions 환경변수**: `.env` 파일은 로컬 전용. GitHub Actions에서는 Repository Secrets → 워크플로우 `env:` 블록으로 주입

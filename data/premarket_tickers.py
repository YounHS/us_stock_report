"""프리마켓 리포트 대상 종목 관리"""

import json
import logging
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List

from config.settings import settings
from config.sp500_tickers import MARKET_ETFS, SECTOR_ETFS
from data.calendar import EarningsCalendar

logger = logging.getLogger(__name__)

# 섹터 ETF 리스트
SECTOR_ETFS_LIST = list(SECTOR_ETFS.values())

# 추천 종목 상태 파일 경로
RECOMMENDATIONS_FILE = Path(__file__).parent.parent / "last_recommendations.json"


def save_recommendations(tickers: List[str]) -> None:
    """
    추천 종목 리스트를 파일에 저장 (main.py에서 호출)

    Args:
        tickers: 추천 종목 티커 리스트
    """
    tz = ZoneInfo(settings.general.timezone)
    data = {
        "tickers": tickers,
        "saved_at": datetime.now(tz).isoformat(),
    }
    RECOMMENDATIONS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    logger.info(f"추천 종목 {len(tickers)}개 저장: {RECOMMENDATIONS_FILE}")


def load_previous_recommendations() -> List[str]:
    """
    전일 추천 종목 로드

    Returns:
        추천 종목 티커 리스트 (없으면 빈 리스트)
    """
    if not RECOMMENDATIONS_FILE.exists():
        logger.info("전일 추천 종목 파일 없음")
        return []

    try:
        data = json.loads(RECOMMENDATIONS_FILE.read_text())
        tickers = data.get("tickers", [])
        saved_at = data.get("saved_at", "")
        logger.info(f"전일 추천 종목 {len(tickers)}개 로드 (저장: {saved_at})")
        return tickers
    except Exception as e:
        logger.warning(f"전일 추천 종목 로드 실패: {e}")
        return []


def get_todays_earnings_tickers() -> List[str]:
    """
    오늘 실적 발표 예정 종목 조회

    Returns:
        오늘 실적 발표 종목 티커 리스트
    """
    try:
        cal = EarningsCalendar()
        earnings = cal.fetch_earnings_calendar()

        tz = ZoneInfo(settings.general.timezone)
        today_str = datetime.now(tz).strftime("%Y-%m-%d")

        tickers = [e["ticker"] for e in earnings if e.get("date") == today_str]
        logger.info(f"오늘 실적 발표 종목: {len(tickers)}개")
        return tickers
    except Exception as e:
        logger.warning(f"실적 발표 종목 조회 실패: {e}")
        return []


def get_premarket_tickers() -> List[str]:
    """
    프리마켓 리포트 대상 종목 합산

    Returns:
        정렬된 티커 리스트 (중복 제거)
    """
    all_tickers = set()

    # 1. 시장 ETF (4개)
    all_tickers.update(MARKET_ETFS)

    # 2. 섹터 ETF (11개)
    all_tickers.update(SECTOR_ETFS_LIST)

    # 3. 전일 추천 종목
    prev_recs = load_previous_recommendations()
    all_tickers.update(prev_recs)

    # 4. 오늘 실적 발표 종목
    earnings = get_todays_earnings_tickers()
    all_tickers.update(earnings)

    result = sorted(all_tickers)
    logger.info(f"프리마켓 대상 종목 합계: {len(result)}개 "
                f"(시장 {len(MARKET_ETFS)} + 섹터 {len(SECTOR_ETFS_LIST)} "
                f"+ 추천 {len(prev_recs)} + 실적 {len(earnings)})")
    return result

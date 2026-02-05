"""경제 캘린더 및 실적 발표 일정 모듈"""

import yfinance as yf
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional
import logging
from bs4 import BeautifulSoup

from config.settings import settings

logger = logging.getLogger(__name__)


def _get_now():
    """타임존이 적용된 현재 시각 반환"""
    tz = ZoneInfo(settings.general.timezone)
    return datetime.now(tz)


class EconomicCalendar:
    """경제 지표 발표 캘린더"""

    # 주요 경제 지표 (수동 관리 - 정기적으로 업데이트 필요)
    RECURRING_EVENTS = {
        "FOMC 금리 결정": {"frequency": "6주", "importance": "high"},
        "비농업 고용지표": {"frequency": "매월 첫째 금요일", "importance": "high"},
        "CPI (소비자물가지수)": {"frequency": "매월", "importance": "high"},
        "ISM 제조업 PMI": {"frequency": "매월 첫째 영업일", "importance": "high"},
        "ISM 서비스업 PMI": {"frequency": "매월", "importance": "medium"},
        "GDP 성장률": {"frequency": "분기", "importance": "high"},
        "소매판매": {"frequency": "매월", "importance": "medium"},
        "실업수당 청구건수": {"frequency": "매주 목요일", "importance": "medium"},
    }

    def __init__(self):
        self.events = []

    def fetch_economic_calendar(self) -> List[Dict]:
        """
        경제 캘린더 데이터 가져오기
        Investing.com 또는 다른 소스에서 스크래핑
        """
        try:
            # 이번 주 + 다음 주 이벤트 가져오기
            events = self._fetch_from_investing_com()
            if events:
                return events
        except Exception as e:
            logger.warning(f"경제 캘린더 스크래핑 실패: {e}")

        # 실패 시 기본 이벤트 반환
        return self._get_default_events()

    def _fetch_from_investing_com(self) -> List[Dict]:
        """Investing.com에서 경제 캘린더 스크래핑"""
        url = "https://www.investing.com/economic-calendar/"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                return []

            soup = BeautifulSoup(response.text, 'html.parser')
            events = []

            # 미국 관련 중요 이벤트만 추출
            rows = soup.select('tr.js-event-item')

            for row in rows[:20]:  # 최대 20개
                try:
                    country = row.get('data-country', '')
                    if country != 'US' and country != '5':  # 미국만
                        continue

                    # 중요도 확인 (별 개수)
                    importance = len(row.select('.grayFullBullishIcon'))
                    if importance < 2:  # 2성급 이상만
                        continue

                    time_elem = row.select_one('.time')
                    event_elem = row.select_one('.event a')
                    date_elem = row.get('data-event-datetime', '')

                    if event_elem:
                        events.append({
                            "date": date_elem[:10] if date_elem else _get_now().strftime("%Y-%m-%d"),
                            "time": time_elem.text.strip() if time_elem else "",
                            "event": event_elem.text.strip(),
                            "importance": "high" if importance >= 3 else "medium",
                        })
                except Exception:
                    continue

            return events

        except Exception as e:
            logger.warning(f"Investing.com 스크래핑 실패: {e}")
            return []

    def _get_default_events(self) -> List[Dict]:
        """기본 경제 이벤트 (스크래핑 실패 시)"""
        today = _get_now()
        events = []

        # 이번 주의 주요 이벤트 생성
        week_events = [
            {"day_offset": 0, "event": "주간 실업수당 청구건수", "importance": "medium"},
            {"day_offset": 1, "event": "ISM 제조업 PMI", "importance": "high"},
            {"day_offset": 3, "event": "비농업 고용지표", "importance": "high"},
        ]

        for evt in week_events:
            event_date = today + timedelta(days=evt["day_offset"])
            # 주말 건너뛰기
            while event_date.weekday() >= 5:
                event_date += timedelta(days=1)

            events.append({
                "date": event_date.strftime("%Y-%m-%d"),
                "day": self._get_day_name(event_date.weekday()),
                "event": evt["event"],
                "importance": evt["importance"],
            })

        return events

    def _get_day_name(self, weekday: int) -> str:
        """요일 이름 반환"""
        days = ["월", "화", "수", "목", "금", "토", "일"]
        return days[weekday]

    def get_week_calendar(self) -> Dict:
        """이번 주 경제 캘린더 반환"""
        today = _get_now()
        start_of_week = today - timedelta(days=today.weekday())

        events = self.fetch_economic_calendar()

        # 날짜별로 그룹화
        calendar = {}
        for i in range(7):
            day = start_of_week + timedelta(days=i)
            day_str = day.strftime("%Y-%m-%d")
            day_name = self._get_day_name(day.weekday())

            calendar[day_str] = {
                "day_name": day_name,
                "date_display": day.strftime("%m/%d"),
                "is_today": day.date() == today.date(),
                "events": []
            }

        for event in events:
            event_date = event.get("date", "")
            if event_date in calendar:
                calendar[event_date]["events"].append(event)

        return calendar


class EarningsCalendar:
    """실적 발표 캘린더"""

    # 주요 종목 리스트 (시가총액 상위)
    MAJOR_STOCKS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
        "JPM", "V", "JNJ", "WMT", "MA", "PG", "HD", "CVX", "MRK", "ABBV",
        "PEP", "KO", "COST", "TMO", "AVGO", "MCD", "WFC", "CSCO", "ACN",
        "ABT", "DHR", "LIN", "NEE", "TXN", "PM", "UNP", "RTX", "LOW"
    ]

    def __init__(self):
        pass

    def fetch_earnings_calendar(self, tickers: Optional[List[str]] = None) -> List[Dict]:
        """
        실적 발표 일정 가져오기

        Args:
            tickers: 조회할 티커 리스트 (없으면 주요 종목)

        Returns:
            실적 발표 일정 리스트
        """
        tickers = tickers or self.MAJOR_STOCKS
        earnings = []

        today = _get_now()
        next_week = today + timedelta(days=14)

        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                calendar = stock.calendar

                if calendar is None or calendar.empty:
                    continue

                # earnings date 추출
                if 'Earnings Date' in calendar.index:
                    earnings_dates = calendar.loc['Earnings Date']
                    if hasattr(earnings_dates, '__iter__') and not isinstance(earnings_dates, str):
                        for ed in earnings_dates:
                            if ed and today <= ed <= next_week:
                                earnings.append({
                                    "ticker": ticker,
                                    "date": ed.strftime("%Y-%m-%d"),
                                    "day": self._get_day_name(ed.weekday()),
                                })
                                break
                    elif earnings_dates:
                        ed = earnings_dates
                        if hasattr(ed, 'strftime') and today <= ed <= next_week:
                            earnings.append({
                                "ticker": ticker,
                                "date": ed.strftime("%Y-%m-%d"),
                                "day": self._get_day_name(ed.weekday()),
                            })

            except Exception as e:
                logger.debug(f"{ticker} 실적 일정 조회 실패: {e}")
                continue

        # 날짜순 정렬
        earnings.sort(key=lambda x: x["date"])
        return earnings

    def _get_day_name(self, weekday: int) -> str:
        """요일 이름 반환"""
        days = ["월", "화", "수", "목", "금", "토", "일"]
        return days[weekday]

    def get_week_calendar(self) -> Dict:
        """이번 주 + 다음 주 실적 캘린더"""
        today = _get_now()
        start_of_week = today - timedelta(days=today.weekday())

        earnings = self.fetch_earnings_calendar()

        # 날짜별로 그룹화 (2주)
        calendar = {}
        for i in range(14):
            day = start_of_week + timedelta(days=i)
            day_str = day.strftime("%Y-%m-%d")
            day_name = self._get_day_name(day.weekday())

            calendar[day_str] = {
                "day_name": day_name,
                "date_display": day.strftime("%m/%d"),
                "is_today": day.date() == today.date(),
                "earnings": []
            }

        for earning in earnings:
            earning_date = earning.get("date", "")
            if earning_date in calendar:
                calendar[earning_date]["earnings"].append(earning["ticker"])

        return calendar

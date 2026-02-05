"""뉴스 수집 모듈"""

import yfinance as yf
from typing import List, Dict
from datetime import datetime
from zoneinfo import ZoneInfo
import logging

from config.settings import settings
from config.sp500_tickers import MARKET_ETFS, SECTOR_ETFS

logger = logging.getLogger(__name__)


class NewsFetcher:
    """시장 뉴스 수집기"""

    def __init__(self):
        self.max_items = settings.general.max_news_items

    def fetch_ticker_news(self, ticker: str) -> List[Dict]:
        """특정 티커의 뉴스 수집"""
        try:
            stock = yf.Ticker(ticker)
            news = stock.news or []

            result = []
            for item in news:
                result.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "publisher": item.get("publisher", ""),
                    "published": self._format_timestamp(item.get("providerPublishTime", 0)),
                    "source_ticker": ticker,
                })

            return result
        except Exception as e:
            logger.warning(f"{ticker} 뉴스 수집 실패: {e}")
            return []

    def fetch_market_news(self) -> List[Dict]:
        """
        주요 시장 지수 관련 뉴스 수집
        SPY, QQQ 등에서 뉴스 수집
        """
        all_news = []
        seen_titles = set()

        # 시장 ETF 뉴스
        for etf in MARKET_ETFS[:2]:  # SPY, QQQ만
            news = self.fetch_ticker_news(etf)
            for item in news:
                if item["title"] not in seen_titles:
                    seen_titles.add(item["title"])
                    all_news.append(item)

        # 시간순 정렬 (최신 먼저)
        all_news.sort(key=lambda x: x["published"], reverse=True)
        return all_news[:self.max_items]

    def fetch_sector_news(self) -> Dict[str, List[Dict]]:
        """
        섹터별 뉴스 수집
        각 섹터 ETF에서 뉴스 수집
        """
        sector_news = {}

        for sector_name, etf in SECTOR_ETFS.items():
            news = self.fetch_ticker_news(etf)
            if news:
                sector_news[sector_name] = news[:3]  # 섹터당 최대 3개

        return sector_news

    def fetch_hot_stocks_news(self, top_gainers: list, top_losers: list) -> List[Dict]:
        """
        핫한 종목(상위 상승/하락) 뉴스 수집

        Args:
            top_gainers: 상위 상승 종목 리스트
            top_losers: 상위 하락 종목 리스트

        Returns:
            뉴스 리스트
        """
        all_news = []
        seen_titles = set()

        # 상위 5개 상승/하락 종목 뉴스
        hot_tickers = []
        for item in top_gainers[:5]:
            hot_tickers.append({"ticker": item.get("ticker", item), "type": "gainer"})
        for item in top_losers[:5]:
            hot_tickers.append({"ticker": item.get("ticker", item), "type": "loser"})

        for item in hot_tickers:
            ticker = item["ticker"]
            news = self.fetch_ticker_news(ticker)
            for n in news[:2]:  # 종목당 최대 2개
                if n["title"] not in seen_titles:
                    seen_titles.add(n["title"])
                    n["stock_type"] = "상승" if item["type"] == "gainer" else "하락"
                    all_news.append(n)

        # 시간순 정렬
        all_news.sort(key=lambda x: x["published"], reverse=True)
        return all_news[:self.max_items]

    def fetch_sector_highlights(self, sector_performance: list) -> List[Dict]:
        """
        핫한 섹터 뉴스 수집 (상위 강세/약세 섹터)

        Args:
            sector_performance: 섹터 성과 리스트

        Returns:
            섹터 뉴스 리스트
        """
        all_news = []
        seen_titles = set()

        if not sector_performance:
            return []

        # 상위 2개 강세/약세 섹터
        sorted_sectors = sorted(sector_performance, key=lambda x: x.daily_return, reverse=True)
        hot_sectors = sorted_sectors[:2] + sorted_sectors[-2:]

        for sector in hot_sectors:
            etf = SECTOR_ETFS.get(sector.name)
            if etf:
                news = self.fetch_ticker_news(etf)
                for n in news[:2]:
                    if n["title"] not in seen_titles:
                        seen_titles.add(n["title"])
                        n["sector"] = sector.name
                        n["sector_return"] = sector.daily_return
                        all_news.append(n)

        all_news.sort(key=lambda x: x["published"], reverse=True)
        return all_news[:6]

    def get_all_news(self, top_movers: Dict = None, sector_performance: list = None) -> Dict:
        """
        모든 뉴스 종합

        Args:
            top_movers: 상위 상승/하락 종목 딕셔너리
            sector_performance: 섹터 성과 리스트

        Returns:
            {
                "market_news": [...],
                "hot_stocks_news": [...],
                "sector_highlights": [...],
                "fetched_at": "..."
            }
        """
        result = {
            "market_news": self.fetch_market_news(),
            "hot_stocks_news": [],
            "sector_highlights": [],
            "fetched_at": datetime.now(ZoneInfo(settings.general.timezone)).strftime("%Y-%m-%d %H:%M:%S"),
        }

        if top_movers:
            result["hot_stocks_news"] = self.fetch_hot_stocks_news(
                top_movers.get("top_gainers", []),
                top_movers.get("top_losers", [])
            )

        if sector_performance:
            result["sector_highlights"] = self.fetch_sector_highlights(sector_performance)

        return result

    def _format_timestamp(self, timestamp: int) -> str:
        """Unix timestamp를 문자열로 변환"""
        if not timestamp:
            return ""
        try:
            dt = datetime.fromtimestamp(timestamp)
            return dt.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return ""

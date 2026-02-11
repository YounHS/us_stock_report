"""프리마켓 가격 데이터 수집"""

import yfinance as yf
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class PreMarketData:
    """프리마켓 가격 데이터"""
    ticker: str
    pre_market_price: Optional[float]
    pre_market_change: Optional[float]
    pre_market_change_pct: Optional[float]
    regular_previous_close: Optional[float]
    has_premarket: bool


class PreMarketFetcher:
    """프리마켓 가격 수집기 (yf.Ticker.info 기반)"""

    def fetch_single(self, ticker: str) -> PreMarketData:
        """
        단일 종목의 프리마켓 가격 조회

        Args:
            ticker: 티커 심볼

        Returns:
            PreMarketData
        """
        try:
            info = yf.Ticker(ticker).info
            pm_price = info.get("preMarketPrice")
            prev_close = info.get("regularMarketPreviousClose")

            if pm_price is not None and prev_close is not None and prev_close > 0:
                change = round(pm_price - prev_close, 2)
                change_pct = round((change / prev_close) * 100, 2)
                return PreMarketData(
                    ticker=ticker,
                    pre_market_price=round(pm_price, 2),
                    pre_market_change=change,
                    pre_market_change_pct=change_pct,
                    regular_previous_close=round(prev_close, 2),
                    has_premarket=True,
                )

            return PreMarketData(
                ticker=ticker,
                pre_market_price=None,
                pre_market_change=None,
                pre_market_change_pct=None,
                regular_previous_close=round(prev_close, 2) if prev_close else None,
                has_premarket=False,
            )

        except Exception as e:
            logger.warning(f"{ticker} 프리마켓 데이터 조회 실패: {e}")
            return PreMarketData(
                ticker=ticker,
                pre_market_price=None,
                pre_market_change=None,
                pre_market_change_pct=None,
                regular_previous_close=None,
                has_premarket=False,
            )

    def fetch_batch(self, tickers: List[str]) -> Dict[str, PreMarketData]:
        """
        여러 종목의 프리마켓 가격 배치 조회

        Args:
            tickers: 티커 심볼 리스트

        Returns:
            {ticker: PreMarketData} 딕셔너리
        """
        logger.info(f"{len(tickers)}개 종목 프리마켓 데이터 수집 시작...")
        result = {}

        for ticker in tickers:
            data = self.fetch_single(ticker)
            result[ticker] = data

        pm_count = sum(1 for d in result.values() if d.has_premarket)
        logger.info(f"프리마켓 데이터 수집 완료: {pm_count}/{len(tickers)}개 프리마켓 가격 확인")
        return result

    def get_significant_movers(
        self, data: Dict[str, PreMarketData], threshold_pct: float = 1.0
    ) -> Dict[str, List[PreMarketData]]:
        """
        프리마켓 주요 변동 종목 필터링

        Args:
            data: 프리마켓 데이터 딕셔너리
            threshold_pct: 변동률 기준 (기본 1.0%)

        Returns:
            {"gainers": [...], "losers": [...]}
        """
        gainers = []
        losers = []

        for ticker, pm in data.items():
            if not pm.has_premarket or pm.pre_market_change_pct is None:
                continue

            if pm.pre_market_change_pct >= threshold_pct:
                gainers.append(pm)
            elif pm.pre_market_change_pct <= -threshold_pct:
                losers.append(pm)

        gainers.sort(key=lambda x: x.pre_market_change_pct, reverse=True)
        losers.sort(key=lambda x: x.pre_market_change_pct)

        return {"gainers": gainers, "losers": losers}

"""프리마켓 가격 데이터 수집"""

import yfinance as yf
import pandas as pd
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
    pre_market_volume: Optional[int] = None
    regular_market_volume: Optional[int] = None
    average_volume: Optional[int] = None
    average_volume_10d: Optional[int] = None


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

            # 거래량 정보 추출
            pm_volume = info.get("preMarketVolume")
            reg_volume = info.get("regularMarketVolume")
            avg_volume = info.get("averageVolume")
            avg_vol_10d = info.get("averageDailyVolume10Day")

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
                    pre_market_volume=pm_volume,
                    regular_market_volume=reg_volume,
                    average_volume=avg_volume,
                    average_volume_10d=avg_vol_10d,
                )

            return PreMarketData(
                ticker=ticker,
                pre_market_price=None,
                pre_market_change=None,
                pre_market_change_pct=None,
                regular_previous_close=round(prev_close, 2) if prev_close else None,
                has_premarket=False,
                pre_market_volume=pm_volume,
                regular_market_volume=reg_volume,
                average_volume=avg_volume,
                average_volume_10d=avg_vol_10d,
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

    def scan_premarket_gainers(
        self, tickers: List[str], min_change_pct: float = 1.5
    ) -> List[str]:
        """
        yf.download() 배치 호출로 프리마켓 상승 종목을 빠르게 스캔

        개별 yf.Ticker().info 호출(종목당 ~0.4초) 대신
        yf.download() 2회(전일종가 + 프리마켓) 배치 호출로 수백 종목을 수초 내 처리.

        Args:
            tickers: 스캔 대상 티커 리스트
            min_change_pct: 최소 변동률 기준 (기본 1.5%)

        Returns:
            PM 상승 종목 티커 리스트 (변동률 내림차순)
        """
        if not tickers:
            return []

        logger.info(f"프리마켓 gainer 배치 스캔: {len(tickers)}개 종목...")

        try:
            # 1. 전일 종가 배치 조회 (단일 HTTP 호출)
            daily = yf.download(
                tickers, period="5d", interval="1d",
                progress=False, auto_adjust=True, threads=True,
            )
            if daily.empty:
                logger.warning("일봉 데이터 없음")
                return []

            if isinstance(daily.columns, pd.MultiIndex):
                prev_close = daily["Close"].iloc[-1]
            else:
                # 단일 종목
                prev_close = pd.Series({tickers[0]: daily["Close"].iloc[-1]})

            # 2. 프리마켓 데이터 배치 조회 (단일 HTTP 호출)
            pm_data = yf.download(
                tickers, period="1d", interval="1m",
                prepost=True, progress=False, auto_adjust=True, threads=True,
            )
            if pm_data.empty:
                logger.warning("프리마켓 데이터 없음")
                return []

            if isinstance(pm_data.columns, pd.MultiIndex):
                latest = pm_data["Close"].iloc[-1]
            else:
                latest = pd.Series({tickers[0]: pm_data["Close"].iloc[-1]})

            # 3. 변동률 계산 및 필터
            change_pct = ((latest - prev_close) / prev_close) * 100
            gainers = change_pct[change_pct >= min_change_pct].dropna().sort_values(ascending=False)

            result = gainers.index.tolist()
            logger.info(f"프리마켓 +{min_change_pct}% 이상: {len(result)}개 종목")
            return result

        except Exception as e:
            logger.warning(f"프리마켓 gainer 스캔 실패: {e}")
            return []

"""Yahoo Finance를 통한 주식 데이터 수집"""

import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

from config.settings import settings

logger = logging.getLogger(__name__)


class StockDataFetcher:
    """S&P 500 주식 데이터 수집기"""

    def __init__(self, lookback_days: Optional[int] = None):
        self.lookback_days = lookback_days or settings.analysis.lookback_days

    def fetch_batch(
        self, tickers: List[str], include_spy: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        여러 종목의 OHLCV 데이터를 배치로 수집

        Args:
            tickers: 티커 심볼 리스트
            include_spy: SPY를 자동으로 포함할지 여부 (상대강도 계산용)

        Returns:
            {ticker: DataFrame} 형태의 딕셔너리
        """
        if not tickers:
            return {}

        # SPY 자동 포함 (상대강도 계산용)
        if include_spy and "SPY" not in tickers:
            tickers = list(tickers) + ["SPY"]

        logger.info(f"{len(tickers)}개 종목 데이터 수집 시작...")

        # 기간 설정
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)

        try:
            # yfinance 배치 다운로드
            data = yf.download(
                tickers,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                group_by="ticker",
                threads=True,
                progress=False,
            )

            result = {}

            if len(tickers) == 1:
                # 단일 종목인 경우 구조가 다름
                ticker = tickers[0]
                if not data.empty:
                    result[ticker] = data
            else:
                # 복수 종목인 경우
                for ticker in tickers:
                    try:
                        if ticker in data.columns.get_level_values(0):
                            ticker_data = data[ticker].dropna(how="all")
                            if not ticker_data.empty:
                                result[ticker] = ticker_data
                    except Exception as e:
                        logger.warning(f"{ticker} 데이터 처리 실패: {e}")
                        continue

            logger.info(f"{len(result)}개 종목 데이터 수집 완료")
            return result

        except Exception as e:
            logger.error(f"배치 데이터 수집 실패: {e}")
            return {}

    def fetch_single(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        단일 종목의 OHLCV 데이터 수집

        Args:
            ticker: 티커 심볼

        Returns:
            OHLCV DataFrame 또는 None
        """
        result = self.fetch_batch([ticker])
        return result.get(ticker)

    def get_latest_prices(self, tickers: List[str]) -> Dict[str, Dict]:
        """
        최신 가격 정보 조회

        Returns:
            {ticker: {"close": float, "change": float, "change_pct": float}}
        """
        data = self.fetch_batch(tickers)
        result = {}

        for ticker, df in data.items():
            if len(df) < 2:
                continue

            try:
                latest_close = df["Close"].iloc[-1]
                prev_close = df["Close"].iloc[-2]
                change = latest_close - prev_close
                change_pct = (change / prev_close) * 100

                result[ticker] = {
                    "close": round(latest_close, 2),
                    "change": round(change, 2),
                    "change_pct": round(change_pct, 2),
                    "volume": int(df["Volume"].iloc[-1]),
                    "date": df.index[-1].strftime("%Y-%m-%d"),
                }
            except Exception as e:
                logger.warning(f"{ticker} 가격 정보 처리 실패: {e}")
                continue

        return result

    def get_market_summary(self) -> Dict[str, Dict]:
        """
        주요 시장 지수 요약 (SPY, QQQ, DIA, IWM)
        """
        from config.sp500_tickers import MARKET_ETFS

        return self.get_latest_prices(MARKET_ETFS)

"""섹터별 분석"""

import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
import logging

from config.sp500_tickers import get_sp500_by_sector, SECTOR_ETFS
from data.fetcher import StockDataFetcher

logger = logging.getLogger(__name__)


@dataclass
class SectorPerformance:
    """섹터 성과 데이터"""
    name: str
    daily_return: float
    weekly_return: float
    monthly_return: float
    advancing: int  # 상승 종목 수
    declining: int  # 하락 종목 수
    strength: str  # "강세", "중립", "약세"


class SectorAnalyzer:
    """섹터별 분석기"""

    def __init__(self, stock_data: Dict[str, pd.DataFrame]):
        """
        Args:
            stock_data: {ticker: DataFrame} 형태의 주가 데이터
        """
        self.stock_data = stock_data
        self.sectors = get_sp500_by_sector()

    def calculate_return(self, close_prices: pd.Series, days: int) -> Optional[float]:
        """특정 기간의 수익률 계산"""
        if len(close_prices) < days + 1:
            return None

        current = close_prices.iloc[-1]
        past = close_prices.iloc[-(days + 1)]
        return ((current - past) / past) * 100

    def analyze_sector(self, sector_name: str) -> Optional[SectorPerformance]:
        """단일 섹터 분석"""
        if sector_name not in self.sectors:
            return None

        tickers = self.sectors[sector_name]
        daily_returns = []
        weekly_returns = []
        monthly_returns = []
        advancing = 0
        declining = 0

        for ticker in tickers:
            if ticker not in self.stock_data:
                continue

            df = self.stock_data[ticker]
            if df.empty or "Close" not in df.columns:
                continue

            close = df["Close"]

            # 일간 수익률
            daily = self.calculate_return(close, 1)
            if daily is not None:
                daily_returns.append(daily)
                if daily > 0:
                    advancing += 1
                elif daily < 0:
                    declining += 1

            # 주간 수익률 (5거래일)
            weekly = self.calculate_return(close, 5)
            if weekly is not None:
                weekly_returns.append(weekly)

            # 월간 수익률 (21거래일)
            monthly = self.calculate_return(close, 21)
            if monthly is not None:
                monthly_returns.append(monthly)

        if not daily_returns:
            return None

        avg_daily = sum(daily_returns) / len(daily_returns)
        avg_weekly = sum(weekly_returns) / len(weekly_returns) if weekly_returns else 0
        avg_monthly = sum(monthly_returns) / len(monthly_returns) if monthly_returns else 0

        # 강도 판별
        if avg_daily > 1.0:
            strength = "강세"
        elif avg_daily < -1.0:
            strength = "약세"
        else:
            strength = "중립"

        return SectorPerformance(
            name=sector_name,
            daily_return=round(avg_daily, 2),
            weekly_return=round(avg_weekly, 2),
            monthly_return=round(avg_monthly, 2),
            advancing=advancing,
            declining=declining,
            strength=strength,
        )

    def analyze_all_sectors(self) -> List[SectorPerformance]:
        """모든 섹터 분석"""
        results = []
        for sector_name in self.sectors.keys():
            perf = self.analyze_sector(sector_name)
            if perf:
                results.append(perf)

        # 일간 수익률 기준 정렬 (높은 순)
        results.sort(key=lambda x: x.daily_return, reverse=True)
        return results

    def get_market_breadth(self) -> Dict:
        """
        시장 전체 상승/하락 종목 수 계산

        Returns:
            {
                "total": int,
                "advancing": int,
                "declining": int,
                "unchanged": int,
                "advance_decline_ratio": float
            }
        """
        advancing = 0
        declining = 0
        unchanged = 0

        for ticker, df in self.stock_data.items():
            if df.empty or len(df) < 2:
                continue

            try:
                close = df["Close"]
                change = close.iloc[-1] - close.iloc[-2]

                if change > 0:
                    advancing += 1
                elif change < 0:
                    declining += 1
                else:
                    unchanged += 1
            except Exception:
                continue

        total = advancing + declining + unchanged
        ad_ratio = advancing / declining if declining > 0 else float("inf")

        return {
            "total": total,
            "advancing": advancing,
            "declining": declining,
            "unchanged": unchanged,
            "advance_decline_ratio": round(ad_ratio, 2),
        }

    def get_top_movers(self, n: int = 10) -> Dict[str, List[Dict]]:
        """
        상승/하락 상위 종목

        Returns:
            {
                "top_gainers": [...],
                "top_losers": [...]
            }
        """
        changes = []

        for ticker, df in self.stock_data.items():
            if df.empty or len(df) < 2:
                continue

            try:
                close = df["Close"]
                change_pct = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100
                changes.append({
                    "ticker": ticker,
                    "close": round(close.iloc[-1], 2),
                    "change_pct": round(change_pct, 2),
                })
            except Exception:
                continue

        # 정렬
        changes.sort(key=lambda x: x["change_pct"], reverse=True)

        return {
            "top_gainers": changes[:n],
            "top_losers": changes[-n:][::-1],  # 역순으로 (가장 하락이 큰 것부터)
        }

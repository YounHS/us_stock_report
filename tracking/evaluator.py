"""
추천 종목 성과 평가 모듈

보유 기간이 만료된 추천 종목의 실제 가격을 조회하여
win/loss/expired 결과를 판정합니다.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "docs" / "data"
RECOMMENDATIONS_FILE = DATA_DIR / "recommendations.json"


class OutcomeEvaluator:
    """pending 상태 추천 종목의 성과를 평가"""

    def evaluate(self) -> int:
        """
        보유 기간이 경과한 pending 종목을 평가합니다.

        Returns:
            평가된 엔트리 수
        """
        if not RECOMMENDATIONS_FILE.exists():
            return 0

        with open(RECOMMENDATIONS_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)

        today = datetime.now()
        today_str = today.strftime("%Y-%m-%d")

        # pending 항목 중 보유 기간 경과한 것만 추출
        to_evaluate = []
        for entry in history:
            if entry["outcome"] != "pending":
                continue
            entry_date = datetime.strptime(entry["date"], "%Y-%m-%d")
            holding_days = entry.get("holding_period_days", 5)
            # 보유 기간을 캘린더 일수로 변환 (trading days → calendar days, ×1.5)
            calendar_days = int(holding_days * 1.5) + 1
            expiry_date = entry_date + timedelta(days=calendar_days)
            if today >= expiry_date:
                to_evaluate.append(entry)

        if not to_evaluate:
            logger.info("평가 대상 추천 종목 없음")
            return 0

        # 필요한 티커 목록 수집
        tickers = list({e["ticker"] for e in to_evaluate})
        logger.info(f"성과 평가 대상: {len(to_evaluate)}건 ({len(tickers)}개 티커)")

        # 가격 데이터 배치 다운로드
        # 가장 오래된 entry 날짜부터 오늘까지
        earliest_date = min(e["date"] for e in to_evaluate)
        start_date = (
            datetime.strptime(earliest_date, "%Y-%m-%d") - timedelta(days=1)
        ).strftime("%Y-%m-%d")

        price_data = {}
        try:
            df = yf.download(
                tickers,
                start=start_date,
                end=today_str,
                auto_adjust=True,
                progress=False,
            )
            if df is not None and not df.empty:
                for ticker in tickers:
                    try:
                        if len(tickers) == 1:
                            ticker_df = df[["High", "Low", "Close"]].copy()
                        elif isinstance(df.columns, pd.MultiIndex):
                            ticker_df = df[["High", "Low", "Close"]].xs(
                                ticker, level=1, axis=1
                            )
                        else:
                            ticker_df = df[["High", "Low", "Close"]].copy()
                        ticker_df = ticker_df.dropna()
                        if not ticker_df.empty:
                            price_data[ticker] = ticker_df
                    except (KeyError, TypeError):
                        continue
        except Exception as e:
            logger.warning(f"가격 데이터 다운로드 실패: {e}")
            return 0

        # 각 항목 평가
        evaluated = 0
        for entry in to_evaluate:
            ticker = entry["ticker"]
            if ticker not in price_data:
                # 데이터 없음 → expired_loss (return 0%)
                entry["outcome"] = "expired_loss"
                entry["return_pct"] = 0.0
                entry["exit_date"] = today_str
                evaluated += 1
                continue

            result = self._determine_outcome(entry, price_data[ticker])
            entry["outcome"] = result["outcome"]
            entry["exit_price"] = result["exit_price"]
            entry["exit_date"] = result["exit_date"]
            entry["return_pct"] = result["return_pct"]
            evaluated += 1

        if evaluated:
            with open(RECOMMENDATIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            logger.info(f"성과 평가 {evaluated}건 완료")

        return evaluated

    def _determine_outcome(self, entry: Dict, price_df) -> Dict:
        """
        개별 추천의 성과를 판정합니다.

        - 보유 기간 내 High >= target → "win" (target price로 청산)
        - 보유 기간 내 Low <= stop → "loss" (stop price로 청산)
        - 같은 날 둘 다 해당 → stop 우선 (보수적)
        - 기간 만료 → 최종 종가 기준 profit/loss
        """
        entry_price = entry["entry_price"]
        target_price = entry.get("target_price")
        stop_loss = entry.get("stop_loss")
        entry_date = datetime.strptime(entry["date"], "%Y-%m-%d")
        holding_days = entry.get("holding_period_days", 5)

        # entry 다음 날부터의 데이터
        mask = price_df.index > entry_date.strftime("%Y-%m-%d")
        period_df = price_df[mask]

        if period_df.empty:
            return {
                "outcome": "expired_loss" if not target_price else "expired_loss",
                "exit_price": entry_price,
                "exit_date": entry["date"],
                "return_pct": 0.0,
            }

        # 거래일 수만큼만 확인
        period_df = period_df.head(holding_days)

        for date_idx, row in period_df.iterrows():
            date_str = date_idx.strftime("%Y-%m-%d")
            high = float(row["High"])
            low = float(row["Low"])
            close = float(row["Close"])

            # 손절 확인 (우선)
            if stop_loss and low <= stop_loss:
                return_pct = ((stop_loss - entry_price) / entry_price) * 100
                return {
                    "outcome": "loss",
                    "exit_price": round(stop_loss, 2),
                    "exit_date": date_str,
                    "return_pct": round(return_pct, 2),
                }

            # 목표가 도달
            if target_price and high >= target_price:
                return_pct = ((target_price - entry_price) / entry_price) * 100
                return {
                    "outcome": "win",
                    "exit_price": round(target_price, 2),
                    "exit_date": date_str,
                    "return_pct": round(return_pct, 2),
                }

        # 기간 만료 → 마지막 종가
        last_row = period_df.iloc[-1]
        last_close = float(last_row["Close"])
        last_date = period_df.index[-1].strftime("%Y-%m-%d")
        return_pct = ((last_close - entry_price) / entry_price) * 100

        outcome = "expired_profit" if return_pct >= 0 else "expired_loss"
        return {
            "outcome": outcome,
            "exit_price": round(last_close, 2),
            "exit_date": last_date,
            "return_pct": round(return_pct, 2),
        }

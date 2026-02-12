"""
추천 종목 통계 요약 모듈

recommendations.json에서 통계를 계산하여 summary.json에 저장합니다.
대시보드에서 클라이언트 사이드로 읽어 표시합니다.
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "docs" / "data"
RECOMMENDATIONS_FILE = DATA_DIR / "recommendations.json"
SUMMARY_FILE = DATA_DIR / "summary.json"


class SummaryGenerator:
    """추천 기록에서 통계를 생성하여 summary.json에 저장"""

    def generate(self) -> bool:
        """
        summary.json을 생성합니다.

        Returns:
            성공 여부
        """
        if not RECOMMENDATIONS_FILE.exists():
            logger.info("추천 기록 파일 없음, 요약 생성 건너뜀")
            return False

        with open(RECOMMENDATIONS_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)

        if not history:
            logger.info("추천 기록이 비어있음")
            return False

        # 완료된 항목만 통계에 포함
        completed = [e for e in history if e["outcome"] != "pending"]
        pending = [e for e in history if e["outcome"] == "pending"]

        summary = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_recommendations": len(history),
            "pending_count": len(pending),
            "evaluated_count": len(completed),
            "overall": self._calc_stats(completed),
            "by_method": {},
            "rolling_30d": {},
            "cumulative_returns": self._calc_cumulative_returns(completed),
        }

        # 방식별 통계
        for method in ["Enhanced", "Kalman", "Long-term", "Legacy"]:
            method_entries = [e for e in completed if e["method"] == method]
            if method_entries:
                summary["by_method"][method] = self._calc_stats(method_entries)

        # 30일 롤링 통계
        cutoff_30d = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        recent = [e for e in completed if e["date"] >= cutoff_30d]
        if recent:
            summary["rolling_30d"] = self._calc_stats(recent)

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(
            f"통계 요약 생성 완료: {len(completed)}건 평가, "
            f"{len(pending)}건 대기"
        )
        return True

    def _calc_stats(self, entries: List[Dict]) -> Dict:
        """통계 계산"""
        if not entries:
            return {
                "count": 0,
                "win_rate": 0,
                "avg_return": 0,
                "best_return": 0,
                "worst_return": 0,
                "profit_factor": 0,
                "win_streak": 0,
                "loss_streak": 0,
            }

        total = len(entries)
        wins = [e for e in entries if e["outcome"] == "win"]
        losses = [e for e in entries if e["outcome"] == "loss"]
        expired_profits = [e for e in entries if e["outcome"] == "expired_profit"]
        expired_losses = [e for e in entries if e["outcome"] == "expired_loss"]

        returns = [e.get("return_pct", 0) or 0 for e in entries]
        positive_returns = [r for r in returns if r > 0]
        negative_returns = [r for r in returns if r < 0]

        win_count = len(wins) + len(expired_profits)
        win_rate = (win_count / total * 100) if total else 0

        avg_return = sum(returns) / len(returns) if returns else 0

        gross_profit = sum(positive_returns) if positive_returns else 0
        gross_loss = abs(sum(negative_returns)) if negative_returns else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
            float("inf") if gross_profit > 0 else 0
        )

        # 연속 승패
        win_streak, loss_streak = self._calc_streaks(entries)

        return {
            "count": total,
            "win_count": win_count,
            "loss_count": total - win_count,
            "win_rate": round(win_rate, 1),
            "avg_return": round(avg_return, 2),
            "best_return": round(max(returns), 2) if returns else 0,
            "worst_return": round(min(returns), 2) if returns else 0,
            "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else 999.99,
            "total_return": round(sum(returns), 2),
            "win_streak": win_streak,
            "loss_streak": loss_streak,
        }

    @staticmethod
    def _calc_streaks(entries: List[Dict]) -> tuple:
        """연속 승/패 기록 계산"""
        # 날짜순 정렬
        sorted_entries = sorted(entries, key=lambda x: x["date"])

        max_win = max_loss = 0
        cur_win = cur_loss = 0

        for e in sorted_entries:
            is_win = e["outcome"] in ("win", "expired_profit")
            if is_win:
                cur_win += 1
                cur_loss = 0
                max_win = max(max_win, cur_win)
            else:
                cur_loss += 1
                cur_win = 0
                max_loss = max(max_loss, cur_loss)

        return max_win, max_loss

    @staticmethod
    def _calc_cumulative_returns(entries: List[Dict]) -> List[Dict]:
        """
        날짜별 누적 수익률 시계열 (차트용)

        각 날짜의 추천 수익률을 순서대로 누적합니다.
        """
        if not entries:
            return []

        # 날짜순 정렬
        sorted_entries = sorted(entries, key=lambda x: x["date"])

        cumulative = 0.0
        series = []
        seen_dates = {}

        for e in sorted_entries:
            ret = e.get("return_pct", 0) or 0
            cumulative += ret
            date = e["date"]
            # 같은 날짜면 마지막 값으로 업데이트
            seen_dates[date] = round(cumulative, 2)

        for date, cum_ret in seen_dates.items():
            series.append({"date": date, "cumulative_return": cum_ret})

        return series

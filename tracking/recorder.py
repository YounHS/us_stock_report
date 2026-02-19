"""
추천 종목 기록 모듈

매일 생성되는 추천 종목(Enhanced, Kalman, Long-term)을 JSON 파일에 기록하여
성과 추적 및 대시보드 표시에 사용합니다.
"""

import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "docs" / "data"
RECOMMENDATIONS_FILE = DATA_DIR / "recommendations.json"


class RecommendationRecorder:
    """추천 종목을 JSON 파일에 기록"""

    def record(
        self,
        recommendation: Optional[Dict],
        recommendation_kalman: Optional[Dict],
        longterm_recommendations: Optional[List[Dict]],
    ) -> int:
        """
        오늘의 추천 종목을 기록합니다.

        Returns:
            새로 기록된 엔트리 수
        """
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")

        # 기존 히스토리 로드
        history = self._load_history()

        new_entries = []

        if recommendation:
            method = recommendation.get("recommendation_method", "Enhanced")
            entry = self._make_entry(recommendation, method, today)
            if entry:
                new_entries.append(entry)

        if recommendation_kalman:
            entry = self._make_entry(recommendation_kalman, "Kalman", today)
            if entry:
                new_entries.append(entry)

        if longterm_recommendations:
            for lt_rec in longterm_recommendations:
                entry = self._make_entry(lt_rec, "Long-term", today)
                if entry:
                    new_entries.append(entry)

        if not new_entries:
            logger.info("기록할 추천 종목 없음")
            return 0

        # 중복 제거 (같은 날 같은 종목/방식이면 교체)
        existing_ids = {e["id"] for e in history}
        for entry in new_entries:
            if entry["id"] in existing_ids:
                history = [e for e in history if e["id"] != entry["id"]]
            history.append(entry)

        # 오래된 항목 제거
        history = self._prune_old_entries(history)

        # 날짜순 정렬
        history.sort(key=lambda x: x["date"], reverse=True)

        self._save_history(history)
        logger.info(f"추천 종목 {len(new_entries)}개 기록 완료")
        return len(new_entries)

    def record_surge(
        self,
        surge_recommendations: Optional[List[Dict]],
    ) -> int:
        """
        개장 급등(Opening Surge) 추천 종목을 기록합니다.

        Args:
            surge_recommendations: get_opening_surge_recommendations() 결과

        Returns:
            새로 기록된 엔트리 수
        """
        if not surge_recommendations:
            return 0

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        history = self._load_history()

        new_entries = []
        for rec in surge_recommendations:
            entry = self._make_surge_entry(rec, today)
            if entry:
                new_entries.append(entry)

        if not new_entries:
            return 0

        existing_ids = {e["id"] for e in history}
        for entry in new_entries:
            if entry["id"] in existing_ids:
                history = [e for e in history if e["id"] != entry["id"]]
            history.append(entry)

        history = self._prune_old_entries(history)
        history.sort(key=lambda x: x["date"], reverse=True)
        self._save_history(history)
        logger.info(f"개장 급등 추천 {len(new_entries)}개 기록 완료")
        return len(new_entries)

    def _make_surge_entry(self, rec: Dict, date_str: str) -> Optional[Dict]:
        """Opening Surge 추천 dict를 트래킹 엔트리로 변환"""
        ticker = rec.get("ticker")
        if not ticker:
            return None

        pm_price = rec.get("pm_price")
        if not pm_price:
            return None

        entry_id = f"{date_str}_{ticker}_Opening Surge"

        return {
            "id": entry_id,
            "date": date_str,
            "ticker": ticker,
            "method": "Opening Surge",
            "entry_price": round(pm_price, 2),
            "target_price": round(rec["target_price"], 2) if rec.get("target_price") else None,
            "stop_loss": round(rec["stop_loss"], 2) if rec.get("stop_loss") else None,
            "score": rec.get("score"),
            "holding_period_days": 1,
            "holding_period_raw": rec.get("holding_period", "30분~1시간"),
            "rsi": rec.get("rsi"),
            "adx": rec.get("adx"),
            "kalman_predicted_price": rec.get("kalman_predicted_price"),
            "score_breakdown": rec.get("score_breakdown"),
            "outcome": "pending",
            "exit_price": None,
            "exit_date": None,
            "return_pct": None,
        }

    def _make_entry(self, rec: Dict, method: str, date_str: str) -> Optional[Dict]:
        """추천 dict를 트래킹 엔트리로 변환"""
        ticker = rec.get("ticker")
        if not ticker:
            return None

        close = rec.get("close")
        if not close:
            return None

        entry_id = f"{date_str}_{ticker}_{method}"

        holding_days = self._parse_holding_period(rec.get("holding_period", ""))

        target_price = rec.get("target_price")
        stop_loss = rec.get("stop_loss")

        # Long-term recs may lack target/stop → use kalman predicted as target
        if method == "Long-term":
            if not target_price:
                target_price = rec.get("kalman_predicted_price")
            if not stop_loss:
                stop_pct = settings.tracking.longterm_default_stop_pct / 100.0
                stop_loss = round(close * (1 - stop_pct), 2)

        return {
            "id": entry_id,
            "date": date_str,
            "ticker": ticker,
            "method": method,
            "entry_price": round(close, 2),
            "target_price": round(target_price, 2) if target_price else None,
            "stop_loss": round(stop_loss, 2) if stop_loss else None,
            "score": rec.get("score"),
            "holding_period_days": holding_days,
            "holding_period_raw": rec.get("holding_period", ""),
            "rsi": rec.get("rsi"),
            "adx": rec.get("adx"),
            "kalman_predicted_price": rec.get("kalman_predicted_price"),
            "score_breakdown": rec.get("score_breakdown"),
            # Outcome fields — filled by evaluator
            "outcome": "pending",
            "exit_price": None,
            "exit_date": None,
            "return_pct": None,
        }

    @staticmethod
    def _parse_holding_period(raw: str) -> int:
        """
        한글 보유 기간 → 최대 거래일 수

        Examples:
            "3-5일" → 5
            "2-3주 (평균회귀 전략)" → 15
            "1-3개월 (추세 추종 전략)" → 63
            "30분~1시간" → 1
        """
        if not raw:
            return 5  # default

        # 분/시간 단위 (인트라데이)
        if "분" in raw or "시간" in raw:
            return 1

        # 개월 단위
        month_match = re.search(r"(\d+)\s*[-~]\s*(\d+)\s*개월", raw)
        if month_match:
            max_months = int(month_match.group(2))
            return max_months * 21  # ~21 trading days per month

        single_month = re.search(r"(\d+)\s*개월", raw)
        if single_month:
            return int(single_month.group(1)) * 21

        # 주 단위
        week_match = re.search(r"(\d+)\s*[-~]\s*(\d+)\s*주", raw)
        if week_match:
            max_weeks = int(week_match.group(2))
            return max_weeks * 5

        single_week = re.search(r"(\d+)\s*주", raw)
        if single_week:
            return int(single_week.group(1)) * 5

        # 일 단위
        day_match = re.search(r"(\d+)\s*[-~]\s*(\d+)\s*일", raw)
        if day_match:
            return int(day_match.group(2))

        single_day = re.search(r"(\d+)\s*일", raw)
        if single_day:
            return int(single_day.group(1))

        return 5  # default

    def _prune_old_entries(self, history: List[Dict]) -> List[Dict]:
        """retention_days보다 오래된 완료 항목 제거"""
        cutoff = (
            datetime.now() - timedelta(days=settings.tracking.retention_days)
        ).strftime("%Y-%m-%d")

        pruned = []
        removed = 0
        for entry in history:
            if entry["date"] < cutoff and entry["outcome"] != "pending":
                removed += 1
                continue
            pruned.append(entry)

        if removed:
            logger.info(f"오래된 트래킹 항목 {removed}개 제거")

        return pruned

    def _load_history(self) -> List[Dict]:
        if RECOMMENDATIONS_FILE.exists():
            try:
                with open(RECOMMENDATIONS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning("추천 기록 파일 읽기 실패, 새로 생성합니다")
        return []

    def _save_history(self, history: List[Dict]):
        with open(RECOMMENDATIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

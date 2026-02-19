"""
추천 종목 파라미터 자동 튜닝 모듈

평가 완료된 추천 성과 데이터를 분석하여 각 추천 방식의
scoring weights 와 minimum score threshold 를 자동 조정합니다.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import settings

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "docs" / "data"
RECOMMENDATIONS_FILE = DATA_DIR / "recommendations.json"
TUNED_PARAMS_FILE = DATA_DIR / "tuned_params.json"


class ParameterOptimizer:
    """평가 결과 기반 scoring weights 자동 튜닝"""

    # method → {score_breakdown_key → settings_weight_key}
    WEIGHT_MAPS = {
        "Enhanced": {
            "rsi_score": "weight_rsi",
            "volume_score": "weight_volume",
            "adx_score": "weight_adx",
            "macd_score": "weight_macd",
            "bollinger_score": "weight_bollinger",
            "relative_strength_score": "weight_relative_strength",
            "week52_score": "weight_52week",
            "obv_score": "weight_obv",
            "stochastic_score": "weight_stochastic",
            "squeeze_score": "weight_squeeze",
        },
        "Long-term": {
            "rsi_score": "longterm_weight_rsi",
            "macd_score": "longterm_weight_macd",
            "bollinger_score": "longterm_weight_bollinger",
            "volume_score": "longterm_weight_volume",
            "adx_score": "longterm_weight_adx",
            "relative_strength_score": "longterm_weight_relative_strength",
            "week52_score": "longterm_weight_week52",
            "kalman_score": "longterm_weight_kalman",
            "obv_score": "longterm_weight_obv",
            "stochastic_score": "longterm_weight_stochastic",
            "squeeze_score": "longterm_weight_squeeze",
        },
        "Opening Surge": {
            "pm_momentum": "surge_weight_pm_momentum",
            "news_catalyst": "surge_weight_news_catalyst",
            "volume_surge": "surge_weight_volume",
            "gap_setup": "surge_weight_gap",
            "squeeze_release": "surge_weight_squeeze",
            "relative_strength": "surge_weight_relative_strength",
            "adx": "surge_weight_adx",
            "stochastic": "surge_weight_stochastic",
        },
    }

    # method → settings attribute for min_score
    MIN_SCORE_ATTRS = {
        "Enhanced": "min_recommendation_score",
        "Kalman": "min_recommendation_score",
        "Long-term": "longterm_min_score",
        "Opening Surge": "surge_min_score",
    }

    # method → min_samples setting attribute
    MIN_SAMPLES_MAP = {
        "Enhanced": "min_samples_shortterm",
        "Kalman": "min_samples_shortterm",
        "Long-term": "min_samples_longterm",
        "Opening Surge": "min_samples_surge",
    }

    def optimize(self) -> Dict:
        """
        전체 방식에 대해 파라미터 튜닝을 수행합니다.

        Returns:
            업데이트된 tuned_params dict
        """
        history = self._load_recommendations()
        if not history:
            logger.info("튜닝: 추천 기록 없음, 스킵")
            return {}

        # 완료된 추천만 필터
        completed = [r for r in history if r.get("outcome") not in ("pending", None)]
        if not completed:
            logger.info("튜닝: 평가 완료 항목 없음, 스킵")
            return {}

        tuned = self.load_tuned_params()
        today = datetime.now().strftime("%Y-%m-%d")
        tuned["last_updated"] = today

        for method in ["Enhanced", "Kalman", "Long-term", "Opening Surge"]:
            method_recs = [r for r in completed if r.get("method") == method]
            result = self._optimize_method(method, method_recs, tuned.get(method, {}))
            if result:
                tuned[method] = result
                logger.info(
                    f"튜닝: {method} 업데이트 (샘플: {result['samples_used']}건, "
                    f"min_score: {result['min_score']})"
                )
            else:
                logger.info(f"튜닝: {method} 샘플 부족, 변경 없음")

        self.save_tuned_params(tuned)
        return tuned

    def _optimize_method(
        self, method: str, completed_recs: List[Dict], existing: Dict
    ) -> Optional[Dict]:
        """
        단일 방식에 대한 가중치/임계값 튜닝

        Returns:
            None if insufficient samples, otherwise updated params dict
        """
        # Kalman reuses Enhanced weight map
        weight_map_key = "Enhanced" if method == "Kalman" else method
        weight_map = self.WEIGHT_MAPS.get(weight_map_key)
        if not weight_map:
            return None

        # score_breakdown 있는 항목만 필터
        recs_with_breakdown = [
            r for r in completed_recs if r.get("score_breakdown")
        ]

        min_samples_attr = self.MIN_SAMPLES_MAP.get(method, "min_samples_shortterm")
        min_samples = getattr(settings.tuning, min_samples_attr, 15)

        if len(recs_with_breakdown) < min_samples:
            return None

        breakdown_keys = list(weight_map.keys())

        # 현재 가중치 (기존 튜닝값 → settings 기본값)
        current_weights = {}
        for bk, sk in weight_map.items():
            if existing and "weights" in existing and sk in existing["weights"]:
                current_weights[sk] = existing["weights"][sk]
            else:
                current_weights[sk] = float(getattr(settings.analysis, sk, 10))

        # 현재 min_score
        min_score_attr = self.MIN_SCORE_ATTRS.get(method, "min_recommendation_score")
        if existing and "min_score" in existing:
            current_min = existing["min_score"]
        else:
            current_min = getattr(settings.analysis, min_score_attr, 35)

        # 효과성 계산
        effectiveness = self._calculate_effectiveness(
            recs_with_breakdown, breakdown_keys
        )

        # 효과성 → 제안 가중치로 변환
        suggested = self._effectiveness_to_weights(effectiveness, weight_map)

        # 스무딩 + 정규화
        alpha = settings.tuning.smoothing_alpha
        smoothed = self._smooth_and_normalize(current_weights, suggested, alpha)

        # min_score 튜닝
        new_min = self._optimize_min_score(
            recs_with_breakdown, current_min
        )

        return {
            "weights": smoothed,
            "min_score": new_min,
            "samples_used": len(recs_with_breakdown),
            "effectiveness": effectiveness,
        }

    def _calculate_effectiveness(
        self, recs: List[Dict], breakdown_keys: List[str]
    ) -> Dict[str, float]:
        """
        각 지표의 효과성 계산: win-rate lift + return lift

        high score vs low score 그룹으로 나누어 비교
        """
        effectiveness = {}

        for key in breakdown_keys:
            scores = []
            for r in recs:
                bd = r.get("score_breakdown", {})
                s = bd.get(key)
                if s is not None:
                    scores.append((s, r))

            if len(scores) < 4:
                effectiveness[key] = 0.0
                continue

            # 중앙값 기준 분리
            score_values = [s for s, _ in scores]
            sorted_vals = sorted(score_values)
            median = sorted_vals[len(sorted_vals) // 2]

            high_group = [r for s, r in scores if s > median]
            low_group = [r for s, r in scores if s <= median]

            # 모든 값이 동일하면 (모두 한쪽 그룹) → 분리 불가
            if not high_group or not low_group:
                # 0이 아닌 값과 0인 값으로 재분리 시도
                high_group = [r for s, r in scores if s > 0]
                low_group = [r for s, r in scores if s <= 0]

            if not high_group or not low_group:
                effectiveness[key] = 0.0
                continue

            # win rate lift
            def _win_rate(group):
                wins = sum(
                    1 for r in group
                    if r.get("outcome") in ("win", "expired_profit")
                )
                return wins / len(group) if group else 0

            wr_high = _win_rate(high_group)
            wr_low = _win_rate(low_group)
            wr_lift = wr_high - wr_low

            # return lift
            def _avg_return(group):
                returns = [
                    r.get("return_pct", 0) or 0 for r in group
                ]
                return sum(returns) / len(returns) if returns else 0

            ret_high = _avg_return(high_group)
            ret_low = _avg_return(low_group)
            ret_lift = ret_high - ret_low

            effectiveness[key] = round(0.6 * wr_lift + 0.4 * ret_lift, 4)

        return effectiveness

    def _effectiveness_to_weights(
        self, effectiveness: Dict[str, float], weight_map: Dict[str, str]
    ) -> Dict[str, float]:
        """효과성 값을 가중치로 변환 (양수 시프트 + 정규화)"""
        if not effectiveness:
            return {}

        # 모든 값을 양수로 시프트
        min_val = min(effectiveness.values()) if effectiveness else 0
        shifted = {}
        for key, val in effectiveness.items():
            shifted[key] = val - min_val + 0.1  # +0.1 to avoid zero

        total = sum(shifted.values())
        if total <= 0:
            return {}

        suggested = {}
        for bk, sk in weight_map.items():
            raw = (shifted.get(bk, 0.1) / total) * 100
            suggested[sk] = raw

        return suggested

    def _optimize_min_score(
        self, recs: List[Dict], current_min: int
    ) -> int:
        """
        win/loss 분포 기반 최소 점수 임계값 튜닝

        승리 그룹 평균 점수와 패배 그룹 평균 점수의 가중 평균에서
        새로운 임계값을 결정합니다.
        """
        win_scores = [
            r["score"] for r in recs
            if r.get("outcome") in ("win", "expired_profit") and r.get("score")
        ]
        loss_scores = [
            r["score"] for r in recs
            if r.get("outcome") in ("loss", "expired_loss") and r.get("score")
        ]

        if not win_scores or not loss_scores:
            return current_min

        avg_win = sum(win_scores) / len(win_scores)
        avg_loss = sum(loss_scores) / len(loss_scores)

        # win 쪽에 60% 가중치, loss 쪽에 40% 가중치
        suggested = avg_loss * 0.4 + avg_win * 0.6

        # EMA 스무딩
        alpha = settings.tuning.smoothing_alpha
        smoothed = (1 - alpha) * current_min + alpha * suggested

        # 바운드
        return int(max(15, min(60, round(smoothed))))

    def _smooth_and_normalize(
        self,
        current: Dict[str, float],
        suggested: Dict[str, float],
        alpha: float,
    ) -> Dict[str, float]:
        """
        EMA 스무딩 + 바운드 적용 + 합계 100 정규화

        new = (1-alpha) * current + alpha * suggested
        """
        if not suggested:
            return current

        floor = settings.tuning.weight_floor
        ceiling = settings.tuning.weight_ceiling

        blended = {}
        for key in current:
            cur = current.get(key, 10.0)
            sug = suggested.get(key, cur)
            blended[key] = (1 - alpha) * cur + alpha * sug

        # 바운드 적용
        for key in blended:
            blended[key] = max(floor, min(ceiling, blended[key]))

        # 합계 100 정규화
        total = sum(blended.values())
        if total > 0:
            for key in blended:
                blended[key] = round((blended[key] / total) * 100, 1)

        return blended

    def load_tuned_params(self) -> Dict:
        """tuned_params.json 로드 (없으면 빈 dict)"""
        try:
            if TUNED_PARAMS_FILE.exists():
                with open(TUNED_PARAMS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"tuned_params.json 로드 실패: {e}")
        return {}

    def save_tuned_params(self, params: Dict):
        """tuned_params.json 저장"""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(TUNED_PARAMS_FILE, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2)
        logger.info(f"튜닝 파라미터 저장: {TUNED_PARAMS_FILE}")

    def _load_recommendations(self) -> List[Dict]:
        """recommendations.json 로드"""
        try:
            if RECOMMENDATIONS_FILE.exists():
                with open(RECOMMENDATIONS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
        return []

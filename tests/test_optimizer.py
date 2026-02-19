"""
ParameterOptimizer 단위 테스트

평가 완료된 추천 데이터로 가중치 튜닝이 올바르게 동작하는지 검증합니다.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from tuning.optimizer import ParameterOptimizer


def _make_rec(
    ticker, method, outcome, score, return_pct, score_breakdown,
):
    """테스트용 추천 레코드 생성"""
    return {
        "id": f"2026-01-01_{ticker}_{method}",
        "date": "2026-01-01",
        "ticker": ticker,
        "method": method,
        "entry_price": 100.0,
        "target_price": 110.0,
        "stop_loss": 95.0,
        "score": score,
        "holding_period_days": 5,
        "outcome": outcome,
        "exit_price": round(100.0 + (return_pct or 0), 2),
        "exit_date": "2026-01-08",
        "return_pct": return_pct,
        "score_breakdown": score_breakdown,
    }


def _make_enhanced_breakdown(
    rsi=0, volume=0, adx=0, macd=0, bollinger=0,
    relative_strength=0, week52=0, obv=0, stochastic=0, squeeze=0,
):
    return {
        "rsi_score": rsi,
        "volume_score": volume,
        "adx_score": adx,
        "macd_score": macd,
        "bollinger_score": bollinger,
        "relative_strength_score": relative_strength,
        "week52_score": week52,
        "obv_score": obv,
        "stochastic_score": stochastic,
        "squeeze_score": squeeze,
    }


def _run_optimizer(recs, tmp_path):
    """헬퍼: 임시 파일로 optimizer 실행"""
    tuned_file = tmp_path / "tuned_params.json"
    with patch.object(ParameterOptimizer, '_load_recommendations', return_value=recs), \
         patch('tuning.optimizer.TUNED_PARAMS_FILE', tuned_file), \
         patch('tuning.optimizer.DATA_DIR', tmp_path):
        optimizer = ParameterOptimizer()
        return optimizer.optimize()


class TestParameterOptimizer:

    def test_insufficient_samples_returns_empty(self, tmp_path):
        """샘플 부족 시 튜닝 스킵"""
        recs = [
            _make_rec(f"AAPL{i}", "Enhanced", "win", 50, 5.0,
                      _make_enhanced_breakdown(rsi=10, macd=8))
            for i in range(3)
        ]
        result = _run_optimizer(recs, tmp_path)
        assert "Enhanced" not in result

    def test_winning_indicators_get_higher_weights(self, tmp_path):
        """승리와 상관관계가 높은 지표에 더 높은 가중치 부여"""
        recs = []
        for i in range(10):
            recs.append(_make_rec(
                f"WIN{i}", "Enhanced", "win", 60, 5.0,
                _make_enhanced_breakdown(rsi=15, volume=2, adx=3, macd=8),
            ))
        for i in range(10):
            recs.append(_make_rec(
                f"LOSS{i}", "Enhanced", "loss", 30, -3.0,
                _make_enhanced_breakdown(rsi=0, volume=8, adx=10, macd=2),
            ))

        result = _run_optimizer(recs, tmp_path)
        assert "Enhanced" in result
        weights = result["Enhanced"]["weights"]
        # RSI는 승리와 상관이 높으므로 volume/adx보다 높아야 함
        assert weights["weight_rsi"] > weights["weight_volume"]

    def test_weights_sum_to_100(self, tmp_path):
        """가중치 합이 100이 되어야 함"""
        recs = []
        for i in range(20):
            outcome = "win" if i % 2 == 0 else "loss"
            ret = 3.0 if outcome == "win" else -2.0
            recs.append(_make_rec(
                f"T{i}", "Enhanced", outcome, 45, ret,
                _make_enhanced_breakdown(
                    rsi=10 if i % 3 == 0 else 0,
                    macd=8 if i % 2 == 0 else 0,
                    adx=5 if i < 10 else 12,
                    volume=7,
                    bollinger=6,
                ),
            ))

        result = _run_optimizer(recs, tmp_path)
        assert "Enhanced" in result
        total = sum(result["Enhanced"]["weights"].values())
        assert abs(total - 100.0) < 1.0

    def test_bounds_respected(self, tmp_path):
        """floor(2.0) ~ ceiling(30.0) 범위 내"""
        recs = []
        for i in range(20):
            outcome = "win" if i < 15 else "loss"
            ret = 10.0 if outcome == "win" else -5.0
            recs.append(_make_rec(
                f"T{i}", "Enhanced", outcome, 50, ret,
                _make_enhanced_breakdown(rsi=15),
            ))

        result = _run_optimizer(recs, tmp_path)
        assert "Enhanced" in result
        for key, val in result["Enhanced"]["weights"].items():
            assert val >= 2.0, f"{key} = {val} < floor 2.0"
            assert val <= 30.0, f"{key} = {val} > ceiling 30.0"

    def test_smoothing_prevents_sudden_jumps(self, tmp_path):
        """EMA 스무딩으로 급격한 변화 방지: alpha=0.3 → alpha=0.0 보다 default에 가까움"""
        recs = []
        for i in range(20):
            outcome = "win" if i % 2 == 0 else "loss"
            ret = 5.0 if outcome == "win" else -3.0
            recs.append(_make_rec(
                f"T{i}", "Enhanced", outcome, 50, ret,
                _make_enhanced_breakdown(
                    rsi=15 if outcome == "win" else 0,
                    macd=5, adx=3,
                ),
            ))

        # alpha=0.3 (default smoothing)
        result_smooth = _run_optimizer(recs, tmp_path)

        # alpha=1.0 (no smoothing, fully suggested)
        tuned_file2 = tmp_path / "tuned_params2.json"
        with patch.object(ParameterOptimizer, '_load_recommendations', return_value=recs), \
             patch('tuning.optimizer.TUNED_PARAMS_FILE', tuned_file2), \
             patch('tuning.optimizer.DATA_DIR', tmp_path), \
             patch('tuning.optimizer.settings.tuning.smoothing_alpha', 1.0):
            optimizer = ParameterOptimizer()
            result_raw = optimizer.optimize()

        if "Enhanced" not in result_smooth or "Enhanced" not in result_raw:
            pytest.skip("Not enough variance in data")

        # 스무딩된 RSI는 raw 보다 기본값(15)에 더 가까워야 함
        default_rsi = 15.0
        smooth_rsi = result_smooth["Enhanced"]["weights"]["weight_rsi"]
        raw_rsi = result_raw["Enhanced"]["weights"]["weight_rsi"]
        assert abs(smooth_rsi - default_rsi) <= abs(raw_rsi - default_rsi) + 0.5

    def test_min_score_bounded(self, tmp_path):
        """min_score가 [15, 60] 범위 내"""
        recs = []
        for i in range(20):
            outcome = "win" if i < 5 else "loss"
            ret = 2.0 if outcome == "win" else -1.0
            recs.append(_make_rec(
                f"T{i}", "Enhanced", outcome, 20 + i, ret,
                _make_enhanced_breakdown(rsi=5, macd=3),
            ))

        result = _run_optimizer(recs, tmp_path)
        if "Enhanced" in result:
            assert 15 <= result["Enhanced"]["min_score"] <= 60

    def test_longterm_method(self, tmp_path):
        """Long-term 방식도 정상 튜닝"""
        recs = []
        for i in range(8):
            outcome = "win" if i < 5 else "expired_loss"
            ret = 8.0 if outcome == "win" else -4.0
            recs.append(_make_rec(
                f"LT{i}", "Long-term", outcome, 55, ret,
                {
                    "rsi_score": 6 if outcome == "win" else 2,
                    "macd_score": 10 if outcome == "win" else 3,
                    "bollinger_score": 5,
                    "volume_score": 4,
                    "adx_score": 12,
                    "relative_strength_score": 8,
                    "week52_score": 5,
                    "kalman_score": 6,
                    "obv_score": 4,
                    "stochastic_score": 3,
                    "squeeze_score": 5,
                },
            ))

        tuned_file = tmp_path / "tuned_params.json"
        with patch.object(ParameterOptimizer, '_load_recommendations', return_value=recs), \
             patch('tuning.optimizer.TUNED_PARAMS_FILE', tuned_file), \
             patch('tuning.optimizer.DATA_DIR', tmp_path), \
             patch('tuning.optimizer.settings.tuning.min_samples_longterm', 5):
            optimizer = ParameterOptimizer()
            result = optimizer.optimize()

        assert "Long-term" in result
        assert result["Long-term"]["samples_used"] == 8
        total = sum(result["Long-term"]["weights"].values())
        assert abs(total - 100.0) < 1.0

    def test_opening_surge_method(self, tmp_path):
        """Opening Surge 방식 튜닝"""
        recs = []
        for i in range(12):
            outcome = "win" if i < 7 else "loss"
            ret = 2.0 if outcome == "win" else -1.0
            recs.append(_make_rec(
                f"SG{i}", "Opening Surge", outcome, 40, ret,
                {
                    "pm_momentum": 20 if outcome == "win" else 5,
                    "news_catalyst": 15 if outcome == "win" else 10,
                    "volume_surge": 10,
                    "gap_setup": 5,
                    "squeeze_release": 8,
                    "relative_strength": 5,
                    "adx": 4,
                    "stochastic": 3,
                },
            ))

        result = _run_optimizer(recs, tmp_path)
        assert "Opening Surge" in result
        total = sum(result["Opening Surge"]["weights"].values())
        assert abs(total - 100.0) < 1.0

    def test_kalman_reuses_enhanced_map(self, tmp_path):
        """Kalman 방식은 Enhanced 가중치 맵 재사용"""
        recs = []
        for i in range(20):
            outcome = "win" if i % 2 == 0 else "expired_loss"
            ret = 3.0 if outcome == "win" else -2.0
            recs.append(_make_rec(
                f"K{i}", "Kalman", outcome, 45, ret,
                _make_enhanced_breakdown(rsi=10 if outcome == "win" else 0, macd=8, adx=5),
            ))

        result = _run_optimizer(recs, tmp_path)
        assert "Kalman" in result
        assert "weight_rsi" in result["Kalman"]["weights"]

    def test_pending_recs_excluded(self, tmp_path):
        """pending 상태 추천은 튜닝 대상에서 제외"""
        recs = [
            _make_rec(f"P{i}", "Enhanced", "pending", 50, 0,
                      _make_enhanced_breakdown(rsi=10))
            for i in range(20)
        ]

        result = _run_optimizer(recs, tmp_path)
        assert "Enhanced" not in result

    def test_effectiveness_calculation(self):
        """효과성 계산 검증: 승리에 기여하는 지표 = 양수 효과성"""
        optimizer = ParameterOptimizer()

        recs = []
        for i in range(10):
            recs.append({
                "outcome": "win", "return_pct": 5.0,
                "score_breakdown": {"rsi_score": 15, "macd_score": 5},
            })
            recs.append({
                "outcome": "loss", "return_pct": -3.0,
                "score_breakdown": {"rsi_score": 0, "macd_score": 5},
            })

        effectiveness = optimizer._calculate_effectiveness(
            recs, ["rsi_score", "macd_score"]
        )
        # RSI는 양수 효과성 (승리 예측): high RSI → win, low RSI → loss
        assert effectiveness["rsi_score"] > 0

    def test_save_load_roundtrip(self, tmp_path):
        """저장/로드 라운드트립"""
        optimizer = ParameterOptimizer()

        test_params = {
            "last_updated": "2026-02-19",
            "Enhanced": {
                "weights": {"weight_rsi": 16.2, "weight_volume": 9.1},
                "min_score": 33,
                "samples_used": 45,
            },
        }

        tuned_file = tmp_path / "tuned_params.json"
        with patch('tuning.optimizer.TUNED_PARAMS_FILE', tuned_file), \
             patch('tuning.optimizer.DATA_DIR', tmp_path):
            optimizer.save_tuned_params(test_params)
            loaded = optimizer.load_tuned_params()

        assert loaded["Enhanced"]["weights"]["weight_rsi"] == 16.2
        assert loaded["Enhanced"]["min_score"] == 33

    def test_empty_history(self, tmp_path):
        """빈 히스토리에서 빈 결과 반환"""
        result = _run_optimizer([], tmp_path)
        assert result == {}

    def test_no_breakdown_recs_skipped(self, tmp_path):
        """score_breakdown 없는 레코드는 무시"""
        recs = []
        for i in range(20):
            r = _make_rec(f"T{i}", "Enhanced", "win", 50, 5.0, None)
            recs.append(r)

        result = _run_optimizer(recs, tmp_path)
        assert "Enhanced" not in result

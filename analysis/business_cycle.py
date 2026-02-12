"""경기 사이클 인디케이터 분석

클래식 섹터 로테이션 모델 기반으로 경기 순환 국면을 판별하고,
6개 복합 팩터를 종합하여 원형 다이어그램 위치를 계산한다.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)

# 4개 국면 정의 (시계 방향, 12시=0도)
PHASES = ["early", "mid", "late", "contraction"]
PHASE_ANGLES = {
    "early": 45.0,        # 12시~3시 (우상단, SVG top-right)
    "mid": 135.0,         # 3시~6시 (우하단, SVG bottom-right)
    "late": 225.0,        # 6시~9시 (좌하단, SVG bottom-left)
    "contraction": 315.0, # 9시~12시 (좌상단, SVG top-left)
}
PHASE_LABELS = {
    "early": "회복기",
    "mid": "확장기",
    "late": "과열기",
    "contraction": "수축기",
}

# 섹터 로테이션 매핑 (어떤 국면에서 어떤 섹터가 주도하는가)
SECTOR_ROTATION = {
    "early": ["Consumer Discretionary", "Financials", "Industrials"],
    "mid": ["Information Technology", "Communication Services"],
    "late": ["Energy", "Materials"],
    "contraction": ["Consumer Staples", "Utilities", "Health Care"],
}

# 역매핑: 섹터 → 주도 국면
SECTOR_TO_PHASE = {}
for phase, sectors in SECTOR_ROTATION.items():
    for sector in sectors:
        SECTOR_TO_PHASE[sector] = phase


@dataclass
class CycleFactorReading:
    """개별 팩터 판독"""
    name: str
    value: Optional[float]
    display_value: str
    phase_signal: Optional[str]  # "early", "mid", "late", "contraction", or None
    weight: int
    description: str


@dataclass
class BusinessCycleResult:
    """경기 사이클 분석 종합 결과"""
    current_phase: str                          # "early", "mid", "late", "contraction"
    phase_position: float                       # 0-360도
    factor_readings: List[CycleFactorReading]
    phase_probabilities: Dict[str, float]       # {"early": 0.3, "mid": 0.4, ...}
    leading_sectors: List[str]
    lagging_sectors: List[str]
    summary: str


class BusinessCycleAnalyzer:
    """경기 순환 사이클 분석기"""

    def __init__(self):
        self.cfg = settings.analysis

    def analyze(
        self,
        sector_performance,
        market_breadth: Dict,
        stock_data: Dict[str, pd.DataFrame],
    ) -> Optional[BusinessCycleResult]:
        """
        경기 사이클 분석 실행

        Args:
            sector_performance: SectorPerformance 리스트 (섹터 로테이션 팩터)
            market_breadth: 시장 너비 데이터 (advance/decline)
            stock_data: 주가 데이터 (^TNX, ^IRX, ^VIX, HYG, TLT, IWM, SPY 포함)

        Returns:
            BusinessCycleResult 또는 None
        """
        try:
            factors = []

            # Factor 1: 섹터 로테이션
            f1 = self._analyze_sector_rotation(sector_performance)
            factors.append(f1)

            # Factor 2: 수익률 곡선
            f2 = self._analyze_yield_curve(stock_data)
            factors.append(f2)

            # Factor 3: 시장 너비
            f3 = self._analyze_market_breadth(market_breadth)
            factors.append(f3)

            # Factor 4: VIX
            f4 = self._analyze_vix(stock_data)
            factors.append(f4)

            # Factor 5: 위험 선호도 (IWM/SPY)
            f5 = self._analyze_risk_appetite(stock_data)
            factors.append(f5)

            # Factor 6: 신용 스프레드 (HYG/TLT)
            f6 = self._analyze_credit_spread(stock_data)
            factors.append(f6)

            # 국면 확률 계산
            phase_probs = self._calculate_phase_probabilities(factors)
            current_phase = max(phase_probs, key=phase_probs.get)
            phase_position = self._calculate_phase_position(phase_probs)

            # 주도/부진 섹터
            leading, lagging = self._get_leading_lagging_sectors(
                sector_performance, current_phase
            )

            summary = self._generate_summary(current_phase, phase_probs, factors)

            return BusinessCycleResult(
                current_phase=current_phase,
                phase_position=phase_position,
                factor_readings=factors,
                phase_probabilities=phase_probs,
                leading_sectors=leading,
                lagging_sectors=lagging,
                summary=summary,
            )
        except Exception as e:
            logger.error(f"경기 사이클 분석 실패: {e}")
            return None

    # ------------------------------------------------------------------
    # Factor 1: 섹터 로테이션
    # ------------------------------------------------------------------
    def _analyze_sector_rotation(self, sector_performance) -> CycleFactorReading:
        """섹터 월간 수익률 순위로 국면 판별"""
        weight = self.cfg.cycle_w_sector

        if not sector_performance:
            return CycleFactorReading(
                name="섹터 로테이션",
                value=None,
                display_value="N/A",
                phase_signal=None,
                weight=weight,
                description="섹터 데이터 없음",
            )

        # 월간 수익률 기준 상위 3개 섹터
        sorted_sectors = sorted(
            sector_performance, key=lambda s: s.monthly_return, reverse=True
        )
        top3_names = [s.name for s in sorted_sectors[:3]]

        # 각 국면에 대해 상위 3개와의 겹침 계산
        phase_scores = {}
        for phase, rotation_sectors in SECTOR_ROTATION.items():
            overlap = len(set(top3_names) & set(rotation_sectors))
            phase_scores[phase] = overlap

        best_phase = max(phase_scores, key=phase_scores.get)
        best_overlap = phase_scores[best_phase]

        if best_overlap == 0:
            phase_signal = None
            desc = f"상위: {', '.join(top3_names[:3])} (명확한 국면 매칭 없음)"
        else:
            phase_signal = best_phase
            desc = f"상위: {', '.join(top3_names[:3])} → {PHASE_LABELS[best_phase]} 패턴"

        return CycleFactorReading(
            name="섹터 로테이션",
            value=float(best_overlap),
            display_value=f"매칭 {best_overlap}/3",
            phase_signal=phase_signal,
            weight=weight,
            description=desc,
        )

    # ------------------------------------------------------------------
    # Factor 2: 수익률 곡선 (10Y-3M spread)
    # ------------------------------------------------------------------
    def _analyze_yield_curve(self, stock_data: Dict) -> CycleFactorReading:
        """10년-3개월 국채 스프레드로 국면 판별"""
        weight = self.cfg.cycle_w_yield

        tnx = stock_data.get("^TNX")  # 10Y yield
        irx = stock_data.get("^IRX")  # 3M yield

        if tnx is None or irx is None or tnx.empty or irx.empty:
            return CycleFactorReading(
                name="수익률 곡선",
                value=None,
                display_value="N/A",
                phase_signal=None,
                weight=weight,
                description="국채 수익률 데이터 없음",
            )

        try:
            tnx_val = float(tnx["Close"].iloc[-1])
            irx_val = float(irx["Close"].iloc[-1])
            spread = tnx_val - irx_val

            # 20일 전 스프레드 (추세)
            spread_trend = None
            if len(tnx) >= 20 and len(irx) >= 20:
                spread_20d_ago = float(tnx["Close"].iloc[-20]) - float(irx["Close"].iloc[-20])
                spread_trend = spread - spread_20d_ago

            steep = self.cfg.cycle_yield_curve_steep
            flat_threshold = self.cfg.cycle_yield_curve_flat
            inversion = self.cfg.cycle_yield_curve_inversion

            if spread < inversion:
                phase_signal = "contraction"
                desc = f"역전 ({spread:+.2f}%) → 수축 경고"
            elif spread < flat_threshold:
                # 플랫: 추세에 따라 과열 또는 수축
                if spread_trend is not None and spread_trend < 0:
                    phase_signal = "late"
                    desc = f"평탄화 중 ({spread:.2f}%, Δ{spread_trend:+.2f}) → 과열 후기"
                else:
                    phase_signal = "late"
                    desc = f"평탄 ({spread:.2f}%) → 과열기 신호"
            elif spread >= steep:
                # 가팔라짐: 회복기
                phase_signal = "early"
                desc = f"급경사 ({spread:.2f}%) → 회복기"
            else:
                # 보통 스프레드: 확장기
                phase_signal = "mid"
                desc = f"정상 ({spread:.2f}%) → 확장기"

            return CycleFactorReading(
                name="수익률 곡선",
                value=spread,
                display_value=f"{spread:.2f}%",
                phase_signal=phase_signal,
                weight=weight,
                description=desc,
            )
        except Exception as e:
            logger.warning(f"수익률 곡선 분석 실패: {e}")
            return CycleFactorReading(
                name="수익률 곡선",
                value=None,
                display_value="N/A",
                phase_signal=None,
                weight=weight,
                description="데이터 처리 오류",
            )

    # ------------------------------------------------------------------
    # Factor 3: 시장 너비 (A/D ratio)
    # ------------------------------------------------------------------
    def _analyze_market_breadth(self, market_breadth: Dict) -> CycleFactorReading:
        """A/D ratio로 국면 판별"""
        weight = self.cfg.cycle_w_breadth

        if not market_breadth:
            return CycleFactorReading(
                name="시장 너비",
                value=None,
                display_value="N/A",
                phase_signal=None,
                weight=weight,
                description="시장 너비 데이터 없음",
            )

        ad_ratio = market_breadth.get("advance_decline_ratio", 1.0)
        advancing = market_breadth.get("advancing", 0)
        declining = market_breadth.get("declining", 0)
        total = market_breadth.get("total", 0)

        strong = self.cfg.cycle_breadth_strong
        weak = self.cfg.cycle_breadth_weak

        if ad_ratio >= strong:
            phase_signal = "early"
            desc = f"강한 너비 (A/D {ad_ratio:.2f}) → 회복기 (광범위한 상승)"
        elif ad_ratio >= 1.0:
            phase_signal = "mid"
            desc = f"양호한 너비 (A/D {ad_ratio:.2f}) → 확장기"
        elif ad_ratio >= weak:
            phase_signal = "late"
            desc = f"약화되는 너비 (A/D {ad_ratio:.2f}) → 과열기 (소수 종목 주도)"
        else:
            phase_signal = "contraction"
            desc = f"빈약한 너비 (A/D {ad_ratio:.2f}) → 수축기"

        pct_advancing = round(advancing / total * 100, 1) if total > 0 else 0

        return CycleFactorReading(
            name="시장 너비",
            value=ad_ratio,
            display_value=f"A/D {ad_ratio:.2f} ({pct_advancing}%↑)",
            phase_signal=phase_signal,
            weight=weight,
            description=desc,
        )

    # ------------------------------------------------------------------
    # Factor 4: VIX
    # ------------------------------------------------------------------
    def _analyze_vix(self, stock_data: Dict) -> CycleFactorReading:
        """VIX 수준 + 추세로 국면 판별"""
        weight = self.cfg.cycle_w_vix

        vix_df = stock_data.get("^VIX")
        if vix_df is None or vix_df.empty:
            return CycleFactorReading(
                name="VIX",
                value=None,
                display_value="N/A",
                phase_signal=None,
                weight=weight,
                description="VIX 데이터 없음",
            )

        try:
            vix = float(vix_df["Close"].iloc[-1])
            vix_low = self.cfg.cycle_vix_low
            vix_high = self.cfg.cycle_vix_high

            # 20일 추세
            vix_trend = None
            if len(vix_df) >= 20:
                vix_20d = float(vix_df["Close"].iloc[-20])
                vix_trend = vix - vix_20d

            if vix > vix_high:
                phase_signal = "contraction"
                desc = f"높은 변동성 ({vix:.1f}) → 수축기 (공포)"
            elif vix > vix_low:
                if vix_trend is not None and vix_trend > 3:
                    phase_signal = "late"
                    desc = f"변동성 상승 중 ({vix:.1f}, Δ{vix_trend:+.1f}) → 과열 후기"
                elif vix_trend is not None and vix_trend < -3:
                    phase_signal = "early"
                    desc = f"변동성 하락 중 ({vix:.1f}, Δ{vix_trend:+.1f}) → 회복기"
                else:
                    phase_signal = "mid"
                    desc = f"보통 변동성 ({vix:.1f}) → 확장기"
            else:
                phase_signal = "mid"
                desc = f"낮은 변동성 ({vix:.1f}) → 확장기 (안정)"

            return CycleFactorReading(
                name="VIX",
                value=vix,
                display_value=f"{vix:.1f}",
                phase_signal=phase_signal,
                weight=weight,
                description=desc,
            )
        except Exception as e:
            logger.warning(f"VIX 분석 실패: {e}")
            return CycleFactorReading(
                name="VIX",
                value=None,
                display_value="N/A",
                phase_signal=None,
                weight=weight,
                description="VIX 처리 오류",
            )

    # ------------------------------------------------------------------
    # Factor 5: 위험 선호도 (IWM/SPY ratio 20일 변화)
    # ------------------------------------------------------------------
    def _analyze_risk_appetite(self, stock_data: Dict) -> CycleFactorReading:
        """IWM/SPY 비율의 20일 변화로 위험 선호도 판별"""
        weight = self.cfg.cycle_w_risk

        iwm = stock_data.get("IWM")
        spy = stock_data.get("SPY")

        if iwm is None or spy is None or iwm.empty or spy.empty:
            return CycleFactorReading(
                name="위험 선호도",
                value=None,
                display_value="N/A",
                phase_signal=None,
                weight=weight,
                description="IWM/SPY 데이터 없음",
            )

        try:
            if len(iwm) < 20 or len(spy) < 20:
                return CycleFactorReading(
                    name="위험 선호도",
                    value=None,
                    display_value="N/A",
                    phase_signal=None,
                    weight=weight,
                    description="데이터 부족 (20일 미만)",
                )

            ratio_now = float(iwm["Close"].iloc[-1]) / float(spy["Close"].iloc[-1])
            ratio_20d = float(iwm["Close"].iloc[-20]) / float(spy["Close"].iloc[-20])
            change_pct = (ratio_now / ratio_20d - 1) * 100

            if change_pct > 2.0:
                phase_signal = "early"
                desc = f"소형주 선호 (IWM/SPY {change_pct:+.2f}%) → 회복기"
            elif change_pct > 0:
                phase_signal = "mid"
                desc = f"소형주 소폭 강세 (IWM/SPY {change_pct:+.2f}%) → 확장기"
            elif change_pct > -2.0:
                phase_signal = "late"
                desc = f"대형주 선호 (IWM/SPY {change_pct:+.2f}%) → 과열기"
            else:
                phase_signal = "contraction"
                desc = f"안전자산 선호 (IWM/SPY {change_pct:+.2f}%) → 수축기"

            return CycleFactorReading(
                name="위험 선호도",
                value=change_pct,
                display_value=f"IWM/SPY {change_pct:+.1f}%",
                phase_signal=phase_signal,
                weight=weight,
                description=desc,
            )
        except Exception as e:
            logger.warning(f"위험 선호도 분석 실패: {e}")
            return CycleFactorReading(
                name="위험 선호도",
                value=None,
                display_value="N/A",
                phase_signal=None,
                weight=weight,
                description="데이터 처리 오류",
            )

    # ------------------------------------------------------------------
    # Factor 6: 신용 스프레드 (HYG/TLT ratio 20일 변화)
    # ------------------------------------------------------------------
    def _analyze_credit_spread(self, stock_data: Dict) -> CycleFactorReading:
        """HYG/TLT 비율의 20일 변화로 신용 환경 판별"""
        weight = self.cfg.cycle_w_credit

        hyg = stock_data.get("HYG")
        tlt = stock_data.get("TLT")

        if hyg is None or tlt is None or hyg.empty or tlt.empty:
            return CycleFactorReading(
                name="신용 스프레드",
                value=None,
                display_value="N/A",
                phase_signal=None,
                weight=weight,
                description="HYG/TLT 데이터 없음",
            )

        try:
            if len(hyg) < 20 or len(tlt) < 20:
                return CycleFactorReading(
                    name="신용 스프레드",
                    value=None,
                    display_value="N/A",
                    phase_signal=None,
                    weight=weight,
                    description="데이터 부족 (20일 미만)",
                )

            ratio_now = float(hyg["Close"].iloc[-1]) / float(tlt["Close"].iloc[-1])
            ratio_20d = float(hyg["Close"].iloc[-20]) / float(tlt["Close"].iloc[-20])
            change_pct = (ratio_now / ratio_20d - 1) * 100

            if change_pct > 1.5:
                phase_signal = "early"
                desc = f"신용 개선 (HYG/TLT {change_pct:+.2f}%) → 회복기"
            elif change_pct > 0:
                phase_signal = "mid"
                desc = f"신용 안정 (HYG/TLT {change_pct:+.2f}%) → 확장기"
            elif change_pct > -1.5:
                phase_signal = "late"
                desc = f"신용 약화 (HYG/TLT {change_pct:+.2f}%) → 과열기"
            else:
                phase_signal = "contraction"
                desc = f"신용 경색 (HYG/TLT {change_pct:+.2f}%) → 수축기"

            return CycleFactorReading(
                name="신용 스프레드",
                value=change_pct,
                display_value=f"HYG/TLT {change_pct:+.1f}%",
                phase_signal=phase_signal,
                weight=weight,
                description=desc,
            )
        except Exception as e:
            logger.warning(f"신용 스프레드 분석 실패: {e}")
            return CycleFactorReading(
                name="신용 스프레드",
                value=None,
                display_value="N/A",
                phase_signal=None,
                weight=weight,
                description="데이터 처리 오류",
            )

    # ------------------------------------------------------------------
    # 국면 확률 / 위치 계산
    # ------------------------------------------------------------------
    def _calculate_phase_probabilities(
        self, factors: List[CycleFactorReading]
    ) -> Dict[str, float]:
        """팩터 투표 가중합으로 국면 확률 계산"""
        votes = {p: 0.0 for p in PHASES}
        total_weight = 0.0

        for f in factors:
            if f.phase_signal is not None and f.phase_signal in votes:
                votes[f.phase_signal] += f.weight
                total_weight += f.weight
            # None인 팩터는 가중치에서 제외 (비례 재분배)

        if total_weight == 0:
            # 모든 팩터가 None → 균등 분배
            return {p: 0.25 for p in PHASES}

        return {p: round(v / total_weight, 3) for p, v in votes.items()}

    def _calculate_phase_position(self, phase_probs: Dict[str, float]) -> float:
        """원형 가중 평균(circular mean)으로 0-360도 위치 계산"""
        sin_sum = 0.0
        cos_sum = 0.0

        for phase, prob in phase_probs.items():
            angle_rad = math.radians(PHASE_ANGLES[phase])
            sin_sum += prob * math.sin(angle_rad)
            cos_sum += prob * math.cos(angle_rad)

        result_angle = math.degrees(math.atan2(sin_sum, cos_sum))
        if result_angle < 0:
            result_angle += 360

        return round(result_angle, 1)

    # ------------------------------------------------------------------
    # 주도/부진 섹터
    # ------------------------------------------------------------------
    def _get_leading_lagging_sectors(
        self, sector_performance, current_phase: str
    ) -> Tuple[List[str], List[str]]:
        """현재 국면에서 예상되는 주도/부진 섹터"""
        leading = SECTOR_ROTATION.get(current_phase, [])

        # 현재 국면의 반대 국면 (대각선)
        opposite = {
            "early": "late",
            "mid": "contraction",
            "late": "early",
            "contraction": "mid",
        }
        opp_phase = opposite.get(current_phase, "contraction")
        lagging = SECTOR_ROTATION.get(opp_phase, [])

        return leading, lagging

    # ------------------------------------------------------------------
    # 요약 문장
    # ------------------------------------------------------------------
    def _generate_summary(
        self,
        current_phase: str,
        phase_probs: Dict[str, float],
        factors: List[CycleFactorReading],
    ) -> str:
        """국면 분석 요약 문장 생성"""
        phase_label = PHASE_LABELS[current_phase]
        prob_pct = round(phase_probs[current_phase] * 100)

        # 유효 팩터 수
        valid_factors = sum(1 for f in factors if f.phase_signal is not None)
        agreeing = sum(
            1 for f in factors
            if f.phase_signal == current_phase
        )

        phase_desc = {
            "early": "경기 바닥을 지나 회복 초기 단계로, 금리 인하 기대와 함께 경기민감 섹터가 주도합니다.",
            "mid": "경기 확장이 안정적으로 진행 중이며, 기업 이익 성장이 시장을 뒷받침합니다.",
            "late": "경기 과열 우려와 인플레이션 압력이 높아지고 있으며, 원자재 섹터가 강세를 보입니다.",
            "contraction": "경기 둔화 또는 수축 국면으로, 방어적 섹터와 안전자산이 선호됩니다.",
        }

        return (
            f"현재 경기 사이클은 {phase_label} 국면으로 판단됩니다 "
            f"(신뢰도 {prob_pct}%, {valid_factors}개 팩터 중 {agreeing}개 일치). "
            f"{phase_desc[current_phase]}"
        )

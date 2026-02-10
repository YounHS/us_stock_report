"""매수 신호 종합 판별"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import logging

from config.settings import settings
from .technical import TechnicalAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class BuySignal:
    """매수 신호 정보"""
    ticker: str
    close: float
    change_pct: float
    signal_type: str  # "rsi", "macd", "sigma", "combined"
    signal_strength: int  # 1-3 (높을수록 강함)
    details: Dict


@dataclass
class ScoreBreakdown:
    """점수 구성 상세"""
    rsi_score: int = 0
    volume_score: int = 0
    adx_score: int = 0
    macd_score: int = 0
    bollinger_score: int = 0
    relative_strength_score: int = 0
    week52_score: int = 0
    obv_score: int = 0
    stochastic_score: int = 0
    squeeze_score: int = 0

    @property
    def total(self) -> int:
        return (
            self.rsi_score
            + self.volume_score
            + self.adx_score
            + self.macd_score
            + self.bollinger_score
            + self.relative_strength_score
            + self.week52_score
            + self.obv_score
            + self.stochastic_score
            + self.squeeze_score
        )


@dataclass
class EnhancedRecommendation:
    """향상된 추천 정보"""
    ticker: str
    score: int  # 0-100
    confidence: str  # "High" (>=70), "Medium" (50-69)
    close: float
    change_pct: float
    target_price: float
    stop_loss: float
    risk_reward_ratio: float
    bullish_factors: List[str]
    warning_factors: List[str]
    score_breakdown: ScoreBreakdown
    holding_period: str
    source: str
    # Technical indicator summary
    rsi: Optional[float] = None
    adx: Optional[float] = None
    volume_ratio: Optional[float] = None
    relative_strength_20d: Optional[float] = None
    week52_position: Optional[float] = None
    # Kalman filter
    kalman_predicted_price: Optional[float] = None
    kalman_trend_velocity: Optional[float] = None
    disclaimer: str = "본 추천은 기술적 분석에 기반한 것으로, 투자 결정은 본인의 판단하에 이루어져야 합니다."


class SignalDetector:
    """매수 신호 감지기"""

    def __init__(self, analysis_results: Dict[str, Dict]):
        """
        Args:
            analysis_results: TechnicalAnalyzer.analyze_batch() 결과
        """
        self.analysis_results = analysis_results

    def detect_rsi_oversold(self) -> List[BuySignal]:
        """RSI 과매도 종목 감지"""
        signals = []

        for ticker, analysis in self.analysis_results.items():
            rsi = analysis.get("rsi")
            if rsi and rsi.is_oversold:
                signals.append(BuySignal(
                    ticker=ticker,
                    close=analysis.get("close", 0),
                    change_pct=analysis.get("change_pct", 0),
                    signal_type="rsi",
                    signal_strength=self._rsi_strength(rsi.value),
                    details={
                        "rsi": rsi.value,
                        "threshold": settings.analysis.rsi_oversold,
                    }
                ))

        # RSI 낮은 순으로 정렬
        signals.sort(key=lambda x: x.details["rsi"])
        return signals

    def detect_macd_cross(self) -> List[BuySignal]:
        """MACD 골든크로스 종목 감지"""
        signals = []

        for ticker, analysis in self.analysis_results.items():
            macd = analysis.get("macd")
            if macd and macd.is_bullish_cross:
                signals.append(BuySignal(
                    ticker=ticker,
                    close=analysis.get("close", 0),
                    change_pct=analysis.get("change_pct", 0),
                    signal_type="macd",
                    signal_strength=2,  # 골든크로스는 중간 강도
                    details={
                        "macd_line": macd.macd_line,
                        "signal_line": macd.signal_line,
                        "histogram": macd.histogram,
                    }
                ))

        # 히스토그램 큰 순으로 정렬
        signals.sort(key=lambda x: x.details["histogram"], reverse=True)
        return signals

    def detect_sigma_reversion(self) -> List[BuySignal]:
        """1시그마 근접 (평균 회귀 후보) 종목 감지"""
        signals = []

        for ticker, analysis in self.analysis_results.items():
            bollinger = analysis.get("bollinger")
            if bollinger and bollinger.is_near_lower_sigma:
                signals.append(BuySignal(
                    ticker=ticker,
                    close=analysis.get("close", 0),
                    change_pct=analysis.get("change_pct", 0),
                    signal_type="sigma",
                    signal_strength=self._sigma_strength(bollinger.z_score),
                    details={
                        "z_score": bollinger.z_score,
                        "sma": bollinger.sma,
                        "lower_band": bollinger.lower_band,
                        "distance_to_sma_pct": round(
                            ((bollinger.sma - analysis.get("close", 0)) / analysis.get("close", 1)) * 100, 2
                        ),
                    }
                ))

        # Z-score 낮은 순으로 정렬 (하단에 가까울수록 먼저)
        signals.sort(key=lambda x: x.details["z_score"])
        return signals

    def detect_combined_signals(self) -> List[BuySignal]:
        """복합 신호 (RSI + MACD 또는 RSI + Sigma) 종목 감지"""
        signals = []

        for ticker, analysis in self.analysis_results.items():
            rsi = analysis.get("rsi")
            macd = analysis.get("macd")
            bollinger = analysis.get("bollinger")

            score = 0
            details = {}

            # RSI 과매도 체크
            if rsi and rsi.is_oversold:
                score += 1
                details["rsi"] = rsi.value

            # MACD 골든크로스 체크
            if macd and macd.is_bullish_cross:
                score += 1
                details["macd_cross"] = True
                details["histogram"] = macd.histogram

            # 1시그마 근접 체크
            if bollinger and bollinger.is_near_lower_sigma:
                score += 1
                details["z_score"] = bollinger.z_score

            # 2개 이상 신호가 겹치면 복합 신호로 등록
            if score >= 2:
                signals.append(BuySignal(
                    ticker=ticker,
                    close=analysis.get("close", 0),
                    change_pct=analysis.get("change_pct", 0),
                    signal_type="combined",
                    signal_strength=score,
                    details=details,
                ))

        # 강도 높은 순으로 정렬
        signals.sort(key=lambda x: x.signal_strength, reverse=True)
        return signals

    def get_all_signals(self) -> Dict[str, List[BuySignal]]:
        """모든 유형의 매수 신호 반환"""
        return {
            "rsi_oversold": self.detect_rsi_oversold(),
            "macd_golden_cross": self.detect_macd_cross(),
            "sigma_reversion": self.detect_sigma_reversion(),
            "combined": self.detect_combined_signals(),
        }

    def _rsi_strength(self, rsi_value: float) -> int:
        """RSI 값에 따른 신호 강도"""
        if rsi_value < 20:
            return 3  # 극심한 과매도
        elif rsi_value < 25:
            return 2
        else:
            return 1

    def _sigma_strength(self, z_score: float) -> int:
        """Z-Score에 따른 신호 강도"""
        if z_score < -1.5:
            return 3  # 2시그마 근접
        elif z_score < -1.1:
            return 2
        else:
            return 1

    def get_top_recommendation(self) -> Dict:
        """
        최종 추천 종목 선정

        선정 기준:
        1. 복합 신호 종목 우선 (2개 이상 조건 충족)
        2. 없으면 RSI 과매도 중 가장 낮은 종목
        3. 추천 근거와 보유 기간 제시

        Returns:
            추천 종목 정보 딕셔너리
        """
        signals = self.get_all_signals()

        recommendation = None
        reasons = []
        holding_period = ""

        # 1순위: 복합 신호 종목
        if signals["combined"]:
            top = signals["combined"][0]
            recommendation = top

            if top.details.get("rsi"):
                reasons.append(f"RSI {top.details['rsi']:.1f} (과매도 구간)")
            if top.details.get("macd_cross"):
                reasons.append("MACD 골든크로스 발생")
            if top.details.get("z_score"):
                reasons.append(f"볼린저밴드 하단 근접 (Z-score: {top.details['z_score']:.2f})")

            holding_period = "2-4주 (복수 신호로 신뢰도 높음)"
            source = "복합 기술적 분석 신호"

        # 2순위: RSI 과매도 중 가장 낮은 종목
        elif signals["rsi_oversold"]:
            top = signals["rsi_oversold"][0]
            recommendation = top

            reasons.append(f"RSI {top.details['rsi']:.1f}로 극심한 과매도 상태")
            reasons.append("단기 반등 가능성 높음")

            if top.details['rsi'] < 20:
                holding_period = "1-2주 (극단적 과매도, 빠른 반등 기대)"
            else:
                holding_period = "2-3주 (과매도 회복 대기)"
            source = "RSI 과매도 신호"

        # 3순위: MACD 골든크로스
        elif signals["macd_golden_cross"]:
            top = signals["macd_golden_cross"][0]
            recommendation = top

            reasons.append("MACD 골든크로스 발생")
            reasons.append(f"히스토그램: {top.details['histogram']:.4f}")

            holding_period = "3-4주 (추세 전환 확인 필요)"
            source = "MACD 골든크로스 신호"

        # 4순위: 1시그마 근접
        elif signals["sigma_reversion"]:
            top = signals["sigma_reversion"][0]
            recommendation = top

            reasons.append(f"볼린저밴드 하단 근접 (Z-score: {top.details['z_score']:.2f})")
            reasons.append(f"20일 이동평균 대비 {top.details['distance_to_sma_pct']}% 상승 여력")

            holding_period = "1-2주 (평균 회귀 전략)"
            source = "볼린저밴드 평균회귀 신호"

        if recommendation:
            # 매도 목표가 계산 (20일 SMA 기준)
            analysis = self.analysis_results.get(recommendation.ticker, {})
            bollinger = analysis.get("bollinger")
            rsi = analysis.get("rsi")
            macd = analysis.get("macd")
            adx = analysis.get("adx")
            volume = analysis.get("volume")
            rs = analysis.get("relative_strength")
            week52 = analysis.get("week52")
            atr = analysis.get("atr")
            kalman = analysis.get("kalman")

            if atr:
                target_price = atr.target_price  # Kalman 블렌딩 목표가
                target_return = round(((target_price - recommendation.close) / recommendation.close) * 100, 2)
            elif bollinger:
                target_price = bollinger.sma  # 20일 SMA
                target_return = round(((target_price - recommendation.close) / recommendation.close) * 100, 2)
            else:
                target_price = round(recommendation.close * 1.05, 2)  # 기본 5% 상승
                target_return = 5.0

            # 손절가 계산 (ATR 기반 또는 기본 5%)
            if atr:
                stop_loss = atr.stop_loss
            else:
                stop_loss = round(recommendation.close * 0.95, 2)

            return {
                "ticker": recommendation.ticker,
                "close": recommendation.close,
                "change_pct": recommendation.change_pct,
                "signal_strength": recommendation.signal_strength,
                "reasons": reasons,
                "holding_period": holding_period,
                "source": source,
                "target_price": round(target_price, 2),
                "target_return": target_return,
                "disclaimer": "본 추천은 기술적 분석에 기반한 것으로, 투자 결정은 본인의 판단하에 이루어져야 합니다.",
                # 기술적 지표 추가
                "rsi": round(rsi.value, 1) if rsi else None,
                "adx": round(adx.adx, 1) if adx else None,
                "volume_ratio": round(volume.volume_ratio, 2) if volume else None,
                "relative_strength_20d": round(rs.rs_20d, 1) if rs else None,
                "week52_position": round(week52.current_position_pct, 1) if week52 else None,
                "stop_loss": stop_loss,
                "risk_reward_ratio": round((target_price - recommendation.close) / (recommendation.close - stop_loss), 2) if recommendation.close > stop_loss else None,
                # MACD 정보
                "macd_histogram": round(macd.histogram, 4) if macd else None,
                "macd_signal": "골든크로스" if macd and macd.is_bullish_cross else None,
                # 볼린저밴드 정보
                "bollinger_z_score": round(bollinger.z_score, 2) if bollinger else None,
                "bollinger_sma": round(bollinger.sma, 2) if bollinger else None,
                # ATR 정보
                "atr_value": round(atr.atr, 2) if atr else None,
                "atr_pct": round(atr.atr / recommendation.close * 100, 2) if atr and recommendation.close else None,
                # Kalman 정보
                "kalman_predicted_price": round(kalman.predicted_price, 2) if kalman else None,
                "kalman_trend_velocity": round(kalman.trend_velocity, 4) if kalman else None,
            }

        return None

    # ========================
    # Enhanced Scoring System
    # ========================

    def _calculate_rsi_score(self, analysis: Dict) -> int:
        """RSI 점수 계산 (최대 weight_rsi점)"""
        rsi = analysis.get("rsi")
        if not rsi:
            return 0

        weight = settings.analysis.weight_rsi
        if rsi.is_oversold:
            if rsi.value < 20:
                return weight  # 극심한 과매도
            elif rsi.value < 25:
                return int(weight * 0.8)
            else:
                return int(weight * 0.6)
        return 0

    def _calculate_volume_score(self, analysis: Dict) -> int:
        """거래량 점수 계산 (최대 weight_volume점)"""
        volume = analysis.get("volume")
        rsi = analysis.get("rsi")

        if not volume:
            return 0

        weight = settings.analysis.weight_volume

        # 과매도 + 거래량 급증 = 바닥 신호 확인
        if rsi and rsi.is_oversold and volume.is_volume_spike:
            if volume.volume_ratio >= 2.0:
                return weight
            else:
                return int(weight * 0.8)

        # 과매도 중 거래량 감소 = 약한 신호 (페널티)
        if rsi and rsi.is_oversold and volume.volume_ratio < 0.7:
            return -int(weight * 0.5)

        return 0

    def _calculate_adx_score(self, analysis: Dict) -> int:
        """ADX 점수 계산 (최대 weight_adx점 또는 페널티)"""
        adx = analysis.get("adx")
        if not adx:
            return 0

        weight = settings.analysis.weight_adx

        # 약한 추세에서 평균회귀 유효 (점수 부여)
        if adx.trend_strength == "weak":
            return weight
        elif adx.trend_strength == "moderate" and adx.trend_direction != "bearish":
            return int(weight * 0.5)

        # 강한 하락 추세 = falling knife 위험 (페널티)
        if adx.trend_strength == "strong" and adx.trend_direction == "bearish":
            return -int(weight * 1.3)  # 20점 페널티

        return 0

    def _calculate_macd_score(self, analysis: Dict) -> int:
        """MACD 점수 계산 (최대 weight_macd점)"""
        macd = analysis.get("macd")
        if not macd:
            return 0

        weight = settings.analysis.weight_macd

        if macd.is_bullish_cross:
            # 골든크로스 발생
            if macd.histogram > 0:
                return weight
            else:
                return int(weight * 0.7)

        # 히스토그램 상승 전환 중 (아직 크로스 전이지만 반등 신호)
        if macd.histogram < 0 and macd.histogram > macd.signal_line * 0.5:
            return int(weight * 0.3)

        return 0

    def _calculate_bollinger_score(self, analysis: Dict) -> int:
        """볼린저밴드 점수 계산 (최대 weight_bollinger점)"""
        bollinger = analysis.get("bollinger")
        if not bollinger:
            return 0

        weight = settings.analysis.weight_bollinger

        if bollinger.is_near_lower_sigma:
            # z-score에 따른 점수 차등
            if bollinger.z_score <= -1.5:
                return weight  # 2시그마 근접
            elif bollinger.z_score <= -1.2:
                return int(weight * 0.8)
            else:
                return int(weight * 0.6)

        return 0

    def _calculate_relative_strength_score(self, analysis: Dict) -> int:
        """상대강도 점수 계산 (최대 weight_relative_strength점)"""
        rs = analysis.get("relative_strength")
        if not rs:
            return 0

        weight = settings.analysis.weight_relative_strength

        if rs.is_outperforming:
            # 시장 대비 아웃퍼폼
            if rs.rs_20d > 5:  # 20일 기준 5% 이상 아웃퍼폼
                return weight
            elif rs.rs_20d > 0:
                return int(weight * 0.7)

        return 0

    def _calculate_week52_score(self, analysis: Dict) -> int:
        """52주 위치 점수 계산 (최대 weight_52week점)"""
        week52 = analysis.get("week52")
        if not week52:
            return 0

        weight = settings.analysis.weight_52week

        if week52.is_near_low:
            # 52주 저점 근처
            if week52.current_position_pct <= 5:
                return weight  # 최저점 근처
            else:
                return int(weight * 0.7)

        return 0

    def _calculate_obv_score(self, analysis: Dict) -> int:
        """OBV 점수 계산 (최대 weight_obv점)"""
        obv = analysis.get("obv")
        rsi = analysis.get("rsi")

        if not obv:
            return 0

        weight = settings.analysis.weight_obv

        # Bullish divergence (가격 하락 중 OBV 상승) = 강한 매집 신호
        if obv.is_bullish_divergence:
            return weight

        # 매집 구간 + 과매도
        if obv.obv_trend == "accumulation":
            if rsi and rsi.is_oversold:
                return int(weight * 0.9)  # 과매도 + 매집 = 바닥 신호
            return int(weight * 0.6)

        # 분산 구간 = 페널티
        if obv.obv_trend == "distribution":
            return -int(weight * 0.5)

        return 0

    def _calculate_stochastic_score(self, analysis: Dict) -> int:
        """Stochastic 점수 계산 (최대 weight_stochastic점)"""
        stochastic = analysis.get("stochastic")
        rsi = analysis.get("rsi")

        if not stochastic:
            return 0

        weight = settings.analysis.weight_stochastic

        # RSI + Stochastic 동시 과매도 = 강한 반등 신호
        if stochastic.is_oversold and rsi and rsi.is_oversold:
            if stochastic.k < 15:  # 극심한 과매도
                return weight
            return int(weight * 0.8)

        # Stochastic bullish cross in oversold zone
        if stochastic.is_bullish_cross:
            return int(weight * 0.7)

        # 단독 Stochastic 과매도
        if stochastic.is_oversold:
            return int(weight * 0.4)

        return 0

    def _calculate_squeeze_score(self, analysis: Dict) -> int:
        """Squeeze 점수 계산 (최대 weight_squeeze점)"""
        squeeze = analysis.get("squeeze")
        rsi = analysis.get("rsi")

        if not squeeze:
            return 0

        weight = settings.analysis.weight_squeeze

        # Squeeze ON + 상승 모멘텀 = 곧 상방 돌파 예상
        if squeeze.is_squeeze_on and squeeze.momentum > 0:
            if squeeze.momentum_direction == "increasing":
                # 연속 squeeze 일수에 따른 보너스 (오래 눌릴수록 폭발력 큼)
                if squeeze.squeeze_count >= 5:
                    return weight
                return int(weight * 0.8)
            return int(weight * 0.5)

        # Squeeze 해제 직후 + 상승 모멘텀 = 돌파 진행 중
        if not squeeze.is_squeeze_on and squeeze.momentum > 0:
            if squeeze.momentum_direction == "increasing":
                return int(weight * 0.6)

        # Squeeze ON + 하락 모멘텀 (과매도 중이면 반등 준비)
        if squeeze.is_squeeze_on and squeeze.momentum < 0:
            if rsi and rsi.is_oversold:
                return int(weight * 0.4)

        return 0

    def _should_filter_out(self, analysis: Dict, score_breakdown: ScoreBreakdown) -> tuple[bool, str]:
        """
        필터링 조건 체크

        Returns:
            (should_filter: bool, reason: str)
        """
        adx = analysis.get("adx")
        volume = analysis.get("volume")
        rsi = analysis.get("rsi")
        atr = analysis.get("atr")

        # 1. ADX > 25 + 하락추세 (falling knife)
        if adx and adx.adx > settings.analysis.adx_weak_trend_threshold:
            if adx.trend_direction == "bearish":
                return True, "강한 하락 추세 (falling knife 위험)"

        # 2. 거래량 감소 중 과매도 (약한 신호)
        if volume and rsi and rsi.is_oversold:
            if volume.volume_ratio < 0.7:
                return True, "과매도 상태이나 거래량 감소 중"

        # 3. R:R < 2:1 (리스크 대비 수익 불리)
        if atr and atr.risk_reward_ratio < settings.analysis.min_risk_reward_ratio:
            return True, f"리스크 대비 수익률 불리 (R:R {atr.risk_reward_ratio:.1f}:1)"

        return False, ""

    def get_enhanced_recommendation(self) -> Optional[EnhancedRecommendation]:
        """
        가중치 점수 시스템 기반 추천 종목 선정

        Returns:
            EnhancedRecommendation or None
        """
        candidates = []

        for ticker, analysis in self.analysis_results.items():
            # 기본 지표가 없으면 스킵
            if not analysis.get("rsi") and not analysis.get("bollinger"):
                continue

            # 점수 계산
            score_breakdown = ScoreBreakdown(
                rsi_score=self._calculate_rsi_score(analysis),
                volume_score=self._calculate_volume_score(analysis),
                adx_score=self._calculate_adx_score(analysis),
                macd_score=self._calculate_macd_score(analysis),
                bollinger_score=self._calculate_bollinger_score(analysis),
                relative_strength_score=self._calculate_relative_strength_score(analysis),
                week52_score=self._calculate_week52_score(analysis),
                obv_score=self._calculate_obv_score(analysis),
                stochastic_score=self._calculate_stochastic_score(analysis),
                squeeze_score=self._calculate_squeeze_score(analysis),
            )

            total_score = score_breakdown.total

            # 최소 점수 미달 시 스킵
            if total_score < settings.analysis.min_recommendation_score:
                continue

            # 필터링 조건 체크
            should_filter, filter_reason = self._should_filter_out(analysis, score_breakdown)
            if should_filter:
                logger.debug(f"{ticker} 필터링됨: {filter_reason}")
                continue

            candidates.append((ticker, total_score, score_breakdown, analysis))

        if not candidates:
            return None

        # 점수 기준 정렬
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_ticker, best_score, best_breakdown, best_analysis = candidates[0]

        # 신뢰도 결정
        if best_score >= 70:
            confidence = "High"
        else:
            confidence = "Medium"

        # 매수 근거 및 주의사항 수집
        bullish_factors = []
        warning_factors = []

        rsi = best_analysis.get("rsi")
        macd = best_analysis.get("macd")
        bollinger = best_analysis.get("bollinger")
        volume = best_analysis.get("volume")
        adx = best_analysis.get("adx")
        rs = best_analysis.get("relative_strength")
        week52 = best_analysis.get("week52")
        atr = best_analysis.get("atr")
        kalman = best_analysis.get("kalman")
        obv = best_analysis.get("obv")
        stochastic = best_analysis.get("stochastic")
        squeeze = best_analysis.get("squeeze")

        # Bullish factors
        if rsi and rsi.is_oversold:
            bullish_factors.append(f"RSI {rsi.value:.1f} (과매도 구간)")

        if macd and macd.is_bullish_cross:
            bullish_factors.append("MACD 골든크로스 발생")

        if bollinger and bollinger.is_near_lower_sigma:
            bullish_factors.append(f"볼린저밴드 하단 근접 (Z-score: {bollinger.z_score:.2f})")

        if volume and volume.is_volume_spike:
            bullish_factors.append(f"거래량 급증 (평균 대비 {volume.volume_ratio:.1f}배)")

        if rs and rs.is_outperforming:
            bullish_factors.append(f"SPY 대비 20일 아웃퍼폼 (+{rs.rs_20d:.1f}%)")

        if week52 and week52.is_near_low:
            bullish_factors.append(f"52주 저점 근처 ({week52.current_position_pct:.1f}% 위치)")

        if adx and adx.trend_strength == "weak":
            bullish_factors.append("약한 추세 (평균회귀 유효)")

        # 새 지표 bullish factors
        if obv and obv.is_bullish_divergence:
            bullish_factors.append("OBV 다이버전스 (매집 신호)")
        elif obv and obv.obv_trend == "accumulation":
            bullish_factors.append("OBV 매집 구간")

        if stochastic and stochastic.is_oversold and rsi and rsi.is_oversold:
            bullish_factors.append(f"Stochastic+RSI 동시 과매도 (%K: {stochastic.k:.1f})")
        elif stochastic and stochastic.is_bullish_cross:
            bullish_factors.append("Stochastic 상향 돌파")

        if squeeze and squeeze.is_squeeze_on and squeeze.momentum > 0:
            bullish_factors.append(f"Squeeze ON + 상승 모멘텀 ({squeeze.squeeze_count}일 연속)")

        # Warning factors
        if adx and adx.trend_direction == "bearish":
            warning_factors.append("하락 추세 지속 중")

        if volume and volume.volume_ratio < 1.0:
            warning_factors.append("거래량 평균 이하")

        if rs and not rs.is_outperforming:
            warning_factors.append("시장 대비 언더퍼폼")

        if week52 and week52.current_position_pct > 50:
            warning_factors.append("52주 중간 이상 위치")

        # 새 지표 warning factors
        if obv and obv.obv_trend == "distribution":
            warning_factors.append("OBV 분산 구간 (매도 압력)")

        if stochastic and stochastic.is_overbought:
            warning_factors.append(f"Stochastic 과매수 (%K: {stochastic.k:.1f})")

        if squeeze and squeeze.momentum < 0 and squeeze.momentum_direction == "decreasing":
            warning_factors.append("Squeeze 하락 모멘텀 악화")

        # 목표가 및 손절가
        close_price = best_analysis.get("close", 0)

        if atr:
            target_price = atr.target_price
            stop_loss = atr.stop_loss
            risk_reward_ratio = atr.risk_reward_ratio
        elif bollinger:
            target_price = bollinger.sma
            stop_loss = round(close_price * 0.95, 2)  # 기본 5% 손절
            risk_reward_ratio = round((target_price - close_price) / (close_price - stop_loss), 2) if close_price > stop_loss else 0
        else:
            target_price = round(close_price * 1.05, 2)
            stop_loss = round(close_price * 0.95, 2)
            risk_reward_ratio = 1.0

        # 보유 기간 결정
        if best_score >= 70:
            holding_period = "2-4주 (복수 신호로 신뢰도 높음)"
        elif rsi and rsi.value < 20:
            holding_period = "1-2주 (극단적 과매도, 빠른 반등 기대)"
        else:
            holding_period = "2-3주 (평균회귀 전략)"

        # 출처 결정
        primary_factors = []
        if best_breakdown.rsi_score > 0:
            primary_factors.append("RSI")
        if best_breakdown.macd_score > 0:
            primary_factors.append("MACD")
        if best_breakdown.bollinger_score > 0:
            primary_factors.append("볼린저밴드")
        if best_breakdown.volume_score > 0:
            primary_factors.append("거래량")
        if best_breakdown.obv_score > 0:
            primary_factors.append("OBV")
        if best_breakdown.stochastic_score > 0:
            primary_factors.append("Stochastic")
        if best_breakdown.squeeze_score > 0:
            primary_factors.append("Squeeze")

        source = f"가중치 점수 시스템 ({', '.join(primary_factors[:3])})"

        return EnhancedRecommendation(
            ticker=best_ticker,
            score=best_score,
            confidence=confidence,
            close=close_price,
            change_pct=best_analysis.get("change_pct", 0),
            target_price=target_price,
            stop_loss=stop_loss,
            risk_reward_ratio=risk_reward_ratio,
            bullish_factors=bullish_factors,
            warning_factors=warning_factors,
            score_breakdown=best_breakdown,
            holding_period=holding_period,
            source=source,
            rsi=rsi.value if rsi else None,
            adx=adx.adx if adx else None,
            volume_ratio=volume.volume_ratio if volume else None,
            relative_strength_20d=rs.rs_20d if rs else None,
            week52_position=week52.current_position_pct if week52 else None,
            kalman_predicted_price=kalman.predicted_price if kalman else None,
            kalman_trend_velocity=kalman.trend_velocity if kalman else None,
        )

    # ============================
    # Kalman Filter Recommendation
    # ============================

    def get_kalman_recommendation(
        self, exclude_tickers: Optional[List[str]] = None
    ) -> Optional[EnhancedRecommendation]:
        """
        칼만 예측가 > 현재가 필터 기반 추천 종목 선정

        기존 Enhanced 점수 시스템을 재사용하되, 칼만 예측가가
        현재가보다 높은 종목만 후보로 선정하여 상승 여력이 있는 종목을 추천.

        Args:
            exclude_tickers: 제외할 종목 리스트 (기존 추천과 중복 방지)

        Returns:
            EnhancedRecommendation or None
        """
        exclude = set(exclude_tickers or [])
        candidates = []

        for ticker, analysis in self.analysis_results.items():
            # 제외 종목 스킵
            if ticker in exclude:
                continue

            # 기본 지표가 없으면 스킵
            if not analysis.get("rsi") and not analysis.get("bollinger"):
                continue

            # 하드 필터: 칼만 예측가 > 현재가
            kalman = analysis.get("kalman")
            close = analysis.get("close", 0)
            if not kalman or not close or kalman.predicted_price <= close:
                continue

            # 점수 계산 (기존 Enhanced와 동일)
            score_breakdown = ScoreBreakdown(
                rsi_score=self._calculate_rsi_score(analysis),
                volume_score=self._calculate_volume_score(analysis),
                adx_score=self._calculate_adx_score(analysis),
                macd_score=self._calculate_macd_score(analysis),
                bollinger_score=self._calculate_bollinger_score(analysis),
                relative_strength_score=self._calculate_relative_strength_score(analysis),
                week52_score=self._calculate_week52_score(analysis),
                obv_score=self._calculate_obv_score(analysis),
                stochastic_score=self._calculate_stochastic_score(analysis),
                squeeze_score=self._calculate_squeeze_score(analysis),
            )

            total_score = score_breakdown.total

            # 최소 점수 미달 시 스킵
            if total_score < settings.analysis.min_recommendation_score:
                continue

            # 필터링 조건 체크
            should_filter, filter_reason = self._should_filter_out(analysis, score_breakdown)
            if should_filter:
                logger.debug(f"{ticker} 칼만 필터링됨: {filter_reason}")
                continue

            candidates.append((ticker, total_score, score_breakdown, analysis))

        if not candidates:
            return None

        # 점수 기준 정렬
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_ticker, best_score, best_breakdown, best_analysis = candidates[0]

        # 신뢰도 결정
        if best_score >= 70:
            confidence = "High"
        else:
            confidence = "Medium"

        # 매수 근거 및 주의사항 수집
        bullish_factors = []
        warning_factors = []

        rsi = best_analysis.get("rsi")
        macd = best_analysis.get("macd")
        bollinger = best_analysis.get("bollinger")
        volume = best_analysis.get("volume")
        adx = best_analysis.get("adx")
        rs = best_analysis.get("relative_strength")
        week52 = best_analysis.get("week52")
        atr = best_analysis.get("atr")
        kalman = best_analysis.get("kalman")
        obv = best_analysis.get("obv")
        stochastic = best_analysis.get("stochastic")
        squeeze = best_analysis.get("squeeze")
        close_price = best_analysis.get("close", 0)

        # 칼만 필터 관련 bullish factor (이 메서드의 핵심)
        if kalman and close_price:
            kalman_upside = ((kalman.predicted_price - close_price) / close_price) * 100
            bullish_factors.append(f"칼만 예측가 ${kalman.predicted_price:.2f} (현재가 대비 +{kalman_upside:.1f}%)")

        # 기존 Enhanced와 동일한 bullish/warning factors
        if rsi and rsi.is_oversold:
            bullish_factors.append(f"RSI {rsi.value:.1f} (과매도 구간)")

        if macd and macd.is_bullish_cross:
            bullish_factors.append("MACD 골든크로스 발생")

        if bollinger and bollinger.is_near_lower_sigma:
            bullish_factors.append(f"볼린저밴드 하단 근접 (Z-score: {bollinger.z_score:.2f})")

        if volume and volume.is_volume_spike:
            bullish_factors.append(f"거래량 급증 (평균 대비 {volume.volume_ratio:.1f}배)")

        if rs and rs.is_outperforming:
            bullish_factors.append(f"SPY 대비 20일 아웃퍼폼 (+{rs.rs_20d:.1f}%)")

        if week52 and week52.is_near_low:
            bullish_factors.append(f"52주 저점 근처 ({week52.current_position_pct:.1f}% 위치)")

        if adx and adx.trend_strength == "weak":
            bullish_factors.append("약한 추세 (평균회귀 유효)")

        if obv and obv.is_bullish_divergence:
            bullish_factors.append("OBV 다이버전스 (매집 신호)")
        elif obv and obv.obv_trend == "accumulation":
            bullish_factors.append("OBV 매집 구간")

        if stochastic and stochastic.is_oversold and rsi and rsi.is_oversold:
            bullish_factors.append(f"Stochastic+RSI 동시 과매도 (%K: {stochastic.k:.1f})")
        elif stochastic and stochastic.is_bullish_cross:
            bullish_factors.append("Stochastic 상향 돌파")

        if squeeze and squeeze.is_squeeze_on and squeeze.momentum > 0:
            bullish_factors.append(f"Squeeze ON + 상승 모멘텀 ({squeeze.squeeze_count}일 연속)")

        # Warning factors
        if adx and adx.trend_direction == "bearish":
            warning_factors.append("하락 추세 지속 중")

        if volume and volume.volume_ratio < 1.0:
            warning_factors.append("거래량 평균 이하")

        if rs and not rs.is_outperforming:
            warning_factors.append("시장 대비 언더퍼폼")

        if week52 and week52.current_position_pct > 50:
            warning_factors.append("52주 중간 이상 위치")

        if obv and obv.obv_trend == "distribution":
            warning_factors.append("OBV 분산 구간 (매도 압력)")

        if stochastic and stochastic.is_overbought:
            warning_factors.append(f"Stochastic 과매수 (%K: {stochastic.k:.1f})")

        if squeeze and squeeze.momentum < 0 and squeeze.momentum_direction == "decreasing":
            warning_factors.append("Squeeze 하락 모멘텀 악화")

        # 목표가 및 손절가
        if atr:
            target_price = atr.target_price
            stop_loss = atr.stop_loss
            risk_reward_ratio = atr.risk_reward_ratio
        elif bollinger:
            target_price = bollinger.sma
            stop_loss = round(close_price * 0.95, 2)
            risk_reward_ratio = round((target_price - close_price) / (close_price - stop_loss), 2) if close_price > stop_loss else 0
        else:
            target_price = round(close_price * 1.05, 2)
            stop_loss = round(close_price * 0.95, 2)
            risk_reward_ratio = 1.0

        # 보유 기간 결정
        if best_score >= 70:
            holding_period = "2-4주 (복수 신호로 신뢰도 높음)"
        elif rsi and rsi.value < 20:
            holding_period = "1-2주 (극단적 과매도, 빠른 반등 기대)"
        else:
            holding_period = "2-3주 (평균회귀 전략)"

        # 출처 결정
        primary_factors = ["칼만필터"]
        if best_breakdown.rsi_score > 0:
            primary_factors.append("RSI")
        if best_breakdown.macd_score > 0:
            primary_factors.append("MACD")
        if best_breakdown.bollinger_score > 0:
            primary_factors.append("볼린저밴드")
        if best_breakdown.volume_score > 0:
            primary_factors.append("거래량")
        if best_breakdown.obv_score > 0:
            primary_factors.append("OBV")
        if best_breakdown.stochastic_score > 0:
            primary_factors.append("Stochastic")
        if best_breakdown.squeeze_score > 0:
            primary_factors.append("Squeeze")

        source = f"칼만 필터 + 가중치 점수 ({', '.join(primary_factors[:3])})"

        return EnhancedRecommendation(
            ticker=best_ticker,
            score=best_score,
            confidence=confidence,
            close=close_price,
            change_pct=best_analysis.get("change_pct", 0),
            target_price=target_price,
            stop_loss=stop_loss,
            risk_reward_ratio=risk_reward_ratio,
            bullish_factors=bullish_factors,
            warning_factors=warning_factors,
            score_breakdown=best_breakdown,
            holding_period=holding_period,
            source=source,
            rsi=rsi.value if rsi else None,
            adx=adx.adx if adx else None,
            volume_ratio=volume.volume_ratio if volume else None,
            relative_strength_20d=rs.rs_20d if rs else None,
            week52_position=week52.current_position_pct if week52 else None,
            kalman_predicted_price=kalman.predicted_price if kalman else None,
            kalman_trend_velocity=kalman.trend_velocity if kalman else None,
        )

    # ============================
    # Long-term Recommendation
    # ============================

    LONGTERM_EXCLUDED = {"SPY", "QQQ", "DIA", "IWM"}

    def _longterm_passes_hard_filters(self, ticker: str, analysis: Dict) -> bool:
        """장기 추천 하드 필터 (하나라도 실패하면 제외)"""
        if ticker in self.LONGTERM_EXCLUDED:
            return False

        kalman = analysis.get("kalman_longterm")
        adx = analysis.get("adx")
        rsi = analysis.get("rsi")
        volume = analysis.get("volume")

        # 1. Kalman velocity > 0 (상승 추세 필수)
        if not kalman or kalman.trend_velocity <= 0:
            return False

        # 2. ADX 방향 != bearish (하락 추세 불가)
        if adx and adx.trend_direction == "bearish":
            return False

        # 3. RSI < 75 (과매수 아님)
        if rsi and rsi.value >= 75:
            return False

        # 4. 거래량 비율 >= 0.7 (거래량 유지)
        if volume and volume.volume_ratio < 0.7:
            return False

        return True

    def _longterm_rsi_score(self, analysis: Dict) -> int:
        """장기 RSI 점수 (최대 longterm_weight_rsi점) — RSI 45-60이 최적"""
        rsi = analysis.get("rsi")
        if not rsi:
            return 0

        weight = settings.analysis.longterm_weight_rsi
        v = rsi.value

        if 45 <= v <= 60:
            return weight  # 건강한 상승대
        elif 40 <= v < 45 or 60 < v <= 65:
            return int(weight * 0.7)
        elif 35 <= v < 40 or 65 < v <= 70:
            return int(weight * 0.4)
        return 0

    def _longterm_macd_score(self, analysis: Dict) -> int:
        """장기 MACD 점수 (최대 longterm_weight_macd점) — MACD > Signal & MACD > 0"""
        macd = analysis.get("macd")
        if not macd:
            return 0

        weight = settings.analysis.longterm_weight_macd
        score = 0

        if macd.macd_line > macd.signal_line:
            score += int(weight * 0.6)
        if macd.macd_line > 0:
            score += int(weight * 0.4)

        return min(score, weight)

    def _longterm_bollinger_score(self, analysis: Dict) -> int:
        """장기 볼린저 점수 (최대 longterm_weight_bollinger점) — Z-score 0.3~1.0 (SMA 상방)"""
        bollinger = analysis.get("bollinger")
        if not bollinger:
            return 0

        weight = settings.analysis.longterm_weight_bollinger
        z = bollinger.z_score

        if 0.3 <= z <= 1.0:
            return weight  # SMA 위, 과열 아님
        elif 0.0 <= z < 0.3:
            return int(weight * 0.6)
        elif 1.0 < z <= 1.5:
            return int(weight * 0.4)
        return 0

    def _longterm_volume_score(self, analysis: Dict) -> int:
        """장기 거래량 점수 (최대 longterm_weight_volume점) — 거래량 >= 1.3x 평균"""
        volume = analysis.get("volume")
        if not volume:
            return 0

        weight = settings.analysis.longterm_weight_volume
        ratio = volume.volume_ratio

        if ratio >= 1.5:
            return weight
        elif ratio >= 1.3:
            return int(weight * 0.8)
        elif ratio >= 1.0:
            return int(weight * 0.5)
        return 0

    def _longterm_adx_score(self, analysis: Dict) -> int:
        """장기 ADX 점수 (최대 longterm_weight_adx점) — ADX >= 30 + bullish"""
        adx = analysis.get("adx")
        if not adx:
            return 0

        weight = settings.analysis.longterm_weight_adx

        if adx.trend_direction == "bullish":
            if adx.adx >= 30:
                return weight  # 강한 상승 추세
            elif adx.adx >= 25:
                return int(weight * 0.7)
            elif adx.adx >= 20:
                return int(weight * 0.4)
        elif adx.trend_direction == "neutral":
            if adx.adx >= 25:
                return int(weight * 0.3)
        return 0

    def _longterm_relative_strength_score(self, analysis: Dict) -> int:
        """장기 상대강도 점수 (최대 longterm_weight_relative_strength점) — RS 20d > 5% 아웃퍼폼"""
        rs = analysis.get("relative_strength")
        if not rs:
            return 0

        weight = settings.analysis.longterm_weight_relative_strength

        if rs.rs_20d > 5:
            return weight  # 강한 아웃퍼폼
        elif rs.rs_20d > 2:
            return int(weight * 0.7)
        elif rs.rs_20d > 0:
            return int(weight * 0.4)
        return 0

    def _longterm_week52_score(self, analysis: Dict) -> int:
        """장기 52주 위치 점수 (최대 longterm_weight_week52점) — 40-70% 위치 (성장 여력)"""
        week52 = analysis.get("week52")
        if not week52:
            return 0

        weight = settings.analysis.longterm_weight_week52
        pos = week52.current_position_pct

        if 40 <= pos <= 70:
            return weight  # 성장 여력 충분
        elif 30 <= pos < 40 or 70 < pos <= 80:
            return int(weight * 0.6)
        elif 20 <= pos < 30 or 80 < pos <= 90:
            return int(weight * 0.3)
        return 0

    def _longterm_kalman_score(self, analysis: Dict) -> int:
        """장기 칼만 점수 (최대 longterm_weight_kalman점) — velocity/price > 0.3%"""
        kalman = analysis.get("kalman_longterm")
        close = analysis.get("close", 0)
        if not kalman or not close:
            return 0

        weight = settings.analysis.longterm_weight_kalman
        velocity_pct = (kalman.trend_velocity / close) * 100

        if velocity_pct > 0.5:
            return weight
        elif velocity_pct > 0.3:
            return int(weight * 0.7)
        elif velocity_pct > 0.1:
            return int(weight * 0.4)
        return 0

    def _longterm_obv_score(self, analysis: Dict) -> int:
        """장기 OBV 점수 (최대 longterm_weight_obv점) — 매집 구간 확인"""
        obv = analysis.get("obv")
        if not obv:
            return 0

        weight = settings.analysis.longterm_weight_obv

        # 매집 구간 = 기관/세력 매수 중
        if obv.obv_trend == "accumulation":
            if obv.obv > obv.obv_sma:  # OBV가 평균 위
                return weight
            return int(weight * 0.7)

        # 다이버전스 (장기에서는 중립적)
        if obv.is_bullish_divergence:
            return int(weight * 0.5)

        # 분산 구간 = 매도 압력
        if obv.obv_trend == "distribution":
            return 0

        return int(weight * 0.3)  # neutral

    def _longterm_stochastic_score(self, analysis: Dict) -> int:
        """장기 Stochastic 점수 (최대 longterm_weight_stochastic점) — 건강한 모멘텀 구간"""
        stochastic = analysis.get("stochastic")
        if not stochastic:
            return 0

        weight = settings.analysis.longterm_weight_stochastic
        k = stochastic.k

        # 장기 투자에서는 50-70 구간이 건강한 상승 모멘텀
        if 50 <= k <= 70:
            return weight
        elif 40 <= k < 50 or 70 < k <= 75:
            return int(weight * 0.6)
        elif 30 <= k < 40:
            return int(weight * 0.3)
        # 과매수 구간 = 주의
        elif k > 80:
            return 0

        return 0

    def _longterm_squeeze_score(self, analysis: Dict) -> int:
        """장기 Squeeze 점수 (최대 longterm_weight_squeeze점) — Squeeze 해제 + 상승 모멘텀"""
        squeeze = analysis.get("squeeze")
        if not squeeze:
            return 0

        weight = settings.analysis.longterm_weight_squeeze

        # Squeeze 해제 직후 + 상승 모멘텀 = 추세 시작
        if not squeeze.is_squeeze_on and squeeze.momentum > 0:
            if squeeze.momentum_direction == "increasing":
                return weight
            return int(weight * 0.7)

        # Squeeze ON + 상승 모멘텀 = 곧 돌파 예상
        if squeeze.is_squeeze_on and squeeze.momentum > 0:
            if squeeze.squeeze_count >= 5:
                return int(weight * 0.8)
            return int(weight * 0.5)

        # 하락 모멘텀
        if squeeze.momentum < 0:
            return 0

        return int(weight * 0.3)

    def get_longterm_recommendations(self) -> List[Dict]:
        """
        장기 투자 추천 종목 선정 (추세 추종 기반)

        Returns:
            추천 종목 리스트 (최대 longterm_top_n개)
        """
        candidates = []

        for ticker, analysis in self.analysis_results.items():
            # 하드 필터
            if not self._longterm_passes_hard_filters(ticker, analysis):
                continue

            # 11개 점수 합산
            rsi_sc = self._longterm_rsi_score(analysis)
            macd_sc = self._longterm_macd_score(analysis)
            bollinger_sc = self._longterm_bollinger_score(analysis)
            volume_sc = self._longterm_volume_score(analysis)
            adx_sc = self._longterm_adx_score(analysis)
            rs_sc = self._longterm_relative_strength_score(analysis)
            week52_sc = self._longterm_week52_score(analysis)
            kalman_sc = self._longterm_kalman_score(analysis)
            obv_sc = self._longterm_obv_score(analysis)
            stochastic_sc = self._longterm_stochastic_score(analysis)
            squeeze_sc = self._longterm_squeeze_score(analysis)

            total = (rsi_sc + macd_sc + bollinger_sc + volume_sc + adx_sc +
                     rs_sc + week52_sc + kalman_sc + obv_sc + stochastic_sc + squeeze_sc)

            if total < settings.analysis.longterm_min_score:
                continue

            candidates.append((ticker, total, {
                "rsi": rsi_sc,
                "macd": macd_sc,
                "bollinger": bollinger_sc,
                "volume": volume_sc,
                "adx": adx_sc,
                "relative_strength": rs_sc,
                "week52": week52_sc,
                "kalman": kalman_sc,
                "obv": obv_sc,
                "stochastic": stochastic_sc,
                "squeeze": squeeze_sc,
            }, analysis))

        # 점수 내림차순
        candidates.sort(key=lambda x: x[1], reverse=True)

        top_n = settings.analysis.longterm_top_n
        results = []

        for ticker, score, breakdown, analysis in candidates[:top_n]:
            rsi = analysis.get("rsi")
            macd = analysis.get("macd")
            bollinger = analysis.get("bollinger")
            volume = analysis.get("volume")
            adx = analysis.get("adx")
            rs = analysis.get("relative_strength")
            week52 = analysis.get("week52")
            kalman = analysis.get("kalman_longterm")
            atr = analysis.get("atr")
            obv = analysis.get("obv")
            stochastic = analysis.get("stochastic")
            squeeze = analysis.get("squeeze")
            close = analysis.get("close", 0)
            change_pct = analysis.get("change_pct", 0)

            # 투자 근거 수집 (상위 5개)
            reasons = []
            if adx and adx.trend_direction == "bullish":
                reasons.append(f"ADX {adx.adx:.1f} — 상승 추세 확인")
            if rs and rs.rs_20d > 0:
                reasons.append(f"SPY 대비 20일 +{rs.rs_20d:.1f}% 아웃퍼폼")
            if macd and macd.macd_line > macd.signal_line:
                reasons.append("MACD 상승 모멘텀 유지")
            if kalman and close:
                n_days = settings.analysis.longterm_prediction_days
                longterm_pred = kalman.predicted_price  # N-step 예측 (이미 계산됨)
                expected_return = ((longterm_pred - close) / close) * 100
                reasons.append(f"칼만 {n_days}일 예측 ${longterm_pred:.2f} ({'+' if expected_return >= 0 else ''}{expected_return:.1f}%)")
            if obv and obv.obv_trend == "accumulation":
                reasons.append("OBV 매집 구간 — 세력 매수 중")
            if squeeze and not squeeze.is_squeeze_on and squeeze.momentum > 0:
                reasons.append("Squeeze 해제 — 상승 돌파 진행")
            if rsi:
                reasons.append(f"RSI {rsi.value:.1f} — 건강한 모멘텀 구간")
            if week52:
                reasons.append(f"52주 {week52.current_position_pct:.0f}% 위치 — 성장 여력")

            reasons = reasons[:5]

            rec = {
                "ticker": ticker,
                "close": close,
                "change_pct": change_pct,
                "score": score,
                "reasons": reasons,
                "holding_period": "1-3개월 (추세 추종 전략)",
                "disclaimer": "본 추천은 기술적 분석에 기반한 것으로, 투자 결정은 본인의 판단하에 이루어져야 합니다.",
                # 기술적 지표
                "rsi": round(rsi.value, 1) if rsi else None,
                "adx": round(adx.adx, 1) if adx else None,
                "macd_signal": "골든크로스" if macd and macd.is_bullish_cross else ("상승" if macd and macd.macd_line > macd.signal_line else None),
                "bollinger_z_score": round(bollinger.z_score, 2) if bollinger else None,
                "volume_ratio": round(volume.volume_ratio, 2) if volume else None,
                "atr_pct": round(atr.atr / close * 100, 2) if atr and close else None,
                "relative_strength_20d": round(rs.rs_20d, 1) if rs else None,
                "week52_position": round(week52.current_position_pct, 1) if week52 else None,
                # 칼만 필터 (장기 전용 N-step 예측)
                "kalman_predicted_price": round(kalman.predicted_price, 2) if kalman else None,
                "kalman_trend_velocity": round(kalman.trend_velocity, 4) if kalman else None,
                # 새 지표
                "obv_trend": obv.obv_trend if obv else None,
                "stochastic_k": round(stochastic.k, 1) if stochastic else None,
                "squeeze_status": "ON" if squeeze and squeeze.is_squeeze_on else ("OFF" if squeeze else None),
                "squeeze_momentum": squeeze.momentum_direction if squeeze else None,
                "score_breakdown": breakdown,
            }

            logger.info(f"   [Long-term] 추천: {ticker} (점수: {score})")
            results.append(rec)

        return results

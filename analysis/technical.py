"""기술적 분석 지표 계산: RSI, MACD, 볼린저밴드"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from config.settings import settings


@dataclass
class RSIResult:
    """RSI 분석 결과"""
    value: float
    is_oversold: bool
    is_overbought: bool


@dataclass
class MACDResult:
    """MACD 분석 결과"""
    macd_line: float
    signal_line: float
    histogram: float
    is_bullish_cross: bool  # 골든크로스


@dataclass
class BollingerResult:
    """볼린저밴드 분석 결과"""
    sma: float
    upper_band: float
    lower_band: float
    z_score: float
    is_near_lower_sigma: bool  # 1시그마 근접


@dataclass
class VolumeResult:
    """거래량 분석 결과"""
    current_volume: int
    avg_volume: float
    volume_ratio: float
    is_volume_spike: bool  # 평균 대비 급증 여부


@dataclass
class ADXResult:
    """ADX 분석 결과"""
    adx: float
    plus_di: float
    minus_di: float
    trend_strength: str  # "strong", "moderate", "weak"
    trend_direction: str  # "bullish", "bearish", "neutral"


@dataclass
class RelativeStrengthResult:
    """SPY 대비 상대강도 결과"""
    rs_5d: float  # 5일 상대강도
    rs_10d: float  # 10일 상대강도
    rs_20d: float  # 20일 상대강도
    is_outperforming: bool  # 시장 대비 아웃퍼폼 여부


@dataclass
class Week52Result:
    """52주 고/저점 분석 결과"""
    high_52w: float
    low_52w: float
    current_position_pct: float  # 0=52주 저점, 100=52주 고점
    is_near_low: bool  # 저점 근처 여부


@dataclass
class ATRResult:
    """ATR 및 리스크 관리 결과"""
    atr: float
    stop_loss: float
    target_price: float
    risk_reward_ratio: float


@dataclass
class KalmanResult:
    """칼만 필터 분석 결과"""
    filtered_price: float      # 노이즈 제거된 현재가
    predicted_price: float     # 다음 스텝 예측가
    trend_velocity: float      # 추세 속도 (일일 변화율)
    blended_target: float      # Bollinger SMA와 블렌딩된 목표가


@dataclass
class OBVResult:
    """OBV (On Balance Volume) 분석 결과"""
    obv: float                 # 현재 OBV 값
    obv_sma: float             # OBV 이동평균
    obv_trend: str             # "accumulation", "distribution", "neutral"
    is_bullish_divergence: bool  # 가격 횡보/하락 중 OBV 상승


@dataclass
class StochasticResult:
    """Stochastic Oscillator 분석 결과"""
    k: float                   # %K 값
    d: float                   # %D 값 (K의 이동평균)
    is_oversold: bool          # %K < 20
    is_overbought: bool        # %K > 80
    is_bullish_cross: bool     # %K가 %D를 상향 돌파 (과매도 구간에서)


@dataclass
class SqueezeResult:
    """TTM Squeeze 분석 결과"""
    is_squeeze_on: bool        # 볼린저가 켈트너 안에 있음 (변동성 수축)
    momentum: float            # 모멘텀 값
    momentum_direction: str    # "increasing", "decreasing"
    squeeze_count: int         # 연속 squeeze 일수


class TechnicalAnalyzer:
    """기술적 분석 지표 계산기"""

    def __init__(self):
        self.rsi_period = settings.analysis.rsi_period
        self.rsi_oversold = settings.analysis.rsi_oversold
        self.rsi_overbought = settings.analysis.rsi_overbought
        self.macd_fast = settings.analysis.macd_fast
        self.macd_slow = settings.analysis.macd_slow
        self.macd_signal = settings.analysis.macd_signal
        self.bollinger_period = settings.analysis.bollinger_period
        self.sigma_threshold = settings.analysis.sigma_threshold

        # New indicators settings
        self.volume_period = settings.analysis.volume_period
        self.volume_spike_threshold = settings.analysis.volume_spike_threshold
        self.adx_period = settings.analysis.adx_period
        self.adx_weak_trend_threshold = settings.analysis.adx_weak_trend_threshold
        self.rs_short_period = settings.analysis.rs_short_period
        self.rs_medium_period = settings.analysis.rs_medium_period
        self.rs_long_period = settings.analysis.rs_long_period
        self.week52_near_low_pct = settings.analysis.week52_near_low_pct
        self.week52_lookback_days = settings.analysis.week52_lookback_days
        self.atr_period = settings.analysis.atr_period
        self.atr_stop_multiplier = settings.analysis.atr_stop_multiplier
        self.kalman_process_variance = settings.analysis.kalman_process_variance
        self.kalman_measurement_variance = settings.analysis.kalman_measurement_variance
        self.kalman_blend_alpha = settings.analysis.kalman_blend_alpha

        # Long-term Kalman settings
        self.longterm_kalman_process_variance = settings.analysis.longterm_kalman_process_variance
        self.longterm_kalman_measurement_variance = settings.analysis.longterm_kalman_measurement_variance
        self.longterm_kalman_blend_alpha = settings.analysis.longterm_kalman_blend_alpha
        self.longterm_kalman_sma_period = settings.analysis.longterm_kalman_sma_period
        self.longterm_prediction_days = settings.analysis.longterm_prediction_days

        # OBV settings
        self.obv_sma_period = settings.analysis.obv_sma_period

        # Stochastic settings
        self.stochastic_k_period = settings.analysis.stochastic_k_period
        self.stochastic_d_period = settings.analysis.stochastic_d_period
        self.stochastic_oversold = settings.analysis.stochastic_oversold
        self.stochastic_overbought = settings.analysis.stochastic_overbought

        # Squeeze settings
        self.squeeze_bb_period = settings.analysis.squeeze_bb_period
        self.squeeze_bb_mult = settings.analysis.squeeze_bb_mult
        self.squeeze_kc_period = settings.analysis.squeeze_kc_period
        self.squeeze_kc_mult = settings.analysis.squeeze_kc_mult

    def calculate_rsi(self, close_prices: pd.Series) -> Optional[RSIResult]:
        """
        RSI (Relative Strength Index) 계산

        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        """
        if len(close_prices) < self.rsi_period + 1:
            return None

        # 일일 변동폭
        delta = close_prices.diff()

        # 상승분/하락분 분리
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # EMA 방식으로 평균 계산
        avg_gain = gain.ewm(span=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_period, adjust=False).mean()

        # RS 계산 (0으로 나누기 방지)
        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))

        latest_rsi = rsi.iloc[-1]

        return RSIResult(
            value=round(latest_rsi, 2),
            is_oversold=latest_rsi < self.rsi_oversold,
            is_overbought=latest_rsi > self.rsi_overbought,
        )

    def calculate_macd(self, close_prices: pd.Series) -> Optional[MACDResult]:
        """
        MACD (Moving Average Convergence Divergence) 계산

        MACD Line = EMA(12) - EMA(26)
        Signal Line = EMA(9) of MACD Line
        Histogram = MACD Line - Signal Line
        """
        if len(close_prices) < self.macd_slow + self.macd_signal:
            return None

        # EMA 계산
        ema_fast = close_prices.ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = close_prices.ewm(span=self.macd_slow, adjust=False).mean()

        # MACD Line
        macd_line = ema_fast - ema_slow

        # Signal Line
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()

        # Histogram
        histogram = macd_line - signal_line

        # 골든크로스 판별 (오늘 MACD > Signal, 어제 MACD <= Signal)
        is_bullish_cross = False
        if len(macd_line) >= 2:
            today_cross = macd_line.iloc[-1] > signal_line.iloc[-1]
            yesterday_below = macd_line.iloc[-2] <= signal_line.iloc[-2]
            is_bullish_cross = today_cross and yesterday_below

        return MACDResult(
            macd_line=round(macd_line.iloc[-1], 4),
            signal_line=round(signal_line.iloc[-1], 4),
            histogram=round(histogram.iloc[-1], 4),
            is_bullish_cross=is_bullish_cross,
        )

    def calculate_bollinger(self, close_prices: pd.Series) -> Optional[BollingerResult]:
        """
        볼린저밴드 및 Z-Score 계산

        Middle Band = SMA(20)
        Upper Band = SMA + 2 * STD
        Lower Band = SMA - 2 * STD
        Z-Score = (현재가 - SMA) / STD
        """
        if len(close_prices) < self.bollinger_period:
            return None

        # 이동평균과 표준편차
        sma = close_prices.rolling(window=self.bollinger_period).mean()
        std = close_prices.rolling(window=self.bollinger_period).std()

        # 밴드 계산
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)

        # Z-Score 계산
        z_score = (close_prices - sma) / std

        latest_z = z_score.iloc[-1]

        # 1시그마 근접 판별 (-1.2 <= z <= -0.8)
        is_near_lower_sigma = -1.2 <= latest_z <= self.sigma_threshold

        return BollingerResult(
            sma=round(sma.iloc[-1], 2),
            upper_band=round(upper_band.iloc[-1], 2),
            lower_band=round(lower_band.iloc[-1], 2),
            z_score=round(latest_z, 2),
            is_near_lower_sigma=is_near_lower_sigma,
        )

    def calculate_volume(self, df: pd.DataFrame) -> Optional[VolumeResult]:
        """
        거래량 분석

        Args:
            df: OHLCV DataFrame (Volume 컬럼 필수)

        Returns:
            VolumeResult or None
        """
        if df.empty or "Volume" not in df.columns:
            return None

        if len(df) < self.volume_period:
            return None

        volume = df["Volume"]
        current_volume = int(volume.iloc[-1])
        avg_volume = volume.rolling(window=self.volume_period).mean().iloc[-1]

        if avg_volume == 0:
            return None

        volume_ratio = current_volume / avg_volume
        is_volume_spike = volume_ratio >= self.volume_spike_threshold

        return VolumeResult(
            current_volume=current_volume,
            avg_volume=round(avg_volume, 0),
            volume_ratio=round(volume_ratio, 2),
            is_volume_spike=is_volume_spike,
        )

    def calculate_adx(self, df: pd.DataFrame) -> Optional[ADXResult]:
        """
        ADX (Average Directional Index) 및 방향 지표 계산

        Args:
            df: OHLCV DataFrame (High, Low, Close 컬럼 필수)

        Returns:
            ADXResult or None
        """
        if df.empty or not all(col in df.columns for col in ["High", "Low", "Close"]):
            return None

        if len(df) < self.adx_period * 2:
            return None

        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        # Smoothed averages
        atr = tr.ewm(span=self.adx_period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=self.adx_period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=self.adx_period, adjust=False).mean() / atr)

        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.inf)
        adx = dx.ewm(span=self.adx_period, adjust=False).mean()

        latest_adx = adx.iloc[-1]
        latest_plus_di = plus_di.iloc[-1]
        latest_minus_di = minus_di.iloc[-1]

        # Trend strength
        if latest_adx >= 40:
            trend_strength = "strong"
        elif latest_adx >= self.adx_weak_trend_threshold:
            trend_strength = "moderate"
        else:
            trend_strength = "weak"

        # Trend direction
        if latest_plus_di > latest_minus_di:
            trend_direction = "bullish"
        elif latest_minus_di > latest_plus_di:
            trend_direction = "bearish"
        else:
            trend_direction = "neutral"

        return ADXResult(
            adx=round(latest_adx, 2),
            plus_di=round(latest_plus_di, 2),
            minus_di=round(latest_minus_di, 2),
            trend_strength=trend_strength,
            trend_direction=trend_direction,
        )

    def calculate_relative_strength(
        self, close: pd.Series, spy_close: pd.Series
    ) -> Optional[RelativeStrengthResult]:
        """
        SPY 대비 상대강도 계산

        Args:
            close: 개별 종목 종가 시리즈
            spy_close: SPY 종가 시리즈

        Returns:
            RelativeStrengthResult or None
        """
        if len(close) < self.rs_long_period or len(spy_close) < self.rs_long_period:
            return None

        # Align series
        min_len = min(len(close), len(spy_close))
        close = close.iloc[-min_len:]
        spy_close = spy_close.iloc[-min_len:]

        # Calculate returns
        stock_returns_5d = (close.iloc[-1] / close.iloc[-self.rs_short_period] - 1) * 100
        spy_returns_5d = (spy_close.iloc[-1] / spy_close.iloc[-self.rs_short_period] - 1) * 100

        stock_returns_10d = (close.iloc[-1] / close.iloc[-self.rs_medium_period] - 1) * 100
        spy_returns_10d = (spy_close.iloc[-1] / spy_close.iloc[-self.rs_medium_period] - 1) * 100

        stock_returns_20d = (close.iloc[-1] / close.iloc[-self.rs_long_period] - 1) * 100
        spy_returns_20d = (spy_close.iloc[-1] / spy_close.iloc[-self.rs_long_period] - 1) * 100

        # Relative strength (stock return - SPY return)
        rs_5d = stock_returns_5d - spy_returns_5d
        rs_10d = stock_returns_10d - spy_returns_10d
        rs_20d = stock_returns_20d - spy_returns_20d

        # Outperforming if positive on at least 2 of 3 timeframes
        outperform_count = sum([rs_5d > 0, rs_10d > 0, rs_20d > 0])
        is_outperforming = outperform_count >= 2

        return RelativeStrengthResult(
            rs_5d=round(rs_5d, 2),
            rs_10d=round(rs_10d, 2),
            rs_20d=round(rs_20d, 2),
            is_outperforming=is_outperforming,
        )

    def calculate_52week(self, close: pd.Series) -> Optional[Week52Result]:
        """
        52주 고/저점 분석

        Args:
            close: 종가 시리즈

        Returns:
            Week52Result or None
        """
        # Use available data up to lookback days
        lookback = min(len(close), self.week52_lookback_days)
        if lookback < 20:  # Minimum data requirement
            return None

        close_52w = close.iloc[-lookback:]
        high_52w = close_52w.max()
        low_52w = close_52w.min()
        current_price = close.iloc[-1]

        # Position as percentage (0 = at 52w low, 100 = at 52w high)
        if high_52w == low_52w:
            current_position_pct = 50.0
        else:
            current_position_pct = ((current_price - low_52w) / (high_52w - low_52w)) * 100

        # Near low if within threshold percentage from 52w low
        is_near_low = current_position_pct <= self.week52_near_low_pct

        return Week52Result(
            high_52w=round(high_52w, 2),
            low_52w=round(low_52w, 2),
            current_position_pct=round(current_position_pct, 2),
            is_near_low=is_near_low,
        )

    def calculate_atr(
        self, df: pd.DataFrame, target_price: float
    ) -> Optional[ATRResult]:
        """
        ATR 및 리스크 관리 계산

        Args:
            df: OHLCV DataFrame
            target_price: 목표 매도가

        Returns:
            ATRResult or None
        """
        if df.empty or not all(col in df.columns for col in ["High", "Low", "Close"]):
            return None

        if len(df) < self.atr_period:
            return None

        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR
        atr = tr.rolling(window=self.atr_period).mean().iloc[-1]

        current_price = close.iloc[-1]
        stop_loss = current_price - (atr * self.atr_stop_multiplier)

        # Risk = current price - stop loss
        # Reward = target price - current price
        risk = current_price - stop_loss
        reward = target_price - current_price

        if risk <= 0:
            risk_reward_ratio = 0.0
        else:
            risk_reward_ratio = reward / risk

        return ATRResult(
            atr=round(atr, 2),
            stop_loss=round(stop_loss, 2),
            target_price=round(target_price, 2),
            risk_reward_ratio=round(risk_reward_ratio, 2),
        )

    def calculate_kalman_filter(
        self, close_prices: pd.Series, bollinger_sma: Optional[float] = None
    ) -> Optional[KalmanResult]:
        """
        1D 칼만 필터로 가격 필터링 및 예측

        State vector: [price, velocity]
        Observation: closing price

        Args:
            close_prices: 종가 시리즈
            bollinger_sma: 볼린저 SMA (블렌딩용, None이면 블렌딩 없음)

        Returns:
            KalmanResult or None
        """
        if len(close_prices) < 10:
            return None

        prices = close_prices.values.astype(float)

        Q = self.kalman_process_variance
        R = self.kalman_measurement_variance
        alpha = self.kalman_blend_alpha

        # State: [price, velocity]
        x = np.array([prices[0], 0.0])

        # Initial covariance
        P = np.array([[1.0, 0.0],
                      [0.0, 1.0]])

        # State transition: price += velocity, velocity stays
        F = np.array([[1.0, 1.0],
                      [0.0, 1.0]])

        # Observation: we observe only price
        H = np.array([[1.0, 0.0]])

        # Process noise covariance
        Q_mat = Q * np.eye(2)

        # Measurement noise covariance
        R_mat = np.array([[R]])

        # Run filter
        for z in prices:
            # Predict
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q_mat

            # Update
            y = z - H @ x_pred
            S = H @ P_pred @ H.T + R_mat
            K = P_pred @ H.T / S[0, 0]

            x = x_pred + K.flatten() * y
            P = (np.eye(2) - K @ H) @ P_pred

        filtered_price = x[0]
        trend_velocity = x[1]

        # 1-step prediction
        x_next = F @ x
        predicted_price = x_next[0]

        # Blend with Bollinger SMA
        if bollinger_sma is not None:
            blended_target = alpha * predicted_price + (1 - alpha) * bollinger_sma
        else:
            blended_target = predicted_price

        return KalmanResult(
            filtered_price=round(filtered_price, 2),
            predicted_price=round(predicted_price, 2),
            trend_velocity=round(trend_velocity, 4),
            blended_target=round(blended_target, 2),
        )

    def calculate_kalman_filter_longterm(
        self, close_prices: pd.Series, sma_anchor: Optional[float] = None
    ) -> Optional[KalmanResult]:
        """
        장기 전용 칼만 필터 (추세 추종에 적합한 파라미터)

        단기 칼만 필터 대비:
        - process_variance 100배 높음 → 추세 변화에 빠르게 반응
        - 50일 SMA와 블렌딩 → 장기 평균으로 앵커링
        - N-step prediction → longterm_prediction_days 거래일 후 예측가

        Args:
            close_prices: 종가 시리즈
            sma_anchor: 50일 SMA (블렌딩용, None이면 블렌딩 없음)

        Returns:
            KalmanResult or None
        """
        if len(close_prices) < 10:
            return None

        prices = close_prices.values.astype(float)

        Q = self.longterm_kalman_process_variance
        R = self.longterm_kalman_measurement_variance
        alpha = self.longterm_kalman_blend_alpha
        N = self.longterm_prediction_days

        # State: [price, velocity]
        x = np.array([prices[0], 0.0])

        # Initial covariance
        P = np.array([[1.0, 0.0],
                      [0.0, 1.0]])

        # State transition: price += velocity, velocity stays
        F = np.array([[1.0, 1.0],
                      [0.0, 1.0]])

        # Observation: we observe only price
        H = np.array([[1.0, 0.0]])

        # Process noise covariance
        Q_mat = Q * np.eye(2)

        # Measurement noise covariance
        R_mat = np.array([[R]])

        # Run filter
        for z in prices:
            # Predict
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q_mat

            # Update
            y = z - H @ x_pred
            S = H @ P_pred @ H.T + R_mat
            K = P_pred @ H.T / S[0, 0]

            x = x_pred + K.flatten() * y
            P = (np.eye(2) - K @ H) @ P_pred

        filtered_price = x[0]
        trend_velocity = x[1]

        # N-step prediction: F^N @ x
        x_n = x.copy()
        for _ in range(N):
            x_n = F @ x_n
        predicted_price = x_n[0]

        # Blend with long-term SMA
        if sma_anchor is not None:
            blended_target = alpha * predicted_price + (1 - alpha) * sma_anchor
        else:
            blended_target = predicted_price

        return KalmanResult(
            filtered_price=round(filtered_price, 2),
            predicted_price=round(predicted_price, 2),
            trend_velocity=round(trend_velocity, 4),
            blended_target=round(blended_target, 2),
        )

    def calculate_obv(self, df: pd.DataFrame) -> Optional[OBVResult]:
        """
        OBV (On Balance Volume) 계산

        OBV는 거래량 누적 지표로, 가격 상승일에는 거래량을 더하고
        가격 하락일에는 거래량을 빼서 매집/분산 신호를 감지

        Args:
            df: OHLCV DataFrame (Close, Volume 컬럼 필수)

        Returns:
            OBVResult or None
        """
        if df.empty or not all(col in df.columns for col in ["Close", "Volume"]):
            return None

        if len(df) < self.obv_sma_period + 1:
            return None

        close = df["Close"]
        volume = df["Volume"]

        # OBV 계산
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = 0

        for i in range(1, len(df)):
            if close.iloc[i] > close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] + volume.iloc[i]
            elif close.iloc[i] < close.iloc[i - 1]:
                obv.iloc[i] = obv.iloc[i - 1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i - 1]

        current_obv = obv.iloc[-1]
        obv_sma = obv.rolling(window=self.obv_sma_period).mean().iloc[-1]

        # OBV 추세 판단 (최근 5일 기준)
        obv_5d_ago = obv.iloc[-5] if len(obv) >= 5 else obv.iloc[0]
        price_5d_ago = close.iloc[-5] if len(close) >= 5 else close.iloc[0]

        obv_change = current_obv - obv_5d_ago
        price_change = close.iloc[-1] - price_5d_ago

        # 추세 판단
        if obv_change > 0 and current_obv > obv_sma:
            obv_trend = "accumulation"
        elif obv_change < 0 and current_obv < obv_sma:
            obv_trend = "distribution"
        else:
            obv_trend = "neutral"

        # Bullish Divergence: 가격 횡보/하락 중 OBV 상승
        is_bullish_divergence = (
            obv_change > 0 and price_change <= 0 and current_obv > obv_sma
        )

        return OBVResult(
            obv=round(current_obv, 0),
            obv_sma=round(obv_sma, 0),
            obv_trend=obv_trend,
            is_bullish_divergence=is_bullish_divergence,
        )

    def calculate_stochastic(
        self, close: pd.Series, high: pd.Series, low: pd.Series
    ) -> Optional[StochasticResult]:
        """
        Stochastic Oscillator 계산

        %K = (현재가 - N일 최저) / (N일 최고 - N일 최저) × 100
        %D = %K의 M일 이동평균

        Args:
            close: 종가 시리즈
            high: 고가 시리즈
            low: 저가 시리즈

        Returns:
            StochasticResult or None
        """
        if len(close) < self.stochastic_k_period + self.stochastic_d_period:
            return None

        # %K 계산
        lowest_low = low.rolling(window=self.stochastic_k_period).min()
        highest_high = high.rolling(window=self.stochastic_k_period).max()

        k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.inf)

        # %D 계산 (K의 SMA)
        d = k.rolling(window=self.stochastic_d_period).mean()

        latest_k = k.iloc[-1]
        latest_d = d.iloc[-1]

        is_oversold = latest_k < self.stochastic_oversold
        is_overbought = latest_k > self.stochastic_overbought

        # Bullish cross in oversold zone
        is_bullish_cross = False
        if len(k) >= 2 and len(d) >= 2:
            # 오늘 K > D, 어제 K <= D (in oversold zone)
            today_above = latest_k > latest_d
            yesterday_below = k.iloc[-2] <= d.iloc[-2]
            in_oversold_zone = latest_k < 30  # 과매도 구간 근처에서의 교차
            is_bullish_cross = today_above and yesterday_below and in_oversold_zone

        return StochasticResult(
            k=round(latest_k, 2),
            d=round(latest_d, 2),
            is_oversold=is_oversold,
            is_overbought=is_overbought,
            is_bullish_cross=is_bullish_cross,
        )

    def calculate_squeeze(self, df: pd.DataFrame) -> Optional[SqueezeResult]:
        """
        TTM Squeeze Indicator 계산

        Squeeze ON: 볼린저밴드가 켈트너 채널 안에 있음 (변동성 수축)
        Squeeze OFF: 볼린저밴드가 켈트너 채널 밖으로 나옴 (변동성 폭발)
        Momentum: 가격 - Donchian 중심선의 선형회귀

        Args:
            df: OHLCV DataFrame (High, Low, Close 컬럼 필수)

        Returns:
            SqueezeResult or None
        """
        if df.empty or not all(col in df.columns for col in ["High", "Low", "Close"]):
            return None

        if len(df) < max(self.squeeze_bb_period, self.squeeze_kc_period) + 5:
            return None

        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        # 볼린저밴드 계산
        bb_sma = close.rolling(window=self.squeeze_bb_period).mean()
        bb_std = close.rolling(window=self.squeeze_bb_period).std()
        bb_upper = bb_sma + (self.squeeze_bb_mult * bb_std)
        bb_lower = bb_sma - (self.squeeze_bb_mult * bb_std)

        # True Range for Keltner Channel
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=self.squeeze_kc_period).mean()

        # Keltner Channel 계산
        kc_middle = close.rolling(window=self.squeeze_kc_period).mean()
        kc_upper = kc_middle + (self.squeeze_kc_mult * atr)
        kc_lower = kc_middle - (self.squeeze_kc_mult * atr)

        # Squeeze 판단 (볼린저가 켈트너 안에 있으면 Squeeze ON)
        squeeze_on = (bb_lower.iloc[-1] > kc_lower.iloc[-1]) and (
            bb_upper.iloc[-1] < kc_upper.iloc[-1]
        )

        # 연속 squeeze 일수 계산
        squeeze_series = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        squeeze_count = 0
        for i in range(len(squeeze_series) - 1, -1, -1):
            if squeeze_series.iloc[i]:
                squeeze_count += 1
            else:
                break

        # Momentum 계산 (Donchian 중심선 기준)
        donchian_mid = (
            high.rolling(window=self.squeeze_bb_period).max()
            + low.rolling(window=self.squeeze_bb_period).min()
        ) / 2
        momentum_basis = (donchian_mid + bb_sma) / 2
        momentum = close - momentum_basis

        current_momentum = momentum.iloc[-1]
        prev_momentum = momentum.iloc[-2] if len(momentum) >= 2 else 0

        # 모멘텀 방향 판단
        if current_momentum > prev_momentum:
            momentum_direction = "increasing"
        elif current_momentum < prev_momentum:
            momentum_direction = "decreasing"
        else:
            momentum_direction = "neutral"

        return SqueezeResult(
            is_squeeze_on=squeeze_on,
            momentum=round(current_momentum, 4),
            momentum_direction=momentum_direction,
            squeeze_count=squeeze_count,
        )

    def analyze(self, df: pd.DataFrame, spy_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        모든 기술적 지표 종합 분석

        Args:
            df: OHLCV DataFrame (Close 컬럼 필수)
            spy_df: SPY OHLCV DataFrame (상대강도 계산용)

        Returns:
            {
                "rsi": RSIResult,
                "macd": MACDResult,
                "bollinger": BollingerResult,
                "volume": VolumeResult,
                "adx": ADXResult,
                "relative_strength": RelativeStrengthResult,
                "week52": Week52Result,
                "close": float,
                "change_pct": float
            }
        """
        if df.empty or "Close" not in df.columns:
            return {}

        close = df["Close"]
        bollinger = self.calculate_bollinger(close)

        result = {
            "rsi": self.calculate_rsi(close),
            "macd": self.calculate_macd(close),
            "bollinger": bollinger,
            "volume": self.calculate_volume(df),
            "adx": self.calculate_adx(df),
            "week52": self.calculate_52week(close),
            "close": round(close.iloc[-1], 2),
        }

        # Relative strength (requires SPY data)
        if spy_df is not None and "Close" in spy_df.columns:
            result["relative_strength"] = self.calculate_relative_strength(
                close, spy_df["Close"]
            )
        else:
            result["relative_strength"] = None

        # Kalman filter
        kalman = self.calculate_kalman_filter(
            close, bollinger.sma if bollinger else None
        )
        result["kalman"] = kalman

        # Long-term Kalman filter (장기 전용 파라미터)
        sma_50 = close.rolling(self.longterm_kalman_sma_period).mean().iloc[-1] if len(close) >= self.longterm_kalman_sma_period else None
        result["kalman_longterm"] = self.calculate_kalman_filter_longterm(close, sma_50)

        # ATR with Kalman-blended target (fallback to Bollinger SMA)
        if kalman and kalman.blended_target > 0:
            result["atr"] = self.calculate_atr(df, kalman.blended_target)
        elif bollinger:
            result["atr"] = self.calculate_atr(df, bollinger.sma)
        else:
            result["atr"] = None

        # OBV (On Balance Volume)
        result["obv"] = self.calculate_obv(df)

        # Stochastic Oscillator
        if "High" in df.columns and "Low" in df.columns:
            result["stochastic"] = self.calculate_stochastic(
                close, df["High"], df["Low"]
            )
        else:
            result["stochastic"] = None

        # TTM Squeeze
        result["squeeze"] = self.calculate_squeeze(df)

        # 일간 변동률 계산
        if len(close) >= 2:
            change_pct = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100
            result["change_pct"] = round(change_pct, 2)
        else:
            result["change_pct"] = 0.0

        return result

    def analyze_batch(
        self, data: Dict[str, pd.DataFrame], spy_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Dict]:
        """
        여러 종목 일괄 분석

        Args:
            data: {ticker: DataFrame} 형태
            spy_df: SPY OHLCV DataFrame (상대강도 계산용)

        Returns:
            {ticker: analysis_result} 형태
        """
        results = {}
        for ticker, df in data.items():
            analysis = self.analyze(df, spy_df)
            if analysis:
                results[ticker] = analysis
        return results

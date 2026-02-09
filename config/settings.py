import os
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()


class SMTPSettings(BaseSettings):
    host: str = Field(default="smtp.gmail.com", alias="SMTP_HOST")
    port: int = Field(default=587, alias="SMTP_PORT")
    user: str = Field(default="", alias="SMTP_USER")
    password: str = Field(default="", alias="SMTP_PASSWORD")


class EmailSettings(BaseSettings):
    from_address: str = Field(default="", alias="EMAIL_FROM")
    recipients: str = Field(default="", alias="EMAIL_RECIPIENTS")
    subject_prefix: str = Field(default="[주식리포트]", alias="EMAIL_SUBJECT_PREFIX")

    @property
    def recipient_list(self) -> list[str]:
        return [r.strip() for r in self.recipients.split(",") if r.strip()]


class AnalysisSettings(BaseSettings):
    lookback_days: int = Field(default=60, alias="DATA_LOOKBACK_DAYS")
    rsi_period: int = Field(default=14, alias="RSI_PERIOD")
    rsi_oversold: int = Field(default=30, alias="RSI_OVERSOLD")
    rsi_overbought: int = Field(default=70, alias="RSI_OVERBOUGHT")
    macd_fast: int = Field(default=12, alias="MACD_FAST")
    macd_slow: int = Field(default=26, alias="MACD_SLOW")
    macd_signal: int = Field(default=9, alias="MACD_SIGNAL")
    bollinger_period: int = Field(default=20, alias="BOLLINGER_PERIOD")
    sigma_threshold: float = Field(default=-1.0, alias="SIGMA_THRESHOLD")

    # Volume
    volume_period: int = Field(default=20, alias="VOLUME_AVG_PERIOD")
    volume_spike_threshold: float = Field(default=1.5, alias="VOLUME_SPIKE_THRESHOLD")

    # ADX
    adx_period: int = Field(default=14, alias="ADX_PERIOD")
    adx_weak_trend_threshold: int = Field(default=25, alias="ADX_WEAK_TREND")

    # Relative Strength
    rs_short_period: int = Field(default=5, alias="RS_SHORT_PERIOD")
    rs_medium_period: int = Field(default=10, alias="RS_MEDIUM_PERIOD")
    rs_long_period: int = Field(default=20, alias="RS_LONG_PERIOD")

    # 52-Week
    week52_near_low_pct: float = Field(default=10.0, alias="WEEK52_NEAR_LOW_PCT")
    week52_lookback_days: int = Field(default=252, alias="WEEK52_LOOKBACK_DAYS")

    # ATR
    atr_period: int = Field(default=14, alias="ATR_PERIOD")
    atr_stop_multiplier: float = Field(default=2.0, alias="ATR_STOP_MULTIPLIER")
    min_risk_reward_ratio: float = Field(default=2.0, alias="MIN_RISK_REWARD_RATIO")

    # Kalman Filter
    kalman_process_variance: float = Field(default=1e-5, alias="KALMAN_PROCESS_VARIANCE")
    kalman_measurement_variance: float = Field(default=1e-2, alias="KALMAN_MEASUREMENT_VARIANCE")
    kalman_blend_alpha: float = Field(default=0.5, alias="KALMAN_BLEND_ALPHA")

    # OBV (On Balance Volume)
    obv_sma_period: int = Field(default=20, alias="OBV_SMA_PERIOD")

    # Stochastic Oscillator
    stochastic_k_period: int = Field(default=14, alias="STOCHASTIC_K_PERIOD")
    stochastic_d_period: int = Field(default=3, alias="STOCHASTIC_D_PERIOD")
    stochastic_oversold: int = Field(default=20, alias="STOCHASTIC_OVERSOLD")
    stochastic_overbought: int = Field(default=80, alias="STOCHASTIC_OVERBOUGHT")

    # TTM Squeeze
    squeeze_bb_period: int = Field(default=20, alias="SQUEEZE_BB_PERIOD")
    squeeze_bb_mult: float = Field(default=2.0, alias="SQUEEZE_BB_MULT")
    squeeze_kc_period: int = Field(default=20, alias="SQUEEZE_KC_PERIOD")
    squeeze_kc_mult: float = Field(default=1.5, alias="SQUEEZE_KC_MULT")

    # Scoring Weights (총합 100점 유지)
    # 기존: RSI(20) + Volume(15) + ADX(15) + MACD(15) + BB(15) + RS(10) + 52W(10) = 100
    # 변경: RSI(15) + Volume(10) + ADX(12) + MACD(12) + BB(12) + RS(8) + 52W(8) + OBV(8) + Stoch(8) + Squeeze(7) = 100
    weight_rsi: int = Field(default=15, alias="WEIGHT_RSI")
    weight_volume: int = Field(default=10, alias="WEIGHT_VOLUME")
    weight_adx: int = Field(default=12, alias="WEIGHT_ADX")
    weight_macd: int = Field(default=12, alias="WEIGHT_MACD")
    weight_bollinger: int = Field(default=12, alias="WEIGHT_BOLLINGER")
    weight_relative_strength: int = Field(default=8, alias="WEIGHT_RELATIVE_STRENGTH")
    weight_52week: int = Field(default=8, alias="WEIGHT_52WEEK")
    weight_obv: int = Field(default=8, alias="WEIGHT_OBV")
    weight_stochastic: int = Field(default=8, alias="WEIGHT_STOCHASTIC")
    weight_squeeze: int = Field(default=7, alias="WEIGHT_SQUEEZE")
    min_recommendation_score: int = Field(default=50, alias="MIN_RECOMMENDATION_SCORE")

    # Long-term Recommendation (총합 100점 유지)
    # 기존: RSI(10) + MACD(15) + BB(10) + Vol(10) + ADX(20) + RS(15) + 52W(10) + Kalman(10) = 100
    # 변경: RSI(8) + MACD(12) + BB(8) + Vol(8) + ADX(15) + RS(12) + 52W(8) + Kalman(8) + OBV(8) + Stoch(5) + Squeeze(8) = 100
    longterm_top_n: int = Field(default=3, alias="LONGTERM_TOP_N")
    longterm_min_score: int = Field(default=40, alias="LONGTERM_MIN_SCORE")
    longterm_weight_rsi: int = Field(default=8, alias="LONGTERM_WEIGHT_RSI")
    longterm_weight_macd: int = Field(default=12, alias="LONGTERM_WEIGHT_MACD")
    longterm_weight_bollinger: int = Field(default=8, alias="LONGTERM_WEIGHT_BOLLINGER")
    longterm_weight_volume: int = Field(default=8, alias="LONGTERM_WEIGHT_VOLUME")
    longterm_weight_adx: int = Field(default=15, alias="LONGTERM_WEIGHT_ADX")
    longterm_weight_relative_strength: int = Field(default=12, alias="LONGTERM_WEIGHT_RELATIVE_STRENGTH")
    longterm_weight_week52: int = Field(default=8, alias="LONGTERM_WEIGHT_52WEEK")
    longterm_weight_kalman: int = Field(default=8, alias="LONGTERM_WEIGHT_KALMAN")
    longterm_weight_obv: int = Field(default=8, alias="LONGTERM_WEIGHT_OBV")
    longterm_weight_stochastic: int = Field(default=5, alias="LONGTERM_WEIGHT_STOCHASTIC")
    longterm_weight_squeeze: int = Field(default=8, alias="LONGTERM_WEIGHT_SQUEEZE")


class SlackSettings(BaseSettings):
    bot_token: str = Field(default="", alias="SLACK_BOT_TOKEN")
    channel: str = Field(default="", alias="SLACK_CHANNEL")


class GeneralSettings(BaseSettings):
    timezone: str = Field(default="Asia/Seoul", alias="TIMEZONE")
    max_news_items: int = Field(default=10, alias="MAX_NEWS_ITEMS")
    top_stocks_count: int = Field(default=10, alias="TOP_STOCKS_COUNT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")


class Settings:
    def __init__(self):
        self.smtp = SMTPSettings()
        self.email = EmailSettings()
        self.analysis = AnalysisSettings()
        self.general = GeneralSettings()
        self.slack = SlackSettings()
        self.base_dir = Path(__file__).parent.parent


settings = Settings()

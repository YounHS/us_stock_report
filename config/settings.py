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

    # Scoring Weights
    weight_rsi: int = Field(default=20, alias="WEIGHT_RSI")
    weight_volume: int = Field(default=15, alias="WEIGHT_VOLUME")
    weight_adx: int = Field(default=15, alias="WEIGHT_ADX")
    weight_macd: int = Field(default=15, alias="WEIGHT_MACD")
    weight_bollinger: int = Field(default=15, alias="WEIGHT_BOLLINGER")
    weight_relative_strength: int = Field(default=10, alias="WEIGHT_RELATIVE_STRENGTH")
    weight_52week: int = Field(default=10, alias="WEIGHT_52WEEK")
    min_recommendation_score: int = Field(default=50, alias="MIN_RECOMMENDATION_SCORE")

    # Long-term Recommendation
    longterm_top_n: int = Field(default=3, alias="LONGTERM_TOP_N")
    longterm_min_score: int = Field(default=40, alias="LONGTERM_MIN_SCORE")
    longterm_weight_rsi: int = Field(default=10, alias="LONGTERM_WEIGHT_RSI")
    longterm_weight_macd: int = Field(default=15, alias="LONGTERM_WEIGHT_MACD")
    longterm_weight_bollinger: int = Field(default=10, alias="LONGTERM_WEIGHT_BOLLINGER")
    longterm_weight_volume: int = Field(default=10, alias="LONGTERM_WEIGHT_VOLUME")
    longterm_weight_adx: int = Field(default=20, alias="LONGTERM_WEIGHT_ADX")
    longterm_weight_relative_strength: int = Field(default=15, alias="LONGTERM_WEIGHT_RELATIVE_STRENGTH")
    longterm_weight_week52: int = Field(default=10, alias="LONGTERM_WEIGHT_52WEEK")
    longterm_weight_kalman: int = Field(default=10, alias="LONGTERM_WEIGHT_KALMAN")


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

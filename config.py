import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    XT_API_KEY: str = os.getenv("XT_API_KEY", "")
    XT_API_SECRET: str = os.getenv("XT_API_SECRET", "")

    AI_API_KEY: str = os.getenv("AI_API_KEY", "")
    AI_BASE_URL: str = os.getenv("AI_BASE_URL", "https://api.openai.com/v1")
    AI_MODEL: str = os.getenv("AI_MODEL", "gpt-4o")

    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_USER_ID: str = os.getenv("TELEGRAM_USER_ID", "")

    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///data/memory.db")

    XT_FUTURES_HOST: str = os.getenv("XT_FUTURES_HOST", "https://fapi.xt.com")

    DEFAULT_SYMBOL: str = "esp_usdt"
    DEFAULT_LEVERAGE: int = 50
    DEFAULT_MARGIN_MODE: str = "CROSSED"
    DEFAULT_TIMEFRAMES: list = ["1m","15m"]
    DEFAULT_MARGIN_AMOUNT_PCT: float = 25.0
    DEFAULT_RISK_PCT: float = 1.0
    SIGNAL_COOLDOWN_MINUTES: int = 5
    MAX_POSITIONS: int = 3
    MIN_CONFIDENCE: int = 80

    # Per-timeframe gate applied before the multi-timeframe vote. The old code
    # reused MIN_CONFIDENCE here, which double-penalised every signal.
    TF_MIN_CONFIDENCE: int = 60

    SCAN_INTERVAL_SEC: int = 60
    GUARD_INTERVAL_SEC: int = 15
    MAX_LOSS_PCT: float = 40.0
    MAX_PROFIT_PCT: float = 500.0
    BREAKEVEN_THRESHOLD_PCT: float = 5.0
    TRAILING_STOP_PCT: float = 10.0

    # Trailing has two independent knobs: the ROI on margin that arms it, and
    # how far behind the mark price the stop sits (raw price percentage).
    TRAILING_TRIGGER_ROI_PCT: float = 10.0
    TRAILING_DISTANCE_PCT: float = 0.5

    # Stop loss must sit well inside the liquidation price, otherwise the
    # position is liquidated before the stop can trigger. 0.5 means the stop
    # is placed at most halfway to liquidation.
    SL_LIQUIDATION_SAFETY: float = 0.5
    # What to do when the exchange rejects the TP/SL order: close the position
    # (default) or keep it and rely on the slower software stop.
    ON_TPSL_FAILURE: str = "close"

    @classmethod
    def validate(cls) -> list:
        missing = []
        required = ["XT_API_KEY", "XT_API_SECRET", "AI_API_KEY",
                    "TELEGRAM_BOT_TOKEN", "TELEGRAM_USER_ID"]
        for key in required:
            if not getattr(cls, key):
                missing.append(key)
        if cls.TELEGRAM_USER_ID and not cls.TELEGRAM_USER_ID.strip().isdigit():
            missing.append("TELEGRAM_USER_ID (must be a numeric Telegram user id)")
        return missing

    @classmethod
    def default_settings(cls) -> dict:
        return {
            "symbol": cls.DEFAULT_SYMBOL,
            "leverage": cls.DEFAULT_LEVERAGE,
            "margin_mode": cls.DEFAULT_MARGIN_MODE,
            "timeframes": ",".join(cls.DEFAULT_TIMEFRAMES),
            "margin_amount_pct": cls.DEFAULT_MARGIN_AMOUNT_PCT,
            "margin_risk_pct": cls.DEFAULT_RISK_PCT,
            "min_confidence": cls.MIN_CONFIDENCE,
            "tf_min_confidence": cls.TF_MIN_CONFIDENCE,
            "cooldown_minutes": cls.SIGNAL_COOLDOWN_MINUTES,
            "max_positions": cls.MAX_POSITIONS,
            "position_mode": "margin",
            "scan_interval_sec": cls.SCAN_INTERVAL_SEC,
            "guard_interval_sec": cls.GUARD_INTERVAL_SEC,
            "max_loss_pct": cls.MAX_LOSS_PCT,
            "max_profit_pct": cls.MAX_PROFIT_PCT,
            "breakeven_threshold_pct": cls.BREAKEVEN_THRESHOLD_PCT,
            "trailing_stop_pct": cls.TRAILING_STOP_PCT,
            "trailing_trigger_roi_pct": cls.TRAILING_TRIGGER_ROI_PCT,
            "trailing_distance_pct": cls.TRAILING_DISTANCE_PCT,
            "sl_liquidation_safety": cls.SL_LIQUIDATION_SAFETY,
            "on_tpsl_failure": cls.ON_TPSL_FAILURE,
        }

    @classmethod
    def to_dict(cls) -> dict:
        redacted = {"XT_API_SECRET", "XT_API_KEY", "AI_API_KEY", "TELEGRAM_BOT_TOKEN"}
        return {k: v for k, v in cls.__dict__.items()
                if not k.startswith("_") and k.isupper() and k not in redacted}

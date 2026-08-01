import logging
import os
import signal
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from config import Config
from bot.memory import LongTermMemory
from bot.ai_chat import AIChat
from bot.trader import XTTrader
from bot.telegram_bot import TelegramBot

is_railway = os.getenv("RAILWAY_ENVIRONMENT") is not None

handlers = [logging.StreamHandler(sys.stdout)]
if not is_railway:
    os.makedirs("logs", exist_ok=True)
    handlers.append(logging.FileHandler("logs/trader.log"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=handlers,
)
logger = logging.getLogger("main")


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, format, *args):
        pass


def _start_health_server():
    port = int(os.getenv("PORT", "8080"))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    logger.info(f"Health check server listening on port {port}")
    server.serve_forever()


def main():
    missing = Config.validate()
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        logger.error("Set them in Railway Variables or .env file.")
        sys.exit(1)

    logger.info("Initializing XT AI Trader...")
    logger.info(f"AI Model: {Config.AI_MODEL}")
    logger.info(f"Railway: {is_railway}")
    logger.info(f"PORT env: {os.getenv('PORT', '<unset>')}")

    # Railway always injects PORT; fall back to it as the deploy signal so the
    # health endpoint comes up even if RAILWAY_ENVIRONMENT is absent.
    should_serve_health = is_railway or os.getenv("PORT") is not None
    if should_serve_health:
        # Bring the health port up before any slow initialisation, otherwise
        # the platform's proxy kills the deploy before it is marked healthy.
        health_thread = threading.Thread(target=_start_health_server, daemon=True)
        health_thread.start()

    database_url = os.getenv("DATABASE_URL", "")
    if not database_url:
        database_url = "sqlite:///data/memory.db"
        logger.warning("DATABASE_URL is not set, falling back to SQLite at "
                       "data/memory.db. On Railway that file lives inside the "
                       "container and is DESTROYED on every deploy, so open trades "
                       "and settings will be lost. Point DATABASE_URL at the MySQL "
                       "service (e.g. DATABASE_URL = ${{ MySQL.MYSQL_URL }} in the "
                       "service's Variables).")
    else:
        masked = database_url.split("@")[-1] if "@" in database_url else database_url
        logger.info(f"DATABASE_URL resolved to host: {masked}")
    logger.info(f"Database: {'MySQL' if 'mysql' in database_url else 'SQLite'}")

    memory = None
    try:
        memory = LongTermMemory(database_url=database_url)
        logger.info("Database schema ready")
    except Exception as e:
        logger.error(f"Database init failed: {e}")
        logger.error("Fix DATABASE_URL / MySQL availability before redeploying.")
        sys.exit(1)

    # Only seed missing keys, otherwise every Railway restart would wipe the
    # settings the user configured over Telegram.
    seeded = [k for k, v in Config.default_settings().items()
              if memory.set_setting_default(k, v)]
    if seeded:
        logger.info(f"Seeded default settings: {', '.join(sorted(seeded))}")
    else:
        logger.info("Existing settings preserved")

    trader = XTTrader(memory=memory)
    ai_chat = AIChat(memory=memory)
    ai_chat.bind_trader(trader)

    try:
        balances = trader.xt.get_balances()
        usdt = next((b for b in balances if str(b.get("coin", "")).upper() == "USDT"), None)
        if usdt:
            logger.info(f"XT connection OK. USDT wallet balance: {usdt.get('walletBalance')}")
        else:
            logger.warning("XT connection OK but no USDT balance found. "
                           "Is the futures account opened?")
    except Exception as e:
        logger.error(f"XT API check failed: {e}")
        logger.error("Verify XT_API_KEY / XT_API_SECRET, that the key has futures "
                     "permissions, and that this server's IP is whitelisted.")

    # A wiped database or a manually opened position would otherwise be
    # invisible to every guard, since they all iterate the local trade table.
    try:
        adopted = trader.position_mgr.adopt_exchange_positions()
        for a in adopted:
            logger.warning(f"Adopted untracked position: {a['symbol']} "
                           f"{a['position_side']} {a['size']}c @ {a['entry_price']} "
                           f"{a['leverage']}x, stop={'yes' if a['has_stop'] else 'NO'}")
        if not adopted:
            logger.info("No untracked exchange positions found")
    except Exception as e:
        logger.error(f"Position adoption failed at startup: {e}")

    telegram_bot = TelegramBot(trader=trader, ai_chat=ai_chat, memory=memory)

    logger.info("XT AI Trader started. Telegram bot is listening...")
    logger.info("Send /start to your bot to begin.")

    try:
        telegram_bot.run()
    except KeyboardInterrupt:
        logger.info("Shutting down via KeyboardInterrupt...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        trader.stop_auto_trade()
        memory.close()
        logger.info("XT AI Trader stopped.")


if __name__ == "__main__":
    main()

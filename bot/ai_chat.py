import json
import time
from openai import OpenAI
from config import Config
from bot.memory import LongTermMemory

SYSTEM_PROMPT = """You are an AI Trading Assistant for XT.com Futures.

Your capabilities:
1. Manage trading settings via function calls (symbol, leverage, margin mode, timeframes, risk, etc.)
2. Analyze market conditions and provide trade recommendations
3. Monitor open positions and suggest management actions
4. Interpret signal scan results and provide clear explanations
5. Remember user preferences and past trading context

AVAILABLE FUNCTIONS:
- get_status() - Get current bot status including open positions, PnL, and settings
- get_pnl() - Get profit/loss summary
- get_balance() - Read the live USDT futures balance from XT
- get_contract_info(symbol) - Contract size, min order in contracts, min notional, max leverage
- set_symbol(symbol) - Change trading pair (e.g. btc_usdt, eth_usdt)
- set_leverage(leverage) - Set leverage
- set_margin_mode(mode) - Set margin mode: CROSSED or ISOLATED
- set_timeframes(timeframes) - Set timeframes for scanning (e.g. "5m,15m,1h")
- set_margin_amount_pct(pct) - Set margin percentage of balance to use per trade (1-100)
- set_margin_risk_pct(pct) - Set risk percentage for position sizing (0.1-10)
- set_min_confidence(confidence) - Set minimum confidence threshold (50-100)
- set_cooldown_minutes(minutes) - Set cooldown minutes after closing position (1-30)
- set_position_mode(mode) - margin (by margin %) or risk (by risk %)
- set_max_loss_pct(pct) / set_max_profit_pct(pct) - Software safety-net ROI limits
- scan_signals() - Run signal scan now
- open_trade(direction, order_type, time_in_force) - Open a trade based on current signals
- close_trade(trade_id) - Close a specific trade
- close_all_trades() - Close all open positions
- mid_manage() - Run mid-position management

IMPORTANT FACTS ABOUT XT FUTURES:
- Order quantity is measured in CONTRACTS, not coin amount. One btc_usdt contract
  is 0.0001 BTC; one doge_usdt contract is 10 DOGE. Use get_contract_info when the
  user asks how much a position is worth.
- Leverage is capped per symbol by a notional-value bracket, so a requested
  leverage may be clamped down. Report the clamped value when that happens.
- Symbols are lowercase with an underscore: btc_usdt, eth_usdt, sol_usdt.
- Every position gets an exchange-side TP/SL order. If TP/SL creation fails the
  bot says so explicitly - treat that as urgent and tell the user.
- ROI shown is return on margin (leverage-amplified), not raw price movement.

IMPORTANT RULES:
- When the user asks to change settings, use the function calls directly.
- Only call open_trade / close_trade / close_all_trades when the user clearly
  asks for that action. Never call them to illustrate what you could do.
- Always explain what you're doing before calling functions.
- If signal strength is high, suggest wider TP and tighter SL.
- If signal strength is low, suggest tighter TP and wider SL.
- Always remind about risk management.
- Format numbers clearly with proper precision.
- Be concise but informative.
"""

FUNCTIONS = [
    {
        "name": "get_status",
        "description": "Get the current status of the trading bot including open positions, PnL, and settings",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "get_pnl",
        "description": "Get profit/loss summary for all trades",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "set_symbol",
        "description": "Change the trading pair symbol",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Trading pair symbol, e.g. btc_usdt, eth_usdt"}
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "set_leverage",
        "description": "Set the trading leverage multiplier",
        "parameters": {
            "type": "object",
            "properties": {
                "leverage": {"type": "integer", "description": "Leverage value, 1-125"}
            },
            "required": ["leverage"]
        }
    },
    {
        "name": "set_margin_mode",
        "description": "Set margin mode for positions",
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["CROSSED", "ISOLATED"], "description": "Margin mode"}
            },
            "required": ["mode"]
        }
    },
    {
        "name": "set_timeframes",
        "description": "Set timeframes for signal scanning",
        "parameters": {
            "type": "object",
            "properties": {
                "timeframes": {"type": "string", "description": "Comma-separated timeframes, e.g. 5m,15m,1h,4h"}
            },
            "required": ["timeframes"]
        }
    },
    {
        "name": "set_margin_amount_pct",
        "description": "Set what percentage of your balance to use as margin per trade",
        "parameters": {
            "type": "object",
            "properties": {
                "pct": {"type": "number", "description": "Percentage 1-100"}
            },
            "required": ["pct"]
        }
    },
    {
        "name": "set_margin_risk_pct",
        "description": "Set what percentage of your balance to risk per trade for position sizing",
        "parameters": {
            "type": "object",
            "properties": {
                "pct": {"type": "number", "description": "Risk percentage 0.1-10"}
            },
            "required": ["pct"]
        }
    },
    {
        "name": "set_min_confidence",
        "description": "Set the minimum confidence threshold for signals to execute",
        "parameters": {
            "type": "object",
            "properties": {
                "confidence": {"type": "integer", "description": "Confidence threshold 50-100"}
            },
            "required": ["confidence"]
        }
    },
    {
        "name": "set_cooldown_minutes",
        "description": "Set cooldown period after closing a position before new signals are accepted",
        "parameters": {
            "type": "object",
            "properties": {
                "minutes": {"type": "integer", "description": "Cooldown minutes 1-30"}
            },
            "required": ["minutes"]
        }
    },
    {
        "name": "set_position_mode",
        "description": "Set position sizing mode: margin (by margin %) or risk (by risk %)",
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {"type": "string", "enum": ["margin", "risk"], "description": "Position sizing mode"}
            },
            "required": ["mode"]
        }
    },
    {
        "name": "set_max_loss_pct",
        "description": "Set the software safety-net max loss as ROI on margin. The exchange stop loss is primary; this is the backup.",
        "parameters": {
            "type": "object",
            "properties": {
                "pct": {"type": "number", "description": "Loss ROI percentage, 1-100"}
            },
            "required": ["pct"]
        }
    },
    {
        "name": "set_max_profit_pct",
        "description": "Set the software safety-net max profit as ROI on margin",
        "parameters": {
            "type": "object",
            "properties": {
                "pct": {"type": "number", "description": "Profit ROI percentage, 1-1000"}
            },
            "required": ["pct"]
        }
    },
    {
        "name": "get_balance",
        "description": "Read the live USDT futures balance from XT",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "get_contract_info",
        "description": "Read contract specs for a symbol: contract size, minimum order in contracts, minimum notional, max leverage, price tick",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Optional, e.g. btc_usdt. Defaults to the active symbol."}
            }
        }
    },
    {
        "name": "scan_signals",
        "description": "Run a signal scan now and return results",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "open_trade",
        "description": "Open a new trade based on current signal scan results",
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {"type": "string", "enum": ["LONG", "SHORT"], "description": "Trade direction"},
                "order_type": {"type": "string", "enum": ["MARKET", "LIMIT"], "description": "Order type"},
                "time_in_force": {"type": "string", "enum": ["GTC", "IOC", "FOK"], "description": "Time in force"}
            },
            "required": ["direction"]
        }
    },
    {
        "name": "close_trade",
        "description": "Close a specific trade by ID",
        "parameters": {
            "type": "object",
            "properties": {
                "trade_id": {"type": "integer", "description": "Trade ID to close"}
            },
            "required": ["trade_id"]
        }
    },
    {
        "name": "close_all_trades",
        "description": "Close all currently open positions",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "mid_manage",
        "description": "Run mid-position management on all open positions (breakeven, trailing stop)",
        "parameters": {"type": "object", "properties": {}}
    },
]


class AIChat:
    def __init__(self, memory: LongTermMemory):
        self.memory = memory
        self.client = OpenAI(
            api_key=Config.AI_API_KEY,
            base_url=Config.AI_BASE_URL,
        )
        self.model = Config.AI_MODEL
        self.trader = None

    def bind_trader(self, trader_instance):
        self.trader = trader_instance

    def execute_function(self, func_name: str, args: dict) -> str:
        if not self.trader and func_name not in ["get_status", "get_pnl"]:
            return "Trader not initialized. Please start the bot first."

        handler_map = {
            "get_status": self._get_status,
            "get_pnl": self._get_pnl,
            "set_symbol": self._set_symbol,
            "set_leverage": self._set_leverage,
            "set_margin_mode": self._set_margin_mode,
            "set_timeframes": self._set_timeframes,
            "set_margin_amount_pct": self._set_margin_amount_pct,
            "set_margin_risk_pct": self._set_margin_risk_pct,
            "set_min_confidence": self._set_min_confidence,
            "set_cooldown_minutes": self._set_cooldown_minutes,
            "set_position_mode": self._set_position_mode,
            "set_max_loss_pct": self._set_max_loss_pct,
            "set_max_profit_pct": self._set_max_profit_pct,
            "get_balance": self._get_balance,
            "get_contract_info": self._get_contract_info,
            "scan_signals": self._scan_signals,
            "open_trade": self._open_trade,
            "close_trade": self._close_trade,
            "close_all_trades": self._close_all_trades,
            "mid_manage": self._mid_manage,
        }

        handler = handler_map.get(func_name)
        if handler:
            return handler(args)
        return f"Unknown function: {func_name}"

    def _get_status(self, args: dict) -> str:
        if not self.trader:
            summary = self.memory.get_trade_summary_for_ai()
            settings = self.memory.get_all_settings()
            return f"{summary}\n\nSettings: {settings}"
        return self.trader.get_status_report()

    def _get_pnl(self, args: dict) -> str:
        pnl = self.memory.get_total_pnl()
        stats = self.memory.get_trade_count()
        return (f"Total PnL: {pnl:.4f} USDT\n"
                f"Total Trades: {stats['total']} | Open: {stats['open']} | Closed: {stats['closed']}\n"
                f"Wins: {stats['wins']} | Losses: {stats['losses']} | "
                f"Flat/Unknown PnL: {stats['flat_or_unknown']} | "
                f"Winrate: {stats['winrate']}%")

    def _set_symbol(self, args: dict) -> str:
        symbol = args["symbol"].lower().strip()
        # Validate against the exchange before saving.
        if self.trader:
            try:
                detail = self.trader.xt.get_symbol_detail(symbol)
                if not detail or not detail.get("contractSize"):
                    return f"Symbol '{symbol}' not found on XT futures. Check the format (e.g. btc_usdt)."
            except Exception as e:
                return f"Could not validate symbol '{symbol}': {e}"
        self.memory.set_setting("symbol", symbol)
        return f"Trading pair set to: {symbol}"

    def _set_leverage(self, args: dict) -> str:
        lev = int(args["leverage"])
        # Clamp to per-symbol max from exchange brackets, not hardcoded 125.
        if self.trader:
            symbol = self.memory.get_setting("symbol", Config.DEFAULT_SYMBOL)
            max_lev = self.trader.risk.get_max_leverage(symbol)
            if max_lev and lev > max_lev:
                return f"Leverage {lev}x exceeds max {max_lev}x for {symbol}. Set to {max_lev}x."
            lev = max(1, min(lev, max_lev or 125))
        else:
            lev = max(1, min(lev, 125))
        self.memory.set_setting("leverage", lev)
        return f"Leverage set to: {lev}x"

    def _set_margin_mode(self, args: dict) -> str:
        mode = args["mode"].upper()
        if mode not in ("CROSSED", "ISOLATED"):
            return "Invalid margin mode. Use CROSSED or ISOLATED."
        self.memory.set_setting("margin_mode", mode)
        return f"Margin mode set to: {mode}"

    def _set_timeframes(self, args: dict) -> str:
        tfs = args["timeframes"].strip()
        valid = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d"]
        tf_list = [t.strip().lower() for t in tfs.split(",")]
        invalid = [t for t in tf_list if t not in valid]
        if invalid:
            return f"Invalid timeframes: {invalid}. Valid: {valid}"
        self.memory.set_setting("timeframes", ",".join(tf_list))
        return f"Timeframes set to: {', '.join(tf_list)}"

    def _set_margin_amount_pct(self, args: dict) -> str:
        pct = float(args["pct"])
        pct = max(1.0, min(pct, 100.0))
        self.memory.set_setting("margin_amount_pct", pct)
        return f"Margin amount percentage set to: {pct}%"

    def _set_margin_risk_pct(self, args: dict) -> str:
        pct = float(args["pct"])
        pct = max(0.1, min(pct, 10.0))
        self.memory.set_setting("margin_risk_pct", pct)
        return f"Risk percentage set to: {pct}%"

    def _set_min_confidence(self, args: dict) -> str:
        conf = int(args["confidence"])
        conf = max(50, min(conf, 100))
        self.memory.set_setting("min_confidence", conf)
        return f"Minimum confidence threshold set to: {conf}%"

    def _set_cooldown_minutes(self, args: dict) -> str:
        mins = int(args["minutes"])
        mins = max(1, min(mins, 30))
        self.memory.set_setting("cooldown_minutes", mins)
        return f"Cooldown period set to: {mins} minutes"

    def _set_position_mode(self, args: dict) -> str:
        mode = args["mode"].lower()
        if mode not in ("margin", "risk"):
            return "Invalid mode. Use margin or risk."
        self.memory.set_setting("position_mode", mode)
        return f"Position sizing mode set to: {mode}"

    def _scan_signals(self, args: dict) -> str:
        if not self.trader:
            return "Trader not running. Cannot scan signals."
        result = self.trader.scanner.scan_and_report()
        report = self.trader.scanner.format_signal_report(result)
        return report

    def _open_trade(self, args: dict) -> str:
        if not self.trader:
            return "Trader not running. Cannot open trade."
        direction = args.get("direction", "")
        if not direction:
            return "Direction (LONG/SHORT) is required"
        order_type = args.get("order_type") or "MARKET"
        # None lets the trader pick the timeInForce the exchange accepts for
        # this order type (MARKET needs IOC, LIMIT defaults to GTC).
        time_in_force = args.get("time_in_force") or None
        return self.trader.execute_trade(direction, order_type, time_in_force)

    def _get_balance(self, args: dict) -> str:
        if not self.trader:
            return "Trader not running. Cannot read balance."
        try:
            item = self.trader.risk._get_usdt_balance(force=True)
        except Exception as e:
            return f"Failed to read balance from XT: {e}"
        if not item:
            return "No USDT balance returned by XT."
        return (f"Wallet: {item.get('walletBalance')} USDT\n"
                f"Available: {item.get('availableBalance')} USDT\n"
                f"Order margin frozen: {item.get('openOrderMarginFrozen')} USDT\n"
                f"Isolated margin: {item.get('isolatedMargin')} USDT\n"
                f"Crossed margin: {item.get('crossedMargin')} USDT")

    def _get_contract_info(self, args: dict) -> str:
        if not self.trader:
            return "Trader not running."
        symbol = (args.get("symbol")
                  or self.memory.get_setting("symbol", Config.DEFAULT_SYMBOL))
        try:
            risk = self.trader.risk
            price = self.trader.scanner.get_current_price(symbol)
            cs = risk.get_contract_size(symbol)
            one = cs * price
            return (f"{symbol}\n"
                    f"Contract size: {cs} ({one:.4f} USDT per contract at {price})\n"
                    f"Min order: {risk.get_min_qty(symbol)} contracts\n"
                    f"Min notional: {risk.get_min_notional(symbol)} USDT\n"
                    f"Max leverage: {risk.get_max_leverage(symbol)}x\n"
                    f"Price precision: {risk.get_price_precision(symbol)} "
                    f"(tick {risk.get_price_step(symbol)})")
        except Exception as e:
            return f"Failed to read contract config for {symbol}: {e}"

    def _set_max_loss_pct(self, args: dict) -> str:
        pct = max(1.0, min(float(args["pct"]), 100.0))
        self.memory.set_setting("max_loss_pct", pct)
        return f"Software max loss (ROI on margin) set to: -{pct}%"

    def _set_max_profit_pct(self, args: dict) -> str:
        pct = max(1.0, min(float(args["pct"]), 1000.0))
        self.memory.set_setting("max_profit_pct", pct)
        return f"Software max profit (ROI on margin) set to: +{pct}%"

    def _close_trade(self, args: dict) -> str:
        if not self.trader:
            return "Trader not running. Cannot close trade."
        trade_id = int(args["trade_id"])
        return self.trader.close_specific_trade(trade_id)

    def _close_all_trades(self, args: dict) -> str:
        if not self.trader:
            return "Trader not running. Cannot close trades."
        return self.trader.close_all_positions()

    def _mid_manage(self, args: dict) -> str:
        if not self.trader:
            return "Trader not running. Cannot manage positions."
        return self.trader.run_mid_management()

    def chat(self, user_message: str, user_id: str = None) -> str:
        self.memory.add_chat_message("user", user_message)
        history = self.memory.get_chat_history(30)
        context = self.memory.get_trade_summary_for_ai()
        ai_context = self.memory.get_ai_context()
        context_msg = f"Current trading context:\n{context}\n\nAI memory context:\n{json.dumps(ai_context, indent=2)}"
        context_msg += f"\n\nValid symbols use format like btc_usdt, eth_usdt (lowercase with underscore)."
        context_msg += f"\nValid timeframes: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 1d"
        context_msg += f"\nMargin modes: CROSSED or ISOLATED"
        context_msg += f"\nOrder types for open_trade: MARKET or LIMIT"
        context_msg += f"\nTime in force for open_trade: GTC, IOC, FOK"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context_msg},
        ]
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
        result = self._call_with_functions(messages)
        self.memory.add_chat_message("assistant", result)
        return result

    def _call_with_functions(self, messages: list, max_rounds: int = 5) -> str:
        for _ in range(max_rounds):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=[{"type": "function", "function": f} for f in FUNCTIONS],
                    tool_choice="auto",
                    timeout=30,
                )
            except Exception as e:
                err_str = str(e)
                if "tool_use_failed" in err_str or "tool" in err_str.lower():
                    messages.append({
                        "role": "user",
                        "content": f"Your last function call failed: {err_str}. Please respond in plain text instead of calling functions. If you need to perform an action, describe it clearly and I'll handle it."
                    })
                    continue
                return f"AI API error: {err_str}"

            choice = response.choices[0]
            message = choice.message

            if message.tool_calls:
                function_result_messages = []
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                    result = self.execute_function(func_name, args)
                    function_result_messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": func_name,
                        "content": result,
                    })
                messages.append({"role": "assistant", "content": None, "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in message.tool_calls
                ]})
                messages.extend(function_result_messages)
            elif getattr(message, "function_call", None):
                func_call = message.function_call
                func_name = func_call.name
                args = json.loads(func_call.arguments) if func_call.arguments else {}
                result = self.execute_function(func_name, args)
                messages.append({
                    "role": "function",
                    "name": func_name,
                    "content": result,
                })
            else:
                return message.content or ""
        return "Max function call rounds exceeded. Please try a more specific request."

    def remember(self, key: str, value: str):
        self.memory.set_ai_context(key, value)

    def recall(self, key: str = None):
        return self.memory.get_ai_context(key)

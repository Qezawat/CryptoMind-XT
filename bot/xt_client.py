import hashlib
import hmac
import json
import logging
import time
import requests

logger = logging.getLogger("xt_client")


class XTError(Exception):
    pass


class XTClient:
    """XT USDT-M futures REST client.

    Returns unwrapped `result` payloads and raises XTError on failure, unlike
    pyxt which returns the raw (code, envelope, error) triple.
    """

    MARKET = "/future/market"
    USER = "/future/user"
    TRADE = "/future/trade"

    def __init__(self, host: str, access_key: str, secret_key: str, timeout: int = 10,
                 sign_style: str = "auto"):
        self.host = host.rstrip("/")
        self._ak = access_key
        self._sk = secret_key
        self.timeout = timeout
        self._sign_style = sign_style
        self._session = requests.Session()

    # ---------- signing ----------

    def _sign_headers(self, path: str, style: str, query: dict = None, body_str: str = None):
        ts = str(int(time.time() * 1000))
        prefix = "xt-validate" if style == "xt" else "validate"
        fixed = f"{prefix}-appkey={self._ak}&{prefix}-timestamp={ts}"
        if query:
            payload = "&".join(f"{k}={query[k]}" for k in sorted(query))
        elif body_str:
            payload = body_str
        else:
            payload = None
        raw = f"{fixed}#{path}#{payload}" if payload else f"{fixed}#{path}"
        sig = hmac.new(self._sk.encode(), raw.encode(), hashlib.sha256).hexdigest()
        return {
            f"{prefix}-appkey": self._ak,
            f"{prefix}-timestamp": ts,
            f"{prefix}-signature": sig,
            # XT's own docs and curl examples spell this header "singature".
            f"{prefix}-singature": sig,
            f"{prefix}-recvwindow": "60000",
            f"{prefix}-algorithms": "HmacSHA256",
        }

    # ---------- transport ----------

    def _public(self, method: str, path: str, params: dict = None):
        url = self.host + path
        params = {k: v for k, v in (params or {}).items() if v is not None}
        resp = self._session.request(method, url, params=params, timeout=self.timeout)
        return self._unwrap(resp, path)

    def _private(self, method: str, path: str, params: dict = None):
        params = {k: v for k, v in (params or {}).items() if v is not None}
        styles = ["xt", "plain"] if self._sign_style == "auto" else [self._sign_style]
        last = None
        for style in styles:
            try:
                out = self._send_signed(method, path, params, style)
            except XTError as e:
                last = e
                if not self._is_auth_error(e) or style == styles[-1]:
                    raise
                logger.warning(f"Signature style '{style}' rejected, retrying with alternate")
                continue
            if self._sign_style == "auto":
                self._sign_style = style
                logger.info(f"XT signature style locked to '{style}'")
            return out
        raise last

    def _send_signed(self, method: str, path: str, params: dict, style: str):
        url = self.host + path
        if method == "GET":
            headers = self._sign_headers(path, style, query=params)
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            resp = self._session.get(url, params=params, headers=headers, timeout=self.timeout)
        else:
            body_str = json.dumps(params)
            headers = self._sign_headers(path, style, body_str=body_str)
            headers["Content-Type"] = "application/json"
            resp = self._session.post(url, data=body_str.encode(), headers=headers,
                                      timeout=self.timeout)
        return self._unwrap(resp, path)

    @staticmethod
    def _is_auth_error(err: XTError) -> bool:
        s = str(err).lower()
        return "403" in s or "signature" in s or "appkey" in s or "auth" in s

    @staticmethod
    def _unwrap(resp, path: str):
        try:
            payload = resp.json()
        except ValueError:
            raise XTError(f"{path} -> HTTP {resp.status_code}, non-JSON body: {resp.text[:200]}")
        if not isinstance(payload, dict):
            return payload
        # /future/user/v1/compat/balance/{coin} uses {rc, mc, ma, result}
        code = payload.get("returnCode", payload.get("rc"))
        # Treat missing returnCode as an error: a valid XT response always has
        # returnCode or rc. Returning the raw envelope silently gives callers
        # garbage data.
        if code is None:
            # Check if this looks like a valid response (has "result" key)
            if "result" not in payload:
                raise XTError(f"{path} -> unexpected response (no returnCode): {str(payload)[:200]}")
        elif code != 0:
            err = payload.get("error") or {}
            msg = payload.get("msgInfo") or payload.get("mc") or ""
            detail = err.get("msg") or err.get("code") or ""
            raise XTError(f"{path} -> returnCode={code} {msg} {detail}".strip())
        if "result" in payload:
            return payload["result"]
        return payload

    # ---------- public market data ----------

    def get_symbol_detail(self, symbol: str) -> dict:
        return self._public("GET", f"{self.MARKET}/v1/public/symbol/detail", {"symbol": symbol})

    def get_klines(self, symbol: str, interval: str, limit: int = None,
                   start_time: int = None, end_time: int = None) -> list:
        return self._public("GET", f"{self.MARKET}/v1/public/q/kline", {
            "symbol": symbol, "interval": interval, "limit": limit,
            "startTime": start_time, "endTime": end_time,
        }) or []

    def get_agg_ticker(self, symbol: str) -> dict:
        return self._public("GET", f"{self.MARKET}/v1/public/q/agg-ticker", {"symbol": symbol}) or {}

    def get_mark_price(self, symbol: str) -> dict:
        return self._public("GET", f"{self.MARKET}/v1/public/q/symbol-mark-price",
                            {"symbol": symbol}) or {}

    def get_leverage_brackets(self, symbol: str) -> list:
        data = self._public("GET", f"{self.MARKET}/v1/public/leverage/bracket/detail",
                            {"symbol": symbol}) or {}
        return data.get("leverageBrackets") or []

    def get_funding_rate(self, symbol: str) -> dict:
        return self._public("GET", f"{self.MARKET}/v1/public/q/funding-rate",
                            {"symbol": symbol}) or {}

    # ---------- account ----------

    def get_balances(self) -> list:
        data = self._private("GET", f"{self.USER}/v1/balance/list")
        return data if isinstance(data, list) else []

    def get_listen_key(self) -> str:
        data = self._private("GET", f"{self.USER}/v1/user/listen-key")
        if isinstance(data, dict):
            return data.get("listenKey") or data.get("accessToken") or ""
        return data or ""

    # ---------- positions ----------

    def get_positions(self, symbol: str = None) -> list:
        """Active positions. Unlike /position/list this includes calMarkPrice and floatingPL."""
        data = self._private("GET", f"{self.USER}/v1/position", {"symbol": symbol})
        return data if isinstance(data, list) else []

    def set_leverage(self, symbol: str, position_side: str, leverage: int):
        return self._private("POST", f"{self.USER}/v1/position/adjust-leverage", {
            "symbol": symbol, "positionSide": position_side, "leverage": leverage,
        })

    def set_position_type(self, symbol: str, position_side: str, position_type: str):
        """CROSSED or ISOLATED. Rejected with position_exists if a position is already open."""
        return self._private("POST", f"{self.USER}/v1/position/change-type", {
            "symbol": symbol, "positionSide": position_side, "positionType": position_type,
        })

    def adjust_margin(self, symbol: str, position_side: str, margin, direction: str):
        """direction is ADD or SUB. Isolated positions only."""
        return self._private("POST", f"{self.USER}/v1/position/margin", {
            "symbol": symbol, "positionSide": position_side,
            "margin": margin, "type": direction,
        })

    def set_auto_margin(self, symbol: str, position_side: str, enabled: bool):
        return self._private("POST", f"{self.USER}/v1/position/auto-margin", {
            "symbol": symbol, "positionSide": position_side, "autoMargin": bool(enabled),
        })

    def get_leverage_info(self, symbol: str) -> list:
        data = self._private("GET", f"{self.TRADE}/v1/position/leverage/list", {"symbol": symbol})
        if isinstance(data, dict):
            return data.get("items") or []
        return data if isinstance(data, list) else []

    # ---------- orders ----------

    def create_order(self, symbol: str, position_side: str, order_side: str,
                     order_type: str, orig_qty: int, price=None,
                     time_in_force: str = None, client_order_id: str = None):
        return self._private("POST", f"{self.TRADE}/v1/order/create", {
            "symbol": symbol, "positionSide": position_side, "orderSide": order_side,
            "orderType": order_type, "origQty": int(orig_qty), "price": price,
            "timeInForce": time_in_force, "clientOrderId": client_order_id,
        })

    def cancel_order(self, order_id):
        return self._private("POST", f"{self.TRADE}/v1/order/cancel", {"orderId": order_id})

    def cancel_all_orders(self, symbol: str):
        return self._private("POST", f"{self.TRADE}/v1/order/cancel-all", {"symbol": symbol})

    def get_order(self, order_id):
        return self._private("GET", f"{self.TRADE}/v1/order/detail", {"orderId": order_id})

    def get_orders(self, symbol: str = None, state: str = "NEW", page: int = 1, size: int = 50):
        data = self._private("GET", f"{self.TRADE}/v1/order/list", {
            "symbol": symbol, "state": state, "page": page, "size": size,
        }) or {}
        return data.get("items") or []

    # ---------- take profit / stop loss ----------

    def create_tpsl(self, symbol: str, position_side: str, orig_qty: int,
                    trigger_profit_price, trigger_stop_price, expire_time_ms: int,
                    profit_order_type: str = "MARKET", stop_order_type: str = "MARKET",
                    profit_tif: str = "IOC", stop_tif: str = "IOC",
                    profit_price=None, stop_price=None):
        """expire_time_ms must be milliseconds. All four delegate params are mandatory."""
        return self._private("POST", f"{self.TRADE}/v1/entrust/create-profit", {
            "symbol": symbol,
            "positionSide": position_side,
            "origQty": int(orig_qty),
            "triggerProfitPrice": trigger_profit_price,
            "triggerStopPrice": trigger_stop_price,
            "expireTime": int(expire_time_ms),
            "profitDelegateOrderType": profit_order_type,
            "profitDelegateTimeInForce": profit_tif,
            "profitDelegatePrice": profit_price,
            "stopDelegateOrderType": stop_order_type,
            "stopDelegateTimeInForce": stop_tif,
            "stopDelegatePrice": stop_price,
        })

    def update_tpsl(self, profit_id, trigger_profit_price=None, trigger_stop_price=None):
        return self._private("POST", f"{self.TRADE}/v1/entrust/update-profit-stop", {
            "profitId": profit_id,
            "triggerProfitPrice": trigger_profit_price,
            "triggerStopPrice": trigger_stop_price,
        })

    def cancel_tpsl(self, profit_id):
        return self._private("POST", f"{self.TRADE}/v1/entrust/cancel-profit-stop",
                             {"profitId": profit_id})

    def cancel_all_tpsl(self, symbol: str):
        return self._private("POST", f"{self.TRADE}/v1/entrust/cancel-all-profit-stop",
                             {"symbol": symbol})

    def get_tpsl_orders(self, symbol: str, state: str = "NOT_TRIGGERED",
                        page: int = 1, size: int = 50) -> list:
        data = self._private("GET", f"{self.TRADE}/v1/entrust/profit-list", {
            "symbol": symbol, "state": state, "page": page, "size": size,
        }) or {}
        return data.get("items") or []

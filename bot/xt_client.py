import hashlib
import hmac
import json
import logging
import time
import requests
from requests.exceptions import RequestException

logger = logging.getLogger("xt_client")


class XTError(Exception):
    pass


class XTClient:
    """XT USDT-M futures REST client with robust rate-limit handling.

    Returns unwrapped `result` payloads and raises XTError on failure.
    Includes automatic retries with exponential backoff for HTTP 429 (Rate Limit)
    and HTTP 5xx (Server Errors) to ensure stability during live trading.
    """

    MARKET = "/future/market"
    USER = "/future/user"
    TRADE = "/future/trade"

    def __init__(self, host: str, access_key: str, secret_key: str, timeout: int = 10):
        self.host = host.rstrip("/")
        self._ak = access_key
        self._sk = secret_key
        self.timeout = timeout
        self._session = requests.Session()
        # XT strictly requires 'xt' style headers for USDT-M futures
        self._prefix = "xt-validate"

    # ---------- signing ----------

    def _sign_headers(self, path: str, query: dict = None, body_str: str = None):
        ts = str(int(time.time() * 1000))
        fixed = f"{self._prefix}-appkey={self._ak}&{self._prefix}-timestamp={ts}"
        
        if query:
            # Sort query params alphabetically for signing as per XT docs
            payload = "&".join(f"{k}={query[k]}" for k in sorted(query))
        elif body_str:
            payload = body_str
        else:
            payload = ""

        # The signed string is "#{path}" with NO trailing "#" when there is no
        # query string / body. XT computes the server-side signature this way
        # (see pyxt _create_sign and XT.Net XTFuturesAuthenticationProvider); a
        # stray trailing "#" changes the HMAC and fails validation on every
        # no-parameter signed call (get_balances, get_listen_key, ...).
        raw = f"{fixed}#{path}#{payload}" if payload else f"{fixed}#{path}"

        sig = hmac.new(self._sk.encode(), raw.encode(), hashlib.sha256).hexdigest()

        headers = {
            "validate-signversion": "2",
            f"{self._prefix}-appkey": self._ak,
            f"{self._prefix}-timestamp": ts,
            f"{self._prefix}-signature": sig,
            f"{self._prefix}-recvwindow": "60000",
            f"{self._prefix}-algorithms": "HmacSHA256",
        }
        # Only the correct "signature" header is sent. The older typo fallback
        # "singature" header was removed.
        return headers

    # ---------- transport ----------

    def _request(self, method: str, path: str, params: dict = None, signed: bool = False):
        url = self.host + path
        params = {k: v for k, v in (params or {}).items() if v is not None}
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                if signed:
                    headers = self._sign_headers(path, query=params if method == "GET" else None, 
                                                  body_str=json.dumps(params) if method != "GET" else None)
                else:
                    headers = {}

                if method == "GET":
                    headers["Content-Type"] = "application/x-www-form-urlencoded"
                    resp = self._session.get(url, params=params, headers=headers, timeout=self.timeout)
                else:
                    body_str = json.dumps(params)
                    headers["Content-Type"] = "application/json"
                    resp = self._session.post(url, data=body_str.encode(), headers=headers, timeout=self.timeout)

                # Handle Rate Limiting (HTTP 429)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 2 ** attempt))
                    logger.warning(f"XT Rate Limit hit (429). Sleeping for {retry_after}s...")
                    time.sleep(retry_after)
                    continue

                # Handle Server Errors (HTTP 5xx)
                if 500 <= resp.status_code < 600:
                    wait_time = 2 ** attempt
                    logger.warning(f"XT Server Error ({resp.status_code}). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                return self._unwrap(resp, path)

            except RequestException as e:
                wait_time = 2 ** attempt
                logger.warning(f"Network error: {e}. Retrying in {wait_time}s...")
                if attempt == max_retries - 1:
                    raise XTError(f"Network error after {max_retries} attempts: {e}")
                time.sleep(wait_time)

        raise XTError(f"Request failed after {max_retries} attempts: {path}")

    def _public(self, method: str, path: str, params: dict = None):
        return self._request(method, path, params, signed=False)

    def _private(self, method: str, path: str, params: dict = None):
        return self._request(method, path, params, signed=True)

    @staticmethod
    def _unwrap(resp, path: str):
        try:
            payload = resp.json()
        except ValueError:
            raise XTError(f"{path} -> HTTP {resp.status_code}, non-JSON body: {resp.text[:200]}")
            
        if not isinstance(payload, dict):
            return payload
            
        code = payload.get("returnCode", payload.get("rc"))
        
        if code is None:
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

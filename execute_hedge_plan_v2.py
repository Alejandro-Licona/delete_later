from pickle import STOP
from typing import Any, Dict, List, Tuple, Set, Optional
import time
import requests
from dotenv import load_dotenv
import json
import numpy as np
from decimal import Decimal

from hyperliquid.info import Info
from hyperliquid.exchange import Exchange

from utils.hyperliquid_client import (
    get_hyperliquid_clients,
    place_market_order,
    place_limit_order,
    place_take_profit_limit,
    quantize_order,
    quantize_size,
)

# Ensure .env in package is loaded when executed from repo root
load_dotenv()

def get_l2_book_snapshot(coin: str, clients: Dict[str, Any], n_levels: int = 20) -> Dict[str, Any]:
    """
    Fetch L2 book snapshot from Hyperliquid /info endpoint.
    
    Args:
        coin: e.g., "BTC"
        clients: From get_hyperliquid_clients(), uses 'info'
        n_levels: Max levels per side (default 20)
    
    Returns:
        Dict with 'bids' (list of [price, size]), 'asks' (list of [price, size]), 'mid_price'.
    """
    info = clients['info']
    request = {
        "type": "l2Book",
        "coin": coin,
        "nSigFigs": 5  # Optional aggregation for precision
    }
    try:
        # Preferred: high-level SDK method if available
        if hasattr(info, "l2_book") and callable(getattr(info, "l2_book")):
            data = info.l2_book(coin)  # type: ignore[attr-defined]
        else:
            # Fallback: direct HTTP POST to the info endpoint
            base_url = getattr(info, "base_url", None) or "https://api.hyperliquid.xyz"
            url = f"{base_url.rstrip('/')}/info"
            resp = requests.post(url, json=request, headers={"Content-Type": "application/json"}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        if 'levels' not in data or len(data['levels']) != 2:
            raise ValueError(f"Invalid L2 response: {data}")
        
        # Parse levels: levels[0] = bids (descending), levels[1] = asks (ascending)
        bids = [[float(level['px']), float(level['sz'])] for level in data['levels'][0][:n_levels]]
        asks = [[float(level['px']), float(level['sz'])] for level in data['levels'][1][:n_levels]]
        
        # Sort bids descending, asks ascending
        bids.sort(key=lambda x: x[0], reverse=True)
        asks.sort(key=lambda x: x[0])
        
        # Mid price
        mid_price = (bids[0][0] + asks[0][0]) / 2 if bids and asks else None
        
        return {
            "bids": bids,
            "asks": asks,
            "mid_price": mid_price,
            "timestamp": data.get('time', int(time.time() * 1000))
        }
    except Exception as e:
        print(f"L2 snapshot error for {coin}: {e}")
        return {"error": str(e)}


def _hl_info_post(*, info: Info, body: Dict[str, Any]) -> Any:
    """
    Helper to POST to Hyperliquid /info endpoint for queries not exposed by SDK.
    """
    base_url = getattr(info, "base_url", None) or "https://api.hyperliquid.xyz"
    url = f"{str(base_url).rstrip('/')}/info"
    resp = requests.post(url, json=body, headers={"Content-Type": "application/json"}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def _get_clearinghouse_state(*, info: Info, address: str) -> Dict[str, Any]:
    """
    Fetch clearinghouseState for address. Tries SDK method first, fallback to direct POST.
    """
    try:
        state = getattr(info, "user_state", None)
        if callable(state):
            state = state(address)  # type: ignore[call-arg]
        if isinstance(state, dict) and ("assetPositions" in state or "marginSummary" in state):
            return state
    except Exception:
        pass
    return _hl_info_post(info=info, body={"type": "clearinghouseState", "user": address})


def get_filled_size_signed(*, info: Info, address: str, symbol: str) -> Decimal:
    """
    Return current signed perp position size (base units) for symbol using clearinghouseState.
    Positive means long, negative means short. Returns Decimal("0") when flat/absent.
    """
    state = _get_clearinghouse_state(info=info, address=address)
    for ap in state.get("assetPositions", []):
        pos = ap.get("position", {})
        if pos.get("coin") == symbol.upper():
            szi = pos.get("szi")
            if szi is not None:
                try:
                    return Decimal(str(szi))
                except Exception:
                    return Decimal("0")
            # Fallback derive from positionValue / entryPx
            pv, px = pos.get("positionValue"), pos.get("entryPx")
            if pv and px:
                try:
                    return Decimal(str(pv)) / Decimal(str(px))
                except Exception:
                    return Decimal("0")
            return Decimal("0")
    return Decimal("0")


def list_open_orders(*, info: Info, address: str) -> List[Dict[str, Any]]:
    """
    Retrieve user's open orders using frontendOpenOrders for richer fields (origSz, reduceOnly).
    Falls back to openOrders if necessary.
    """
    try:
        data = _hl_info_post(info=info, body={"type": "frontendOpenOrders", "user": address})
        if isinstance(data, list):
            return data
    except Exception:
        pass
    fallback = _hl_info_post(info=info, body={"type": "openOrders", "user": address})
    return fallback if isinstance(fallback, list) else []


def get_symbol_side_pending_size(
    *, info: Info, address: str, symbol: str, is_buy: bool
) -> Tuple[Decimal, List[int]]:
    """
    Sum remaining open order sizes (base units) for this symbol and side. 
    Returns (pending_size, matching_oids).
    Assumes side mapping B=buy, A=sell per Hyperliquid docs.
    """
    pending = Decimal("0")
    oids: List[int] = []
    orders = list_open_orders(info=info, address=address)
    side_code = "B" if is_buy else "A"
    for order in orders:
        if order.get("coin") != symbol.upper():
            continue
        if order.get("side") != side_code:
            continue
        try:
            size_d = Decimal(str(order.get("sz", "0")))
        except Exception:
            size_d = Decimal("0")
        if size_d <= 0:
            continue
        pending += size_d
        # oid may be int or str; keep only int-like
        oid = order.get("oid")
        if isinstance(oid, int):
            oids.append(oid)
    return pending, oids


def get_run_pending_size(
    *, info: Info, address: str, symbol: str, is_buy: bool, run_oids: Set[int]
) -> Decimal:
    """
    Sum remaining open order sizes for this symbol/side but only for oids placed by this run.
    """
    if not run_oids:
        return Decimal("0")
    side_code = "B" if is_buy else "A"
    pending = Decimal("0")
    orders = list_open_orders(info=info, address=address)
    for order in orders:
        if order.get("coin") != symbol.upper():
            continue
        if order.get("side") != side_code:
            continue
        oid = order.get("oid")
        if not isinstance(oid, int) or oid not in run_oids:
            continue
        try:
            size_d = Decimal(str(order.get("sz", "0")))
        except Exception:
            size_d = Decimal("0")
        if size_d > 0:
            pending += size_d
    return pending


def _directional_executed_delta(*, start: Decimal, current: Decimal, is_buy: bool) -> Decimal:
    """
    Directional executed since baseline, clamped to >= 0.
    - For buy runs: max(0, current - start)
    - For sell runs: max(0, start - current)
    """
    if is_buy:
        return max(Decimal("0"), current - start)
    return max(Decimal("0"), start - current)


def _cancel_oids_safely(*, exchange: Exchange, oids: List[int]) -> None:
    """
    Cancel a list of order IDs safely, ignoring errors.
    """
    for oid in oids:
        try:
            exchange.post("/exchange/cancel", {"oid": oid})
        except Exception:
            continue

def hybrid_twap_vwap_order(
    *,
    symbol: str,
    total_size: float,
    is_buy: bool,
    clients: Dict[str, Any],
    max_time_sec: int = 180,  # 5 min
    tranche_interval_sec: int = 5,  # TWAP slice every 30s
    impact_threshold: float = 0.05,  # Max 15% of level size per tranche
    aggressive_offset_pct: float = 0.001,  # 0.05% inside book for limits
    max_levels_deep: int = 5,  # Max book levels to scan
    fallback_market: bool = True,  # Fallback to market if liquidity low
    poll_fills_sec: float = 1.0,
    poll_interval_sec: float = 0.25,
    enable_position_guard: bool = True,  # NEW: Enable position-aware guard
    tolerance: float = 1e-6,  # NEW: Tolerance for completion check
) -> Dict[str, Any]:
    """
    Hybrid TWAP/VWAP order placer: Slices total_size into liquidity-weighted tranches,
    placed as aggressive limits (TWAP-timed), adapting velocity based on L2 book.
    
    NOW INCLUDES POSITION-AWARE GUARD to prevent overtrading by tracking:
    - Actual filled position (via clearinghouseState)
    - Pending orders (via openOrders)
    - Only submits orders for remaining needed size
    
    Args:
        symbol: e.g., "BTC"
        total_size: Absolute quantity to fill
        is_buy: True for buy (long), False for sell (short)
        clients: From get_hyperliquid_clients()
        max_time_sec: Total execution time limit
        tranche_interval_sec: Time between tranches
        impact_threshold: Max tranche size as % of book level
        aggressive_offset_pct: % inside book for limit prices
        max_levels_deep: Max book levels to scan for liquidity
        fallback_market: Use market order if liquidity < total_size / 10
        enable_position_guard: If True, tracks position to prevent overtrading
        tolerance: Size tolerance for completion check
    
    Returns:
        Dict with 'tranches' (list of executed orders), 'fills' (list of statuses),
    """
    exchange = clients['exchange']
    info = clients['info']
    address = clients.get('address', '')
    
    results = {
        "tranches": [],
        "fills": [],
        "resting_oids": [],
        "avg_book_utilization_pct": 0.0,
        "avg_px_intended": None,
        "avg_px_fills": None,
        "total_filled": 0.0,
        "execution_time_sec": 0.0,
        "position_guard_enabled": enable_position_guard,
        "start_position": None,
        "final_position": None,
        "pending_size": 0.0,
    }
    
    # NEW: Track starting position if guard enabled
    start_pos = Decimal("0")
    run_oids: Set[int] = set()
    target_decimal = Decimal(str(total_size))
    tol_decimal = Decimal(str(tolerance))
    
    if enable_position_guard and address:
        try:
            start_pos = get_filled_size_signed(info=info, address=address, symbol=symbol)
            results["start_position"] = float(start_pos)
        except Exception as e:
            print(f"Warning: Could not get starting position: {e}")
            enable_position_guard = False
    
    # Fetch initial L2 snapshot
    book = get_l2_book_snapshot(symbol, clients)
    if "error" in book:
        if fallback_market:
            res = place_market_order(
                exchange=exchange,
                symbol=symbol,
                is_buy=is_buy,
                size=total_size,
                slippage=None,
                reduce_only=False,
                info=info,
            )
            results["tranches"].append({"type": "market_fallback", "size": total_size, "response": res})
            results["fills"].append(res)
        return results
    
    # Determine side: bids for buys (support), asks for sells (resistance)
    relevant_levels = book['bids'] if is_buy else book['asks']
    mid_price = book['mid_price']
    
    if not relevant_levels or not mid_price:
        raise ValueError(f"No book levels or mid price for {symbol} on {'buy' if is_buy else 'sell'} side")
    
    # Calculate cumulative liquidity and tranche sizes (VWAP-weighted)
    cum_liquidity = sum(level[1] for level in relevant_levels[:max_levels_deep])
    tranche_sizes = []
    remaining_size = total_size
    
    for level in relevant_levels[:max_levels_deep]:
        level_size = level[1]
        tranche_size = min(level_size * impact_threshold, remaining_size)
        if tranche_size > 0:
            tranche_sizes.append(tranche_size)
            remaining_size -= tranche_size
        if remaining_size <= 0:
            break
    
    # If insufficient liquidity, fallback or adjust
    if remaining_size > total_size * 0.5 and fallback_market:
        res = place_market_order(
            exchange=exchange,
            symbol=symbol,
            is_buy=is_buy,
            size=remaining_size,
            slippage=None,
            reduce_only=False,
            info=info,
        )
        tranche_sizes.append(remaining_size)
        results["tranches"].append({"type": "market_remainder", "size": remaining_size, "response": res})
        results["fills"].append(res)
    
    # Number of tranches (TWAP): Divide by interval, cap at max_time
    n_tranches = min(len(tranche_sizes), max_time_sec // tranche_interval_sec)
    if n_tranches == 0:
        n_tranches = 1
    
    # Resize tranches (VWAP weights, TWAP timing)
    if len(tranche_sizes) < n_tranches:
        tranche_sizes += [0] * (n_tranches - len(tranche_sizes))
    else:
        weights = np.array([s for s in tranche_sizes[:n_tranches] if s > 0])
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(weights)) / len(weights)
        tranche_sizes = weights * total_size
    
    start_time = time.time()
    filled = 0.0
    book_utilization_accum = 0.0
    intended_notional = 0.0
    intended_qty = 0.0
    fills_notional = 0.0
    fills_qty = 0.0
    # Post-execution processing buffers
    seen_fill_keys: Set[Tuple[int, float, float]] = set()
    post_statuses: List[Any] = []
    
    for i, tranche_size in enumerate(tranche_sizes[:n_tranches]):
        if tranche_size <= 0:
            continue
        
        # NEW: Check if we already hit target (position guard)
        if enable_position_guard and address:
            try:
                current_pos = get_filled_size_signed(info=info, address=address, symbol=symbol)
                executed_dir = _directional_executed_delta(start=start_pos, current=current_pos, is_buy=is_buy)
                pending_run = get_run_pending_size(info=info, address=address, symbol=symbol, is_buy=is_buy, run_oids=run_oids)
                
                remaining_needed = target_decimal - (executed_dir + pending_run)
                
                # If we've already filled enough, stop
                if remaining_needed <= tol_decimal:
                    print(f"Target reached: executed={float(executed_dir)}, pending={float(pending_run)}, target={total_size}")
                    # Cancel any leftover orders
                    if run_oids:
                        _cancel_oids_safely(exchange=exchange, oids=list(run_oids))
                    break
                
                # Cap tranche size to remaining needed
                if Decimal(str(tranche_size)) > remaining_needed:
                    tranche_size = float(remaining_needed)
                    print(f"Capping tranche {i+1} to remaining: {tranche_size}")
                
            except Exception as e:
                print(f"Warning: Position check failed at tranche {i+1}: {e}")
        
        # Refresh book per tranche
        book = get_l2_book_snapshot(symbol, clients)
        if "error" in book:
            print(f"Failed to refresh book at tranche {i+1}: {book['error']}")
            break
        
        relevant_levels = book['bids'] if is_buy else book['asks']
        if not relevant_levels:
            break
        
        # Select level: Deeper if more liquidity
        cum_liq = 0.0
        selected_level = relevant_levels[0]
        for level in relevant_levels[:max_levels_deep]:
            cum_liq += level[1]
            if cum_liq >= tranche_size / impact_threshold:
                selected_level = level
                break
        
        # Aggressive limit price: 0.1% inside book
        level_price = selected_level[0]
        offset = aggressive_offset_pct * level_price
        limit_price = level_price + offset if is_buy else level_price - offset
        
        # Quantize intended order to exact exchange increments
        q_size, q_price = quantize_order(info=info, symbol=symbol, size=tranche_size, price=limit_price)

        # Place limit order with quantized values
        res = place_limit_order(
            exchange=exchange,
            symbol=symbol,
            is_buy=is_buy,
            size=q_size,
            price=q_price,
            tif="Gtc",
            reduce_only=False,
            info=info,
        )
        
        tranche_result = {
            "tranche": i + 1,
            "size": q_size,
            "price": q_price,
            "level_price": selected_level[0],
            "level_size": selected_level[1],
            "response": res,
            "timestamp": time.time()
        }
        results["tranches"].append(tranche_result)
        # Defer response processing until after all orders sent
        post_statuses.append(res)
        
        # NEW: Extract and track resting OIDs immediately for position guard
        if enable_position_guard:
            try:
                if isinstance(res, dict) and res.get("status") == "ok":
                    statuses = res.get("response", {}).get("data", {}).get("statuses", [])
                    for st in statuses:
                        if "resting" in st and "oid" in st["resting"]:
                            oid = st["resting"]["oid"]
                            if isinstance(oid, int):
                                run_oids.add(oid)
            except Exception:
                pass
        
        # Book utilization: (quantized_size / selected_level_size) %
        utilization = (q_size / selected_level[1]) * 100 if selected_level[1] > 0 else 0.0
        book_utilization_accum += utilization / n_tranches

        # Intended notional from quantized tranche price and size
        intended_notional += q_price * q_size
        intended_qty += q_size
        
        # Wait for tranche interval
        sleep_time = min(tranche_interval_sec, max_time_sec - (time.time() - start_time))
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    # Cleanup: Cancel unfilled orders if time exceeds max
    if results["resting_oids"] and time.time() - start_time >= max_time_sec:
        for oid in results["resting_oids"]:
            exchange.post("/exchange/cancel", json={"oid": oid})
        # Market remainder
        remainder = total_size - filled
        if remainder > 0 and fallback_market:
            fallback_res = place_market_order(
                exchange=exchange,
                symbol=symbol,
                is_buy=is_buy,
                size=remainder,
                slippage=None,
                reduce_only=False,
                info=info,
            )
            results["tranches"].append({"type": "final_market", "size": remainder, "response": fallback_res})
            # Defer processing; include in post-execution responses
            post_statuses.append(fallback_res)

    # After sending all orders: extract resting OIDs from responses
    for res in post_statuses:
        try:
            if isinstance(res, dict) and res.get("status") == "ok":
                statuses = res.get("response", {}).get("data", {}).get("statuses", [])
                for st in statuses:
                    if "resting" in st and "oid" in st["resting"]:
                        results["resting_oids"].append(st["resting"]["oid"])
        except Exception:
            pass

    # Post-execution: extract immediate fills from responses
    for res in post_statuses:
        for fill_px, fill_sz in _extract_px_sz_pairs(res):
            key = (0, fill_px, fill_sz)  # OID may be absent here
            if fill_sz > 0 and key not in seen_fill_keys:
                seen_fill_keys.add(key)
                filled += fill_sz
                if fill_px > 0:
                    fills_notional += fill_px * fill_sz
                fills_qty += fill_sz
                results["fills"].append({"fill": {"px": fill_px, "sz": fill_sz}})

    # Poll for fills that complete shortly after placement
    if poll_fills_sec > 0 and results["resting_oids"]:
        end_poll = time.time() + poll_fills_sec
        address = clients.get("address")
        while time.time() < end_poll:
            for oid in list(results["resting_oids"]):
                try:
                    status = info.query_order_by_oid(address, oid)
                except Exception:
                    continue
                for fill_px, fill_sz in _extract_px_sz_pairs(status):
                    key = (oid, fill_px, fill_sz)
                    if fill_sz > 0 and key not in seen_fill_keys:
                        seen_fill_keys.add(key)
                        filled += fill_sz
                        if fill_px > 0:
                            fills_notional += fill_px * fill_sz
                        fills_qty += fill_sz
                        results["fills"].append({"oid": oid, "fill": {"px": fill_px, "sz": fill_sz}})
            time.sleep(max(0.0, poll_interval_sec))
    
    results["total_filled"] = filled
    results["avg_book_utilization_pct"] = book_utilization_accum
    # Compute avgPx metrics
    results["avg_px_intended"] = (intended_notional / intended_qty) if intended_qty > 0 else None
    results["avg_px_fills"] = (fills_notional / fills_qty) if fills_qty > 0 else None
    results["execution_time_sec"] = time.time() - start_time
    
    # NEW: Final position check if guard enabled
    if enable_position_guard and address:
        try:
            final_pos = get_filled_size_signed(info=info, address=address, symbol=symbol)
            executed_total = _directional_executed_delta(start=start_pos, current=final_pos, is_buy=is_buy)
            pending_final, _ = get_symbol_side_pending_size(info=info, address=address, symbol=symbol, is_buy=is_buy)
            
            results["final_position"] = float(final_pos)
            results["executed_directional"] = float(executed_total)
            results["pending_size"] = float(pending_final)
            results["target_vs_executed_diff"] = total_size - float(executed_total)
            
            # Log warning if we significantly overtraded
            if executed_total > target_decimal * Decimal("1.05"):  # 5% overshoot
                print(f"WARNING: Overtraded! Target: {total_size}, Executed: {float(executed_total)}")
        except Exception as e:
            print(f"Warning: Final position check failed: {e}")
    
    return results


def _extract_px_sz_pairs(obj: Any) -> List[Tuple[float, float]]:
    """
    Recursively extract (px, sz) pairs from any nested dict/list structure.
    """
    pairs: List[Tuple[float, float]] = []
    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if "px" in node and "sz" in node:
                try:
                    px = float(node.get("px", 0) or 0)
                    sz = float(node.get("sz", 0) or 0)
                    pairs.append((px, sz))
                except Exception:
                    pass
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(obj)
    return pairs


def enforce_target_execution(
    *,
    symbol: str,
    target_size: float,
    is_buy: bool,
    clients: Dict[str, Any],
    tolerance: float = 1e-6,
    poll_fills_sec: float = 1.0,
    aggressive_offset_pct: float = 0.001,
    max_levels_deep: int = 5,
    market_remainder_threshold: float = 0.15,
    max_seconds: Optional[float] = 300.0,
) -> Dict[str, Any]:
    """
    Position-aware execution guard that ensures total executed size converges to target_size
    without over-trading or leaving ghost orders.
    
    This wrapper continuously reconciles:
    - Filled position (from clearinghouseState)
    - Pending orders (from openOrders)
    - Remaining needed size
    
    And ensures we never submit more orders than needed.
    
    Args:
        symbol: Perp symbol (e.g., "ARB")
        target_size: Total size to execute (absolute value)
        is_buy: True for buy/long, False for sell/short
        clients: From get_hyperliquid_clients()
        tolerance: Size tolerance for completion
        poll_fills_sec: How often to check fills
        aggressive_offset_pct: Offset for limit prices
        max_levels_deep: Max book levels to scan
        market_remainder_threshold: Use market order if remaining < this % of target
        max_seconds: Max execution time (None = no limit)
    
    Returns:
        Dict with execution summary including executed, pending, and run_oids
    """
    info: Info = clients["info"]
    exchange: Exchange = clients["exchange"]
    address: str = clients["address"]

    tol = Decimal(str(tolerance))
    target = Decimal(str(target_size))

    start_pos = get_filled_size_signed(info=info, address=address, symbol=symbol)
    run_oids: Set[int] = set()

    results: Dict[str, Any] = {
        "symbol": symbol,
        "is_buy": is_buy,
        "target": float(target),
        "executed": 0.0,
        "pending": 0.0,
        "run_oids": [],
        "iterations": 0,
        "canceled_oids": [],
        "orders": [],
    }

    deadline = (time.time() + float(max_seconds)) if max_seconds is not None else None

    while True:
        # 1) Reconcile executed and pending
        current_pos = get_filled_size_signed(info=info, address=address, symbol=symbol)
        executed_dir = _directional_executed_delta(start=start_pos, current=current_pos, is_buy=is_buy)
        # Only count pending from this run to avoid double counting
        pending_run = get_run_pending_size(info=info, address=address, symbol=symbol, is_buy=is_buy, run_oids=run_oids)

        remaining = target - (executed_dir + pending_run)
        
        if remaining <= tol:
            # Target reached - cancel any leftover orders
            if run_oids:
                _cancel_oids_safely(exchange=exchange, oids=list(run_oids))
            results["executed"] = float(executed_dir)
            results["pending"] = float(pending_run)
            results["run_oids"] = list(run_oids)
            print(f"✓ Target execution complete: {float(executed_dir)}/{float(target)}")
            return results

        # 2) Submit new tranche capped by remaining
        tranche_remaining_dec = max(Decimal("0"), remaining)
        tranche_remaining = float(tranche_remaining_dec)
        
        if tranche_remaining > 0:
            # If small remainder, use market order to finish
            STOP_MARKET_ORDER = False
            # if tranche_remaining_dec <= Decimal(str(market_remainder_threshold)) * target:
            if STOP_MARKET_ORDER:
                try:
                    print(f"Using market order for remainder: {tranche_remaining}")
                    mr = place_market_order(
                        exchange=exchange,
                        symbol=symbol,
                        is_buy=is_buy,
                        size=tranche_remaining,
                        slippage=None,
                        reduce_only=False,
                        info=info,
                    )
                    results["orders"].append({"type": "market_remainder", "response": mr, "size": tranche_remaining})
                except Exception as e:
                    results["orders"].append({"type": "market_remainder_error", "error": str(e)})
            else:
                print(f"Using hybrid_twap_vwap_order for larger remaining size: {tranche_remaining}")
                # Use hybrid TWAP/VWAP for larger remaining size
                order_res = hybrid_twap_vwap_order(
                    symbol=symbol,
                    total_size=tranche_remaining,
                    is_buy=is_buy,
                    clients=clients,
                    enable_position_guard=True,  # Nested guard
                )
                results["orders"].append(order_res)
                for oid in order_res.get("resting_oids", []) or []:
                    if isinstance(oid, int):
                        run_oids.add(oid)

        results["iterations"] += 1

        # 3) Exit conditions: time or sleep
        if deadline is not None and time.time() >= deadline:
            # Time up: cancel remaining orders and exit
            print(f"Execution timeout reached. Canceling remaining orders.")
            if run_oids:
                _cancel_oids_safely(exchange=exchange, oids=list(run_oids))
            # Refresh before exit
            current_pos = get_filled_size_signed(info=info, address=address, symbol=symbol)
            executed_dir = _directional_executed_delta(start=start_pos, current=current_pos, is_buy=is_buy)
            pending_side, _ = get_symbol_side_pending_size(info=info, address=address, symbol=symbol, is_buy=is_buy)
            results["executed"] = float(executed_dir)
            results["pending"] = float(pending_side)
            results["run_oids"] = list(run_oids)
            return results

        time.sleep(max(0.0, float(poll_fills_sec)))


def execute_orders(
    *,
    symbol: str,
    entry_is_buy: bool,
    market_orders: List[Dict[str, Any]],
    limit_orders: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Execute a hedge plan on Hyperliquid.

    Each market order dict expects: {"quantity", "market_price", "take_profit"?}
    Each limit order dict expects: {"quantity", "limit_price", "take_profit"?}
    """

    clients = get_hyperliquid_clients()
    exchange = clients["exchange"]
    address = clients["address"]
    info = clients["info"]

    results: Dict[str, Any] = {"market": [], "limit": [], "take_profit": [], "resting_oids": []}

    for mo in market_orders:
        qty = float(mo.get("quantity", 0))
        if qty == 0:
            continue
        
        # Naive way to place Market orders use IOC by default - enable for testing 
        # res = place_market_order(
        #     exchange=exchange,
        #     symbol=symbol,
        #     is_buy=entry_is_buy,
        #     size=abs(qty),
        #     slippage=None,
        #     reduce_only=False,
        # )
        # results["market"].append(res)

        # White Star hybrid TWAP/VWAP to increase trade velocity 
        order_result = hybrid_twap_vwap_order(
            symbol=symbol,
            total_size=abs(qty),
            is_buy=entry_is_buy,
            clients=clients
        )

        results["market"].append(order_result)

        # Optional take-profit as reduce-only opposite limit
        tp = mo.get("take_profit")
        if tp is not None:
            tp_res = place_take_profit_limit(
                exchange=exchange,
                entry_is_buy=entry_is_buy,
                symbol=symbol,
                size=abs(qty),
                take_profit_price=float(tp),
                tif="Gtc",
                info=info,
            )
            results["take_profit"].append(tp_res)

    for lo in limit_orders:
        qty = float(lo.get("quantity", 0))
        if qty == 0:
            continue
        price = float(lo["limit_price"])  # must exist
        # Quantize to avoid float_to_wire rounding errors
        q_size, q_price = quantize_order(info=info, symbol=symbol, size=abs(qty), price=price)
        res = place_limit_order(
            exchange=exchange,
            symbol=symbol,
            is_buy=entry_is_buy,
            size=q_size,
            price=q_price,
            tif="Gtc",
            reduce_only=False,
            info=info,
        )
        results["limit"].append(res)
        try:
            if isinstance(res, dict) and res.get("status") == "ok":
                statuses = (
                    res.get("response", {})
                    .get("data", {})
                    .get("statuses", [])
                )
                for st in statuses:
                    if "resting" in st and "oid" in st["resting"]:
                        results["resting_oids"].append(st["resting"]["oid"])
        except Exception:
            pass

        tp = lo.get("take_profit")
        if tp is not None:
            tp_res = place_take_profit_limit(
                exchange=exchange,
                entry_is_buy=entry_is_buy,
                symbol=symbol,
                size=abs(qty),
                take_profit_price=float(tp),
                tif="Gtc",
                info=info,
            )
            results["take_profit"].append(tp_res)

    return results


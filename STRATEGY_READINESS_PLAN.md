# Strategy Readiness Plan

## Objective

Build evidence and operational controls for each strategy before considering
automated orders. This plan does not authorize live trading. The current
Breakout order option submits an entry only and does not manage exits; keep it
disabled.

## Release gates (apply to every strategy)

1. **Specification and data integrity**
   - Freeze each strategy's entry, invalidation, exit, time-window, instrument,
     and sizing rules.
   - Require completed candles and point-in-time inputs; record missing,
     stale, or delayed quotes as errors rather than silently substituting data.
   - Unit-test long/short directions, session boundaries, gaps, duplicate
     signals, missing data, and stop/target ambiguity.
2. **Historical validation**
   - Run chronological, walk-forward/out-of-sample tests across multiple market
     regimes; do not tune thresholds on the final holdout.
   - Include realistic brokerage, taxes, spread, slippage, expiry/roll rules,
     and next-available execution prices. Report trade count, net expectancy,
     profit factor, drawdown, tail loss, exposure, and results by regime.
   - Require positive net expectancy out of sample and risk limits that the
     owner approves. A win-rate target alone (including 70–80%) is not a
     readiness criterion or a promise of future results.
3. **Forward paper validation**
   - Run at least 60 market sessions and 100 eligible recommendations per
     strategy where enough signals occur; otherwise continue until sample
     minimums are met.
   - Store each signal, quote, entry-touch/fill assumption, exit, costs, and
     data-quality status. Reconcile paper results against raw broker candles
     and manually review a sample of days.
   - Treat ambiguous OHLC ordering conservatively; do not count a touch as a
     fill when the sequence cannot be established.
4. **Shadow and limited-live approval**
   - Shadow the complete proposed order lifecycle without submitting orders.
     Verify contract selection, quote freshness, sizing, order-state
     reconciliation, exit supervision, restart recovery, alerts, and kill
     switch behavior.
   - Require explicit user approval of per-trade risk, daily loss limit,
     max concurrent positions, max exposure, allowed products, and operating
     hours before any live mode is implemented or enabled.
   - If approved, start with the smallest practical size and human monitoring.
     Automatically stop on stale data, unknown order state, rejected/missing
     protective exit, risk-limit breach, or failed reconciliation.

## Strategy sequence

### 1. Breakout

- **Already in place:** completed 5-minute close plus live price must remain
  beyond the trigger; paper tracking waits for reference-entry touch.
- **First implementation:** keep the backtest gate aligned with live logic and
  make its close-price proxy limitation explicit. Do not claim historical
  tests prove an intrabar live-price reclaim filter.
- **Next:** replay signals with the exact live schedule, then expand
  out-of-sample tests; collect timestamped option bid/ask and realistic option
  P&L rather than treating underlying T1 hits as option wins.
- **Before live:** build and test order/fill reconciliation, protective exits,
  position limits, restart recovery, and a kill switch. The current optional
  path only submits an entry and is not ready for unattended use.

### 2. Trend Pullback

- Freeze EMA trend, pullback/confirmation, VWAP, volume, Nifty alignment,
  extension, option-liquidity, and cutoff definitions.
- Build or verify a strategy-specific chronological backtest that uses the
  same completed-candle decisions and option/exit assumptions as production.
- Validate it independently using the historical and forward-paper gates;
  compare results by volatility/trend regime before considering any parameter
  changes.
- Only after Breakout has a tested shared order lifecycle, add Trend Pullback
  as a separately enabled strategy with its own identifiers and limits.

### 3. Daily/hourly/15-minute Long Option

- Preserve daily/hourly direction and completed 15-minute pullback/confirmation
  rules; test for candle lookahead and stale option quotes.
- Continue paper entry/exit monitoring; verify same-bar stop priority,
  forced 15:15 exit, restart recovery, and notification/report persistence.
- Add fill-aware option P&L and historical bid/ask/slippage assumptions before
  judging profitability.
- Do not connect live orders until the shared lifecycle can supervise both
  entry and same-day exit and recover safely after process/API failures.

### 4. Swing Long Option

- Validate daily pullback/breakout candidates and next-session trigger using
  chronological data and point-in-time option-chain information.
- Implement persistent paper-position lifecycle: confirmed entry reference,
  underlying and option prices, stop/target, three-session maximum hold,
  Friday exit, expiry handling, and restart-safe notifications.
- Compare underlying-based and option-premium exits with realistic costs and
  unavailable-data reporting. Do not convert candidate confirmation into an
  order before this lifecycle is reliable.

## Implementation order and decision points

1. Breakout backtest/live-rule parity and reliable evidence collection.
2. Trend Pullback validation and paper evidence.
3. 15-minute position-lifecycle reliability and paper evidence.
4. Swing persistent exit monitoring and paper evidence.
5. Shared broker order-management, reconciliation, portfolio risk controls,
   kill switch, and human approval workflow.
6. Strategy-by-strategy shadow run, then explicit limited-live approval.

At each gate, publish the sample size, net results, limitations, and unresolved
failures. Do not proceed to the next gate on a calendar deadline alone.

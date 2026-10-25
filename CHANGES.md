# Backend fixes

Every finding from the review is fixed except authentication/CORS (left as-is: the app is hosted locally).

## Before you run it

1. **Never run against Postgres yet.** `main.py`, `db.py`, `price_history_sync.py` were compile-checked and their pure logic is unit-tested, but the SQL has not been executed. Back up the `postgres_data` volume, then start the API and hit `/health` and `/sync/holdings`.
2. **Schema:** new columns/tables are created automatically at startup by `ensure_schema()` (`risk_scores.history_days / beta_observations / median_daily_turnover`, `recommendations.model_status / model_name / llm_recommendation / policy_recommendation`, `fii_dii_flows`). I could not see your schema file, so confirm these unique constraints exist, since several `ON CONFLICT` clauses depend on them: `stocks(symbol)`, `holdings(stock_id)`, `risk_scores(stock_id)`, `price_history(stock_id, date)`.
3. **Re-run in order** after deploying: `/sync/holdings` -> `/risk/score-stocks` -> `/sentiment/refresh` -> `/recommendations/run`. Old scores were computed with the beta bug (every beta was 1.0).
4. **Windows without Docker:** add `tzdata` to your dependencies.
5. **Frontend:** make sure `liquidity_needs` is reverse-coded (5 = "won't need the money"). See README.

## What changed, by finding

| Finding | Fix | File |
|---|---|---|
| Beta was 1.0 for every stock (tz-aware vs naive join -> 0 rows) | Both series reduced to plain dates before joining; no overlap now gives `None`, never 1.0 | `beta_calculator.py`, `main.py` |
| NaN / meaningless volatility on short history; missing data scored as "Low" | `None` below 60 returns; stock skipped with a reason; stale score deleted | `volatility_calculator.py`, `main.py` |
| `volatility_1y` / `beta_1y` actually used 3 years | Trailing 252 days | both calculators |
| Conservative investors could never be "within profile" | Falls out of the beta fix (the constant 50 was inflating every score) | - |
| Liquidity used share count | Median daily traded value over 60 sessions; threshold Rs 1 cr (`LIQUIDITY_MIN_TURNOVER_INR`) | `liquidity_flagger.py` |
| Sold positions stayed in `holdings` | Pruned inside the sync transaction; empty Kite response leaves data alone | `main.py` |
| Partial intraday bar frozen forever | Today's bar ignored until 16:00 IST; last 30 days re-fetched and upserted (`lookback_days` now used) | `market_hours.py`, `price_history_sync.py` |
| Splits/bonuses created price jumps | Overlapping closes compared; a >1% disagreement rebuilds that stock's history | `price_history_sync.py` |
| `.NS` hard-coded | BSE -> `.BO`; `-BE`-style series suffixes stripped | `price_history_sync.py` |
| NSE+BSE rows overwrote each other | Merged by symbol, quantity-weighted average cost | `main.py` |
| Multi-statement writes not atomic | Holdings sync and recommendation inserts each in one transaction | `db.py`, `main.py` |
| Unscored holdings dropped silently | `coverage_pct`, `scored_holdings`, `unscored_symbols` in the portfolio summary | `portfolio_aggregation.py` |
| Duplicate risk-label cut-offs | One source of truth: `composite_risk_label` | `portfolio_aggregation.py` |
| HHI went `None` if one stock lacked a sector | Computed over covered holdings | `portfolio_aggregation.py` |
| News keywords matched substrings ("against" -> "gain") | Whole-word matching | `news_sentiment.py` |
| VIX cliff at 20; score saturated; missing input counted neutral | Linear VIX ramp; score normalised by available weight; `coverage` reported | `quant_sentiment.py` |
| FII/DII was one day, mixed signs scored 0 | Daily history stored; 5-day rolling sum | `db.py`, `main.py` |
| One failed source gave a 502 | Sources fetched independently, failures reported as `warnings` | `market_data.py` |
| Sentiment label wording/thresholds differed | Shared `composite_sentiment_label` (65/35) | `quant_sentiment.py`, `main.py` |
| Stale sentiment used silently; stock sentiment never read | Market <= 3 days, stock <= 7 days old; both fed to recommendations (sector sentiment is still stored but not used) | `main.py` |
| Fallback policy inverted and mostly dead | New risk-aware policy using the tier's limit, position/sector concentration and sentiment | `recommendation_policy.py` |
| LLM action never checked | Model bounded by the policy; overrides recorded as `ollama_overridden` | `recommendation_agent.py` |
| Fallback differed from LLM path; `model_status` not stored | Same policy both ways; engine, model name and both suggestions stored | `main.py`, `db.py` |
| Recommendations ignored position size | Position %, sector %, unrealised P&L, portfolio coverage added to the inputs | `main.py` |
| One long sequential request | Parallel model calls (`RECOMMENDATION_MAX_WORKERS`, default 4), optional `?background=true`, one run at a time | `main.py` |
| Review returns not comparable, no benchmark | Excess return vs Nifty, hit/miss, `too_early` under 5 days; entry date in IST | `review_harness.py`, `main.py` |
| `/debug/env` always said token missing | Checks the database | `main.py` |
| `score_stocks` always COMPLETED | Reports PARTIAL when stocks were skipped | `main.py` |
| DDL on every request; runs stuck in RUNNING after a crash | `ensure_schema()` once per process; startup marks interrupted runs FAILED | `db.py`, `main.py` |
| `WHEN %s IS NULL` may fail in psycopg3 | Explicit `::boolean` cast | `main.py` |

## Other changes that are not review findings

Compared against your original files, these also differ:

- **New endpoint** `GET /recommendations/runs/{run_id}` and a `?background=true` option on `POST /recommendations/run`.
- **New response fields only; no field was removed or renamed.** Added: `/portfolio/summary` (`coverage_pct`, `scored_holdings`, `total_holdings`, `unscored_symbols`), `/risk/scores` (`history_days`, `beta_observations`, `median_daily_turnover`), `/recommendations` and the run response (`model_status`, `model_name`, `llm_recommendation`, `policy_recommendation`), `/recommendations/review` (`holding_days`, `benchmark_return_pct`, `excess_return_pct`, `outcome`, `outcome_basis`), `/sync/holdings` (`removed_sold_positions`, `rebuilt_after_split_or_bonus`).
- **`/risk/score-stocks` can now return `"status": "partial"`** (and its run is recorded as PARTIAL) when a stock was skipped. I checked `index.tsx`: it does not look at that status, so nothing breaks, and the updated page now reads the skipped stocks and shows why.
- **`/sync/holdings`** no longer overwrites a stored ISIN with an empty string, and price rows with a null close are ignored when scoring.
- **`recommendation_agent.py`** builds messages directly instead of through `ChatPromptTemplate`, and the prompt no longer asks the model to echo `inputs_snapshot` (Python attaches it). `choose_recommendation` keeps its old signature but now returns INCREASE where it used to return BUY.
- **`kite_client.py`**: docstrings only (the old one said the token goes in `.env`; it is stored in Postgres). No logic change; you can skip this file.
- **`db.py`**: Kite login only creates its own `kite_sessions` table, so signing in never depends on the other tables existing.

## Dashboard (`frontend/index.tsx`) and smooth Kite re-login

Replace your `index.tsx` route with `frontend/index.tsx`. It was type-checked in strict mode against stand-ins for React, lucide and your UI components (the package registry was blocked in my sandbox), so run your own `tsc`/`npm run build` once.

**Kite re-login.** Kite's redirect URL has to keep pointing at the API, so the browser used to end on a JSON page. Now:
1. `Login to Kite`, or `Sync holdings` hitting an expired session, calls `/login?return_to=<this page>`.
2. The API checks `return_to` (local addresses or `FRONTEND_URL` only; tested against `//evil.com`, `localhost@evil.com`, backslash and header-injection tricks), keeps it in a 10-minute cookie, and sends you to Kite.
3. After sign-in the callback redirects back with `?kite=success` (or `?kite=error&message=...`). No token is placed in the URL.
4. The page removes the flag from the address bar, then runs the sync you had asked for automatically. A guard prevents an endless login loop if Kite keeps rejecting the session.
- New file `frontend_redirect.py` (+ `tests/test_frontend_redirect.py`); `main.py` `/login` accepts `return_to`, the callback redirects.
- New optional setting `FRONTEND_URL`. Without it and without `return_to`, the callback behaves exactly as before (JSON).

**Other dashboard changes**
- Removed the 5-minute market-sentiment auto-refresh (it hit NSE, Tavily and the AI model ~288 times a day and stored a row each time). Refresh is by the button only; the card shows the date of the reading, a "Stale" tag after 3 days, and "Based on partial data" when a source failed.
- After **Run**, stocks that could not be scored are listed, and hovering the Risk column shows why.
- Recommendations show an **Adjusted** / **Standard wording** tag when the AI was overridden or unavailable, and the modal explains it.
- New line under the top cards: portfolio risk, whether it fits the profile, and how many holdings it covers.
- Sync message now reports removed (sold) positions, stocks without price data, and rebuilt histories.
- Sentiment wording is Positive / Neutral / Negative everywhere (matches the API). BUY removed from the filter (the API stores INCREASE).
- API validation errors no longer show as "[object Object]". Profile card shows the score as "x / 25".
- `/holdings` now includes `sector` (the table already tried to show it).

## Plain-language recommendations (stocks and ETFs)

- New `plain_language.py`. Every recommendation is now stored as three short parts - **headline** (what to do), **why**, **what to keep in mind** - in everyday words. No scores, decimals, "volatility", "beta", "sentiment", "liquidity" or "tier".
- The AI model is shown everyday-word facts only ("its price has big ups and downs", "it already makes up about 14% of your investments"), never raw numbers, so it has no jargon to repeat. If its reply still contains technical words, decimals, or is too long, it is replaced by the standard wording (`model_status = ollama_reworded`). The policy still decides the action; the model can only word it.
- Works for both cases: **holdings** (SELL / INCREASE / HOLD) and **ETFs you do not own** (BUY / WATCH / AVOID). ETF candidates are judged on what adding a small amount (assumed 5%) does to your overall portfolio risk, not on the ETF's risk alone. Overall market mood can hold a BUY back to WATCH for equity ETFs only.
- ETFs that could not be rated say so in plain words ("There is not enough price history to rate it fairly yet") instead of a technical reason.
- The ETF scan uses the standard wording by default (fast, no model call per ETF). Set `ETF_USE_LLM=1` to have the model word each ETF instead.
- `/etfs` now returns `explanation`; stored stock recommendations carry it inside `inputs_snapshot.explanation`.
- Dashboard: recommendation rows show the headline; the detail view shows Why / What to keep in mind with a plain action label (e.g. "Keep holding"); ETF rows open the same view; ETF column headings and the "How actions are determined" text are in plain words.

## Not changed

- Authentication and CORS (by your decision).
- Sector-level sentiment is still stored but not used in recommendations.
- Policy numbers (SELL when 15+ points over the tier limit, INCREASE only at score <= 35 and 5+ under the limit, 10% position and 35% sector caps, Rs 1 cr liquidity floor) are my defaults, not an industry standard. Tune them at the top of `recommendation_policy.py` and `liquidity_flagger.py`.

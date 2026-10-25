# Stock Portfolio Analyzer

A local FastAPI backend and React dashboard for Indian equity holdings, risk analysis, market sentiment, auditable recommendations, and forward-price review.

## Setup

1. Copy `.env.example` to `.env`; set `KITE_API_KEY`, `KITE_API_SECRET`, `OLLAMA_API_KEY`, and `TAVILY_API_KEY` as needed.
2. Start the backend and database: `docker compose up --build -d`.
3. The API is available at `http://localhost:8000` and Postgres at `localhost:5434`. Start the React dashboard from its own folder (for example `npm run dev`) and open the address it prints. If the dashboard is not on the same machine as the API, set `VITE_STOCK_API_URL`.

To stop the application, run `docker compose down`. Postgres data is retained in the `postgres_data` volume. To run the API outside Docker for development, install dependencies with `uv sync --extra dev`, then:

```powershell
uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

Running outside Docker on Windows also needs the `tzdata` package (`uv add tzdata`): the market-hours logic uses `zoneinfo` with `Asia/Kolkata`, and Windows has no system timezone database.

Ollama Cloud is the default. LangChain `ChatOllama` connects directly to `https://ollama.com` using `OLLAMA_API_KEY` as a Bearer token. The default model is `gemma4:31b`; availability and free monthly credits depend on Ollama's current plan and quotas. Set `OLLAMA_MODE=local` and `OLLAMA_BASE_URL` to use local Ollama instead.

Optional Langfuse Cloud tracing covers Ollama recommendation and news-sentiment calls. Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in `.env` to enable it; `LANGFUSE_BASE_URL` defaults to the EU Cloud endpoint (`https://cloud.langfuse.com`). For a US-region project, set it to `https://us.cloud.langfuse.com`. Recommendation traces from one run share a session ID. Prompts and model outputs are sent to Langfuse when tracing is enabled, so avoid putting secrets in traced inputs.

News search uses the official `tavily-python` SDK. Headline text is passed as untrusted context for qualitative summarization; never include brokerage credentials or tokens in prompts.

## Daily Kite Login

In the Kite developer console, open the Kite Connect app whose API key is in this project's `KITE_API_KEY` setting. Set that app's Redirect URL to `http://localhost:8000/auth/kite/callback` (this stays pointing at the API). This is an OAuth callback for this analyzer's Kite Connect app; it does not configure the normal Kite website or any other app. Keep the redirect URL for your original/other app unchanged, and use a separate Kite Connect app/API key for this analyzer if necessary.

From this dashboard, click **Login to Kite** or **Sync holdings**. The dashboard starts this analyzer's Kite Connect login and sends you to Kite; after sign-in, Kite returns to the callback above, and the analyzer sends you back to the dashboard. The API remembers the dashboard's validated `return_to` address in a 10-minute cookie and uses it to return with `?kite=success` or `?kite=error&message=...`. Only addresses on `localhost`, `127.0.0.1`, `::1`, or the origin in `FRONTEND_URL` are accepted. The access token itself is never put in a URL; it is stored in this project's Postgres database.

Signing in to the normal Kite website, or using another Kite Connect app, should continue to open that site/app. If that flow instead lands on this analyzer, check which Kite Connect app/API key was used and restore that other app's own Redirect URL; this analyzer's callback URL belongs only on this analyzer's Kite Connect app. Opening this API's `/login` directly is also an analyzer login, not the normal Kite website; use the dashboard's **Login to Kite** action for the intended return-to-dashboard flow.

The callback returns to the dashboard only when the login was initiated by this dashboard and its short-lived return cookie is present. Opening `/login` directly, or arriving at this callback from another flow, will not automatically redirect to this dashboard; the API returns a JSON result instead. `FRONTEND_URL` is used only to allow your deployed dashboard's origin for validated return addresses, not as an automatic redirect destination. You can always call `POST /sync/holdings` yourself.

The callback URL belongs to the Kite Connect app configured with this analyzer's API key. If another application uses that same Kite Connect app, Kite will still send its login response to the registered callback URL; this analyzer cannot select the other application's callback. Configure that application's own Kite Connect app/API key and Redirect URL separately.

## Data Workflow

- `POST /sync/holdings` stores Kite holdings in one transaction, merges the same symbol held on NSE and BSE, removes positions you have since sold (their history stays in `holdings_history`), and runs an incremental price sync. If Kite returns no holdings at all, stored holdings are left untouched. Only completed sessions are stored: today's bar is ignored until after 16:00 IST. Each sync re-fetches and re-checks the last 30 days; if the provider has re-based prices (split or bonus), that stock's history is rebuilt and reported under `rebuilt_after_split_or_bonus`. Stocks without history get up to three years from yfinance. Unsupported instruments may return `no_data`, which is reported instead of inventing prices.
- Import the current official AMFI stock ranking/category file through `POST /stocks/metadata/import`. Each row accepts `symbol`, `market_cap_rank`, `sector`, and optional `trade_to_trade_flag`, `isin`, and `exchange`. The app does not infer missing ranks or sectors.
- Complete the risk questionnaire, then call `POST /risk/score-stocks`. Volatility and beta use the trailing 252 trading days. A stock needs at least 60 daily returns (and 60 days overlapping the Nifty 50 for beta) to be scored; otherwise it is **skipped with a reason** and any older score is removed, instead of being given a default. Liquidity is judged on median daily traded value (rupees), not share count.
- `POST /sentiment/refresh` with `{"scope_type":"MARKET"}` fetches Nifty/VIX via yfinance, FII/DII flows from NSE, and news via Tavily, then persists the combined signal. If one source is down the score is computed from the others (reweighted) and the failure is listed under `warnings`; it only errors if nothing could be fetched. FII/DII flows are accumulated daily in `fii_dii_flows` and summed over the last 5 stored days once 3 are available.
- `POST /recommendations/run` uses saved risk, market sentiment (if under 3 days old), stock sentiment (if under 7 days old), position and sector weights, and portfolio alignment. The deterministic policy in `recommendation_policy.py` bounds the model: the LLM may explain, or be more cautious, but cannot recommend a risk-increasing action the numbers do not support. Each stored recommendation records `model_status`, `model_name`, the model's own suggestion and the policy's. Add `?background=true` to return immediately and poll `GET /recommendations/runs/{run_id}`. Every recommendation is written for a non-expert: a short headline, why, and what to keep in mind (`plain_language.py`); model text containing technical words is replaced by standard wording.
- `POST /etfs/analyze?background=true` refreshes the NSE ETF market-watch universe, then analyzes only equity ETFs whose NSE underlying description contains “Smallcap” or “Small Cap” (the temporary small-cap test scope). The equal-weight risk score uses six trailing-year factors: volatility, absolute beta, liquidity, Sortino ratio, maximum drawdown, and annualized tracking error versus Nifty 50. Sortino scores are 20/50/80/100 at ratios >=1.5, >=0.5, >0, and <=0; drawdown scores are 20/50/80 below 15%, through 30%, and above 30%; tracking error scores are 20/50/80 below 5%, through 10%, and above 10%. These are initial screening cutoffs for testing, not regulatory standards. Unavailable factors are omitted and the remaining weights are renormalized. At least 60 daily returns are still required for the beta/volatility-based analysis. Current portfolio/sector/market-cap concentration indexes are reported as context, not mixed into the ETF-only score.
- ETF output also includes the NSE portfolio-turnover ratio, iNAV and price premium/discount, and yield-versus-repo-rate gap only when those source values are present. ETF traded-value liquidity remains separate from the fund portfolio-turnover ratio. The equity small-cap test does not score the yield/repo gap: it is a comparison, not a validated equity risk measure, and the current NSE response may not provide either value. ETF history comes from yfinance, so provider gaps are reported as skipped symbols instead of carrying forward stale scores. Poll `GET /etfs/runs/{run_id}`; `GET /etfs` returns the small-cap catalogue and latest scores/actions.
- ETF risk-fit recommendations do not simulate a purchase amount or constituent-level overlap with the portfolio. They are not comparable to a full fund-risk assessment, especially for debt, commodity, and international ETFs, which are listed but not scored in this first version. The NSE feed and yfinance are external sources and may be incomplete or unavailable.
- `GET /recommendations/review` compares each recommendation with the first later close and the latest close, and with the Nifty 50 over the same window. Outcomes are judged on excess return over Nifty, and calls younger than 5 days are marked `too_early`. These are price returns, not dividend-adjusted total returns.

## Useful Routes

| Route | Purpose |
|---|---|
| `GET /stocks/metadata/status` | Show rank, sector, and T2T metadata coverage |
| `POST /stocks/metadata/import` | Import official reference metadata |
| `POST /risk/questionnaire` | Save a five-dimension investor profile |
| `POST /risk/score-stocks`, `GET /risk/scores` | Calculate and read per-stock risk (with history length and beta observations) |
| `GET /portfolio/summary` | Value-weighted score, HHI, profile alignment, and how much of the portfolio the score covers |
| `POST /sentiment/refresh`, `GET /sentiment/latest` | Fetch, compute, persist, and read sentiment |
| `POST /recommendations/run` | Generate and persist an auditable recommendation run |
| `GET /recommendations`, `GET /recommendations/runs`, `GET /recommendations/runs/{id}` | Read recommendations and run history |
| `GET /recommendations/review` | Review forward price changes against Nifty 50 |
| `POST /etfs/analyze?background=true` | Refresh the NSE ETF universe and score supported equity ETFs |
| `GET /etfs`, `GET /etfs/runs/{id}` | Read the latest ETF catalogue/scores and poll an analysis run |

## Questionnaire scoring convention

Every answer is 1-5 where a higher number means more able or willing to take risk. Questions whose natural wording runs the other way must be reverse-coded by the frontend. In particular `liquidity_needs`: 5 means "I will not need this money for a long time", 1 means "I may need it soon".

## Tests

`python -m pytest tests`. These cover the pure logic (risk maths, policy, sentiment, review) and need no database or network.

Volatility, beta, liquidity, sentiment thresholds, and profile-alignment bands are conventions/tunable policy choices, not SEBI-mandated standards. The market-cap rank categorization is the official standard referenced here.

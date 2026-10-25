import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  Check,
  ChevronLeft,
  ChevronRight,
  CircleHelp,
  Eye,
  EyeOff,
  LoaderCircle,
  RefreshCw,
  Search,
  ShieldCheck,
  Sparkles,
  Wallet,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const API_BASE = (import.meta.env.VITE_STOCK_API_URL || "http://localhost:8000").replace(/\/+$/, "");

type Holding = {
  symbol: string;
  quantity: number | string;
  avg_cost: number | string;
  last_price: number | string;
  exchange?: string | null;
  sector?: string | null;
};

type PortfolioSummary = {
  total_value?: number | string | null;
  weighted_average_risk_score?: number | null;
  risk_label?: string | null;
  profile_alignment?: string | null;
  coverage_pct?: number | null;
  scored_holdings?: number | null;
  total_holdings?: number | null;
  unscored_symbols?: string[] | null;
  [key: string]: unknown;
};
type RiskProfile = {
  total_score?: number | null;
  risk_tier?: string | null;
  capacity_score?: number | null;
  tolerance_score?: number | null;
  answers?: Record<string, number> | null;
  created_at?: string | null;
} | null;
type RiskScore = { symbol: string; composite_score?: number | null; risk_label?: string | null };
type Sentiment = {
  scope_type?: string;
  scope_value?: string;
  composite_sentiment?: number | null;
  label?: string | null;
  created_at?: string | null;
  source_summary?: unknown;
};
type Explanation = { headline?: string; why?: string; watch?: string };
type Recommendation = {
  symbol?: string;
  stock_symbol?: string;
  recommendation?: string;
  reasoning?: string;
  explanation?: Explanation | null;
  inputs_snapshot?: { explanation?: Explanation | null } | null;
  created_at?: string;
  model_status?: string | null;
  llm_recommendation?: string | null;
  policy_recommendation?: string | null;
};
type SyncResult = {
  synced?: number;
  message?: string;
  removed_sold_positions?: string[];
  price_history?: {
    issues?: Array<{ symbol?: string; status?: string; reason?: string; error?: string }>;
    rebuilt_after_split_or_bonus?: string[];
  };
};
type ScoreResult = {
  status?: string;
  records?: Array<{ symbol?: string; status?: string; reason?: string }>;
};
type ETFResult = {
  symbol: string;
  underlying?: string | null;
  asset_class?: string | null;
  last_price?: number | string | null;
  nav?: number | string | null;
  inav?: number | string | null;
  inav_premium_discount_pct?: number | string | null;
  traded_value?: number | string | null;
  portfolio_turnover_ratio?: number | string | null;
  yield_pct?: number | string | null;
  repo_rate_pct?: number | string | null;
  yield_repo_gap_pct?: number | string | null;
  portfolio_hhi?: number | string | null;
  sector_hhi?: number | string | null;
  market_cap_hhi?: number | string | null;
  recommendation?: string | null;
  reasoning?: string | null;
  explanation?: Explanation | null;
  composite_score?: number | string | null;
  risk_label?: string | null;
  volatility_1y?: number | string | null;
  beta_1y?: number | string | null;
  sortino_1y?: number | string | null;
  max_drawdown_1y_pct?: number | string | null;
  tracking_error_1y_pct?: number | string | null;
  liquidity_flag?: boolean | null;
  history_days?: number | null;
};
type ETFAnalysisRun = {
  status: string;
  notes?: string | null;
  recommendations?: ETFResult[];
};

type ApiResult<T> = { value?: T; error?: Error };

class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

// FastAPI returns `detail` as a string for most errors but as a list of
// {loc, msg, type} objects for request-validation errors.
function formatErrorDetail(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) =>
        item && typeof item === "object" && "msg" in item ? String((item as { msg: unknown }).msg) : String(item),
      )
      .join("; ");
  }
  if (detail && typeof detail === "object") return JSON.stringify(detail);
  return String(detail);
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    const detail =
      payload && typeof payload === "object" && "detail" in payload
        ? formatErrorDetail(payload.detail)
        : payload && typeof payload === "object" && "error" in payload
          ? formatErrorDetail(payload.error)
          : `Request failed (${response.status})`;
    throw new ApiError(response.status, detail);
  }
  return payload as T;
}

// ---- Kite sign-in round trip ------------------------------------------------
// Kite's redirect URL points at the API, so signing in leaves this page. These
// keys let the page remember what the user was doing and pick it up on return.
const KITE_PENDING_KEY = "meridian.kite.pending"; // action to resume after login
const KITE_STARTED_KEY = "meridian.kite.started"; // when the login redirect began
const KITE_LOOP_GUARD_MS = 2 * 60 * 1000;

const session = {
  get(key: string): string | null {
    try {
      return window.sessionStorage.getItem(key);
    } catch {
      return null;
    }
  },
  set(key: string, value: string) {
    try {
      window.sessionStorage.setItem(key, value);
    } catch {
      /* storage unavailable: the page still works, it just won't auto-resume */
    }
  },
  remove(key: string) {
    try {
      window.sessionStorage.removeItem(key);
    } catch {
      /* ignore */
    }
  },
};

// Stops an endless login loop if Kite keeps rejecting the session.
function kiteLoginAttemptedRecently(): boolean {
  const started = Number(session.get(KITE_STARTED_KEY));
  return Number.isFinite(started) && started > 0 && Date.now() - started < KITE_LOOP_GUARD_MS;
}

const QUESTIONS = [
  {
    key: "income_stability",
    title: "How stable is your income?",
    low: "Varies often",
    high: "Very steady",
  },
  {
    key: "investment_experience",
    title: "How much investing experience do you have?",
    low: "Just starting",
    high: "Very experienced",
  },
  {
    key: "time_horizon",
    title: "When might you need this money?",
    low: "Within a year",
    high: "More than 10 years",
  },
  {
    key: "drawdown_reaction",
    title: "If your investments fell sharply, what would you do?",
    low: "Sell to limit losses",
    high: "Stay invested",
  },
  {
    key: "liquidity_needs",
    title: "How likely are you to need these funds soon?",
    low: "Likely soon",
    high: "Not for a long time",
  },
] as const;

type Answers = Record<(typeof QUESTIONS)[number]["key"], number>;

const emptyAnswers = (): Answers => ({
  income_stability: 0,
  investment_experience: 0,
  time_horizon: 0,
  drawdown_reaction: 0,
  liquidity_needs: 0,
});

function asNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim() && Number.isFinite(Number(value))) return Number(value);
  return null;
}

function summaryNumber(summary: PortfolioSummary | null, keys: string[]): number | null {
  if (!summary) return null;
  for (const key of keys) {
    const value = asNumber(summary[key]);
    if (value !== null) return value;
  }
  return null;
}

function money(value: unknown): string {
  const num = asNumber(value);
  if (num == null) return "—";
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 2,
  }).format(num);
}

function numberText(value: unknown): string {
  const num = asNumber(value);
  if (num == null) return "—";
  return new Intl.NumberFormat("en-IN", { maximumFractionDigits: 2 }).format(num);
}

function shortDate(value?: string | null): string {
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.valueOf())
    ? ""
    : new Intl.DateTimeFormat("en-IN", { day: "numeric", month: "short", year: "numeric" }).format(date);
}

function ageInDays(value?: string | null): number | null {
  if (!value) return null;
  const time = new Date(value).valueOf();
  return Number.isNaN(time) ? null : (Date.now() - time) / 86_400_000;
}

function listSymbols(symbols: string[], max = 3): string {
  const shown = symbols.slice(0, max).join(", ");
  return symbols.length > max ? `${shown} +${symbols.length - max} more` : shown;
}

function getRecommendationColor(recommendation?: string): string {
  if (!recommendation) return "bg-foreground/5 text-market-slate ring-border";
  const upper = recommendation.toUpperCase();

  if (upper.includes("BUY") || upper.includes("INCREASE")) {
    return "bg-market-gain/15 text-market-gain ring-market-gain/30";
  }
  if (upper.includes("SELL")) {
    return "bg-market-loss/15 text-market-loss ring-market-loss/30";
  }

  return "bg-foreground/10 text-market-slate ring-border";
}

// Say when the action is not purely the AI's call. The backend keeps the model
// inside the investor's risk rules and falls back to the rules if it is down.
// The plain-language explanation saved with each recommendation (older rows only have `reasoning`).
function explanationOf(rec: Recommendation | null): Explanation | null {
  const found = rec?.explanation ?? rec?.inputs_snapshot?.explanation ?? null;
  return found && (found.headline || found.why || found.watch) ? found : null;
}

const PLAIN_ACTION: Record<string, string> = {
  SELL: "Consider selling",
  INCREASE: "Buy a little more",
  HOLD: "Keep holding",
  BUY: "Good to add",
  WATCH: "Wait for now",
  AVOID: "Skip for now",
};

function plainAction(recommendation?: string | null): string | null {
  return recommendation ? PLAIN_ACTION[recommendation.toUpperCase()] ?? null : null;
}

function engineNote(rec: Recommendation | null): { label: string; detail: string } | null {
  if (!rec) return null;
  if (rec.model_status === "ollama_overridden") {
    const suggested = rec.llm_recommendation ? `suggested ${rec.llm_recommendation}` : "suggested a different action";
    return {
      label: "Adjusted",
      detail: `The AI ${suggested}, but your own risk limits point to ${rec.recommendation?.toUpperCase() ?? "a different action"}, so that was used.`,
    };
  }
  if (rec.model_status === "deterministic_fallback") {
    return { label: "Standard wording", detail: "The AI model was unavailable, so this uses standard wording based on your risk limits." };
  }
  return null;
}

// Same wording as the backend label (Positive / Neutral / Negative) and the same 35 / 65 cut-offs.
const getSentimentDetails = (score: number | null | undefined) => {
  if (score == null || isNaN(score)) return { label: "Unknown", color: "text-market-slate", marker: 50 };
  if (score <= 35) return { label: "Negative", color: "text-market-loss", marker: score };
  if (score >= 65) return { label: "Positive", color: "text-market-gain", marker: score };
  return { label: "Neutral", color: "text-market-sky", marker: score };
};

// `source_summary` is stored by the API as JSON (sometimes returned as a string).
function sentimentQuality(raw: unknown): { partial: boolean; notes: string[] } {
  let data: unknown = raw;
  if (typeof raw === "string") {
    try {
      data = JSON.parse(raw);
    } catch {
      return { partial: false, notes: [] };
    }
  }
  const quant =
    data && typeof data === "object"
      ? (data as { quantitative?: { warnings?: unknown; coverage?: unknown } | null }).quantitative
      : null;
  const warnings = Array.isArray(quant?.warnings) ? quant.warnings.map(String) : [];
  const coverage = asNumber(quant?.coverage);
  return { partial: warnings.length > 0 || (coverage !== null && coverage < 1), notes: warnings };
}

function describeSync(result: SyncResult): string {
  const synced = result.synced ?? 0;
  if (synced === 0 && result.message) return result.message;

  const parts = [`Synced ${synced} holding${synced === 1 ? "" : "s"}.`];
  const removed = result.removed_sold_positions ?? [];
  if (removed.length) parts.push(`Removed ${removed.length} sold position${removed.length === 1 ? "" : "s"}: ${listSymbols(removed)}.`);
  const issues = (result.price_history?.issues ?? []).filter((issue) => issue.symbol);
  if (issues.length) parts.push(`No usable price data for ${listSymbols(issues.map((issue) => String(issue.symbol)))}.`);
  const rebuilt = result.price_history?.rebuilt_after_split_or_bonus ?? [];
  if (rebuilt.length) parts.push(`Price history rebuilt after a split or bonus: ${listSymbols(rebuilt)}.`);
  return parts.join(" ");
}

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Meridian — Stock Portfolio Analyzer" },
      {
        name: "description",
        content: "Review your Indian stock holdings, investor risk profile, and portfolio analysis in one place.",
      },
    ],
  }),
  component: Dashboard,
});

function Dashboard() {
  const [holdings, setHoldings] = useState<Holding[]>([]);
  const [summary, setSummary] = useState<PortfolioSummary | null>(null);
  const [profile, setProfile] = useState<RiskProfile>(null);
  const [risks, setRisks] = useState<RiskScore[]>([]);
  const [etfResults, setETFResults] = useState<ETFResult[]>([]);
  const [etfAnalyzing, setETFAnalyzing] = useState(false);
  const [etfSearch, setETFSearch] = useState("");
  const [etfActionFilter, setETFActionFilter] = useState("ALL");
  const [etfPage, setETFPage] = useState(0);
  const [etfNotice, setETFNotice] = useState("");
  const [etfError, setETFError] = useState("");
  const [sentiments, setSentiments] = useState<Sentiment[]>([]);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [portfolioValueHidden, setPortfolioValueHidden] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [analysisRunning, setAnalysisRunning] = useState(false);
  const [sentimentRefreshing, setSentimentRefreshing] = useState(false);
  const [kiteRedirecting, setKiteRedirecting] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [notice, setNotice] = useState("");
  // Why a holding has no risk score, from the most recent analysis run.
  const [skipReasons, setSkipReasons] = useState<Record<string, string>>({});

  // Questionnaire states
  const [questionnaireOpen, setQuestionnaireOpen] = useState(false);
  const [answers, setAnswers] = useState<Answers>(emptyAnswers);
  const [savingAnswers, setSavingAnswers] = useState(false);
  const [questionnaireError, setQuestionnaireError] = useState("");

  // Pagination & Filter states
  const [holdingPage, setHoldingPage] = useState(0);
  const [selectedRec, setSelectedRec] = useState<Recommendation | null>(null);
  const [holdingRiskFilter, setHoldingRiskFilter] = useState("ALL");
  const [recFilter, setRecFilter] = useState("ALL");

  const HOLDINGS_PER_PAGE = 5;
  const ETF_RESULTS_PER_PAGE = 10;
  // The backend stores INCREASE for "add to an existing holding"; BUY no longer appears.
  const REC_FILTER_OPTIONS = ["INCREASE", "HOLD", "SELL"];
  const ETF_ACTION_FILTER_OPTIONS = ["BUY", "WATCH", "AVOID", "UNSCORED"];

  const loadDashboard = useCallback(async (isRefresh = false) => {
    if (isRefresh) setRefreshing(true);
    else setLoading(true);
    setErrorMessage("");
    const tasks: Array<Promise<ApiResult<unknown>>> = [
      api<Holding[]>("/holdings").then((value) => ({ value })).catch((error: Error) => ({ error })),
      api<PortfolioSummary>("/portfolio/summary").then((value) => ({ value })).catch((error: Error) => ({ error })),
      api<{ data: RiskProfile }>("/risk/questionnaire/latest")
        .then((value) => ({ value }))
        .catch((error: Error) => ({ error })),
      api<RiskScore[]>("/risk/scores").then((value) => ({ value })).catch((error: Error) => ({ error })),
      api<ETFResult[]>("/etfs").then((value) => ({ value })).catch((error: Error) => ({ error })),
      api<Sentiment[]>("/sentiment/latest").then((value) => ({ value })).catch((error: Error) => ({ error })),
      api<Recommendation[]>("/recommendations")
        .then((value) => ({ value }))
        .catch((error: Error) => ({ error })),
    ];
    const [holdingsResult, summaryResult, profileResult, risksResult, etfsResult, sentimentResult, recsResult] =
      await Promise.all(tasks);

    if (holdingsResult.value) {
      setHoldings(holdingsResult.value as Holding[]);
      setHoldingPage(0);
    }
    if (summaryResult.value) setSummary(summaryResult.value as PortfolioSummary);
    if (profileResult.value) setProfile((profileResult.value as { data: RiskProfile }).data);
    if (risksResult.value) setRisks(risksResult.value as RiskScore[]);
    if (etfsResult?.value) setETFResults(etfsResult.value as ETFResult[]);
    if (sentimentResult.value) setSentiments(sentimentResult.value as Sentiment[]);

    if (recsResult.value) {
      // The API returns newest first, so the first row per symbol is the latest.
      const rawRecs = recsResult.value as Recommendation[];
      const deduplicated: Recommendation[] = [];
      const seen = new Set<string>();

      for (const rec of rawRecs) {
        const symbol = (rec.symbol || rec.stock_symbol || "").trim().toUpperCase();
        if (symbol && !seen.has(symbol)) {
          seen.add(symbol);
          deduplicated.push(rec);
        }
      }
      setRecommendations(deduplicated);
    }

    const firstError = [
      holdingsResult,
      summaryResult,
      profileResult,
      risksResult,
      sentimentResult,
      recsResult,
    ].find((result) => result.error)?.error;

    if (firstError) {
      setErrorMessage(
        firstError instanceof TypeError
          ? `Can't reach the stock analyzer at ${API_BASE}. Start the API or set VITE_STOCK_API_URL.`
          : firstError.message,
      );
    }
    setLoading(false);
    setRefreshing(false);
  }, []);

  // Market sentiment is refreshed on request only. Each refresh calls NSE, a news
  // search and the AI model and is stored as a new reading, and the underlying
  // data (daily flows, 20-day return, moving averages) does not move minute to minute.
  const refreshSentiment = useCallback(async () => {
    setSentimentRefreshing(true);
    setNotice("");
    try {
      await api("/sentiment/refresh", {
        method: "POST",
        body: JSON.stringify({ scope_type: "MARKET" }),
      });
      setNotice("Market sentiment refreshed.");
      const result = await api<Sentiment[]>("/sentiment/latest").catch(() => null);
      if (result) setSentiments(result);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Market sentiment could not be refreshed.");
    } finally {
      setSentimentRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void loadDashboard();
  }, [loadDashboard]);

  // Leave for Kite and come straight back: the API sends the browser to this page
  // again once the login is done (see `return_to`), and the effect below resumes.
  const startKiteLogin = useCallback((resumeSync: boolean) => {
    session.set(KITE_STARTED_KEY, String(Date.now()));
    if (resumeSync) session.set(KITE_PENDING_KEY, "sync");
    setKiteRedirecting(true);
    setNotice("Taking you to Kite to sign in. You'll come straight back here.");
    const returnTo = `${window.location.origin}${window.location.pathname}`;
    window.location.assign(`${API_BASE}/login?return_to=${encodeURIComponent(returnTo)}`);
  }, []);

  const syncHoldings = useCallback(
    async (afterLogin = false) => {
      setRefreshing(true);
      setNotice(afterLogin ? "Logged in to Kite. Syncing your holdings…" : "");
      try {
        const result = await api<SyncResult>("/sync/holdings", { method: "POST" });
        session.remove(KITE_STARTED_KEY);
        setNotice(describeSync(result));
        await loadDashboard(true);
      } catch (error) {
        const message = error instanceof Error ? error.message.toLowerCase() : "";
        const needsLogin =
          error instanceof ApiError &&
          (error.status === 401 ||
            (error.status === 502 && (message.includes("token") || message.includes("permission"))));
        if (needsLogin) {
          if (afterLogin || kiteLoginAttemptedRecently()) {
            // We have only just signed in. Do not bounce the user to Kite again.
            setNotice("Kite did not accept the login. Please use “Login to Kite” to try again.");
            setRefreshing(false);
            return;
          }
          startKiteLogin(true);
          return;
        }
        setNotice(error instanceof Error ? error.message : "Holdings could not be synced.");
        setRefreshing(false);
      }
    },
    [loadDashboard, startKiteLogin],
  );

  // Back from Kite: the API redirects here with ?kite=success or ?kite=error.
  const returnHandled = useRef(false);
  useEffect(() => {
    if (returnHandled.current) return;
    returnHandled.current = true;

    const params = new URLSearchParams(window.location.search);
    const kite = params.get("kite");
    if (!kite) return;

    const message = params.get("message");
    params.delete("kite");
    params.delete("message");
    const query = params.toString();
    // Remove the flag so a page refresh does not repeat this.
    window.history.replaceState(null, "", `${window.location.pathname}${query ? `?${query}` : ""}${window.location.hash}`);

    const pending = session.get(KITE_PENDING_KEY);
    session.remove(KITE_PENDING_KEY);

    if (kite === "success") {
      if (pending === "sync") {
        void syncHoldings(true);
      } else {
        session.remove(KITE_STARTED_KEY);
        setNotice("Logged in to Kite. Use “Sync holdings” to refresh your portfolio.");
      }
    } else {
      session.remove(KITE_STARTED_KEY);
      setNotice(message ? `Kite login failed: ${message}` : "Kite login failed. Please try again.");
    }
  }, [syncHoldings]);

  // Derived Values
  const riskBySymbol = useMemo(() => new Map(risks.map((risk) => [risk.symbol, risk])), [risks]);
  const totalValue = summaryNumber(summary, ["portfolio_value", "total_value", "current_value", "total_portfolio_value"]) ??
    holdings.reduce((total, item) => total + (asNumber(item.quantity) || 0) * (asNumber(item.last_price) || 0), 0);
  const totalCost = holdings.reduce((total, item) => total + (asNumber(item.quantity) || 0) * (asNumber(item.avg_cost) || 0), 0);
  const totalPnl = summaryNumber(summary, ["total_pnl", "pnl", "profit_loss", "unrealized_pnl"]) ?? totalValue - totalCost;
  const hasHoldings = holdings.length > 0;
  const marketSentiment = sentiments.find(
    (item) => item.scope_type === "MARKET" && (!item.scope_value || item.scope_value === "INDIA"),
  );
  const sentimentScore = asNumber(marketSentiment?.composite_sentiment);
  const sentimentDetails = getSentimentDetails(sentimentScore);
  const sentimentAge = ageInDays(marketSentiment?.created_at);
  // The API ignores market sentiment older than 3 days when making recommendations.
  const sentimentStale = sentimentAge !== null && sentimentAge > 3;
  const sentimentQualityInfo = sentimentQuality(marketSentiment?.source_summary);
  const completedQuestions = Object.values(answers).filter((answer) => answer > 0).length;
  const hasProfile = Boolean(profile);

  // Portfolio-level risk line
  const portfolioRisk = asNumber(summary?.weighted_average_risk_score);
  const scoredCount = asNumber(summary?.scored_holdings);
  const holdingCount = asNumber(summary?.total_holdings);
  const coveragePct = asNumber(summary?.coverage_pct);
  const unscoredSymbols = Array.isArray(summary?.unscored_symbols) ? summary.unscored_symbols.map(String) : [];
  const alignment = summary?.profile_alignment;
  const alignmentText =
    alignment === "within_profile"
      ? `within your ${profile?.risk_tier ?? ""} profile`.replace("  ", " ")
      : alignment === "above_profile"
        ? `above what your ${profile?.risk_tier ?? ""} profile is comfortable with`.replace("  ", " ")
        : "";

  // Dynamic filter options
  const latestRecommendationBySymbol = useMemo(
    () =>
      new Map(
        recommendations.map((recommendation) => [
          (recommendation.symbol || recommendation.stock_symbol || "").trim().toUpperCase(),
          recommendation,
        ]),
      ),
    [recommendations],
  );
  const uniqueRisks = useMemo(() => {
    const rSet = new Set<string>();
    holdings.forEach((h) => {
      const r = riskBySymbol.get(h.symbol)?.risk_label;
      rSet.add(r ? r.toUpperCase() : "UNSCORED");
    });
    return Array.from(rSet).sort();
  }, [holdings, riskBySymbol]);

  // Filter Arrays
  const filteredHoldings = useMemo(() => {
    return holdings.filter((h) => {
      const r = riskBySymbol.get(h.symbol)?.risk_label;
      const riskStr = r ? r.toUpperCase() : "UNSCORED";
      if (holdingRiskFilter !== "ALL" && riskStr !== holdingRiskFilter) return false;
      if (recFilter !== "ALL") {
        const recommendation = latestRecommendationBySymbol.get(h.symbol.trim().toUpperCase());
        return recommendation?.recommendation?.toUpperCase() === recFilter;
      }
      return true;
    });
  }, [
    holdings,
    riskBySymbol,
    holdingRiskFilter,
    recFilter,
    latestRecommendationBySymbol,
  ]);

  const totalHoldingPages = Math.ceil(filteredHoldings.length / HOLDINGS_PER_PAGE);
  const paginatedHoldings = filteredHoldings.slice(holdingPage * HOLDINGS_PER_PAGE, (holdingPage + 1) * HOLDINGS_PER_PAGE);

  const openQuestionnaire = () => {
    const saved = profile?.answers;
    const nextAnswers = emptyAnswers();
    for (const question of QUESTIONS) {
      const value = saved?.[question.key];
      if (typeof value === "number" && value >= 1 && value <= 5) nextAnswers[question.key] = value;
    }
    setAnswers(nextAnswers);
    setQuestionnaireError("");
    setQuestionnaireOpen(true);
  };

  const submitQuestionnaire = async () => {
    if (completedQuestions !== QUESTIONS.length) {
      setQuestionnaireError("Please answer all five questions before saving.");
      return;
    }
    setSavingAnswers(true);
    setQuestionnaireError("");
    try {
      await api("/risk/questionnaire", {
        method: "POST",
        body: JSON.stringify(answers),
      });
      setQuestionnaireOpen(false);
      setNotice("Your risk profile has been updated.");
      await loadDashboard(true);
    } catch (error) {
      setQuestionnaireError(error instanceof Error ? error.message : "Your answers could not be saved.");
    } finally {
      setSavingAnswers(false);
    }
  };

  const runAnalysis = async () => {
    setAnalysisRunning(true);
    setNotice("");
    let skippedNote = "";
    try {
      const scoring = await api<ScoreResult>("/risk/score-stocks", { method: "POST" });
      const skipped = (scoring.records ?? []).filter((record) => record.status === "skipped" && record.symbol);
      const reasons: Record<string, string> = {};
      for (const record of skipped) reasons[String(record.symbol)] = record.reason || "Skipped";
      setSkipReasons(reasons);
      if (skipped.length) {
        skippedNote = ` ${skipped.length} stock${skipped.length === 1 ? " was" : "s were"} not scored (${listSymbols(
          skipped.map((record) => String(record.symbol)),
        )}). Hover the Risk column for the reason.`;
      }

      await api("/recommendations/run", { method: "POST" });
      setNotice(`Stock risk scores and recommendations are ready.${skippedNote}`);
      await loadDashboard(true);
    } catch (error) {
      const base = error instanceof Error ? error.message : "Analysis could not be completed.";
      setNotice(`${base}${skippedNote}`);
      // Scores may have been saved even though the recommendations step failed.
      await loadDashboard(true);
    } finally {
      setAnalysisRunning(false);
    }
  };

  const runETFAnalysis = async () => {
    setETFAnalyzing(true);
    setETFPage(0);
    setETFError("");
    setETFNotice(
      "Refreshing NSE listings and analyzing small-cap ETF price histories…",
    );
    try {
      const started = await api<{ run_id: number }>("/etfs/analyze?background=true", { method: "POST" });
      const startedAt = Date.now();
      let run: ETFAnalysisRun;
      do {
        await new Promise((resolve) => window.setTimeout(resolve, 1800));
        run = await api<ETFAnalysisRun>(`/etfs/runs/${started.run_id}`);
        if (run.recommendations) setETFResults(run.recommendations);
        if (Date.now() - startedAt > 15 * 60 * 1000) throw new Error("ETF analysis is taking longer than expected. Check the run again shortly.");
      } while (run.status === "RUNNING");

      if (run.status === "FAILED") throw new Error(run.notes || "ETF analysis failed.");
      const notes = run.notes
        ? (JSON.parse(run.notes) as {
            scored_count?: number;
            small_cap_etf_count?: number;
            skipped?: Array<{ symbol: string }>;
          })
        : {};
      const skippedCount = notes.skipped?.length ?? 0;
      setETFNotice(
        `${notes.scored_count ?? run.recommendations?.length ?? 0} small-cap ETFs scored${skippedCount ? ` · ${skippedCount} skipped for insufficient or unavailable data` : ""}.`,
      );
      await loadDashboard(true);
    } catch (error) {
      setETFError(error instanceof Error ? error.message : "ETF analysis could not be completed.");
      setETFNotice("");
    } finally {
      setETFAnalyzing(false);
    }
  };

  const selectedEngineNote = engineNote(selectedRec);
  const selectedExplanation = explanationOf(selectedRec);
  const equityETFResults = etfResults.filter(
    (item) =>
      item.asset_class === "Equity" &&
      /small\s*cap/i.test(item.underlying || "") &&
      (!etfSearch ||
        `${item.symbol} ${item.underlying || ""}`
          .toLowerCase()
          .includes(etfSearch.toLowerCase())),
  );
  const filteredETFResults = equityETFResults.filter((item) => {
    const unscored = asNumber(item.composite_score) == null;
    if (etfActionFilter === "UNSCORED") return unscored;
    if (etfActionFilter === "ALL") return true;
    return !unscored && item.recommendation?.toUpperCase() === etfActionFilter;
  });
  const totalETFPages = Math.ceil(filteredETFResults.length / ETF_RESULTS_PER_PAGE);
  const visibleETFPage = Math.min(etfPage, Math.max(totalETFPages - 1, 0));
  const paginatedETFResults = filteredETFResults.slice(
    visibleETFPage * ETF_RESULTS_PER_PAGE,
    (visibleETFPage + 1) * ETF_RESULTS_PER_PAGE,
  );
  const scoredETFCount = etfResults.filter(
    (item) =>
      item.asset_class === "Equity" &&
      /small\s*cap/i.test(item.underlying || "") &&
      asNumber(item.composite_score) !== null,
  ).length;

  return (
    <main className="relative min-h-screen overflow-hidden bg-background text-foreground">
      <div className="pointer-events-none absolute inset-x-0 top-0 h-80 bg-[radial-gradient(ellipse_at_top,oklch(0.46_0.13_255_/_0.2),transparent_70%)]" />
      <div className="relative mx-auto max-w-[1340px] px-4 py-5 sm:px-6 lg:px-8">

        {/* Navigation Bar */}
        <header className="mb-6 flex flex-wrap items-center justify-between gap-4 border-b border-border/50 pb-5">
          <div className="flex items-center gap-3">
            <div className="grid size-10 place-items-center rounded-lg bg-market-sky/15 ring-1 ring-market-sky/30">
              <span className="font-display text-base font-bold text-market-sky">M</span>
            </div>
            <div>
              <div className="font-display text-lg font-semibold leading-none">Meridian</div>
              <div className="mt-1 text-xs text-market-slate">Portfolio desk · NSE / BSE</div>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2.5">
            <span className="inline-flex items-center gap-2 rounded-full bg-foreground/5 px-3 py-1.5 text-[11px] text-market-slate ring-1 ring-border">
              <span className={`size-1.5 rounded-full ${hasHoldings ? "bg-market-gain" : "bg-market-slate"}`} />
              {hasHoldings ? `${holdings.length} positions loaded` : "Kite · not synced"}
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={() => startKiteLogin(true)}
              disabled={kiteRedirecting}
              className="h-9 rounded-lg px-3.5 text-xs font-semibold shadow-none"
            >
              {kiteRedirecting ? <LoaderCircle className="size-3.5 animate-spin mr-1.5" /> : null}
              Login to Kite
            </Button>
            <Button
              size="sm"
              onClick={() => void syncHoldings()}
              disabled={refreshing || kiteRedirecting}
              className="h-9 rounded-lg bg-primary px-3.5 text-xs font-semibold text-primary-foreground shadow-none hover:bg-primary/90"
            >
              {refreshing ? <LoaderCircle className="size-3.5 animate-spin mr-1.5" /> : <RefreshCw className="size-3.5 mr-1.5" />}
              Sync holdings
            </Button>
          </div>
        </header>

        {/* Global Notice / Error */}
        {(errorMessage || notice) && (
          <div className="mb-5 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-border bg-card/80 px-4 py-2.5 text-xs text-market-slate" role="status">
            <span>{notice || errorMessage}</span>
            {errorMessage && !notice && (
              <Button variant="outline" size="sm" className="h-7 text-xs" onClick={() => void loadDashboard(true)} disabled={refreshing}>
                Try again
              </Button>
            )}
          </div>
        )}

        {/* TOP DECK: 3 Clean High-Level Summary Cards */}
        <section className="mb-6 grid grid-cols-1 gap-4 md:grid-cols-12 dashboard-rise">

          {/* Card 1: Portfolio Value & Return */}
          <div className="flex flex-col justify-between rounded-xl bg-card/75 p-4 ring-1 ring-border backdrop-blur-xl md:col-span-5">
            <div>
              <div className="flex items-center justify-between">
                <span className="font-mono text-[10px] uppercase text-market-slate">Portfolio value</span>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 w-6 p-0"
                  onClick={() => setPortfolioValueHidden((hidden) => !hidden)}
                  aria-label={portfolioValueHidden ? "Show portfolio values" : "Hide portfolio values"}
                  title={portfolioValueHidden ? "Show portfolio values" : "Hide portfolio values"}
                >
                  {portfolioValueHidden ? <Eye className="size-3.5" /> : <EyeOff className="size-3.5" />}
                </Button>
              </div>
              <div className="mt-1 flex flex-wrap items-baseline gap-2.5">
                <h1 className="font-display text-3xl font-bold">
                  {portfolioValueHidden ? "••••••" : hasHoldings ? money(totalValue) : "—"}
                </h1>
                {hasHoldings && !portfolioValueHidden && (
                  <span className={`inline-flex items-center gap-0.5 rounded px-1.5 py-0.5 font-mono text-[11px] ring-1 ${totalPnl >= 0 ? "bg-market-gain/10 text-market-gain ring-market-gain/20" : "bg-market-loss/10 text-market-loss ring-market-loss/20"}`}>
                    {totalPnl >= 0 ? <ArrowUpRight className="size-3" /> : <ArrowDownRight className="size-3" />}
                    {money(Math.abs(totalPnl))}
                  </span>
                )}
              </div>
            </div>
            <div className="mt-3 flex items-center justify-between border-t border-border/50 pt-2.5 font-mono text-[11px] text-market-slate">
              <span>Invested {portfolioValueHidden ? "••••••" : money(totalCost)}</span>
              <span>{portfolioValueHidden ? "••••" : holdings.length} positions</span>
            </div>
          </div>

          {/* Card 2: Market Sentiment (refreshed on request) */}
          <div className="flex flex-col justify-between rounded-xl bg-card/75 p-4 ring-1 ring-border backdrop-blur-xl md:col-span-4">
            <div className="flex items-center justify-between">
              <span className="font-mono text-[10px] uppercase text-market-slate">Market sentiment</span>
              <div className="flex items-center gap-1.5">
                {sentimentStale && (
                  <span
                    className="rounded bg-market-loss/10 px-1.5 py-0.5 font-mono text-[10px] text-market-loss ring-1 ring-market-loss/20"
                    title="Older than 3 days: recommendations ignore it until you refresh."
                  >
                    Stale
                  </span>
                )}
                <span className={`rounded px-1.5 py-0.5 font-mono text-[10px] ring-1 ${marketSentiment ? "bg-market-sky/10 text-market-sky ring-market-sky/20" : "bg-foreground/5 text-market-slate ring-border"}`}>
                  {marketSentiment?.label || "Not refreshed"}
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 w-6 p-0"
                  onClick={() => void refreshSentiment()}
                  disabled={sentimentRefreshing}
                  title="Refresh market sentiment"
                >
                  <RefreshCw className={`size-3 ${sentimentRefreshing ? "animate-spin" : ""}`} />
                </Button>
              </div>
            </div>

            {marketSentiment ? (
              <div className="my-1.5">
                <div className="flex items-baseline justify-between mb-1">
                  <div className="flex items-baseline gap-1.5">
                    <span className={`font-display text-2xl font-bold ${sentimentDetails.color}`}>
                      {sentimentScore == null ? "—" : Math.round(sentimentScore)}
                    </span>
                    <span className="font-mono text-[10px] text-market-slate">/ 100</span>
                  </div>
                  <span className={`font-mono text-[11px] font-medium uppercase ${sentimentDetails.color}`}>
                    {sentimentDetails.label}
                  </span>
                </div>

                <div className="relative h-1.5 w-full rounded-full bg-gradient-to-r from-red-500 via-gray-400 to-emerald-500">
                  {sentimentScore != null && (
                    <div
                      className="absolute top-1/2 -mt-1.5 h-3 w-1 rounded-full bg-foreground shadow ring-1 ring-background transition-all duration-500"
                      style={{ left: `calc(${Math.min(100, Math.max(0, sentimentDetails.marker))}% - 2px)` }}
                    />
                  )}
                </div>
                {sentimentQualityInfo.partial && (
                  <p className="mt-1.5 text-[10px] text-market-slate" title={sentimentQualityInfo.notes.join("\n")}>
                    Based on partial data: some sources were unavailable.
                  </p>
                )}
              </div>
            ) : (
              <div className="my-2 flex items-center gap-1.5 text-xs text-market-slate">
                <Activity className="size-3.5" /> Refresh for market signal
              </div>
            )}

            <div className="border-t border-border/50 pt-2 font-mono text-[10px] text-market-slate flex justify-between">
              <span>NSE / Sensex</span>
              <span>{shortDate(marketSentiment?.created_at) || "Not refreshed"}</span>
            </div>
          </div>

          {/* Card 3: Investor Profile */}
          <div className="flex flex-col justify-between rounded-xl bg-card/75 p-4 ring-1 ring-border backdrop-blur-xl md:col-span-3">
            <div className="flex items-center justify-between">
              <span className="font-mono text-[10px] uppercase text-market-slate">Investor profile</span>
              <span className={`rounded px-1.5 py-0.5 font-mono text-[10px] ring-1 ${hasProfile ? "bg-market-teal/10 text-market-teal ring-market-teal/20" : "bg-primary/10 text-primary ring-primary/20"}`}>
                {hasProfile ? "On file" : "Needed"}
              </span>
            </div>

            <div className="my-1">
              <h2 className="font-display text-lg font-bold truncate">
                {profile?.risk_tier || "Unprofiled"}
              </h2>
              <p className="text-[11px] text-market-slate truncate">
                {profile?.total_score != null ? `Questionnaire score ${numberText(profile.total_score)} / 25` : "Take questionnaire to assess"}
              </p>
            </div>

            <Button
              variant="outline"
              size="sm"
              className="h-7 w-full text-[11px] font-medium shadow-none"
              onClick={openQuestionnaire}
            >
              {hasProfile ? "Retake survey" : "Start questionnaire"}
            </Button>
          </div>
        </section>

        <Tabs defaultValue="holdings" className="w-full">
          <TabsList aria-label="Portfolio views">
            <TabsTrigger value="holdings">
              <Wallet className="mr-2 size-4" aria-hidden="true" />
              Holdings
            </TabsTrigger>
            <TabsTrigger value="etfs">
              <Activity className="mr-2 size-4" aria-hidden="true" />
              ETFs
            </TabsTrigger>
          </TabsList>

          <TabsContent value="holdings" className="mt-4">
            {/* Portfolio risk against the investor's profile */}
            {hasHoldings && (
              <div className="mb-6 flex flex-wrap items-center gap-x-3 gap-y-1 rounded-xl bg-card/75 px-4 py-2.5 text-xs ring-1 ring-border backdrop-blur-xl">
                <ShieldCheck
                  className={`size-4 shrink-0 ${alignment === "above_profile" ? "text-market-loss" : "text-market-teal"}`}
                  aria-hidden="true"
                />
                {portfolioRisk !== null ? (
                  <>
                    <span className="font-medium text-foreground">
                      Portfolio risk {Math.round(portfolioRisk)} / 100{summary?.risk_label ? ` (${summary.risk_label})` : ""}
                    </span>
                    {alignmentText && (
                      <span className={alignment === "above_profile" ? "text-market-loss" : "text-market-slate"}>
                        {alignmentText}
                      </span>
                    )}
                    {scoredCount !== null && holdingCount !== null && scoredCount < holdingCount && (
                      <span
                        className="text-market-slate"
                        title={unscoredSymbols.length ? `Not scored: ${unscoredSymbols.join(", ")}` : undefined}
                      >
                        · based on {scoredCount} of {holdingCount} holdings
                        {coveragePct !== null ? ` (${Math.round(coveragePct)}% of value)` : ""}
                      </span>
                    )}
                  </>
                ) : (
                  <span className="text-market-slate">Portfolio risk isn't available yet. Run the analysis to score your holdings.</span>
                )}
              </div>
            )}

            {/* Combined holdings ledger and latest analysis */}
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-12 items-start">
          <section className="rounded-xl bg-card/75 p-4 ring-1 ring-border backdrop-blur-xl sm:p-5 lg:col-span-12">
            <div>
              <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <BarChart3 className="size-4 text-market-sky" />
                  <h2 className="font-display text-sm font-semibold">Holdings &amp; latest analysis</h2>
                  <span className="rounded-full bg-foreground/5 px-2 py-0.5 font-mono text-[10px] text-market-slate">{filteredHoldings.length} items</span>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  {uniqueRisks.length > 0 && (
                    <select
                      className="h-7 rounded-md border border-border bg-foreground/5 px-2 font-mono text-[10px] text-market-slate outline-none focus:ring-1 focus:ring-primary"
                      value={holdingRiskFilter}
                      onChange={(e) => {
                        setHoldingRiskFilter(e.target.value);
                        setHoldingPage(0);
                      }}
                      title="Filter by risk label"
                    >
                      <option value="ALL">All Risks</option>
                      {uniqueRisks.map((r) => (
                        <option key={r} value={r}>{r}</option>
                      ))}
                    </select>
                  )}
                  <select
                    className="h-7 rounded-md border border-border bg-foreground/5 px-2 font-mono text-[10px] text-market-slate outline-none focus:ring-1 focus:ring-primary"
                    value={recFilter}
                    onChange={(e) => {
                      setRecFilter(e.target.value);
                      setHoldingPage(0);
                    }}
                    title="Filter holdings by latest analysis action"
                  >
                    <option value="ALL">All Actions</option>
                    {REC_FILTER_OPTIONS.map((r) => (
                      <option key={r} value={r}>{r}</option>
                    ))}
                  </select>

                  <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={() => void loadDashboard(true)} disabled={refreshing} title="Reload">
                    <RefreshCw className={`size-3.5 ${refreshing ? "animate-spin" : ""}`} />
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-7 px-2 text-xs shadow-none"
                    onClick={() => void runAnalysis()}
                    disabled={analysisRunning || !hasProfile || !hasHoldings}
                    title={!hasProfile ? "Complete risk profile first" : !hasHoldings ? "Sync holdings first" : "Run analysis"}
                  >
                    {analysisRunning ? <LoaderCircle className="size-3 animate-spin mr-1" /> : <Sparkles className="size-3 mr-1" />}
                    Run
                  </Button>
                </div>
              </div>

              <div className="w-full overflow-x-auto">
                <table className="w-full border-collapse text-xs">
                  <thead>
                    <tr className="border-b border-border/80 font-mono text-[10px] uppercase text-market-slate">
                      <th className="pb-2 text-left font-medium">Instrument</th>
                      <th className="pb-2 text-right font-medium">Qty</th>
                      <th className="pb-2 text-right font-medium">Avg cost</th>
                      <th className="pb-2 text-right font-medium">Last price</th>
                      <th className="pb-2 text-right font-medium">P&amp;L</th>
                      <th className="pb-2 text-right font-medium">Risk</th>
                      <th className="pb-2 text-right font-medium">Latest action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border/50">
                    {loading && holdings.length === 0 ? (
                      <tr>
                        <td colSpan={7} className="py-12 text-center text-market-slate">
                          <LoaderCircle className="mx-auto mb-2 size-4 animate-spin" />
                          Loading holdings…
                        </td>
                      </tr>
                    ) : filteredHoldings.length === 0 ? (
                      <tr>
                        <td colSpan={7} className="py-12 text-center text-market-slate">
                          <Wallet className="mx-auto mb-2 size-5" />
                          <p className="font-medium text-xs">
                            {hasHoldings ? "No holdings match filter" : "Sync holdings to see your portfolio"}
                          </p>
                        </td>
                      </tr>
                    ) : (
                      paginatedHoldings.map((holding) => {
                        const qty = asNumber(holding.quantity) || 0;
                        const last = asNumber(holding.last_price) || 0;
                        const avg = asNumber(holding.avg_cost) || 0;
                        const pnl = (last - avg) * qty;
                        const risk = riskBySymbol.get(holding.symbol);
                        const recommendation = latestRecommendationBySymbol.get(holding.symbol.trim().toUpperCase());
                        const symbol = holding.symbol;
                        return (
                          <tr key={holding.symbol} className="hover:bg-foreground/[0.02]">
                            <td className="py-3 pr-2">
                              <button
                                type="button"
                                className="text-left font-semibold text-foreground underline-offset-2 hover:underline disabled:cursor-default disabled:no-underline"
                                onClick={() => recommendation && setSelectedRec(recommendation)}
                                disabled={!recommendation}
                                title={recommendation ? "View latest analysis" : "No analysis available for this holding"}
                              >
                                {symbol}
                              </button>
                              <div className="text-[10px] text-market-slate">
                                {[holding.exchange || "NSE", holding.sector].filter(Boolean).join(" · ")}
                              </div>
                            </td>
                            <td className="py-3 text-right font-mono text-[11px] text-market-slate">{numberText(qty)}</td>
                            <td className="py-3 text-right font-mono text-[11px] text-market-slate">{money(avg)}</td>
                            <td className="py-3 text-right font-mono text-[11px]">{money(last)}</td>
                            <td className={`py-3 text-right font-mono text-[11px] ${pnl >= 0 ? "text-market-gain" : "text-market-loss"}`}>
                              {money(pnl)}
                            </td>
                            <td className="py-3 text-right">
                              {risk ? (
                                <span className="inline-flex rounded bg-foreground/5 px-1.5 py-0.5 font-mono text-[9px] text-market-slate">
                                  {risk.risk_label || numberText(risk.composite_score)}
                                </span>
                              ) : (
                                <span
                                  className="cursor-help text-market-slate text-[10px]"
                                  title={skipReasons[holding.symbol] ?? "Not scored yet. Run the analysis."}
                                >
                                  —
                                </span>
                              )}
                            </td>
                            <td className="py-3 text-right">
                              {recommendation ? (
                                <span className={`rounded px-2 py-0.5 font-mono text-[10px] font-medium ring-1 ${getRecommendationColor(recommendation.recommendation)}`}>
                                  {recommendation.recommendation?.toUpperCase() || "REVIEWED"}
                                </span>
                              ) : (
                                <span className="text-[10px] text-market-slate">No analysis yet</span>
                              )}
                            </td>
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>
            </div>

            {hasHoldings && (
              <div className="mt-4 flex items-center justify-between border-t border-border/60 pt-3">
                <div className="flex items-center gap-1.5">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7 px-2 text-xs"
                    disabled={holdingPage === 0}
                    onClick={() => setHoldingPage((p) => Math.max(0, p - 1))}
                  >
                    <ChevronLeft className="size-3" />
                  </Button>
                  <span className="font-mono text-[10px] text-market-slate">
                    {totalHoldingPages > 0 ? holdingPage + 1 : 0} / {totalHoldingPages || 1}
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7 px-2 text-xs"
                    disabled={holdingPage >= totalHoldingPages - 1}
                    onClick={() => setHoldingPage((p) => Math.min(totalHoldingPages - 1, p + 1))}
                  >
                    <ChevronRight className="size-3" />
                  </Button>
                </div>
                <div className="font-mono text-[10px] text-market-slate">
                  Page {totalHoldingPages > 0 ? holdingPage + 1 : 0} of {totalHoldingPages || 1}
                </div>
              </div>
            )}
          </section>
            </div>
          </TabsContent>

          <TabsContent value="etfs" className="mt-4">
            {/* ETF risk-fit scan */}
            <section className="rounded-xl bg-card/75 p-4 ring-1 ring-border backdrop-blur-xl sm:p-5" aria-labelledby="etf-analyzer-title">
          <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
            <div>
              <div className="flex items-center gap-2">
                <BarChart3 className="size-4 text-market-teal" aria-hidden="true" />
                <h2 id="etf-analyzer-title" className="font-display text-sm font-semibold">ETF analyzer</h2>
                <span className="rounded bg-foreground/5 px-1.5 py-0.5 font-mono text-[9px] text-market-slate ring-1 ring-border">NSE equity</span>
              </div>
              <p className="mt-1 text-[11px] text-market-slate">
                Risk-fit ranking only. Constituent overlap and purchase-size effects are not modeled.
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              className="h-8 shadow-none"
              onClick={() => void runETFAnalysis()}
              disabled={etfAnalyzing || !hasProfile}
              title={!hasProfile ? "Complete the investor profile first" : "Refresh NSE listings and analyze available ETF histories"}
            >
              {etfAnalyzing ? <LoaderCircle className="mr-1.5 size-3.5 animate-spin" /> : <Activity className="mr-1.5 size-3.5" />}
              {etfAnalyzing ? "Analyzing" : "Analyze ETFs"}
            </Button>
          </div>

          <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
            <div className="relative w-full sm:max-w-xs">
              <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-market-slate" aria-hidden="true" />
              <input
                type="search"
                value={etfSearch}
                onChange={(event) => {
                  setETFSearch(event.target.value);
                  setETFPage(0);
                }}
                placeholder="Filter symbol or underlying"
                aria-label="Filter ETF symbol or underlying"
                className="h-8 w-full rounded-md border border-border bg-background/50 pl-8 pr-3 text-xs outline-none focus:ring-1 focus:ring-primary"
              />
            </div>
            <select
              value={etfActionFilter}
              onChange={(event) => {
                setETFActionFilter(event.target.value);
                setETFPage(0);
              }}
              aria-label="Filter ETFs by action"
              className="h-8 rounded-md border border-border bg-background/50 px-2 text-xs outline-none focus:ring-1 focus:ring-primary"
            >
              <option value="ALL">All actions</option>
              {ETF_ACTION_FILTER_OPTIONS.map((action) => <option key={action} value={action}>{action}</option>)}
            </select>
            <span className="font-mono text-[10px] text-market-slate">
              {scoredETFCount} scored · {filteredETFResults.length} matches
            </span>
          </div>

          {(etfNotice || etfError) && (
            <p className={`mb-3 text-xs ${etfError ? "text-market-loss" : "text-market-slate"}`} role={etfError ? "alert" : "status"}>
              {etfError || etfNotice}
            </p>
          )}

          <div className="w-full rounded-md border border-border/70">
            <table className="w-full table-fixed border-collapse text-xs">
              <thead className="bg-card text-[10px] uppercase text-market-slate">
                <tr className="border-b border-border/80 font-mono">
                  <th className="w-[32%] px-1.5 py-2 text-left font-medium sm:px-3">Small-cap ETF / metrics</th>
                  <th className="w-[21%] px-1.5 py-2 text-right font-medium sm:px-3">Price / NAV / iNAV</th>
                  <th className="w-[14%] px-1.5 py-2 text-right font-medium sm:px-3" title="Overall risk out of 100. Lower is calmer.">Risk</th>
                  <th className="w-[13%] px-1.5 py-2 text-right font-medium sm:px-3" title="Annualized price swings and maximum peak-to-trough decline">Volatility / drawdown</th>
                  <th className="w-[9%] px-1.5 py-2 text-right font-medium sm:px-3" title="How closely it moves with the overall market. About 1 means in step.">Market link</th>
                  <th className="w-[11%] px-1.5 py-2 text-right font-medium sm:px-3">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/50">
                {paginatedETFResults.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="px-3 py-10 text-center text-market-slate">
                      {etfResults.length ? "                      No small-cap ETFs match these filters." : "Run an ETF analysis to load current NSE listings."}
                    </td>
                  </tr>
                ) : paginatedETFResults.map((item) => (
                  <tr key={item.symbol} className="hover:bg-foreground/2">
                    <td className="max-w-70 px-1.5 py-2.5 sm:px-3">
                      <button
                        type="button"
                        className="block w-full text-left"
                        onClick={() => setSelectedRec({ ...item, symbol: item.symbol, recommendation: asNumber(item.composite_score) == null ? undefined : item.recommendation ?? undefined, reasoning: item.reasoning ?? undefined })}
                        aria-label={`See why for ${item.symbol}`}
                      >
                        <div className="truncate font-semibold" title={item.symbol}>{item.symbol}</div>
                        <div className="truncate text-[10px] text-market-slate" title={item.underlying || undefined}>{item.underlying || "Underlying not supplied"}</div>
                        <div className="truncate text-[10px] text-foreground/80" title="Sortino, maximum drawdown, and tracking error">
                          Sortino {asNumber(item.sortino_1y) == null ? "—" : numberText(asNumber(item.sortino_1y))}
                          {" · Drawdown "}{asNumber(item.max_drawdown_1y_pct) == null ? "—" : `${numberText(asNumber(item.max_drawdown_1y_pct))}%`}
                          {" · Tracking error "}{asNumber(item.tracking_error_1y_pct) == null ? "—" : `${numberText(asNumber(item.tracking_error_1y_pct))}%`}
                        </div>
                      </button>
                    </td>
                    <td className="break-all px-1.5 py-2.5 text-right font-mono text-[9px] sm:px-3 sm:text-[10px]">
                      {item.last_price == null ? "—" : money(asNumber(item.last_price) ?? 0)}
                      <div className="text-market-slate">NAV {item.nav == null ? "—" : money(asNumber(item.nav) ?? 0)}</div>
                      <div className="text-market-slate">iNAV {item.inav == null ? "—" : money(asNumber(item.inav) ?? 0)}{asNumber(item.inav_premium_discount_pct) == null ? "" : ` (${numberText(asNumber(item.inav_premium_discount_pct))}%)`}</div>
                      <div className="text-market-slate">Turnover ratio {asNumber(item.portfolio_turnover_ratio) == null ? "—" : numberText(asNumber(item.portfolio_turnover_ratio))}</div>
                      <div className="text-market-slate">Yield − repo {asNumber(item.yield_repo_gap_pct) == null ? "—" : `${numberText(asNumber(item.yield_repo_gap_pct))}%`}</div>
                      <div className="text-market-slate">Yield / repo {asNumber(item.yield_pct) == null ? "—" : `${numberText(asNumber(item.yield_pct))}%`} / {asNumber(item.repo_rate_pct) == null ? "—" : `${numberText(asNumber(item.repo_rate_pct))}%`}</div>
                    </td>
                    <td className="break-all px-1.5 py-2.5 text-right sm:px-3">
                      {asNumber(item.composite_score) == null ? "—" : (
                        <span className="font-mono" title={item.risk_label || undefined}>
                          {numberText(asNumber(item.composite_score))} <span className="text-market-slate">/ 100</span>
                        </span>
                      )}
                    </td>
                    <td className="break-all px-1.5 py-2.5 text-right font-mono text-[9px] sm:px-3 sm:text-[10px]">
                      {asNumber(item.volatility_1y) == null ? "—" : `${numberText(asNumber(item.volatility_1y))}%`}
                      <div className="text-market-slate">{asNumber(item.max_drawdown_1y_pct) == null ? "Max —" : `Max ${numberText(asNumber(item.max_drawdown_1y_pct))}%`}</div>
                    </td>
                    <td className="break-all px-1.5 py-2.5 text-right font-mono text-[9px] sm:px-3 sm:text-[10px]">
                      {asNumber(item.beta_1y) == null ? "—" : numberText(asNumber(item.beta_1y))}
                      <div className="text-market-slate" title="Sortino ratio">{asNumber(item.sortino_1y) == null ? "Sortino —" : `Sortino ${numberText(asNumber(item.sortino_1y))}`}</div>
                      <div className="text-market-slate" title="Annualized tracking error versus Nifty 50">{asNumber(item.tracking_error_1y_pct) == null ? "TE —" : `TE ${numberText(asNumber(item.tracking_error_1y_pct))}%`}</div>
                      <div className="text-market-slate" title="Current portfolio Herfindahl concentration index">{asNumber(item.portfolio_hhi) == null ? "Portfolio HHI —" : `Portfolio HHI ${numberText(asNumber(item.portfolio_hhi))}`}</div>
                      <div className="text-market-slate" title="Current sector concentration index">{asNumber(item.sector_hhi) == null ? "Sector HHI —" : `Sector HHI ${numberText(asNumber(item.sector_hhi))}`}</div>
                    </td>
                    <td className="px-1.5 py-2.5 text-right sm:px-3">
                      <span
                        title={asNumber(item.composite_score) == null ? item.reasoning || "Not enough data to score this ETF." : undefined}
                        className={`inline-flex rounded px-1.5 py-0.5 font-mono text-[9px] ring-1 ${getRecommendationColor(asNumber(item.composite_score) == null ? undefined : item.recommendation || undefined)}`}
                      >
                        {asNumber(item.composite_score) == null ? "NOT RATED" : item.recommendation || "REVIEW"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {filteredETFResults.length > 0 && (
            <div className="mt-3 flex items-center justify-between border-t border-border/60 pt-2">
              <Button
                variant="ghost"
                size="sm"
                className="h-7 px-2 text-xs"
                disabled={visibleETFPage === 0}
                onClick={() => setETFPage((page) => Math.max(page - 1, 0))}
                aria-label="Previous ETF results page"
              >
                <ChevronLeft className="mr-1 size-3" /> Prev
              </Button>
              <span className="font-mono text-[10px] text-market-slate">
                {visibleETFPage + 1} / {totalETFPages}
              </span>
              <Button
                variant="ghost"
                size="sm"
                className="h-7 px-2 text-xs"
                disabled={visibleETFPage >= totalETFPages - 1}
                onClick={() => setETFPage((page) => Math.min(page + 1, totalETFPages - 1))}
                aria-label="Next ETF results page"
              >
                Next <ChevronRight className="ml-1 size-3" />
              </Button>
            </div>
          )}
          <details className="mt-3 border-t border-border/60 pt-2 text-[11px] text-market-slate">
            <summary className="flex cursor-pointer list-none items-center gap-1.5 font-medium text-foreground">
              <CircleHelp className="size-3.5 text-market-teal" aria-hidden="true" /> How actions are determined
            </summary>
            <div className="mt-2 space-y-1.5 leading-relaxed">
              <p><strong className="text-foreground">How we decide.</strong> For this small-cap test, the risk number equally combines price swings, market sensitivity, liquidity, Sortino, maximum drawdown, and tracking error. Portfolio concentration is shown separately as context; it does not change the ETF score. Yield and repo-rate comparison is shown only when NSE supplies both values, and is not scored for equity ETFs.</p>
              <p><strong className="text-foreground">BUY</strong> means the ETF fits your comfort level and adding a small amount would keep your overall investments about as safe, or safer. <strong className="text-foreground">WATCH</strong> means wait: it is a close call, hard to trade, or would not help your overall risk. <strong className="text-foreground">AVOID</strong> means it is much riskier than you said you are comfortable with.</p>
              <p>If the overall market looks gloomy, a BUY may be held back to WATCH. This is a screening aid, not personal investment advice.</p>
            </div>
          </details>
          <p className="mt-2 text-[10px] text-market-slate">
            Only equity ETFs with “Smallcap” or “Small Cap” in the underlying name are rated during this test.
          </p>
            </section>
          </TabsContent>
        </Tabs>

        {/* Reasoning Modal */}
        <Dialog open={!!selectedRec} onOpenChange={(open) => !open && setSelectedRec(null)}>
          <DialogContent className="border-border bg-popover text-popover-foreground sm:max-w-md max-h-[85vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-3">
                <span className="font-display text-xl">{selectedRec?.symbol || selectedRec?.stock_symbol}</span>
                <span className={`rounded px-2 py-0.5 font-mono text-xs ring-1 ${getRecommendationColor(selectedRec?.recommendation)}`}>
                  {selectedRec?.recommendation?.toUpperCase()}
                </span>
                {plainAction(selectedRec?.recommendation) && (
                  <span className="text-xs font-normal text-market-slate">{plainAction(selectedRec?.recommendation)}</span>
                )}
              </DialogTitle>
              <DialogDescription className="text-xs text-market-slate">
                Analysis generated {shortDate(selectedRec?.created_at) || "recently"}
              </DialogDescription>
            </DialogHeader>
            <div className="py-4">
              {selectedEngineNote && (
                <p className="mb-3 rounded-lg bg-foreground/5 px-3 py-2 text-xs text-market-slate ring-1 ring-border">
                  <span className="font-semibold text-foreground">{selectedEngineNote.label}: </span>
                  {selectedEngineNote.detail}
                </p>
              )}
              {selectedExplanation ? (
                <div className="space-y-3">
                  <p className="text-base font-semibold leading-snug text-foreground">{selectedExplanation.headline}</p>
                  {selectedExplanation.why && (
                    <div>
                      <p className="text-[11px] font-semibold uppercase tracking-wide text-market-slate">Why</p>
                      <p className="text-sm leading-relaxed text-foreground/90">{selectedExplanation.why}</p>
                    </div>
                  )}
                  {selectedExplanation.watch && (
                    <div>
                      <p className="text-[11px] font-semibold uppercase tracking-wide text-market-slate">What to keep in mind</p>
                      <p className="text-sm leading-relaxed text-foreground/90">{selectedExplanation.watch}</p>
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-sm leading-relaxed whitespace-pre-wrap text-market-slate">
                  {selectedRec?.reasoning || "No detailed reasoning provided."}
                </p>
              )}
            </div>
            <DialogFooter>
              <Button onClick={() => setSelectedRec(null)} variant="secondary" size="sm">Close</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Questionnaire Modal */}
        <Dialog open={questionnaireOpen} onOpenChange={setQuestionnaireOpen}>
          <DialogContent className="max-h-[88vh] overflow-y-auto border-border bg-popover text-popover-foreground sm:max-w-xl">
            <DialogHeader>
              <DialogTitle className="font-display text-xl">{hasProfile ? "Retake your risk questionnaire" : "Your investor risk profile"}</DialogTitle>
              <DialogDescription className="text-market-slate">Answer the 5 questions to calibrate recommendations.</DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-2">
              {QUESTIONS.map((question, index) => (
                <fieldset key={question.key}>
                  <legend className="mb-2 text-xs font-semibold">{index + 1}. {question.title}</legend>
                  <div className="grid grid-cols-5 gap-2">
                    {[1, 2, 3, 4, 5].map((value) => (
                      <Button
                        key={value}
                        type="button"
                        variant={answers[question.key] === value ? "default" : "outline"}
                        className="h-9 px-2 text-xs"
                        aria-pressed={answers[question.key] === value}
                        onClick={() => setAnswers((previous) => ({ ...previous, [question.key]: value }))}
                      >
                        {value}
                      </Button>
                    ))}
                  </div>
                  <div className="mt-1 flex justify-between gap-2 text-[9px] text-market-slate">
                    <span>{question.low}</span><span className="text-right">{question.high}</span>
                  </div>
                </fieldset>
              ))}
            </div>
            {questionnaireError && <p className="text-xs text-destructive" role="alert">{questionnaireError}</p>}
            <DialogFooter className="gap-2 sm:gap-2">
              <span className="mr-auto inline-flex items-center gap-1 text-xs text-market-slate"><CircleHelp className="size-3.5" /> {completedQuestions} / 5 answered</span>
              <Button variant="outline" size="sm" onClick={() => setQuestionnaireOpen(false)}>Cancel</Button>
              <Button size="sm" onClick={() => void submitQuestionnaire()} disabled={savingAnswers}>
                {savingAnswers ? <LoaderCircle className="size-3.5 animate-spin mr-1.5" /> : <Check className="size-3.5 mr-1.5" />}
                Save profile
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </main>
  );
}

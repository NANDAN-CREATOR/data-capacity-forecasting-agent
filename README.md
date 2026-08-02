# Data Capacity Forecasting Agent (Sample / Demo)

**Day 9** of the "Agentic AI in Data Engineering" series. Days 1-8 all
reasoned about the **present**: an incident that already happened, data
that already looks off, a change already proposed, a job already
running, a request already made. This agent reasons about the
**future**: given a resource's historical usage trend, it forecasts
when capacity will run out and recommends when to act.

It runs entirely on a **local model via [Ollama](https://ollama.com)** —
no cloud API key required.

> **This is a sample, not a production system.** The "production systems"
> it investigates (a usage-metrics history store, a capacity/quota
> registry, a planned-events calendar, a provisioning-process
> lead-time reference, a past capacity-incident archive) are replaced
> with small mock backends returning fixed, hand-crafted data across
> four illustrative scenarios. See
> [Adapting This to Real Systems](#adapting-this-to-real-systems) for
> what pointing this at a real environment would actually take.

> **This agent never provisions, reserves, or purchases anything.**
> Every terminal action is a recommendation for a human to review and
> action.

---

## Why This Agent Needs a Different Guardrail Than the Rest of the Series

Every prior agent reasoned about something that already happened or was
already true. This one has to reason about something that **hasn't
happened yet** — and that changes what "safe" means:

**The core risk is false precision about something inherently
uncertain.** A forecast is an extrapolation, not a fact. This agent is
required to:

- **Never state a forecast as a single certain date.** Trend
  extrapolation always has uncertainty — a forecast presented as a
  guarantee sets everyone up to be surprised when reality deviates from
  the trend line.
- **Separate steady-state growth from one-time events before
  extrapolating anything.** A completed backfill, a planned purge, or
  an unconfirmed migration can make a naive trend line wildly wrong.
- **Explicitly subtract provisioning lead time from the breach date.**
  The right moment to act isn't when capacity runs out — it's the
  forecasted breach date *minus* however long it actually takes to get
  more capacity approved and provisioned. Waiting until the breach date
  is imminent defeats the entire purpose of forecasting.
- **Escalate rather than guess** when a known future event is
  significant enough to swing the forecast either way, but its size or
  timing isn't confirmed yet.
- **Weigh the asymmetric cost of being wrong.** Running out of capacity
  is usually far worse than provisioning a bit early — especially for a
  resource with a documented history of past capacity incidents, which
  should pull the recommendation toward the earlier, more conservative
  side.

---

## What It Demonstrates

Four independent, runnable scenarios, each testing a different terminal
action and a different forecasting guardrail:

| Scenario | What happens | Correct terminal action |
|---|---|---|
| `clean_linear_trend_recommend_provisioning` | Steady, well-behaved growth, no confounding events | `recommend_provisioning_action`, with an honest range and lead-time-adjusted date |
| `temporary_spike_excluded_no_action` | Recent usage looks alarming (200 GB/day) — but it's a completed one-time backfill; real steady-state rate is ~5 GB/day | `no_action_needed`, based on the real rate, not the naive spike |
| `uncertain_major_migration_escalate` | A planned migration could range from "modest" to "doubling load," and isn't yet confirmed | `escalate_to_human`, rather than guessing a number for it |
| `urgent_short_runway_recommend_immediate_action` | Fast compounding growth, already at 91% capacity, a 5-week lead time, and a documented past incident that caused real data loss | `recommend_provisioning_action`, urgently — the lead time nearly exceeds the likely breach window |

The second and fourth scenarios are worth reading closely: the second
shows the agent correctly *ignoring* an alarming-looking trend once it's
explained by a one-time event; the fourth shows it correctly recognizing
that "recommend action for some future date" isn't good enough when the
math shows the lead time itself is nearly as long as the runway.

---

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) installed and running locally
- A tool-calling-capable model pulled in Ollama, for example:

  ```bash
  ollama pull llama3.1
  ```

  Other models known to support tool calling in Ollama at the time of
  writing include `qwen2.5`, `mistral-nemo`, and `firefunction-v2` — check
  the [Ollama model library](https://ollama.com/library) for current
  tool-calling support. Watch in particular whether a smaller model
  states a single confident date instead of a range, and whether it
  correctly subtracts lead time from the breach date rather than
  recommending action only once capacity is already nearly exhausted —
  those are the two most useful signals this repo's scenario design is
  built to surface.

---

## Setup

```bash
git clone https://github.com/NANDAN-CREATOR/data-capacity-forecasting-agent.git
cd data-capacity-forecasting-agent
pip install -r requirements.txt

# In a separate terminal:
ollama serve
ollama pull llama3.1
```

## Running the Demo

```bash
python agent.py --scenario clean_linear_trend_recommend_provisioning
python agent.py --scenario temporary_spike_excluded_no_action
python agent.py --scenario uncertain_major_migration_escalate
python agent.py --scenario urgent_short_runway_recommend_immediate_action
```

Or with a different model:

```bash
python agent.py --scenario urgent_short_runway_recommend_immediate_action --model qwen2.5
```

You'll see a step-by-step trace: which tool the model calls at each step,
what it returns, and finally one of the three terminal-action records.

**Note on output:** the exact number of steps and their order can vary
between runs and models. What should stay consistent is *which terminal
action* it reaches for each scenario, and — critically — whether
`confidence_range` is genuinely expressed as a range rather than a
point estimate, and whether `recommended_action_date` in the fourth
scenario reflects genuine urgency given how close the lead time is to
the likely breach window.

---

## Project Structure

```
data-capacity-forecasting-agent/
├── agent.py           # mock tools, four scenarios, schemas, system prompt, agent loop
├── requirements.txt
├── LICENSE
└── README.md
```

Kept as a single file for a sample project like this — see
[Adapting This to Real Systems](#adapting-this-to-real-systems) for how
you'd split it up for a real deployment.

---

## How the Guardrails Work

- **No point-estimate forecasts.** The system prompt forbids stating a
  forecast as a single certain date — `confidence_range` must express a
  range or explicit uncertainty every time.
- **One-time events excluded before extrapolation.** `get_known_future_events`
  is checked before treating recent usage as "the growth rate" — a
  completed backfill or planned purge must not be naively extrapolated.
- **Lead time subtracted, not added as an afterthought.** `get_provisioning_lead_time`
  is required before recommending any action date, and the recommended
  date must already account for it.
- **Escalation for materially uncertain future events.** If a known
  future event's size or timing isn't confirmed and could swing the
  forecast either way, the agent must escalate rather than assign it a
  guessed number.
- **Past-incident-aware conservatism.** `search_past_capacity_incidents`
  is checked, and a documented history of real harm from running out
  should shift the recommendation earlier, reflecting the asymmetric
  cost of the two kinds of mistakes.
- **Prompt-injection awareness.** The system prompt instructs the model
  to treat all gathered material as data to analyze, never as commands
  to follow.
- **Step limit.** A hard cap (`MAX_STEPS`, default 16) forces an
  escalation rather than an unbounded investigation.

---

## Adapting This to Real Systems

Only the **bodies** of these five read-only functions in `agent.py` need
to change to point this at a real environment:

| Function | Would call, in a real deployment |
|---|---|
| `get_historical_usage_trend` | Your monitoring/observability platform's usage metrics history |
| `get_current_capacity` | Your cloud provider's quota API or infrastructure-as-code config |
| `get_known_future_events` | A shared planning calendar, project roadmap, or migration-tracking system |
| `get_provisioning_lead_time` | Your procurement/budget-approval process documentation, or historical time-to-provision data |
| `search_past_capacity_incidents` | Your incident tracker, filtered to capacity-exhaustion causes |

The tool **schemas**, the **system prompt**, and the **agent loop** don't
need to change. You'd also want to, at minimum:

- Replace the qualitative trend descriptions in this demo with an
  actual statistical forecasting method (e.g. a fitted trend line with
  a genuine confidence interval, or a time-series forecasting model)
  once real historical data is available — this repo keeps the
  underlying math qualitative deliberately, to keep the mock data
  simple, but a real deployment should get more rigorous and precise
- Route recommendations to wherever your team actually reviews
  capacity-planning decisions (a FinOps/infra planning process), and
  always treat `recommended_action_date` as the start of a request
  process, not a guarantee capacity will land exactly on that day
- Build a scenario-based test suite exactly like this repo's four
  scenarios, but drawn from your own team's real resources, including
  ones with a real history of past capacity incidents, to calibrate how
  conservative your recommendations should actually be
- Decide, with whoever owns budget/procurement, how much safety margin
  beyond raw lead time is appropriate — real procurement timelines often
  have their own variance too, which this demo simplifies away

---

## License

MIT — see [LICENSE](LICENSE).

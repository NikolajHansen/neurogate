# Neurogate — Hybrid Neurosymbolic Inference Engine

## The Problem

Modern AI systems burn enormous compute on training and inference — even for inputs that follow deterministic, predictable rules. A large language model answering `if temperature > 100 then boiling` is wasteful by design. Expert systems solved this class of problem elegantly in the 1980s, but failed at scale because encoding an entire domain by hand is intractable and rule bases become unmanageable.

The insight: use a neural model to *synthesize* rules for predictable patterns, and fall back to the neural model only for genuinely ambiguous input.

---

## Core Idea

Route every inference through the cheapest capable path. The neural model (via Ollama) is the fallback, not the default.

```
Input
  │
  ▼
┌─────────────────────┐
│  Confidence Gate    │  ← lightweight classifier / voting ensemble
│  "Is this a known   │
│   rule domain?"     │
└──────┬──────────────┘
       │
   Yes (high confidence)        No (ambiguous / unknown)
       │                               │
       ▼                               ▼
┌─────────────┐               ┌───────────────────┐
│ Rule Engine │               │  Ollama (neural)  │
│ (declarative│               │  full inference   │
│  fast, O(1))│               └────────┬──────────┘
└─────────────┘                        │
                                       ▼
                               ┌───────────────────┐
                               │  Telemetry Log    │
                               │  input + output   │
                               └────────┬──────────┘
                                        │
                               ┌────────▼──────────┐
                               │  Pattern Cluster  │
                               │  + Stability Check│
                               └────────┬──────────┘
                                        │
                               N occurrences, >X% consistent
                                        │
                                        ▼
                               ┌───────────────────┐
                               │  Rule Candidate   │──► (human review) ──► Rule Engine
                               └───────────────────┘
```

---

## Architecture Layers

### Layer 1 — Rule Engine (Declarative)

Simple, deterministic rules. Hand-written or AI-synthesized. Evaluated locally with zero model invocation.

```
boiling(X)    :- temperature(X, T), T > 100.
fraud_risk(X) :- transaction(X, amount, A), A > 10000, country(X, C), high_risk_country(C).
syntax_error  :- token(T), unexpected_context(T).
```

- Zero training cost
- Fully auditable and explainable
- O(1) or O(log n) evaluation
- Can be replicated on every node in a cluster at near-zero cost

### Layer 2 — Confidence Gate (Router)

Before invoking the neural model, classify the input:

- **Statistical classifier** (naive bayes, decision tree, or small embedding model) — is this input within a known rule's domain?
- **Confidence threshold** — if confidence > configurable threshold (e.g. 0.95), apply matching rule directly
- **Voting ensemble** — multiple lightweight classifiers vote; majority wins; escalate to neural on split vote or low confidence

The gate itself is cheap enough to run on every inference node.

### Layer 3 — Neural Backend (Ollama)

Invoked only when:
- No rule matches
- Confidence gate fails or returns low confidence
- Input is genuinely ambiguous or novel

Ollama provides the model backend, supporting local deployment on CPU/GPU and ARM (e.g. Raspberry Pi clusters).

### Layer 4 — Rule Promotion Pipeline

The system gets cheaper over time as the rule base matures:

1. **Telemetry** — log all neural inferences: input, output, confidence, timestamp
2. **Clustering** — group similar inputs with similar outputs (embedding similarity)
3. **Stability check** — when a pattern appears N times (configurable) with >X% output consistency → mark as rule candidate
4. **Review gate** — human review before promotion (especially important in regulated domains: medicine, finance, legal)
5. **Promotion** — approved candidates become declarative rules in the rule engine

---

## Distribution Strategy (Pi Cluster / Kubernetes)

Designed with heterogeneous clusters in mind (e.g. 10× Raspberry Pi 4 8GB on Kubernetes):

| Component        | Deployment         | Why                                      |
|------------------|--------------------|------------------------------------------|
| Rule Engine      | Every node         | Stateless, cheap, eliminates network hop |
| Confidence Gate  | Every node         | Small model, fast local evaluation       |
| Ollama backend   | Sharded across nodes | Expensive, only hit on cache/rule miss  |
| Telemetry store  | Central or distributed | Write-heavy, read for clustering       |

Hot paths (high-confidence rule matches) never leave the local node. Neural invocations are rare and can tolerate cross-node latency.

### Avoiding Hotspots

Inspired by JIT hotspot compilation: profile neural inference to identify frequently-activated model layers/patterns for a given domain. Replicate those on multiple nodes. For bounded-domain workloads (code, SQL, medical records, financial transactions), input distributions are narrow enough that static hotspot approximation is effective.

---

## Why This Works Best for Bounded Domains

The more rule-bound and predictable the input distribution, the more effective the system:

| Domain              | Why it works                                      |
|---------------------|---------------------------------------------------|
| **Code completion** | Formal grammar, narrow vocabulary, repeated idioms |
| **SQL queries**     | Very constrained syntax                           |
| **Medical records** | Structured templates, repeated terminology        |
| **Financial transactions** | Limited vocabulary, predictable patterns  |
| **Log analysis**    | Highly repetitive formats                         |

General-purpose chat is the worst case. Code is the best case.

---

## Key Properties

- **Gets cheaper over time** — opposite of pure neural (fixed cost forever)
- **Explainable** — rule engine decisions are auditable; critical in regulated industries
- **Graceful degradation** — if rule engine has no match, neural handles it transparently
- **Domain-agnostic** — rules and classifiers are pluggable per domain
- **Cluster-friendly** — rule engine replication eliminates single points of failure for hot paths

---

## Metrics to Track

| Metric                  | Purpose                                        |
|-------------------------|------------------------------------------------|
| Rule hit rate           | Are rules covering enough traffic?             |
| Confidence gate accuracy | Is routing working correctly?                 |
| Neural invocation rate  | Should decrease over time as rules mature      |
| Rule conflict rate      | Knowledge base health indicator                |
| Promotion acceptance rate | Quality of AI-synthesized rule candidates   |
| p50/p99 inference latency | Rule path vs neural path comparison         |

---

## Ollama Integration

Ollama provides:
- Local model serving (no cloud dependency)
- REST API compatible with OpenAI spec
- ARM support (Pi4, Apple Silicon)
- Model hot-swap without service restart
- Streaming responses

Neurogate wraps Ollama as the neural backend, adding the routing, rule engine, telemetry, and promotion pipeline on top.

---

## Origins

This architecture emerged from a discussion about:
- The inefficiency of training large models for deterministic rules
- The failure modes of 1980s expert systems (hand-crafted, unscalable knowledge bases)
- The insight that AI can *synthesize* the rules that humans struggled to write manually
- JIT hotspot compilation as a model for identifying and replicating high-frequency inference paths
- Running inference on a Raspberry Pi Kubernetes cluster where compute is genuinely constrained

The name **Neurogate** reflects the core mechanic: a gate that routes between neural and symbolic reasoning, directing each inference to the appropriate layer.

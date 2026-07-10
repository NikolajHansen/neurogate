# Neurogate

Hybrid neurosymbolic inference engine. Routes inputs through a declarative rule engine first; falls back to a local neural model (via [Ollama](https://ollama.ai)) only when no rule applies. Automatically promotes stable neural inference patterns into new rules over time.

## Why

Large models are wasteful for deterministic, rule-bound inputs. A rule engine is O(1) and fully auditable. The goal is to combine the best of both: symbolic reasoning for predictable inputs, neural reasoning for everything else — with the system getting cheaper over time as the rule base grows.

## Status

Early concept / design phase. See [docs/idea.md](docs/idea.md) for the full architecture.

## License

MIT

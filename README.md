# Self–Other Asymmetry in LLM Epistemic Auditing

Apart Research Sprint — **AI × Epistemics** (13–15 November 2026).  
**Sprint note:** all code and stimuli developed during the sprint; no prior work.

## Research question

Are LLMs better at detecting epistemic errors in **third-party** answers than in **their own**, when the error text is identical?

## Setup

```bash
pip install -r requirements.txt
export DEEPSEEK_API_KEY=...
export OPENAI_BASE_URL=https://api.deepseek.com
python generate_stimuli.py && python experiment.py && python analyze.py
```

Never commit API keys. Prefer `DEEPSEEK_API_KEY`; also supports OpenAI / Groq / Anthropic / mock.

## Prompt templates (soft_v3 — current)

Matched structure; **only attribution differs**. No forced auditor/`DETECTO` stance.

### SELF

```
Question:
{pregunta}

Your answer:
"""{respuesta}"""

Is your answer correct?
First line: YES or NO
Second line: brief reason (≤40 words).
```

### OTHER

```
Question:
{pregunta}

Another model's answer:
"""{respuesta}"""

Is that answer correct?
First line: YES or NO
Second line: brief reason (≤40 words).
```

Metrics: YES → no detection (0); NO → detection (1). Parser in `experiment.py`.

## Stimuli

- `stimuli.json` — v2 harder planted errors (≥20)
- `stimuli_v1_easy.json` — easy backup
- `stimuli_controls.json` — 20 correct-answer FPR controls (matched categories; `es_correcta=true`)

FPR run example:

```bash
export STIMULI_JSON=stimuli_controls.json
export RESULTS_CSV=results_controls_deepseek.csv
export META_JSON=run_meta_controls_deepseek.json
export AUDIT_PROVIDER=deepseek
python experiment.py
```

## Current live status (lever 3 — other models)

| Provider | Model | Self | Other | Gap | Notes |
|----------|-------|-----:|------:|----:|-------|
| Groq | `allam-2-7b` | 0.05 | 0.05 | 0 | Live; `llama-3.1-8b-instant` unavailable on account |
| Ollama cloud | — | — | — | — | Chat **402** (no credits); models listable |
| DeepSeek (prior) | `deepseek-chat` | 1.00 | 1.00 | 0 | Soft_v3 ceiling |

Per-provider: `results_groq.csv`, `run_meta_groq.json`, `analysis_groq.png`, `run_meta_ollama.json`.

**Remaining levers:** FPR controls; multi-turn true self; Ollama after credits.

## Artifacts

`results.csv`, `analysis.png`, `stats_summary.txt`, `run_meta.json`, `paper.md`  
User summary: `../self-other-preliminary-results.md`

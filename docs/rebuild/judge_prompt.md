# WP6 judge prompt and protocol

Status: pilot executed 2026-11-11 on the `rebuild` branch. Full 900-row runs
are not analyzed in this document. It publishes the exact prompts, decoding
settings, seed handling, parsing rules, and artifact locations for the
LLM-judge component of WP6. The prompt is defined in
`src/models/rebuild_judge.py` (`SYSTEM_PROMPT`, `USER_TEMPLATE`); the text
below is reproduced verbatim from those constants.

Dataset: `ragtruth_qa_test`, 900 rows, prevalence 0.178 (160 hallucinated,
740 grounded). The judge sees the natural-prevalence sample, not a balanced
sample. Labels are RAGTruth response-level hallucination labels (1 =
hallucinated).

## Models and decoding

| Backend | Model | Serving | Output cap | Seed | Temperature |
|---|---|---|---|---|---|
| `openai` | `gpt-5.6-luna` (`OPENAI_MODEL`) | OpenAI chat completions API | `max_completion_tokens=512` (auto-retry up to 2,048) | `42`, `123`, `456` per repeat | not sent; the model rejects `temperature=0` |
| `qwen` | `qwen3.5:4b` Q4_K_M (`QWEN_MODEL`) | local Ollama `/api/chat` | `num_predict=300` | `42`, `123`, `456` per repeat | `0` |

OpenAI request settings:

- `response_format={"type": "json_object"}`. If the API rejects that parameter
  with a 400 error, the client retries the same request without it; parsing
  remains the same.
- `max_completion_tokens=512`, which covers hidden reasoning tokens and the
  JSON object. The pilot used 158 completion tokens per call on average, of
  which 106 were reasoning tokens.
- If the provider returns a 400 max-token error, the client retries with a
  doubled cap (up to 2,048) before giving up. The per-call record stores the
  cap that actually produced the response.
- `seed` is passed on every call. A seed is a request, not a guarantee: the
  model is a reasoning model and repeated calls can still differ. Run-to-run
  agreement is therefore measured, not assumed.
- No `temperature` parameter is sent. `gpt-5.6-luna` rejects
  `temperature=0`; omitting the parameter uses the provider default.

Qwen request settings:

- `stream=false`, `think=false` to disable the thinking channel.
- `options={"temperature": 0, "seed": <42|123|456>, "num_predict": 300,
  "num_ctx": 8192}`.
- The pilot produced identical labels and confidences for all three seeds.
  That is an empirical property of this local setup (temperature 0), not an
  assumption used anywhere in the metrics.

Local setup (only needed when Ollama is not already running):

```powershell
ollama serve
ollama pull qwen3.5:4b
```

The server URL defaults to `http://localhost:11434` and can be changed with
`HALU_QWEN_URL`. The model name defaults to `qwen3.5:4b` and can be changed
with `QWEN_MODEL` or `--model`. The installed tag is `qwen3.5:4b` with
`Q4_K_M` quantization (4.2B parameters, 262,144-token context). The runner
checks `/api/tags` before the first call and exits with the available model
list if the requested model is missing.

## Exact system prompt

Token count: 1,150 tokens with the `o200k_base` tokenizer (845 words).
SHA-256 of the UTF-8 string:
`5688460de11af09e053037b0eb1c4f8d4fcdabb6aa3cea10cf44d3c41244156e`.
The static prefix is identical for every row, seed, and backend, so it is
eligible for provider prefix caching (the pilot cache hit rate was 0.892).

```text
You are a factual-consistency judge for a research evaluation of hallucination detection. In each request you receive one QUESTION, one CONTEXT, and one ANSWER. Your task is to decide whether the ANSWER is grounded in the CONTEXT or hallucinated with respect to it.

DEFINITIONS
- A checkable claim is a factual assertion in the ANSWER that can be compared with the CONTEXT. Examples include statements about entities, events, numbers, dates, quantities, locations, causes, and relationships. Opinions, greetings, hedges, and procedural filler are not checkable claims.
- GROUNDED means every substantive checkable claim in the ANSWER is supported by at least one passage in the CONTEXT. Support can be a direct restatement, a paraphrase, or a combination of passages. The ANSWER does not need to reuse the wording of the CONTEXT.
- HALLUCINATED means at least one substantive checkable claim in the ANSWER is contradicted by the CONTEXT, or is not supported by any passage in the CONTEXT. A single unsupported or contradicted detail is enough, even when the rest of the ANSWER is correct.

DECISION PROCEDURE
1. Read the QUESTION to understand what the ANSWER is responding to. The QUESTION is not evidence. A claim that merely repeats the QUESTION is not automatically supported.
2. Read the CONTEXT carefully. It may contain several passages, repeated text, web boilerplate, or irrelevant material. Any single passage may support a claim.
3. List the checkable claims in the ANSWER internally. Do not show this list in your output.
4. For each claim, ask whether the CONTEXT entails it under a reasonable reading. Paraphrase and ordinary inference are allowed. World knowledge is not allowed.
5. If every checkable claim is supported, output grounded. If any checkable claim is contradicted or unsupported, output hallucinated.
6. If the ANSWER contains no checkable factual claim, output grounded.

RULES AND EDGE CASES
- Use only the CONTEXT. A claim that is true in the real world but absent from the CONTEXT is unsupported.
- Do not penalize the ANSWER for omitting information that is present in the CONTEXT. Completeness is not required.
- Do not reward fluency, length, detail, or confidence. A long, polished answer with one unsupported detail is hallucinated. A short, plain answer that is fully supported is grounded.
- Treat numbers, dates, names, locations, and quantities as substantive. If the ANSWER reports a different number, date, name, location, or quantity from the CONTEXT, treat the claim as contradicted unless the CONTEXT itself is ambiguous.
- Approximate quantities are supported when the CONTEXT states the same approximate quantity or a range that contains it. Rounding is acceptable when the rounded value is consistent with the CONTEXT.
- A claim that combines supported facts into a relationship the CONTEXT does not state is unsupported. For example, if the CONTEXT says A happened before B and C happened before D, do not infer that A happened before D.
- If the CONTEXT explicitly says that information is unknown, unavailable, or disputed, an ANSWER that asserts a definite value is unsupported.
- If two passages conflict, a claim supported by one of them is supported. Do not require the CONTEXT to be internally consistent.
- If the ANSWER contradicts itself, judge each substantive claim against the CONTEXT. A self-contradictory answer normally contains at least one unsupported claim.
- If the ANSWER is a refusal, a request for clarification, or a statement that the CONTEXT does not contain the answer, treat it as having no checkable claim and output grounded.
- If the ANSWER repeats the QUESTION without adding a factual assertion, output grounded.
- Do not evaluate grammar, style, helpfulness, safety, or whether the ANSWER follows the requested format.

WORKED EXAMPLES
- CONTEXT: "The bridge opened in 1937 and carries four lanes." ANSWER: "The bridge opened in 1937 and carries six lanes." Judgment: hallucinated, because the lane count contradicts the CONTEXT.
- CONTEXT: "The study enrolled 120 adults and reported a 12 percent reduction." ANSWER: "The study enrolled 120 adults." Judgment: grounded, because the stated claim is supported and the omission is not penalized.
- CONTEXT: "The plant operates in Ohio." ANSWER: "The plant operates in Ohio and is the largest in the country." Judgment: hallucinated, because the size claim is not supported even if it is true in the real world.

CONFIDENCE
Report confidence as your probability that your own judgment is correct, not the probability that the ANSWER is hallucinated. Use values near 0.5 when the CONTEXT is ambiguous or the claim is borderline, values near 0.7 when the judgment is plausible but debatable, and values above 0.9 only when the CONTEXT clearly supports or clearly contradicts the claim. Do not use 1.0.

OUTPUT
Return one JSON object and nothing else. Do not use markdown fences, preamble, or trailing commentary. The object must have exactly these keys:
{"judgment": "grounded" or "hallucinated", "confidence": <number between 0 and 1>, "reason": "<at most 25 words naming the deciding claim>"}
The reason must name the claim that decided the judgment. If the judgment is grounded, name the main supported claim. Keep the reason under 25 words.

```

## Exact user prompt

```text
QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}
```

`{question}`, `{context}`, and `{answer}` are the RAGTruth QA fields joined by
`sample_id` from `data/processed/unified_records.parquet`. The fields are
stripped of leading and trailing whitespace; no truncation is applied on the
API path. The pilot prompt averaged 1,622 tokens per call including the system
prompt, with a minimum of about 1,500 and a maximum under 2,000 tokens.

## Parsing rules

The parser in `parse_judge_response` applies these rules in order:

1. Strip whitespace and an optional Markdown code fence (```` ```json ```` or
   ```` ``` ````).
2. Take the substring from the first `{` to the last `}` and parse it as JSON.
3. Read `judgment` (also accepts `judgement`). Map any string containing
   `hallucinat`, `unsupported`, `not grounded`, `contradict`, or `unfaithful`
   to `hallucinated`. Map any string containing `grounded`, `supported`,
   `faithful`, `not hallucinated`, or `no hallucination` to `grounded`.
4. Read `confidence` as a float. A value greater than 2 and at most 100 is
   treated as a percentage and divided by 100. Values are clipped to
   `[0, 1]`. Missing or non-numeric values become 0.5.
5. Read `reason` (also accepts `reasoning`) and truncate it to 300
   characters. The reason is stored for audit only and is not used in any
   metric.
6. If no valid JSON object or valid judgment is found, fall back to keyword
   matching on the raw text: `hallucinated` without `grounded` means
   hallucinated, `grounded` means grounded, otherwise grounded at confidence
   0.5. Such records carry `parse_ok=false`; the pilot had zero parse
   failures on both backends.

The binary prediction is `pred = 1` for `judgment == "hallucinated"` and `0`
otherwise. The AUROC score is the model's confidence that the answer is
hallucinated:

```text
score = confidence                     if judgment == "hallucinated"
score = 1 - confidence                 if judgment == "grounded"
```

This conversion treats "grounded with confidence 0.95" as a hallucination
score of 0.05. It is the only place where the confidence semantics are
transformed; the prompt asks for confidence in the judgment, not in the
hallucination label.

## Metrics and uncertainty

- Accuracy, precision, recall, and F1 use the binary prediction at the
  model's own decision boundary (no threshold tuning).
- AUROC uses the converted confidence score.
- The predicted-positive rate is the flag rate, which is directly comparable
  to the 0.178 prevalence.
- 95% confidence intervals use 1,000 row-bootstrap resamples with
  `BOOTSTRAP_SEED=777`. Resamples with one class are skipped for AUROC.
- Run-to-run agreement uses pairwise exact match and Cohen's kappa for each
  pair of seeds, three-way exact agreement, and Fleiss' kappa across the
  three repeats.

## Token accounting and cost

The OpenAI path records `prompt_tokens`, `cached_tokens`, `completion_tokens`,
and `reasoning_tokens` from the API response for every call. Cost is computed
as:

```text
cost = (prompt_tokens - cached_tokens) / 1e6 * 0.20
     + cached_tokens / 1e6 * 0.02
     + completion_tokens / 1e6 * 1.20
```

The prices are assumptions: GPT-5.6-Luna input at $0.20 per million uncached
tokens and $0.02 per million cached tokens (AGENTS.md), output at $1.20 per
million tokens (the value already used by `src/models/eval_llm_judge.py`).
Override them with `HALU_JUDGE_INPUT_USD_PER_MTOK`,
`HALU_JUDGE_CACHED_INPUT_USD_PER_MTOK`, and
`HALU_JUDGE_OUTPUT_USD_PER_MTOK`. Local Qwen inference is recorded at $0 API
cost; electricity and hardware are excluded.

## Run commands

Pilot (30 rows, three repeats):

```powershell
& .venv\Scripts\python.exe src\models\rebuild_judge.py --backend openai --limit 30 --repeats 3 --concurrency 6 --out-tag pilot
& .venv\Scripts\python.exe src\models\rebuild_judge.py --backend qwen --limit 30 --repeats 3 --concurrency 1 --out-tag pilot
```

Full natural-prevalence run (900 rows, three repeats):

```powershell
& .venv\Scripts\python.exe src\models\rebuild_judge.py --backend openai --limit 0 --repeats 3 --concurrency 6 --out-tag full
& .venv\Scripts\python.exe src\models\rebuild_judge.py --backend qwen --limit 0 --repeats 3 --concurrency 1 --out-tag full
```

If the Qwen temperature-0 determinism observed in the pilot holds, a single
Qwen repeat is enough for point estimates:

```powershell
& .venv\Scripts\python.exe src\models\rebuild_judge.py --backend qwen --limit 0 --repeats 1 --concurrency 1 --out-tag full
```

Comparison rows and encoder timing:

```powershell
& .venv\Scripts\python.exe src\models\rebuild_judge.py --comparisons-only
& .venv\Scripts\python.exe src\models\rebuild_judge.py --time-encoder --encoder-rows 200
```

Outputs are written to `artifacts/results/rebuild/judge/`:

- `calls_<backend>_<tag>.jsonl`: one record per call, including tokens and
  latency; resumable.
- `predictions_<backend>_<tag>.csv`: parsed predictions for completed calls.
- `metrics_<backend>_<tag>.json`: per-seed metrics, CIs, agreement, tokens,
  cost, latency, and errors.
- `comparison_rows.csv` and `comparison_rows.json`: EC-XGB (m3) and
  length-only rows on the 900 test rows.
- `encoder_timing.json`: RoBERTa-base timing.

## Scope

This is an LLM-as-judge evaluation, not a human evaluation. No annotator
review, inter-annotator agreement with humans, or adjudication is claimed.
Pilot numbers (30 rows, 5 positives) are unstable and are not final; only the
`--limit 0` run produces the WP6 table.

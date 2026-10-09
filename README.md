# agentos

A multi-agent assistant that carries out operating-system tasks on Linux using open-weight chat models
(Qwen, Hermes) served behind any OpenAI-compatible endpoint: llama.cpp `llama-server`, vLLM or Ollama.
Written in Rust.

```
user task
   │
   ▼
┌─────────┐  JSON plan   ┌──────────┐  tool calls   ┌───────────────┐
│ planner │─────────────▶│ executor │──────────────▶│ tools + policy │
└─────────┘              └──────────┘◀──────────────└───────────────┘
                              │  step reports           observations
                              ▼
                         ┌────────┐  retry + feedback (bounded)
                         │ critic │────────────────────────▶ executor
                         └────────┘
                              │ pass
                              ▼
                        final answer
```

## How the work is split

| Role | Job | Tools |
|------|-----|-------|
| planner | Turns the goal into at most N short, checkable steps (JSON). Falls back to a single step if the model's output is not parseable. | none |
| executor | Carries out one step at a time in a tool loop (`run_shell`, `read_file`, `write_file`, `list_dir`, `system_info`). Ends a step with a `RESULT:` message. Bounded by `--max-tool-rounds`. | all |
| critic | Reads the goal, the step reports and the list of tool calls actually made, and returns `pass` or `retry` plus the final answer. A `retry` triggers a bounded repair pass (`--max-replans`, default 1). | none |

Each role has its own model and, optionally, its own server, so the planner and critic can run on one model and the executor on
another, or on a different GPU:

```bash
agentos run "find what is using most disk space under /var and summarise it" \
  --planner-model qwen2.5-7b-instruct --planner-url http://gpu0:8080/v1 \
  --executor-model hermes-3-llama-3.1-8b --executor-url http://gpu1:8080/v1 \
  --critic-model qwen2.5-7b-instruct --critic-url http://gpu0:8080/v1 -v
```

## Safety model

Every shell command is classified before it runs (`src/policy.rs`):

* **read-only** (`ls`, `df`, `ps`, `grep`, `git status`, pipelines of those) runs immediately;
* **needs approval** (anything not provably read-only, redirects, `$(...)`, `find -delete`, package installs, ...) prompts on stdin;
* **denied** (`rm -rf /`, `mkfs`, `dd` to a device, fork bombs, `curl | sh`, power control) is never run, even with `--approve-all`.

`write_file` always asks and cannot leave the working directory (`..` and symlink escapes are rejected). Commands run in their own
process group with a timeout, and the whole group is killed on expiry. Output returned to the model is truncated.

This is a guard rail against a model making a mistake, not a security boundary against a hostile one. Run it in a container or VM
if the task or the model is not trusted.

## Context management

Each executor conversation is kept under `--context-budget` estimated tokens (about four characters per token). When it goes
over, `src/context.rs` first cuts old tool outputs to a head and tail snippet, then replaces the oldest whole turns with a one-line
note. The system prompt, the first user message and the most recent turns are always kept, and a tool result is never separated
from the assistant message that requested it. Qwen3-style `<think>` blocks are stripped from history.

## Tool-call formats

Both are handled: native `tool_calls` from the server, and the `<tool_call>{"name": ..., "arguments": ...}</tool_call>` text that
Hermes and Qwen chat templates produce when the runtime does not translate it.

## Serving a model

```bash
# llama.cpp (--jinja enables the model's tool-call template)
llama-server -m Qwen2.5-7B-Instruct-Q4_K_M.gguf --jinja -c 8192 -ngl 99 --port 8080

# vLLM
vllm serve Qwen/Qwen2.5-7B-Instruct --enable-auto-tool-choice --tool-call-parser hermes

# Ollama (base URL http://localhost:11434/v1, model name qwen2.5:7b)
```

## Build and run

```bash
cargo build --release
export AGENTOS_BASE_URL=http://localhost:8080/v1
export AGENTOS_MODEL=qwen2.5-7b-instruct
./target/release/agentos run "report the 5 largest files in this directory" --workdir . -v
./target/release/agentos tools        # print the tool schemas sent to the model
```

## Measuring serving speed

```bash
./target/release/agentos bench --runs 5 --max-tokens 256
```

Streams real requests (after one warm-up) and reports time-to-first-token and decode tokens/second per run and as a median.
Decode rate excludes prefill. Token counts come from the server's `usage` block when it sends one; otherwise stream chunks are
counted and the output says so. The numbers describe whatever server and hardware you point it at, so record the runtime, GPU,
quantization and context length alongside them.

## Tests

```bash
cargo test
```

43 tests, none of which need a model. The end-to-end tests (`tests/e2e.rs`) run the whole planner, executor and critic loop against
a scripted in-process OpenAI-compatible server and check what the executor was actually sent: real tool output fed back,
approval and denial, the Hermes inline format, a bounded repair pass, the tool-round limit, per-role servers, context compaction
that keeps tool/assistant pairs valid, HTTP error surfacing, and streaming bench statistics.

What this does **not** test is model quality: how well a given Qwen or Hermes checkpoint follows the planner and executor prompts
has to be checked against a real server.

## Limitations

* Unix only (`sh`, process groups, `/proc`).
* The token estimate is a heuristic, not the model's tokenizer.
* Tool calls are not streamed; only `bench` uses streaming.
* The shell policy is a conservative allowlist, not a parser.

use agentos::agents::Orchestrator;
use agentos::bench::{run_bench, DEFAULT_PROMPT};
use agentos::config::{Endpoint, Settings};
use agentos::llm::ChatClient;
use agentos::tools::{self, FixedApprover, StdinApprover, ToolContext};
use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "agentos",
    version,
    about = "Multi-agent OS assistant for open-weight models"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Args, Clone)]
struct Server {
    /// OpenAI-compatible base URL (llama.cpp: :8080/v1, vLLM: :8000/v1, Ollama: :11434/v1).
    #[arg(
        long,
        env = "AGENTOS_BASE_URL",
        default_value = "http://localhost:8080/v1"
    )]
    base_url: String,
    #[arg(long, env = "AGENTOS_API_KEY", hide_env_values = true)]
    api_key: Option<String>,
}

#[derive(Subcommand)]
#[allow(clippy::large_enum_variant)]
enum Command {
    /// Plan, execute and review a task.
    Run {
        task: String,
        #[command(flatten)]
        server: Server,
        /// Model for all roles unless a role-specific model is given.
        #[arg(long, env = "AGENTOS_MODEL", default_value = "qwen2.5-7b-instruct")]
        model: String,
        #[arg(long)]
        planner_model: Option<String>,
        #[arg(long)]
        executor_model: Option<String>,
        #[arg(long)]
        critic_model: Option<String>,
        /// Separate server for a role, e.g. the executor on a different GPU.
        #[arg(long)]
        planner_url: Option<String>,
        #[arg(long)]
        executor_url: Option<String>,
        #[arg(long)]
        critic_url: Option<String>,
        #[arg(long, default_value = ".")]
        workdir: PathBuf,
        /// Approve every state-changing action without asking. Hard-denied commands are still blocked.
        #[arg(long)]
        approve_all: bool,
        #[arg(long, default_value_t = 6)]
        max_steps: usize,
        #[arg(long, default_value_t = 8)]
        max_tool_rounds: usize,
        #[arg(long, default_value_t = 1)]
        max_replans: usize,
        /// Estimated-token budget for one executor conversation before it is compacted.
        #[arg(long, default_value_t = 6000)]
        context_budget: usize,
        #[arg(long, default_value_t = 0.2)]
        temperature: f32,
        #[arg(long, default_value_t = 1024)]
        max_tokens: u32,
        #[arg(long, default_value_t = 30)]
        shell_timeout_secs: u64,
        #[arg(short, long)]
        verbose: bool,
    },
    /// Measure time-to-first-token and decode tokens/second against a running server.
    Bench {
        #[command(flatten)]
        server: Server,
        #[arg(long, env = "AGENTOS_MODEL", default_value = "qwen2.5-7b-instruct")]
        model: String,
        #[arg(long, default_value_t = 5)]
        runs: usize,
        #[arg(long, default_value_t = 256)]
        max_tokens: u32,
        #[arg(long, default_value = DEFAULT_PROMPT)]
        prompt: String,
    },
    /// Print the tool schemas sent to the model.
    Tools,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Tools => {
            println!("{}", serde_json::to_string_pretty(&tools::specs())?);
            Ok(())
        }
        Command::Bench {
            server,
            model,
            runs,
            max_tokens,
            prompt,
        } => {
            let client = ChatClient::new(&server.base_url, server.api_key);
            println!("server: {}\nmodel:  {model}\nruns:   {runs} (after 1 warm-up), max_tokens {max_tokens}\n", server.base_url);
            let summary = run_bench(&client, &model, &prompt, runs, max_tokens, |i, s| {
                println!(
                    "run {i}: ttft {:>7.1} ms | decode {:>7.2} tok/s | {} prompt tok, {} completion tok",
                    s.ttft.as_secs_f64() * 1000.0,
                    s.decode_tps(),
                    s.prompt_tokens,
                    s.completion_tokens
                );
            })
            .context("benchmark failed")?;
            println!(
                "\nmedian: ttft {:.1} ms | decode {:.2} tok/s",
                summary.ttft_ms_median(),
                summary.decode_tps_median()
            );
            if !summary.all_counts_from_server() {
                println!("note: the server sent no usage block, so tokens were counted from stream chunks (approximate).");
            }
            Ok(())
        }
        Command::Run {
            task,
            server,
            model,
            planner_model,
            executor_model,
            critic_model,
            planner_url,
            executor_url,
            critic_url,
            workdir,
            approve_all,
            max_steps,
            max_tool_rounds,
            max_replans,
            context_budget,
            temperature,
            max_tokens,
            shell_timeout_secs,
            verbose,
        } => {
            let workdir = workdir.canonicalize().context("workdir does not exist")?;
            let endpoint = |url: Option<String>, m: Option<String>| Endpoint {
                base_url: url
                    .unwrap_or_else(|| server.base_url.clone())
                    .trim_end_matches('/')
                    .to_string(),
                api_key: server.api_key.clone(),
                model: m.unwrap_or_else(|| model.clone()),
            };
            let mut settings = Settings::single(&server.base_url, &model, workdir.clone());
            settings.planner = endpoint(planner_url, planner_model);
            settings.executor = endpoint(executor_url, executor_model);
            settings.critic = endpoint(critic_url, critic_model);
            settings.max_plan_steps = max_steps;
            settings.max_tool_rounds = max_tool_rounds;
            settings.max_replans = max_replans;
            settings.context_budget_tokens = context_budget;
            settings.temperature = temperature;
            settings.max_tokens = max_tokens;
            settings.shell_timeout = std::time::Duration::from_secs(shell_timeout_secs);
            settings.verbose = verbose;

            let approver: Box<dyn tools::Approver> = if approve_all {
                Box::new(FixedApprover(true))
            } else {
                Box::new(StdinApprover)
            };
            let ctx = ToolContext {
                workdir,
                approver,
                shell_timeout: settings.shell_timeout,
                max_output_bytes: settings.max_tool_output_bytes,
            };
            let mut orchestrator = Orchestrator::new(settings, ctx);
            let report = orchestrator.run(&task)?;

            println!("\n{}", report.final_answer);
            eprintln!(
                "\n[{} LLM calls, {} tool calls, {} prompt + {} completion tokens, reviewer: {}, repair passes: {}]",
                report.llm_calls, report.tool_calls, report.prompt_tokens, report.completion_tokens,
                report.critic_verdict, report.replans
            );
            Ok(())
        }
    }
}

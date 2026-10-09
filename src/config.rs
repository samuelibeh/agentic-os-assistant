use std::path::PathBuf;
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct Endpoint {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
}

#[derive(Clone, Debug)]
pub struct Settings {
    pub planner: Endpoint,
    pub executor: Endpoint,
    pub critic: Endpoint,
    pub workdir: PathBuf,
    pub temperature: f32,
    pub max_tokens: u32,
    pub max_plan_steps: usize,
    pub max_tool_rounds: usize,
    pub max_replans: usize,
    pub context_budget_tokens: usize,
    pub shell_timeout: Duration,
    pub max_tool_output_bytes: usize,
    pub verbose: bool,
}

impl Settings {
    /// One endpoint and one model for all three roles.
    pub fn single(base_url: &str, model: &str, workdir: PathBuf) -> Self {
        let ep = Endpoint {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: None,
            model: model.to_string(),
        };
        Settings {
            planner: ep.clone(),
            executor: ep.clone(),
            critic: ep,
            workdir,
            temperature: 0.2,
            max_tokens: 1024,
            max_plan_steps: 6,
            max_tool_rounds: 8,
            max_replans: 1,
            context_budget_tokens: 6000,
            shell_timeout: Duration::from_secs(30),
            max_tool_output_bytes: 8 * 1024,
            verbose: false,
        }
    }
}

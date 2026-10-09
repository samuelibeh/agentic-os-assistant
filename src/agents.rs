use crate::config::{Endpoint, Settings};
use crate::context::{compact, shorten};
use crate::llm::{strip_think, ChatClient, Message, Reply};
use crate::tools::{self, ToolContext};
use anyhow::Result;
use serde_json::Value;

const PLANNER_PROMPT: &str = "You are the planner of a Linux operations assistant. Break the user's \
goal into at most {max} short, concrete steps that an operator with a shell can carry out and check. \
Prefer read-only inspection before any change. Reply with JSON only, no prose: {\"steps\": [\"...\", \"...\"]}";

const EXECUTOR_PROMPT: &str = "You are the executor of a Linux operations assistant working in {workdir}. \
You are given one step at a time. Use the tools to carry it out. Rules: inspect before you change anything; \
make one small change at a time; if a command fails, read the error and adjust; never guess file contents or \
command output. When the step is done, reply with plain text beginning with `RESULT:` and the facts you \
established. Do not call a tool in that final message.";

const CRITIC_PROMPT: &str = "You are the reviewer of a Linux operations assistant. You see the user's goal, \
the plan, what each step reported, and the tool calls that were made. Decide whether the goal is met by the \
evidence shown (not by what was merely claimed). Reply with JSON only: \
{\"verdict\": \"pass\" or \"retry\", \"feedback\": \"what is missing or wrong, empty when pass\", \
\"summary\": \"the final answer for the user, based on the evidence\"}";

#[derive(Clone, Copy, Debug)]
enum RoleKind {
    Planner,
    Executor,
    Critic,
}

#[derive(Debug, Default, Clone)]
pub struct RunReport {
    pub plan: Vec<String>,
    pub step_results: Vec<(String, String)>,
    pub critic_verdict: String,
    pub replans: usize,
    pub final_answer: String,
    pub llm_calls: usize,
    pub tool_calls: usize,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

pub struct Orchestrator {
    settings: Settings,
    pub tools: ToolContext,
    stats: RunReport,
    actions: Vec<String>,
}

impl Orchestrator {
    pub fn new(settings: Settings, tools: ToolContext) -> Self {
        Orchestrator {
            settings,
            tools,
            stats: RunReport::default(),
            actions: Vec::new(),
        }
    }

    fn log(&self, line: &str) {
        if self.settings.verbose {
            eprintln!("{line}");
        }
    }

    fn endpoint(&self, kind: RoleKind) -> &Endpoint {
        match kind {
            RoleKind::Planner => &self.settings.planner,
            RoleKind::Executor => &self.settings.executor,
            RoleKind::Critic => &self.settings.critic,
        }
    }

    fn ask(&mut self, kind: RoleKind, msgs: &[Message], tools: Option<&[Value]>) -> Result<Reply> {
        let ep = self.endpoint(kind).clone();
        let client = ChatClient::new(&ep.base_url, ep.api_key.clone());
        let reply = client.chat(
            &ep.model,
            msgs,
            tools,
            self.settings.temperature,
            self.settings.max_tokens,
        )?;
        self.stats.llm_calls += 1;
        self.stats.prompt_tokens += reply.usage.prompt_tokens;
        self.stats.completion_tokens += reply.usage.completion_tokens;
        Ok(reply)
    }

    pub fn run(&mut self, goal: &str) -> Result<RunReport> {
        self.stats = RunReport::default();
        self.actions.clear();

        let plan = self.plan(goal)?;
        self.log(&format!(
            "[planner] {} step(s): {}",
            plan.len(),
            plan.join(" | ")
        ));
        self.stats.plan = plan.clone();

        let mut done: Vec<(String, String)> = Vec::new();
        for (i, step) in plan.iter().enumerate() {
            self.log(&format!("[executor] step {}: {step}", i + 1));
            let result = self.run_step(goal, &done, step)?;
            self.log(&format!(
                "[executor] step {} result: {}",
                i + 1,
                shorten(&result, 300)
            ));
            done.push((step.clone(), result));
        }

        let mut verdict = self.critique(goal, &plan, &done)?;
        while verdict.verdict == "retry" && self.stats.replans < self.settings.max_replans {
            self.stats.replans += 1;
            self.log(&format!("[critic] retry requested: {}", verdict.feedback));
            let step = format!("Repair pass. The reviewer said: {}", verdict.feedback);
            let result = self.run_step(goal, &done, &step)?;
            done.push((step, result));
            verdict = self.critique(goal, &plan, &done)?;
        }
        self.log(&format!("[critic] verdict: {}", verdict.verdict));

        let mut report = self.stats.clone();
        report.final_answer = if verdict.summary.trim().is_empty() {
            done.last().map(|(_, r)| r.clone()).unwrap_or_default()
        } else {
            verdict.summary
        };
        report.critic_verdict = verdict.verdict;
        report.step_results = done;
        Ok(report)
    }

    fn plan(&mut self, goal: &str) -> Result<Vec<String>> {
        let system = PLANNER_PROMPT.replace("{max}", &self.settings.max_plan_steps.to_string());
        let msgs = [Message::system(system), Message::user(goal)];
        let reply = self.ask(RoleKind::Planner, &msgs, None)?;
        Ok(parse_plan(
            &reply.message.content,
            goal,
            self.settings.max_plan_steps,
        ))
    }

    fn run_step(&mut self, goal: &str, done: &[(String, String)], step: &str) -> Result<String> {
        let system =
            EXECUTOR_PROMPT.replace("{workdir}", &self.settings.workdir.display().to_string());
        let mut brief = format!("Overall goal: {goal}\n");
        if !done.is_empty() {
            brief.push_str("Completed so far:\n");
            for (i, (s, r)) in done.iter().enumerate() {
                brief.push_str(&format!("{}. {} -> {}\n", i + 1, s, shorten(r, 500)));
            }
        }
        brief.push_str(&format!("Current step: {step}"));
        let mut msgs = vec![Message::system(system), Message::user(brief)];
        let specs = tools::specs();

        for _ in 0..self.settings.max_tool_rounds {
            let report = compact(&mut msgs, self.settings.context_budget_tokens);
            if report.truncated_outputs + report.dropped_messages > 0 {
                self.log(&format!(
                    "[context] truncated {} outputs, dropped {} messages",
                    report.truncated_outputs, report.dropped_messages
                ));
            }
            let reply = self.ask(RoleKind::Executor, &msgs, Some(&specs))?;
            let mut assistant = reply.message;
            assistant.content = strip_think(&assistant.content);
            if assistant.tool_calls.is_empty() {
                return Ok(assistant.content);
            }
            msgs.push(assistant.clone());
            for call in &assistant.tool_calls {
                let (name, args) = (&call.function.name, &call.function.arguments);
                self.log(&format!("[tool] {name} {}", shorten(args, 160)));
                let output = tools::execute(&mut self.tools, name, args);
                self.stats.tool_calls += 1;
                self.actions.push(format!("{name} {}", shorten(args, 160)));
                msgs.push(Message::tool(call.id.clone(), name.clone(), output));
            }
        }
        Ok("incomplete: the tool-round limit was reached before the step finished".into())
    }

    fn critique(
        &mut self,
        goal: &str,
        plan: &[String],
        done: &[(String, String)],
    ) -> Result<CriticReply> {
        let mut text = format!("Goal: {goal}\n\nPlan:\n");
        for (i, s) in plan.iter().enumerate() {
            text.push_str(&format!("{}. {s}\n", i + 1));
        }
        text.push_str("\nStep reports:\n");
        for (i, (s, r)) in done.iter().enumerate() {
            text.push_str(&format!("{}. {s}\n   report: {}\n", i + 1, shorten(r, 700)));
        }
        text.push_str("\nTool calls made:\n");
        for a in self.actions.iter().rev().take(20).rev() {
            text.push_str(&format!("- {a}\n"));
        }
        let msgs = [Message::system(CRITIC_PROMPT), Message::user(text)];
        let reply = self.ask(RoleKind::Critic, &msgs, None)?;
        Ok(parse_critic(&reply.message.content))
    }
}

#[derive(Debug, PartialEq)]
pub struct CriticReply {
    pub verdict: String,
    pub feedback: String,
    pub summary: String,
}

/// Extracts the first JSON object from model output, tolerating code fences and surrounding prose.
pub fn extract_json(text: &str) -> Option<Value> {
    let text = strip_think(text);
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    serde_json::from_str(text.get(start..=end)?).ok()
}

pub fn parse_plan(text: &str, goal: &str, max_steps: usize) -> Vec<String> {
    let steps: Vec<String> = extract_json(text)
        .and_then(|v| v["steps"].as_array().cloned())
        .map(|arr| {
            arr.iter()
                .filter_map(|s| match s {
                    Value::String(s) => Some(s.trim().to_string()),
                    Value::Object(o) => ["step", "goal", "description", "task"]
                        .iter()
                        .find_map(|k| o.get(*k).and_then(Value::as_str))
                        .map(|s| s.trim().to_string()),
                    _ => None,
                })
                .filter(|s| !s.is_empty())
                .take(max_steps)
                .collect()
        })
        .unwrap_or_default();
    if steps.is_empty() {
        vec![goal.to_string()]
    } else {
        steps
    }
}

pub fn parse_critic(text: &str) -> CriticReply {
    match extract_json(text) {
        Some(v) => {
            let verdict = match v["verdict"].as_str().map(|s| s.trim().to_lowercase()) {
                Some(s) if s == "retry" || s == "fail" => "retry".to_string(),
                Some(s) if s == "pass" => "pass".to_string(),
                _ => "unparsed".to_string(),
            };
            CriticReply {
                verdict,
                feedback: v["feedback"].as_str().unwrap_or("").to_string(),
                summary: v["summary"].as_str().unwrap_or("").to_string(),
            }
        }
        None => CriticReply {
            verdict: "unparsed".into(),
            feedback: String::new(),
            summary: String::new(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_parses_fenced_json_and_objects() {
        let text = "Sure!\n```json\n{\"steps\": [\"check disk\", {\"step\": \"find big files\"}, 7, \"\"]}\n```";
        assert_eq!(
            parse_plan(text, "g", 6),
            vec!["check disk", "find big files"]
        );
    }

    #[test]
    fn plan_is_capped_and_falls_back_to_the_goal() {
        assert_eq!(
            parse_plan("{\"steps\":[\"a\",\"b\",\"c\"]}", "g", 2),
            vec!["a", "b"]
        );
        assert_eq!(
            parse_plan("no json here", "do the thing", 6),
            vec!["do the thing"]
        );
        assert_eq!(
            parse_plan("{\"steps\":[]}", "do the thing", 6),
            vec!["do the thing"]
        );
    }

    #[test]
    fn critic_verdicts() {
        assert_eq!(
            parse_critic("{\"verdict\":\"pass\",\"summary\":\"ok\"}").verdict,
            "pass"
        );
        assert_eq!(
            parse_critic("<think>x</think>{\"verdict\":\"Retry\",\"feedback\":\"f\"}").verdict,
            "retry"
        );
        assert_eq!(parse_critic("looks fine to me").verdict, "unparsed");
    }
}

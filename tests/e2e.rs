mod common;

use agentos::agents::Orchestrator;
use agentos::bench::run_bench;
use agentos::config::Settings;
use agentos::llm::ChatClient;
use agentos::tools::{FixedApprover, ToolContext};
use common::*;
use serde_json::{json, Value};
use std::fs;
use std::path::Path;
use std::time::Duration;

fn orchestrator(url: &str, dir: &Path, approve: bool) -> Orchestrator {
    let mut s = Settings::single(url, "planner-model", dir.to_path_buf());
    s.executor.model = "executor-model".into();
    s.critic.model = "critic-model".into();
    let ctx = ToolContext {
        workdir: dir.to_path_buf(),
        approver: Box::new(FixedApprover(approve)),
        shell_timeout: Duration::from_secs(5),
        max_output_bytes: 4096,
    };
    Orchestrator::new(s, ctx)
}

fn model(req: &Value) -> &str {
    req["model"].as_str().unwrap()
}

#[test]
fn plan_execute_review_with_real_tool_output() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("a.txt"), "").unwrap();
    fs::write(dir.path().join("b.txt"), "").unwrap();

    let server = spawn(|_, req| match model(req) {
        "planner-model" => {
            text_reply(r#"{"steps":["Count the .txt files in the working directory"]}"#)
        }
        "executor-model" if last_role(req) == "tool" => {
            text_reply("RESULT: there are 2 .txt files")
        }
        "executor-model" => {
            tool_call_reply("c1", "run_shell", json!({"command": "ls *.txt | wc -l"}))
        }
        "critic-model" => text_reply(
            r#"{"verdict":"pass","feedback":"","summary":"The directory holds 2 .txt files."}"#,
        ),
        other => panic!("unexpected model {other}"),
    });

    let report = orchestrator(&server.base_url, dir.path(), false)
        .run("how many txt files are here?")
        .unwrap();

    assert_eq!(report.plan.len(), 1);
    assert_eq!(report.critic_verdict, "pass");
    assert_eq!(report.final_answer, "The directory holds 2 .txt files.");
    assert_eq!(report.tool_calls, 1);
    assert_eq!(report.llm_calls, 4);
    assert_eq!(report.prompt_tokens, 20 + 30 + 20 + 20);

    let reqs = server.requests.lock().unwrap();
    let second_exec = reqs
        .iter()
        .filter(|r| model(r) == "executor-model")
        .nth(1)
        .unwrap();
    let tool_msg = second_exec["messages"].as_array().unwrap().last().unwrap();
    assert_eq!(tool_msg["role"], "tool");
    assert_eq!(tool_msg["tool_call_id"], "c1");
    assert!(
        tool_msg["content"]
            .as_str()
            .unwrap()
            .contains("[exit 0]\n2"),
        "{tool_msg}"
    );
    assert!(
        second_exec["tools"].as_array().unwrap().len() >= 5,
        "tool schemas must be sent"
    );
}

#[test]
fn each_role_can_use_a_different_server_and_model() {
    let dir = tempfile::tempdir().unwrap();
    let planner = spawn(|_, _| text_reply(r#"{"steps":["do it"]}"#));
    let executor = spawn(|_, _| text_reply("RESULT: done"));
    let critic = spawn(|_, _| text_reply(r#"{"verdict":"pass","summary":"fine"}"#));

    let mut s = Settings::single(&planner.base_url, "qwen", dir.path().to_path_buf());
    s.executor.base_url = executor.base_url.clone();
    s.executor.model = "hermes".into();
    s.critic.base_url = critic.base_url.clone();
    let mut o = Orchestrator::new(
        s,
        ToolContext {
            workdir: dir.path().to_path_buf(),
            approver: Box::new(FixedApprover(false)),
            shell_timeout: Duration::from_secs(5),
            max_output_bytes: 4096,
        },
    );
    let report = o.run("goal").unwrap();
    assert_eq!(report.final_answer, "fine");
    assert_eq!(planner.requests.lock().unwrap().len(), 1);
    assert_eq!(executor.requests.lock().unwrap()[0]["model"], "hermes");
    assert_eq!(critic.requests.lock().unwrap().len(), 1);
}

#[test]
fn hermes_inline_tool_call_text_is_executed() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("note.txt"), "secret-token-42").unwrap();

    let server = spawn(|_, req| {
        match model(req) {
        "planner-model" => text_reply(r#"{"steps":["Read note.txt"]}"#),
        "executor-model" if last_role(req) == "tool" => {
            let content = req["messages"].as_array().unwrap().last().unwrap()["content"].as_str().unwrap().to_string();
            text_reply(&format!("RESULT: {content}"))
        }
        "executor-model" => text_reply(
            "<think>I should read it.</think><tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"note.txt\"}}\n</tool_call>",
        ),
        _ => text_reply(r#"{"verdict":"pass","summary":"ok"}"#),
    }
    });

    let report = orchestrator(&server.base_url, dir.path(), false)
        .run("read the note")
        .unwrap();
    assert_eq!(report.tool_calls, 1);
    assert_eq!(report.step_results[0].1, "RESULT: secret-token-42");
}

#[test]
fn state_changing_command_is_not_run_when_the_user_declines() {
    let dir = tempfile::tempdir().unwrap();
    let server = spawn(|_, req| match model(req) {
        "planner-model" => text_reply(r#"{"steps":["Create marker.txt"]}"#),
        "executor-model" if last_role(req) == "tool" => text_reply("RESULT: could not create it"),
        "executor-model" => {
            tool_call_reply("c1", "run_shell", json!({"command": "touch marker.txt"}))
        }
        _ => text_reply(r#"{"verdict":"pass","summary":"not created"}"#),
    });
    orchestrator(&server.base_url, dir.path(), false)
        .run("make a marker")
        .unwrap();
    assert!(!dir.path().join("marker.txt").exists());

    let reqs = server.requests.lock().unwrap();
    let fed_back = reqs
        .iter()
        .filter(|r| model(r) == "executor-model")
        .nth(1)
        .unwrap()["messages"]
        .as_array()
        .unwrap()
        .last()
        .unwrap()["content"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(fed_back.contains("denied by user"), "{fed_back}");
}

#[test]
fn approved_command_runs() {
    let dir = tempfile::tempdir().unwrap();
    let server = spawn(|_, req| match model(req) {
        "planner-model" => text_reply(r#"{"steps":["Create marker.txt"]}"#),
        "executor-model" if last_role(req) == "tool" => text_reply("RESULT: created"),
        "executor-model" => {
            tool_call_reply("c1", "run_shell", json!({"command": "touch marker.txt"}))
        }
        _ => text_reply(r#"{"verdict":"pass","summary":"created"}"#),
    });
    orchestrator(&server.base_url, dir.path(), true)
        .run("make a marker")
        .unwrap();
    assert!(dir.path().join("marker.txt").exists());
}

#[test]
fn critic_retry_triggers_one_repair_pass() {
    let dir = tempfile::tempdir().unwrap();
    let server = spawn(|idx, req| match model(req) {
        "planner-model" => text_reply(r#"{"steps":["Do the thing"]}"#),
        "executor-model" => text_reply("RESULT: attempted"),
        "critic-model" => {
            let first = idx < 4;
            if first {
                text_reply(
                    r#"{"verdict":"retry","feedback":"you never verified the result","summary":""}"#,
                )
            } else {
                text_reply(r#"{"verdict":"retry","feedback":"still not verified","summary":""}"#)
            }
        }
        _ => unreachable!(),
    });
    let report = orchestrator(&server.base_url, dir.path(), false)
        .run("goal")
        .unwrap();
    assert_eq!(report.replans, 1, "max_replans is 1 so the loop must stop");
    assert_eq!(report.step_results.len(), 2);
    assert!(report.step_results[1]
        .0
        .contains("you never verified the result"));
    assert_eq!(report.critic_verdict, "retry");
    assert_eq!(report.final_answer, "RESULT: attempted");
}

#[test]
fn tool_round_limit_stops_a_looping_executor() {
    let dir = tempfile::tempdir().unwrap();
    let server = spawn(|_, req| match model(req) {
        "planner-model" => text_reply(r#"{"steps":["loop forever"]}"#),
        "executor-model" => tool_call_reply("c", "system_info", json!({})),
        _ => text_reply(r#"{"verdict":"pass","summary":"gave up"}"#),
    });
    let mut o = orchestrator(&server.base_url, dir.path(), false);
    let report = o.run("goal").unwrap();
    assert_eq!(report.tool_calls, 8);
    assert!(report.step_results[0].1.starts_with("incomplete"));
}

fn run_with_budget(budget: usize) -> Vec<Value> {
    let dir = tempfile::tempdir().unwrap();
    let server = spawn(|idx, req| match model(req) {
        "planner-model" => text_reply(r#"{"steps":["dump numbers"]}"#),
        "executor-model" if idx < 8 => tool_call_reply(
            &format!("c{idx}"),
            "run_shell",
            json!({"command": "seq 1 2000"}),
        ),
        "executor-model" => text_reply("RESULT: done"),
        _ => text_reply(r#"{"verdict":"pass","summary":"ok"}"#),
    });
    let mut s = Settings::single(&server.base_url, "planner-model", dir.path().to_path_buf());
    s.executor.model = "executor-model".into();
    s.context_budget_tokens = budget;
    s.max_tool_rounds = 10;
    let mut o = Orchestrator::new(
        s,
        ToolContext {
            workdir: dir.path().to_path_buf(),
            approver: Box::new(FixedApprover(true)),
            shell_timeout: Duration::from_secs(5),
            max_output_bytes: 4096,
        },
    );
    o.run("goal").unwrap();
    let reqs = server.requests.lock().unwrap();
    let last_exec = reqs.iter().rfind(|r| model(r) == "executor-model").unwrap();
    last_exec["messages"].as_array().unwrap().clone()
}

fn assert_tool_messages_have_owners(msgs: &[Value]) {
    for (i, m) in msgs.iter().enumerate().filter(|(_, m)| m["role"] == "tool") {
        let owner = msgs[..i]
            .iter()
            .rev()
            .find(|x| x["role"] == "assistant")
            .unwrap();
        let ids: Vec<_> = owner["tool_calls"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["id"].clone())
            .collect();
        assert!(
            ids.contains(&m["tool_call_id"]),
            "tool message {i} lost its assistant message"
        );
    }
}

#[test]
fn moderate_budget_truncates_old_tool_output_but_keeps_every_turn() {
    let msgs = run_with_budget(3500);
    let tool_msgs: Vec<_> = msgs.iter().filter(|m| m["role"] == "tool").collect();
    assert_eq!(
        tool_msgs.len(),
        7,
        "no turn should be dropped when truncation is enough"
    );
    assert!(
        tool_msgs[0]["content"]
            .as_str()
            .unwrap()
            .contains("chars omitted"),
        "oldest output not truncated"
    );
    assert!(
        tool_msgs.last().unwrap()["content"].as_str().unwrap().len() > 3000,
        "latest output must stay intact"
    );
    assert!(!msgs.iter().any(|m| m["content"]
        .as_str()
        .unwrap_or("")
        .contains("context compacted")));
    assert_tool_messages_have_owners(&msgs);
}

#[test]
fn tight_budget_drops_old_turns_with_a_note_and_keeps_pairs_valid() {
    let msgs = run_with_budget(1200);
    assert!(
        msgs.iter().any(|m| m["content"]
            .as_str()
            .unwrap_or("")
            .contains("context compacted")),
        "no compaction note"
    );
    assert!(
        msgs.iter().filter(|m| m["role"] == "tool").count() < 7,
        "old turns should have been dropped"
    );
    assert_eq!(msgs[0]["role"], "system");
    assert_eq!(msgs.last().unwrap()["role"], "tool");
    assert_tool_messages_have_owners(&msgs);
}

#[test]
fn http_errors_surface_with_the_server_message() {
    let dir = tempfile::tempdir().unwrap();
    let server = spawn(|_, _| MockReply::Status(500, "model not loaded".into()));
    let err = orchestrator(&server.base_url, dir.path(), false)
        .run("goal")
        .unwrap_err();
    let msg = format!("{err:#}");
    assert!(
        msg.contains("500") && msg.contains("model not loaded"),
        "{msg}"
    );
}

#[test]
fn streaming_bench_reports_ttft_and_decode_rate() {
    let chunk = |t: &str| json!({"choices":[{"delta":{"content":t}}]});
    let server = spawn(move |_, _| {
        let mut chunks: Vec<Value> = (0..10).map(|i| chunk(&format!("tok{i} "))).collect();
        chunks.push(json!({"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":10}}));
        MockReply::Sse(chunks)
    });
    let client = ChatClient::new(&server.base_url, None);
    let mut seen = 0;
    let summary = run_bench(&client, "m", "hi", 3, 64, |_, _| seen += 1).unwrap();
    assert_eq!(seen, 3);
    assert_eq!(summary.runs.len(), 3);
    assert!(summary.all_counts_from_server());
    let r = &summary.runs[0];
    assert_eq!((r.prompt_tokens, r.completion_tokens), (7, 10));
    assert!(r.text.starts_with("tok0 tok1"));
    // 9 inter-token gaps of about 20 ms each: roughly 50 tok/s, with generous scheduler slack.
    let tps = r.decode_tps();
    assert!(tps > 15.0 && tps < 80.0, "decode tps {tps}");
    assert_eq!(
        server.requests.lock().unwrap().len(),
        4,
        "1 warm-up + 3 measured"
    );
}

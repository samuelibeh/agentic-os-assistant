use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionCall {
    pub name: String,
    #[serde(deserialize_with = "args_as_string")]
    pub arguments: String,
}

/// OpenAI sends arguments as a JSON string, Ollama and some templates send an object.
fn args_as_string<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    Ok(match Value::deserialize(d)? {
        Value::String(s) => s,
        Value::Null => "{}".to_string(),
        other => other.to_string(),
    })
}

fn null_to_empty<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    Ok(Option::<String>::deserialize(d)?.unwrap_or_default())
}

fn function_kind() -> String {
    "function".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    #[serde(default)]
    pub id: String,
    #[serde(rename = "type", default = "function_kind")]
    pub kind: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct Message {
    pub role: String,
    #[serde(default, deserialize_with = "null_to_empty")]
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl Message {
    pub fn system(s: impl Into<String>) -> Self {
        Message {
            role: "system".into(),
            content: s.into(),
            ..Default::default()
        }
    }
    pub fn user(s: impl Into<String>) -> Self {
        Message {
            role: "user".into(),
            content: s.into(),
            ..Default::default()
        }
    }
    pub fn tool(call_id: String, name: String, content: String) -> Self {
        Message {
            role: "tool".into(),
            content,
            tool_call_id: Some(call_id),
            name: Some(name),
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

#[derive(Debug, Clone)]
pub struct Reply {
    pub message: Message,
    pub usage: Usage,
    pub finish_reason: String,
    pub elapsed: Duration,
}

pub struct ChatClient {
    base_url: String,
    api_key: Option<String>,
    agent: ureq::Agent,
}

impl ChatClient {
    pub fn new(base_url: &str, api_key: Option<String>) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(10))
            .timeout_read(Duration::from_secs(300))
            .build();
        ChatClient {
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            agent,
        }
    }

    fn post(&self, path: &str, body: Value) -> Result<ureq::Response> {
        let mut req = self.agent.post(&format!("{}{}", self.base_url, path));
        if let Some(k) = &self.api_key {
            req = req.set("Authorization", &format!("Bearer {k}"));
        }
        match req.send_json(body) {
            Ok(r) => Ok(r),
            Err(ureq::Error::Status(code, resp)) => {
                let text = resp.into_string().unwrap_or_default();
                bail!(
                    "server returned HTTP {code}: {}",
                    text.chars().take(500).collect::<String>()
                )
            }
            Err(e) => Err(anyhow!(e)).context(format!("could not reach {}", self.base_url)),
        }
    }

    pub fn chat(
        &self,
        model: &str,
        messages: &[Message],
        tools: Option<&[Value]>,
        temperature: f32,
        max_tokens: u32,
    ) -> Result<Reply> {
        let mut body = json!({
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": false,
        });
        if let Some(t) = tools {
            body["tools"] = Value::Array(t.to_vec());
            body["tool_choice"] = json!("auto");
        }
        let started = Instant::now();
        let v: Value = self.post("/chat/completions", body)?.into_json()?;
        parse_reply(&v, started.elapsed())
    }

    /// Streams a completion and records time-to-first-token and decode throughput.
    pub fn stream_completion(
        &self,
        model: &str,
        prompt: &str,
        temperature: f32,
        max_tokens: u32,
    ) -> Result<StreamStats> {
        let body = json!({
            "model": model,
            "messages": [Message::user(prompt)],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": true,
            "stream_options": {"include_usage": true},
        });
        let started = Instant::now();
        let resp = self.post("/chat/completions", body)?;
        let reader = BufReader::new(resp.into_reader());

        let mut first: Option<Duration> = None;
        let mut chunks = 0u64;
        let mut usage: Option<Usage> = None;
        let mut text = String::new();
        for line in reader.lines() {
            let line = line?;
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let data = data.trim();
            if data == "[DONE]" {
                break;
            }
            let Ok(v) = serde_json::from_str::<Value>(data) else {
                continue;
            };
            if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
                usage = Some(parse_usage(u));
            }
            let piece = v["choices"][0]["delta"]["content"].as_str().unwrap_or("");
            if !piece.is_empty() {
                first.get_or_insert_with(|| started.elapsed());
                chunks += 1;
                text.push_str(piece);
            }
        }
        let total = started.elapsed();
        let from_server = usage.map(|u| u.completion_tokens > 0).unwrap_or(false);
        let u = usage.unwrap_or_default();
        Ok(StreamStats {
            ttft: first.unwrap_or(total),
            total,
            prompt_tokens: u.prompt_tokens,
            completion_tokens: if from_server {
                u.completion_tokens
            } else {
                chunks
            },
            tokens_from_server: from_server,
            text,
        })
    }
}

#[derive(Debug, Clone)]
pub struct StreamStats {
    pub ttft: Duration,
    pub total: Duration,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// False when the server sent no usage block and chunks were counted instead.
    pub tokens_from_server: bool,
    pub text: String,
}

impl StreamStats {
    /// Tokens per second after the first token, excluding prefill.
    pub fn decode_tps(&self) -> f64 {
        let decode = self.total.saturating_sub(self.ttft).as_secs_f64();
        if self.completion_tokens < 2 || decode <= 0.0 {
            return 0.0;
        }
        (self.completion_tokens - 1) as f64 / decode
    }
}

fn parse_usage(u: &Value) -> Usage {
    Usage {
        prompt_tokens: u["prompt_tokens"].as_u64().unwrap_or(0),
        completion_tokens: u["completion_tokens"].as_u64().unwrap_or(0),
    }
}

pub fn parse_reply(v: &Value, elapsed: Duration) -> Result<Reply> {
    let choice = v["choices"]
        .get(0)
        .ok_or_else(|| anyhow!("response has no choices: {v}"))?;
    let mut message: Message =
        serde_json::from_value(choice["message"].clone()).context("malformed assistant message")?;
    if message.tool_calls.is_empty() {
        let (rest, calls) = parse_inline_tool_calls(&message.content);
        if !calls.is_empty() {
            message.content = rest;
            message.tool_calls = calls;
        }
    }
    for (i, c) in message.tool_calls.iter_mut().enumerate() {
        if c.id.is_empty() {
            c.id = format!("call_{i}");
        }
    }
    Ok(Reply {
        message,
        usage: v.get("usage").map(parse_usage).unwrap_or_default(),
        finish_reason: choice["finish_reason"].as_str().unwrap_or("").to_string(),
        elapsed,
    })
}

/// Hermes and Qwen templates emit `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`
/// in the text when the serving runtime does not translate it into `tool_calls`.
pub fn parse_inline_tool_calls(content: &str) -> (String, Vec<ToolCall>) {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";
    let mut calls = Vec::new();
    let mut rest = String::new();
    let mut cursor = content;
    while let Some(start) = cursor.find(OPEN) {
        rest.push_str(&cursor[..start]);
        let after = &cursor[start + OPEN.len()..];
        let (body, next) = match after.find(CLOSE) {
            Some(end) => (&after[..end], &after[end + CLOSE.len()..]),
            None => (after, ""),
        };
        if let Ok(v) = serde_json::from_str::<Value>(body.trim()) {
            if let Some(name) = v["name"].as_str() {
                let args = match &v["arguments"] {
                    Value::String(s) => s.clone(),
                    Value::Null => "{}".to_string(),
                    other => other.to_string(),
                };
                calls.push(ToolCall {
                    id: format!("call_inline_{}", calls.len()),
                    kind: function_kind(),
                    function: FunctionCall {
                        name: name.to_string(),
                        arguments: args,
                    },
                });
            }
        }
        cursor = next;
    }
    rest.push_str(cursor);
    (rest.trim().to_string(), calls)
}

/// Removes `<think>...</think>` reasoning blocks that Qwen3-style models prepend.
pub fn strip_think(s: &str) -> String {
    let mut out = String::new();
    let mut cursor = s;
    while let Some(start) = cursor.find("<think>") {
        out.push_str(&cursor[..start]);
        match cursor[start..].find("</think>") {
            Some(end) => cursor = &cursor[start + end + "</think>".len()..],
            None => {
                cursor = "";
                break;
            }
        }
    }
    out.push_str(cursor);
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_openai_tool_calls_with_string_arguments() {
        let v = json!({"choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","content":null,
            "tool_calls":[{"id":"c1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a\"}"}}]}}],
            "usage":{"prompt_tokens":10,"completion_tokens":5}});
        let r = parse_reply(&v, Duration::ZERO).unwrap();
        assert_eq!(
            r.message.tool_calls[0].function.arguments,
            "{\"path\":\"a\"}"
        );
        assert_eq!(r.usage.completion_tokens, 5);
        assert_eq!(r.message.content, "");
    }

    #[test]
    fn parses_object_arguments() {
        let v = json!({"choices":[{"message":{"role":"assistant","content":"",
            "tool_calls":[{"function":{"name":"list_dir","arguments":{"path":"."}}}]}}]});
        let r = parse_reply(&v, Duration::ZERO).unwrap();
        assert_eq!(
            r.message.tool_calls[0].function.arguments,
            "{\"path\":\".\"}"
        );
        assert_eq!(r.message.tool_calls[0].id, "call_0");
    }

    #[test]
    fn parses_hermes_inline_tool_calls() {
        let text = "Checking.\n<tool_call>\n{\"name\": \"run_shell\", \"arguments\": {\"command\": \"df -h\"}}\n</tool_call>";
        let (rest, calls) = parse_inline_tool_calls(text);
        assert_eq!(rest, "Checking.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "run_shell");
        assert!(calls[0].function.arguments.contains("df -h"));
    }

    #[test]
    fn inline_parser_ignores_garbage() {
        let (rest, calls) = parse_inline_tool_calls("<tool_call>not json</tool_call>done");
        assert!(calls.is_empty());
        assert_eq!(rest, "done");
    }

    #[test]
    fn strips_think_blocks() {
        assert_eq!(strip_think("<think>hmm</think>\nanswer"), "answer");
        assert_eq!(strip_think("a<think>x</think>b<think>unterminated"), "ab");
    }

    #[test]
    fn decode_tps_excludes_prefill() {
        let s = StreamStats {
            ttft: Duration::from_millis(500),
            total: Duration::from_millis(1500),
            prompt_tokens: 10,
            completion_tokens: 51,
            tokens_from_server: true,
            text: String::new(),
        };
        assert!((s.decode_tps() - 50.0).abs() < 1e-9);
    }
}

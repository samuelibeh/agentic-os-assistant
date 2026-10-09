#![allow(dead_code)]
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

pub enum MockReply {
    Json(Value),
    Sse(Vec<Value>),
    Status(u16, String),
}

pub struct MockServer {
    pub base_url: String,
    pub requests: Arc<Mutex<Vec<Value>>>,
}

/// Minimal OpenAI-compatible server. `handler` receives the request index and body.
pub fn spawn(handler: impl Fn(usize, &Value) -> MockReply + Send + Sync + 'static) -> MockServer {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base_url = format!("http://{}/v1", listener.local_addr().unwrap());
    let requests = Arc::new(Mutex::new(Vec::new()));
    let log = requests.clone();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            let body = read_request(&stream);
            let idx = {
                let mut l = log.lock().unwrap();
                l.push(body.clone());
                l.len() - 1
            };
            respond(stream, handler(idx, &body));
        }
    });
    MockServer { base_url, requests }
}

fn read_request(stream: &TcpStream) -> Value {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut len = 0usize;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        if line == "\r\n" || line.is_empty() {
            break;
        }
        if let Some(v) = line.to_lowercase().strip_prefix("content-length:") {
            len = v.trim().parse().unwrap();
        }
    }
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).unwrap();
    serde_json::from_slice(&buf).unwrap_or(Value::Null)
}

fn respond(mut stream: TcpStream, reply: MockReply) {
    match reply {
        MockReply::Json(v) => {
            let body = v.to_string();
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
        }
        MockReply::Status(code, text) => {
            let _ = write!(
                stream,
                "HTTP/1.1 {code} Error\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{text}",
                text.len()
            );
        }
        MockReply::Sse(chunks) => {
            let _ = write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n"
            );
            for c in chunks {
                let _ = write!(stream, "data: {c}\n\n");
                let _ = stream.flush();
                thread::sleep(Duration::from_millis(20));
            }
            let _ = write!(stream, "data: [DONE]\n\n");
        }
    }
    let _ = stream.flush();
}

pub fn text_reply(content: &str) -> MockReply {
    MockReply::Json(
        json!({"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":content}}],
        "usage":{"prompt_tokens":20,"completion_tokens":10}}),
    )
}

pub fn tool_call_reply(id: &str, name: &str, args: Value) -> MockReply {
    MockReply::Json(
        json!({"choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","content":null,
        "tool_calls":[{"id":id,"type":"function","function":{"name":name,"arguments":args.to_string()}}]}}],
        "usage":{"prompt_tokens":30,"completion_tokens":12}}),
    )
}

pub fn last_role(req: &Value) -> &str {
    req["messages"]
        .as_array()
        .and_then(|m| m.last())
        .and_then(|m| m["role"].as_str())
        .unwrap_or("")
}

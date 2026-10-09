use crate::llm::Message;

/// Cheap token estimate (about four characters per token) used to decide when to compact.
pub fn estimate_tokens(msgs: &[Message]) -> usize {
    msgs.iter()
        .map(|m| {
            let calls: usize = m
                .tool_calls
                .iter()
                .map(|c| c.function.name.len() + c.function.arguments.len())
                .sum();
            (m.content.len() + calls) / 4 + 4
        })
        .sum()
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct CompactionReport {
    pub truncated_outputs: usize,
    pub dropped_messages: usize,
}

const KEEP_RECENT_GROUPS: usize = 2;
const TRUNCATED_OUTPUT_CHARS: usize = 400;

/// Shrinks `msgs` to fit `budget` tokens in two passes:
/// 1. tool outputs older than the last few turns are cut to a head and tail snippet;
/// 2. whole oldest turns (an assistant message plus its tool results) are replaced by a
///    single note. The system prompt, the first user message and the latest turn are kept,
///    and a tool result is never separated from the assistant message that requested it.
pub fn compact(msgs: &mut Vec<Message>, budget: usize) -> CompactionReport {
    let mut report = CompactionReport::default();
    if estimate_tokens(msgs) <= budget {
        return report;
    }

    let protected = protected_prefix(msgs);
    let groups = group_boundaries(msgs, protected);

    let recent_start = groups
        .len()
        .checked_sub(KEEP_RECENT_GROUPS)
        .map(|i| groups[i].0)
        .unwrap_or(protected);
    for m in msgs[protected..recent_start]
        .iter_mut()
        .filter(|m| m.role == "tool")
    {
        if m.content.chars().count() > TRUNCATED_OUTPUT_CHARS {
            m.content = shorten(&m.content, TRUNCATED_OUTPUT_CHARS);
            report.truncated_outputs += 1;
        }
    }
    if estimate_tokens(msgs) <= budget {
        return report;
    }

    let mut groups = groups;
    let mut drop_until = protected;
    while groups.len() > 1 && estimate_tokens(msgs) - freed(msgs, protected, drop_until) > budget {
        let (_, end) = groups.remove(0);
        drop_until = end;
    }
    if drop_until > protected {
        report.dropped_messages = drop_until - protected;
        let note = Message::user(format!(
            "[context compacted: {} earlier messages were removed to stay within the context window]",
            report.dropped_messages
        ));
        msgs.splice(protected..drop_until, std::iter::once(note));
    }
    report
}

fn freed(msgs: &[Message], from: usize, to: usize) -> usize {
    estimate_tokens(&msgs[from..to])
}

/// Leading system messages plus the first user message.
fn protected_prefix(msgs: &[Message]) -> usize {
    let mut i = 0;
    while i < msgs.len() && msgs[i].role == "system" {
        i += 1;
    }
    if i < msgs.len() && msgs[i].role == "user" {
        i += 1;
    }
    i
}

/// (start, end) index pairs; an assistant message and the tool messages after it form one group.
fn group_boundaries(msgs: &[Message], from: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = from;
    while i < msgs.len() {
        let start = i;
        i += 1;
        if msgs[start].role == "assistant" {
            while i < msgs.len() && msgs[i].role == "tool" {
                i += 1;
            }
        }
        out.push((start, i));
    }
    out
}

pub fn shorten(s: &str, max_chars: usize) -> String {
    let n = s.chars().count();
    if n <= max_chars {
        return s.to_string();
    }
    let head: String = s.chars().take(max_chars * 2 / 3).collect();
    let tail: String = s.chars().skip(n - max_chars / 3).collect();
    format!(
        "{head}\n...[{} chars omitted]...\n{tail}",
        n - head.chars().count() - tail.chars().count()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{FunctionCall, ToolCall};

    fn assistant_call(id: &str) -> Message {
        Message {
            role: "assistant".into(),
            tool_calls: vec![ToolCall {
                id: id.into(),
                kind: "function".into(),
                function: FunctionCall {
                    name: "run_shell".into(),
                    arguments: "{\"command\":\"ls\"}".into(),
                },
            }],
            ..Default::default()
        }
    }

    fn convo(turns: usize, output_len: usize) -> Vec<Message> {
        let mut v = vec![Message::system("sys"), Message::user("goal")];
        for i in 0..turns {
            let id = format!("c{i}");
            v.push(assistant_call(&id));
            v.push(Message::tool(
                id,
                "run_shell".into(),
                "x".repeat(output_len),
            ));
        }
        v
    }

    #[test]
    fn under_budget_is_untouched() {
        let mut m = convo(2, 100);
        let before = m.clone();
        assert_eq!(compact(&mut m, 10_000), CompactionReport::default());
        assert_eq!(m, before);
    }

    #[test]
    fn old_tool_output_is_truncated_before_anything_is_dropped() {
        let mut m = convo(5, 4000);
        let budget = estimate_tokens(&m) - 2000;
        let r = compact(&mut m, budget);
        assert!(r.truncated_outputs > 0);
        assert_eq!(r.dropped_messages, 0);
        assert!(estimate_tokens(&m) <= budget);
        assert_eq!(
            m.last().unwrap().content.len(),
            4000,
            "latest output stays intact"
        );
    }

    #[test]
    fn oldest_turns_are_dropped_and_pairs_stay_intact() {
        let mut m = convo(10, 4000);
        let r = compact(&mut m, 1500);
        assert!(r.dropped_messages > 0);
        assert_eq!(m[0].role, "system");
        assert_eq!(m[1].content, "goal");
        assert!(m[2].content.contains("context compacted"));
        for (i, msg) in m.iter().enumerate() {
            if msg.role == "tool" {
                let id = msg.tool_call_id.as_ref().unwrap();
                let owner = m[..i].iter().rev().find(|x| x.role == "assistant").unwrap();
                assert!(
                    owner.tool_calls.iter().any(|c| &c.id == id),
                    "orphaned tool message at {i}"
                );
            }
        }
        assert_eq!(m.last().unwrap().role, "tool");
    }

    #[test]
    fn latest_turn_survives_an_impossible_budget() {
        let mut m = convo(4, 4000);
        compact(&mut m, 10);
        assert!(m.iter().any(|x| x.role == "assistant"));
        assert_eq!(m.last().unwrap().role, "tool");
    }

    #[test]
    fn shorten_keeps_head_and_tail() {
        let s = format!("{}{}", "a".repeat(500), "z".repeat(500));
        let out = shorten(&s, 300);
        assert!(out.starts_with('a') && out.ends_with('z'));
        assert!(out.contains("chars omitted"));
    }
}

use crate::context::shorten;
use crate::policy::{self, Verdict};
use serde_json::{json, Value};
use std::fs;
use std::io::{Read, Write};
use std::os::unix::process::CommandExt;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const MAX_READ_BYTES: u64 = 64 * 1024;
const MAX_DIR_ENTRIES: usize = 200;

pub trait Approver: Send {
    fn approve(&mut self, prompt: &str) -> bool;
}

/// Approves or denies everything without asking (`--approve-all`, tests).
pub struct FixedApprover(pub bool);

impl Approver for FixedApprover {
    fn approve(&mut self, _prompt: &str) -> bool {
        self.0
    }
}

pub struct StdinApprover;

impl Approver for StdinApprover {
    fn approve(&mut self, prompt: &str) -> bool {
        eprint!("\n{prompt}\nAllow? [y/N] ");
        let _ = std::io::stderr().flush();
        let mut line = String::new();
        if std::io::stdin().read_line(&mut line).is_err() {
            return false;
        }
        matches!(line.trim().to_lowercase().as_str(), "y" | "yes")
    }
}

pub struct ToolContext {
    pub workdir: PathBuf,
    pub approver: Box<dyn Approver>,
    pub shell_timeout: Duration,
    pub max_output_bytes: usize,
}

pub fn specs() -> Vec<Value> {
    fn spec(name: &str, description: &str, properties: Value, required: &[&str]) -> Value {
        json!({"type": "function", "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        }})
    }
    vec![
        spec(
            "run_shell",
            "Run a shell command in the working directory. Read-only inspection commands run \
             immediately; anything that may change the system asks the user first. Times out after \
             a fixed limit.",
            json!({"command": {"type": "string", "description": "The command line to run with sh -c."}}),
            &["command"],
        ),
        spec(
            "read_file",
            "Read a text file (first 64 KiB). Relative paths are resolved against the working directory.",
            json!({"path": {"type": "string"}}),
            &["path"],
        ),
        spec(
            "write_file",
            "Create or overwrite a file inside the working directory. Always asks the user first.",
            json!({"path": {"type": "string"}, "content": {"type": "string"}}),
            &["path", "content"],
        ),
        spec(
            "list_dir",
            "List a directory with entry types and sizes.",
            json!({"path": {"type": "string", "description": "Defaults to the working directory."}}),
            &[],
        ),
        spec(
            "system_info",
            "Report kernel, CPU count, load average, uptime and memory.",
            json!({}),
            &[],
        ),
    ]
}

/// Runs a tool and always returns text for the model, including errors.
pub fn execute(ctx: &mut ToolContext, name: &str, arguments: &str) -> String {
    let args: Value = match serde_json::from_str(if arguments.trim().is_empty() {
        "{}"
    } else {
        arguments
    }) {
        Ok(v) => v,
        Err(e) => return format!("error: arguments are not valid JSON ({e})"),
    };
    let result = match name {
        "run_shell" => str_arg(&args, "command").and_then(|c| run_shell(ctx, &c)),
        "read_file" => str_arg(&args, "path").and_then(|p| read_file(ctx, &p)),
        "write_file" => str_arg(&args, "path")
            .and_then(|p| str_arg(&args, "content").map(|c| (p, c)))
            .and_then(|(p, c)| write_file(ctx, &p, &c)),
        "list_dir" => list_dir(ctx, args["path"].as_str().unwrap_or(".")),
        "system_info" => Ok(system_info()),
        other => Err(format!("unknown tool `{other}`")),
    };
    match result {
        Ok(out) => out,
        Err(e) => format!("error: {e}"),
    }
}

fn str_arg(args: &Value, key: &str) -> Result<String, String> {
    args[key]
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| format!("missing string argument `{key}`"))
}

fn run_shell(ctx: &mut ToolContext, command: &str) -> Result<String, String> {
    match policy::classify(command) {
        Verdict::Deny(reason) => {
            return Ok(format!(
                "denied by policy: {reason}. The command was not run."
            ))
        }
        Verdict::NeedsApproval(reason) => {
            let prompt = format!("The agent wants to run:\n  $ {command}\n({reason})");
            if !ctx.approver.approve(&prompt) {
                return Ok("denied by user. The command was not run.".into());
            }
        }
        Verdict::ReadOnly => {}
    }
    let mut child = Command::new("sh")
        .arg("-c")
        .arg(command)
        .current_dir(&ctx.workdir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0)
        .spawn()
        .map_err(|e| format!("failed to start shell: {e}"))?;

    let mut out_pipe = child.stdout.take().unwrap();
    let mut err_pipe = child.stderr.take().unwrap();
    let out_thread = thread::spawn(move || {
        let mut b = Vec::new();
        let _ = out_pipe.read_to_end(&mut b);
        b
    });
    let err_thread = thread::spawn(move || {
        let mut b = Vec::new();
        let _ = err_pipe.read_to_end(&mut b);
        b
    });

    let deadline = Instant::now() + ctx.shell_timeout;
    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break Some(s),
            Ok(None) if Instant::now() >= deadline => {
                timed_out = true;
                unsafe { libc::kill(-(child.id() as i32), libc::SIGKILL) };
                let _ = child.wait();
                break None;
            }
            Ok(None) => thread::sleep(Duration::from_millis(10)),
            Err(e) => return Err(format!("wait failed: {e}")),
        }
    };
    let stdout = String::from_utf8_lossy(&out_thread.join().unwrap_or_default()).into_owned();
    let stderr = String::from_utf8_lossy(&err_thread.join().unwrap_or_default()).into_owned();

    let header = if timed_out {
        format!(
            "[timed out after {}s and was killed]",
            ctx.shell_timeout.as_secs()
        )
    } else {
        format!(
            "[exit {}]",
            status
                .and_then(|s| s.code())
                .map_or("signal".into(), |c| c.to_string())
        )
    };
    let mut text = header;
    if !stdout.trim().is_empty() {
        text.push('\n');
        text.push_str(stdout.trim_end());
    }
    if !stderr.trim().is_empty() {
        text.push_str("\n[stderr]\n");
        text.push_str(stderr.trim_end());
    }
    Ok(shorten(&text, ctx.max_output_bytes))
}

fn read_file(ctx: &ToolContext, path: &str) -> Result<String, String> {
    let full = if Path::new(path).is_absolute() {
        PathBuf::from(path)
    } else {
        ctx.workdir.join(path)
    };
    let file = fs::File::open(&full).map_err(|e| format!("cannot open {}: {e}", full.display()))?;
    let mut buf = Vec::new();
    file.take(MAX_READ_BYTES + 1)
        .read_to_end(&mut buf)
        .map_err(|e| format!("cannot read {}: {e}", full.display()))?;
    let truncated = buf.len() as u64 > MAX_READ_BYTES;
    buf.truncate(MAX_READ_BYTES as usize);
    let mut text = String::from_utf8_lossy(&buf).into_owned();
    if truncated {
        text.push_str("\n[truncated at 64 KiB]");
    }
    Ok(text)
}

fn write_file(ctx: &mut ToolContext, path: &str, content: &str) -> Result<String, String> {
    let full = resolve_inside(&ctx.workdir, path)?;
    let prompt = format!(
        "The agent wants to write {} bytes to {}",
        content.len(),
        full.display()
    );
    if !ctx.approver.approve(&prompt) {
        return Ok("denied by user. The file was not written.".into());
    }
    if let Some(parent) = full.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
    }
    fs::write(&full, content).map_err(|e| format!("cannot write {}: {e}", full.display()))?;
    Ok(format!(
        "wrote {} bytes to {}",
        content.len(),
        full.display()
    ))
}

fn list_dir(ctx: &ToolContext, path: &str) -> Result<String, String> {
    let full = if Path::new(path).is_absolute() {
        PathBuf::from(path)
    } else {
        ctx.workdir.join(path)
    };
    let mut entries: Vec<_> = fs::read_dir(&full)
        .map_err(|e| format!("cannot list {}: {e}", full.display()))?
        .filter_map(Result::ok)
        .collect();
    entries.sort_by_key(|e| e.file_name());
    let total = entries.len();
    let mut lines: Vec<String> = entries
        .iter()
        .take(MAX_DIR_ENTRIES)
        .map(|e| {
            let meta = e.metadata().ok();
            let kind = match &meta {
                Some(m) if m.is_dir() => "dir ",
                Some(m) if m.file_type().is_symlink() => "link",
                _ => "file",
            };
            format!(
                "{kind} {:>10}  {}",
                meta.map_or(0, |m| m.len()),
                e.file_name().to_string_lossy()
            )
        })
        .collect();
    if total > MAX_DIR_ENTRIES {
        lines.push(format!("... {} more entries", total - MAX_DIR_ENTRIES));
    }
    if lines.is_empty() {
        return Ok("(empty directory)".into());
    }
    Ok(lines.join("\n"))
}

fn system_info() -> String {
    let read = |p: &str| fs::read_to_string(p).unwrap_or_default().trim().to_string();
    let meminfo = read("/proc/meminfo");
    let mem = |key: &str| {
        meminfo
            .lines()
            .find(|l| l.starts_with(key))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
            .map_or("unknown".to_string(), |kb| format!("{} MiB", kb / 1024))
    };
    format!(
        "kernel: {} {}\nhostname: {}\ncpus: {}\nload average: {}\nuptime seconds: {}\nmemory total: {}\nmemory available: {}",
        read("/proc/sys/kernel/ostype"),
        read("/proc/sys/kernel/osrelease"),
        read("/proc/sys/kernel/hostname"),
        thread::available_parallelism().map_or(0, |n| n.get()),
        read("/proc/loadavg"),
        read("/proc/uptime").split_whitespace().next().unwrap_or("unknown"),
        mem("MemTotal"),
        mem("MemAvailable"),
    )
}

/// Resolves `path` against `workdir` and rejects anything that ends up outside it,
/// including `..` segments and symlinks on the existing part of the path.
pub fn resolve_inside(workdir: &Path, path: &str) -> Result<PathBuf, String> {
    let joined = if Path::new(path).is_absolute() {
        PathBuf::from(path)
    } else {
        workdir.join(path)
    };
    let mut norm = PathBuf::new();
    for c in joined.components() {
        match c {
            Component::ParentDir => {
                norm.pop();
            }
            Component::CurDir => {}
            other => norm.push(other.as_os_str()),
        }
    }
    let root = workdir
        .canonicalize()
        .map_err(|e| format!("bad workdir: {e}"))?;

    let mut existing = norm.as_path();
    let mut tail = Vec::new();
    while !existing.exists() {
        tail.push(existing.file_name().ok_or("invalid path")?.to_owned());
        existing = existing.parent().ok_or("invalid path")?;
    }
    let mut full = existing
        .canonicalize()
        .map_err(|e| format!("cannot resolve path: {e}"))?;
    for part in tail.into_iter().rev() {
        full.push(part);
    }
    if full.starts_with(&root) {
        Ok(full)
    } else {
        Err(format!(
            "path {} is outside the working directory {}",
            full.display(),
            root.display()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(dir: &Path, approve: bool) -> ToolContext {
        ToolContext {
            workdir: dir.to_path_buf(),
            approver: Box::new(FixedApprover(approve)),
            shell_timeout: Duration::from_secs(5),
            max_output_bytes: 4096,
        }
    }

    #[test]
    fn read_only_command_runs_without_approval() {
        let d = tempfile::tempdir().unwrap();
        fs::write(d.path().join("a.txt"), "hi").unwrap();
        let mut c = ctx(d.path(), false);
        let out = execute(&mut c, "run_shell", r#"{"command":"ls"}"#);
        assert!(
            out.starts_with("[exit 0]") && out.contains("a.txt"),
            "{out}"
        );
    }

    #[test]
    fn mutating_command_is_blocked_when_user_declines() {
        let d = tempfile::tempdir().unwrap();
        let mut c = ctx(d.path(), false);
        let out = execute(&mut c, "run_shell", r#"{"command":"touch made.txt"}"#);
        assert!(out.contains("denied by user"));
        assert!(!d.path().join("made.txt").exists());
    }

    #[test]
    fn mutating_command_runs_when_approved() {
        let d = tempfile::tempdir().unwrap();
        let mut c = ctx(d.path(), true);
        execute(&mut c, "run_shell", r#"{"command":"touch made.txt"}"#);
        assert!(d.path().join("made.txt").exists());
    }

    #[test]
    fn denied_commands_never_run_even_with_approve_all() {
        let d = tempfile::tempdir().unwrap();
        let mut c = ctx(d.path(), true);
        let out = execute(&mut c, "run_shell", r#"{"command":"rm -rf /"}"#);
        assert!(out.contains("denied by policy"), "{out}");
    }

    #[test]
    fn timeout_kills_the_whole_process_group() {
        let d = tempfile::tempdir().unwrap();
        let mut c = ctx(d.path(), true);
        c.shell_timeout = Duration::from_millis(300);
        let started = Instant::now();
        let out = execute(&mut c, "run_shell", r#"{"command":"sleep 30 & sleep 30"}"#);
        assert!(out.contains("timed out"), "{out}");
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn large_output_is_truncated() {
        let d = tempfile::tempdir().unwrap();
        let mut c = ctx(d.path(), true);
        let out = execute(&mut c, "run_shell", r#"{"command":"seq 1 100000"}"#);
        assert!(out.len() < 5000 && out.contains("chars omitted"));
    }

    #[test]
    fn write_file_is_confined_to_workdir() {
        let d = tempfile::tempdir().unwrap();
        let mut c = ctx(d.path(), true);
        let out = execute(
            &mut c,
            "write_file",
            r#"{"path":"../escape.txt","content":"x"}"#,
        );
        assert!(out.contains("outside the working directory"), "{out}");
        let out = execute(
            &mut c,
            "write_file",
            r#"{"path":"/etc/agentos-test","content":"x"}"#,
        );
        assert!(out.contains("outside the working directory"), "{out}");
        let out = execute(
            &mut c,
            "write_file",
            r#"{"path":"sub/ok.txt","content":"x"}"#,
        );
        assert!(out.starts_with("wrote"), "{out}");
        assert_eq!(
            fs::read_to_string(d.path().join("sub/ok.txt")).unwrap(),
            "x"
        );
    }

    #[test]
    fn write_file_rejects_symlink_escape() {
        let d = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        std::os::unix::fs::symlink(outside.path(), d.path().join("link")).unwrap();
        let mut c = ctx(d.path(), true);
        let out = execute(
            &mut c,
            "write_file",
            r#"{"path":"link/x.txt","content":"x"}"#,
        );
        assert!(out.contains("outside the working directory"), "{out}");
        assert!(!outside.path().join("x.txt").exists());
    }

    #[test]
    fn write_file_respects_denial() {
        let d = tempfile::tempdir().unwrap();
        let mut c = ctx(d.path(), false);
        let out = execute(&mut c, "write_file", r#"{"path":"a.txt","content":"x"}"#);
        assert!(out.contains("denied by user"));
        assert!(!d.path().join("a.txt").exists());
    }

    #[test]
    fn bad_arguments_return_errors_not_panics() {
        let d = tempfile::tempdir().unwrap();
        let mut c = ctx(d.path(), false);
        assert!(execute(&mut c, "run_shell", "{not json").starts_with("error:"));
        assert!(execute(&mut c, "run_shell", "{}").contains("missing string argument"));
        assert!(execute(&mut c, "nope", "{}").contains("unknown tool"));
    }

    #[test]
    fn list_and_read_and_sysinfo_work() {
        let d = tempfile::tempdir().unwrap();
        fs::write(d.path().join("a.txt"), "hello").unwrap();
        fs::create_dir(d.path().join("sub")).unwrap();
        let mut c = ctx(d.path(), false);
        let listing = execute(&mut c, "list_dir", "{}");
        assert!(
            listing.contains("a.txt") && listing.contains("dir "),
            "{listing}"
        );
        assert_eq!(execute(&mut c, "read_file", r#"{"path":"a.txt"}"#), "hello");
        assert!(execute(&mut c, "system_info", "{}").contains("cpus:"));
    }
}

use crate::llm::{ChatClient, StreamStats};
use anyhow::Result;

pub const DEFAULT_PROMPT: &str =
    "Explain how a Linux process scheduler decides which task runs next. \
Be thorough and give a concrete example.";

pub struct BenchSummary {
    pub runs: Vec<StreamStats>,
}

impl BenchSummary {
    pub fn ttft_ms_median(&self) -> f64 {
        median(
            self.runs
                .iter()
                .map(|r| r.ttft.as_secs_f64() * 1000.0)
                .collect(),
        )
    }
    pub fn decode_tps_median(&self) -> f64 {
        median(self.runs.iter().map(StreamStats::decode_tps).collect())
    }
    pub fn all_counts_from_server(&self) -> bool {
        self.runs.iter().all(|r| r.tokens_from_server)
    }
}

/// One warm-up request (not counted) followed by `runs` measured streaming requests.
pub fn run_bench(
    client: &ChatClient,
    model: &str,
    prompt: &str,
    runs: usize,
    max_tokens: u32,
    mut on_run: impl FnMut(usize, &StreamStats),
) -> Result<BenchSummary> {
    client.stream_completion(model, prompt, 0.0, max_tokens.min(16))?;
    let mut out = Vec::with_capacity(runs);
    for i in 0..runs {
        let stats = client.stream_completion(model, prompt, 0.0, max_tokens)?;
        on_run(i + 1, &stats);
        out.push(stats);
    }
    Ok(BenchSummary { runs: out })
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::median;

    #[test]
    fn median_of_odd_even_empty() {
        assert_eq!(median(vec![3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(vec![4.0, 1.0, 2.0, 3.0]), 2.5);
        assert_eq!(median(vec![]), 0.0);
    }
}

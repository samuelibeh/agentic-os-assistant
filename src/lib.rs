//! Multi-agent OS assistant for open-weight chat models (Qwen, Hermes) served
//! behind an OpenAI-compatible endpoint (llama.cpp server, vLLM, Ollama).

pub mod agents;
pub mod bench;
pub mod config;
pub mod context;
pub mod llm;
pub mod policy;
pub mod tools;

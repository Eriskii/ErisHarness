//! ErisHarness runs many agents with little memory each. An agent is a transcript on disk
//! plus a machine to act on: an ErisSandbox sandbox that exists as processes only while the
//! agent is doing something, or this machine directly.

pub mod agent;
mod harness;
pub mod machine;
pub mod provider;
mod store;
pub mod tools;

pub use harness::{Harness, HarnessBuilder, Recipient};

//! ErisHarness runs many agents with little memory each. An agent is a transcript on disk
//! plus a sandbox that exists as processes only while the agent is doing something.

pub mod agent;
mod bootstrap;
mod cgroup;
mod harness;
pub mod provider;
pub mod rootfs;
pub mod sandbox;
mod store;
pub mod tools;

pub use bootstrap::{Host, bootstrap};
pub use harness::{Harness, HarnessBuilder};

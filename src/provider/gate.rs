//! Paces one account's model calls. A provider's rate limit belongs to the account, so every
//! agent calling through it shares one gate.
//!
//! Calls take a slot for their whole duration and wait for one in the order they came. The
//! number of slots adapts to what the account sustains: a rate limit halves it and pauses every
//! call until the provider's retry time, and each successful call adds a fraction of a slot,
//! about one per round of successes.

use super::Hold;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, watch};
use tokio::time::{Instant, sleep_until};

pub struct RateGate {
    slots: Arc<Semaphore>,
    state: Mutex<State>,
    /// When the pause after the latest rate limit ends.
    pause: watch::Sender<Option<Instant>>,
    max: usize,
}

struct State {
    limit: f64,
    /// Slots calls may hold: floor(limit), at most `max`.
    capacity: usize,
    /// Slots held by calls beyond `capacity` since a rate limit; they are retired when released.
    debt: usize,
}

impl RateGate {
    pub fn new(initial: usize, max: usize) -> Arc<Self> {
        let max = max.max(1);
        let initial = initial.clamp(1, max);
        Arc::new(Self {
            slots: Arc::new(Semaphore::new(initial)),
            state: Mutex::new(State { limit: initial as f64, capacity: initial, debt: 0 }),
            pause: watch::channel(None).0,
            max,
        })
    }

    /// How many calls may run at once.
    pub fn slots(&self) -> usize {
        self.state.lock().unwrap().capacity
    }

    /// Waits for a slot and for any pause to end. `report` hears each [`Hold`] as it begins,
    /// and `None` once the call goes on; a call that never waits hears nothing.
    pub async fn acquire(self: &Arc<Self>, report: &(dyn Fn(Option<Hold>) + Sync)) -> Permit {
        let mut pause = self.pause.subscribe();
        let mut shown = None;
        let mut show = |hold: Option<Hold>| {
            if hold != shown {
                report(hold);
                shown = hold;
            }
        };
        let limited = |until: Instant| Hold::RateLimited { until: unix_millis(until) };
        let slot = self.slots.clone().acquire_owned();
        tokio::pin!(slot);
        let permit = loop {
            let until = active(*pause.borrow_and_update());
            tokio::select! {
                biased;
                permit = &mut slot => break permit.expect("the gate never closes"),
                _ = std::future::ready(()) => {}
            }
            show(Some(until.map_or(Hold::Queued, limited)));
            tokio::select! {
                biased;
                permit = &mut slot => break permit.expect("the gate never closes"),
                _ = pause.changed() => {}
                _ = sleep_until(until.unwrap_or_else(Instant::now)), if until.is_some() => {}
            }
        };
        loop {
            let until = active(*pause.borrow_and_update());
            let Some(until) = until else { break };
            show(Some(limited(until)));
            tokio::select! {
                _ = sleep_until(until) => {}
                _ = pause.changed() => {}
            }
        }
        show(None);
        Permit { gate: self.clone(), permit: Some(permit) }
    }

    fn succeeded(&self) {
        let mut state = self.state.lock().unwrap();
        state.limit = (state.limit + 1.0 / state.limit).min(self.max as f64);
        while state.capacity < state.limit as usize {
            if state.debt > 0 {
                state.debt -= 1;
            } else {
                self.slots.add_permits(1);
            }
            state.capacity += 1;
        }
    }

    fn rate_limited(&self, wait: Duration) {
        {
            let mut state = self.state.lock().unwrap();
            let target = (state.capacity / 2).max(1);
            state.limit = target as f64;
            let excess = state.capacity - target;
            state.capacity = target;
            let retired = self.slots.forget_permits(excess);
            state.debt += excess - retired;
        }
        let until = Instant::now() + wait;
        self.pause.send_modify(|pause| {
            if pause.is_none_or(|current| current < until) {
                *pause = Some(until);
            }
        });
    }
}

/// A call's slot. Dropping it frees the slot without changing how many there are.
pub struct Permit {
    gate: Arc<RateGate>,
    permit: Option<OwnedSemaphorePermit>,
}

impl Permit {
    /// The call succeeded: the account may take a little more.
    pub fn succeeded(self) {
        self.gate.succeeded();
    }

    /// The provider rate-limited the call and asked to wait `wait` before trying again.
    pub fn rate_limited(self, wait: Duration) {
        self.gate.rate_limited(wait);
    }
}

impl Drop for Permit {
    fn drop(&mut self) {
        let Some(permit) = self.permit.take() else { return };
        let mut state = self.gate.state.lock().unwrap();
        if state.debt > 0 {
            state.debt -= 1;
            permit.forget();
        }
    }
}

fn active(pause: Option<Instant>) -> Option<Instant> {
    pause.filter(|until| *until > Instant::now())
}

fn unix_millis(until: Instant) -> u64 {
    let at = SystemTime::now() + until.saturating_duration_since(Instant::now());
    at.duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::advance;

    /// Each report: "queued", "limited", or "free" once the call goes on.
    type Seen = Arc<Mutex<Vec<&'static str>>>;

    fn reports() -> (Seen, impl Fn(Option<Hold>) + Send + Sync + Clone) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        let report = move |hold: Option<Hold>| {
            sink.lock().unwrap().push(match hold {
                Some(Hold::Queued) => "queued",
                Some(Hold::RateLimited { .. }) => "limited",
                None => "free",
            })
        };
        (seen, report)
    }

    #[tokio::test(start_paused = true)]
    async fn calls_wait_for_a_slot_in_the_order_they_came() {
        let gate = RateGate::new(2, 8);
        let (seen, report) = reports();
        let a = gate.acquire(&report).await;
        let _b = gate.acquire(&report).await;
        assert!(seen.lock().unwrap().is_empty(), "a call with a free slot is never held");
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut waiting = Vec::new();
        for n in 0..3 {
            let (gate, order, report) = (gate.clone(), order.clone(), report.clone());
            waiting.push(tokio::spawn(async move {
                let permit = gate.acquire(&report).await;
                order.lock().unwrap().push(n);
                permit.succeeded();
            }));
            tokio::task::yield_now().await;
        }
        tokio::task::yield_now().await;
        assert!(order.lock().unwrap().is_empty(), "both slots are taken");
        drop(a);
        for task in waiting {
            task.await.unwrap();
        }
        assert_eq!(*order.lock().unwrap(), [0, 1, 2]);
        assert_eq!(*seen.lock().unwrap(), ["queued", "queued", "queued", "free", "free", "free"]);
    }

    #[tokio::test(start_paused = true)]
    async fn a_rate_limit_halves_the_slots_and_pauses_every_call() {
        let gate = RateGate::new(8, 64);
        let (seen, report) = reports();
        let mut held: Vec<_> = Vec::new();
        for _ in 0..8 {
            held.push(gate.acquire(&report).await);
        }
        let limited = held.pop().unwrap();
        limited.rate_limited(Duration::from_secs(30));
        assert_eq!(gate.slots(), 4);
        drop(held);
        let started = Instant::now();
        let next = tokio::spawn({
            let (gate, report) = (gate.clone(), report.clone());
            async move { gate.acquire(&report).await }
        });
        tokio::task::yield_now().await;
        assert!(!next.is_finished(), "the pause holds every call");
        advance(Duration::from_secs(31)).await;
        let _next = next.await.unwrap();
        assert!(started.elapsed() >= Duration::from_secs(30));
        assert_eq!(*seen.lock().unwrap(), ["limited", "free"], "the wait is reported and ends");
        let mut running = Vec::new();
        for _ in 0..3 {
            running.push(gate.acquire(&report).await);
        }
        let fifth = tokio::spawn({
            let (gate, report) = (gate.clone(), report.clone());
            async move { gate.acquire(&report).await }
        });
        tokio::task::yield_now().await;
        assert!(!fifth.is_finished(), "only four calls run at once");
    }

    #[tokio::test(start_paused = true)]
    async fn successes_add_about_a_slot_per_round_up_to_the_maximum() {
        let gate = RateGate::new(2, 3);
        let (_, report) = reports();
        // From two slots: 2 → 2.5 → 2.9 → 3.2, about one slot per round of successes.
        for _ in 0..3 {
            gate.acquire(&report).await.succeeded();
        }
        assert_eq!(gate.slots(), 3);
        for _ in 0..10 {
            gate.acquire(&report).await.succeeded();
        }
        assert_eq!(gate.slots(), 3);
        let a = gate.acquire(&report).await;
        a.rate_limited(Duration::from_secs(1));
        a_limit_never_drops_below_one(&gate, &report).await;
    }

    async fn a_limit_never_drops_below_one(gate: &Arc<RateGate>, report: &(impl Fn(Option<Hold>) + Sync)) {
        for _ in 0..4 {
            advance(Duration::from_secs(2)).await;
            gate.acquire(report).await.rate_limited(Duration::from_secs(1));
        }
        assert_eq!(gate.slots(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn calls_already_waiting_see_a_pause_begin() {
        let gate = RateGate::new(1, 4);
        let (_, quiet) = reports();
        let first = gate.acquire(&quiet).await;
        let (seen, report) = reports();
        let queued = tokio::spawn({
            let gate = gate.clone();
            async move { gate.acquire(&report).await }
        });
        tokio::task::yield_now().await;
        assert_eq!(*seen.lock().unwrap(), ["queued"]);
        first.rate_limited(Duration::from_secs(10));
        tokio::task::yield_now().await;
        assert_eq!(*seen.lock().unwrap(), ["queued", "limited"]);
        advance(Duration::from_secs(11)).await;
        queued.await.unwrap();
        assert_eq!(*seen.lock().unwrap(), ["queued", "limited", "free"]);
    }
}

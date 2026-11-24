//! A queue's timeline: its work in submission order, held back behind a
//! GPU-side wait (`ID3D12CommandQueue::Wait`) until the fence reaches the
//! value.
//!
//! Work runs on the thread that submits it, as before, unless the queue is
//! held back: then the submission returns at once and the work runs later,
//! on the thread that moves the fence (a signal or a bottom-of-pipe write
//! from another queue, or the title's CPU signal). Nothing blocks a
//! submitting thread behind a wait, so a thread may submit a queue's wait
//! before the signal it waits for.
//!
//! Locks: a timeline's state is taken before a fence's waiters, never
//! after; work runs with no lock held.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use crate::fence::Held;

/// One step of a queue's work.
pub(crate) enum Op {
    /// Hold back what follows until the fence reaches the value.
    Wait(Held, u64),
    /// Command lists to run, a fence to signal.
    Run(Box<dyn FnOnce() + Send>),
}

#[derive(Default)]
pub(crate) struct Timeline {
    state: Mutex<State>,
    /// Signalled whenever work is done or the queue stops running it.
    moved: Condvar,
    /// With [`frames_in_flight`]: signalled when there is work for the
    /// queue's own thread.
    work: Condvar,
    /// The queue's own thread was started.
    worker: std::sync::atomic::AtomicBool,
}

/// `VICTORY_ASYNC_SUBMIT=N` (opt-in, 1-3): submissions return at once and
/// a thread of the queue's own runs its work (prepare, GPU, copy-back),
/// so the title records its next lists while the GPU runs the last ones,
/// as on the console; at most N presents may be pending before a present
/// waits. 0 (the default): work runs on the submitting thread, which
/// waits for it.
pub(crate) fn frames_in_flight() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        let n = std::env::var("VICTORY_ASYNC_SUBMIT")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(0)
            .min(3);
        if n > 0 {
            tracing::info!(frames = n, "VICTORY_ASYNC_SUBMIT: queues run their work on their own threads");
        }
        n
    })
}

#[derive(Default)]
struct State {
    ops: VecDeque<Op>,
    /// Ops submitted and ops done (a submission's ticket is its count).
    pushed: u64,
    done: u64,
    /// A thread is running the queue's work.
    busy: bool,
    /// Tickets of presents submitted and not done yet (async only).
    frames: VecDeque<u64>,
    /// The first op is a wait not reached yet, registered with its fence.
    held: bool,
}

impl Timeline {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Adds work to the queue and runs it unless the queue is held back.
    /// Work another thread is running for the queue is waited for, as a
    /// submission's own work is.
    pub(crate) fn submit(self: &Arc<Self>, op: Op) {
        if frames_in_flight() > 0 && self.start_worker() {
            return self.submit_async(op, false);
        }
        let mut s = self.lock();
        s.ops.push_back(op);
        s.pushed += 1;
        let ticket = s.pushed;
        while s.busy && s.done < ticket {
            s = self.moved.wait(s).unwrap_or_else(|e| e.into_inner());
        }
        if s.done >= ticket {
            return;
        }
        // Still held back unless the value was reached by a path that does
        // not wake us (a CPU write to the fence's memory).
        if s.held
            && let Some(Op::Wait(fence, value)) = s.ops.front()
            && !fence.reached(*value)
        {
            return;
        }
        self.run(s);
    }

    /// A present: as [`Timeline::submit`], and with
    /// [`frames_in_flight`] the title waits here while that many earlier
    /// presents are not done yet.
    pub(crate) fn submit_present(self: &Arc<Self>, op: Op) {
        if frames_in_flight() > 0 && self.start_worker() {
            self.submit_async(op, true);
        } else {
            self.submit(op);
        }
    }

    fn submit_async(self: &Arc<Self>, op: Op, present: bool) {
        let mut s = self.lock();
        s.ops.push_back(op);
        s.pushed += 1;
        if present {
            let ticket = s.pushed;
            s.frames.push_back(ticket);
            while s.frames.len() as u64 > frames_in_flight() {
                let oldest = s.frames[0];
                if s.done >= oldest {
                    s.frames.pop_front();
                    continue;
                }
                // Held behind a wait (maybe for a fence this very thread
                // signals later): the title must not wait for it.
                if s.held {
                    break;
                }
                self.work.notify_all();
                s = self
                    .moved
                    .wait_timeout(s, std::time::Duration::from_millis(5))
                    .unwrap_or_else(|e| e.into_inner())
                    .0;
            }
        }
        self.work.notify_all();
    }

    /// Starts the queue's own thread once; whether it runs.
    fn start_worker(self: &Arc<Self>) -> bool {
        use std::sync::atomic::Ordering;
        static FAILED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if FAILED.load(Ordering::Relaxed) {
            return false;
        }
        if self.worker.swap(true, Ordering::AcqRel) {
            return true;
        }
        let me = self.clone();
        let spawned = std::thread::Builder::new()
            .name("victory-queue".into())
            .spawn(move || {
                loop {
                    let mut s = me.lock();
                    loop {
                        let runnable = !s.busy
                            && match s.ops.front() {
                                None => false,
                                Some(Op::Wait(fence, value)) => !s.held || fence.reached(*value),
                                Some(Op::Run(_)) => true,
                            };
                        if runnable {
                            break;
                        }
                        // A timeout as well: a fence reached by a CPU write
                        // to its memory wakes no one.
                        s = me
                            .work
                            .wait_timeout(s, std::time::Duration::from_millis(2))
                            .unwrap_or_else(|e| e.into_inner())
                            .0;
                    }
                    me.run(s);
                }
            });
        match spawned {
            Ok(_) => true,
            Err(e) => {
                tracing::warn!(error = %e, "no queue thread; work runs on the submitting thread");
                FAILED.store(true, Ordering::Relaxed);
                false
            }
        }
    }

    /// The fence this queue waits for moved: runs what it held back.
    pub(crate) fn wake(self: &Arc<Self>) {
        if frames_in_flight() > 0 {
            self.work.notify_all();
            return;
        }
        let s = self.lock();
        if !s.busy {
            self.run(s);
        }
    }

    fn run<'a>(self: &'a Arc<Self>, mut s: MutexGuard<'a, State>) {
        s.busy = true;
        s.held = false;
        while let Some(op) = s.ops.pop_front() {
            match op {
                Op::Wait(fence, value) => {
                    // Checked and registered under the fence's lock, so a
                    // signal after this check wakes us.
                    if !fence.wait_on_gpu(value, self) {
                        s.ops.push_front(Op::Wait(fence, value));
                        s.held = true;
                        break;
                    }
                }
                Op::Run(work) => {
                    drop(s);
                    work();
                    s = self.lock();
                }
            }
            s.done += 1;
            self.moved.notify_all();
        }
        s.busy = false;
        self.moved.notify_all();
    }
}

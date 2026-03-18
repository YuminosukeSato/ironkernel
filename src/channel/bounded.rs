use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crossbeam_channel::{bounded, Receiver, Sender, TryRecvError};

use crate::buffer::inner::Buffer;
use crate::error::{ParsecError, ParsecResult};

/// Message types that can flow through a channel.
#[derive(Debug, Clone)]
pub enum Msg {
    Buffer(Buffer),
    Closed,
}

/// A bounded channel that carries Buffer handles.
///
/// `close()` marks the channel as closed. After close:
/// - `send()` returns `ChannelClosed` immediately
/// - `recv()` drains remaining items, then returns `ChannelClosed`
#[derive(Clone)]
pub struct Channel {
    sender: Sender<Msg>,
    receiver: Receiver<Msg>,
    closed: Arc<AtomicBool>,
}

impl Channel {
    /// Create a bounded channel with the given capacity.
    pub fn new(capacity: usize) -> Self {
        let (sender, receiver) = bounded(capacity);
        Channel {
            sender,
            receiver,
            closed: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Close the channel. Idempotent: subsequent calls are no-ops.
    ///
    /// Sends `Msg::Closed` to wake any blocking receiver.
    /// After close, `send()` fails and `recv()` drains then fails.
    pub fn close(&self) {
        if self
            .closed
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            // Best-effort: send Closed to wake blocking receivers.
            // Ignore error if channel is already disconnected.
            let _ = self.sender.send(Msg::Closed);
        }
    }

    /// Returns true if `close()` has been called.
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    /// Send a buffer through the channel. Blocks if full.
    pub fn send(&self, buf: Buffer) -> ParsecResult<()> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(ParsecError::ChannelClosed);
        }
        self.sender
            .send(Msg::Buffer(buf))
            .map_err(|_| ParsecError::ChannelClosed)
    }

    /// Receive a buffer from the channel. Blocks until available.
    pub fn recv(&self) -> ParsecResult<Buffer> {
        match self.receiver.recv() {
            Ok(Msg::Buffer(buf)) => Ok(buf),
            Ok(Msg::Closed) => Err(ParsecError::ChannelClosed),
            Err(_) => Err(ParsecError::ChannelClosed),
        }
    }

    /// Try to receive without blocking.
    pub fn try_recv(&self) -> ParsecResult<Option<Buffer>> {
        match self.receiver.try_recv() {
            Ok(Msg::Buffer(buf)) => Ok(Some(buf)),
            Ok(Msg::Closed) => Err(ParsecError::ChannelClosed),
            Err(TryRecvError::Empty) => Ok(None),
            Err(TryRecvError::Disconnected) => Err(ParsecError::ChannelClosed),
        }
    }

    /// Get the underlying receiver for use with select.
    pub fn receiver(&self) -> &Receiver<Msg> {
        &self.receiver
    }

    /// Get the underlying sender.
    pub fn sender(&self) -> &Sender<Msg> {
        &self.sender
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn send_recv_basic() {
        let ch = Channel::new(10);
        let buf = Buffer::from_f64_vec(vec![1.0, 2.0, 3.0]);
        ch.send(buf).unwrap();
        let received = ch.recv().unwrap();
        assert_eq!(received.as_f64_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn send_recv_multiple() {
        let ch = Channel::new(10);
        for i in 0..5 {
            ch.send(Buffer::from_f64_vec(vec![i as f64])).unwrap();
        }
        for i in 0..5 {
            let buf = ch.recv().unwrap();
            assert_eq!(buf.as_f64_slice(), &[i as f64]);
        }
    }

    #[test]
    fn try_recv_empty() {
        let ch = Channel::new(10);
        assert!(ch.try_recv().unwrap().is_none());
    }

    #[test]
    fn try_recv_has_data() {
        let ch = Channel::new(10);
        ch.send(Buffer::from_f64_vec(vec![42.0])).unwrap();
        let buf = ch.try_recv().unwrap().unwrap();
        assert_eq!(buf.as_f64_slice(), &[42.0]);
    }

    #[test]
    fn channel_clone_shares() {
        let ch1 = Channel::new(10);
        let ch2 = ch1.clone();
        ch1.send(Buffer::from_f64_vec(vec![1.0])).unwrap();
        let buf = ch2.recv().unwrap();
        assert_eq!(buf.as_f64_slice(), &[1.0]);
    }

    #[test]
    fn concurrent_send_recv() {
        let ch = Channel::new(100);
        let ch_send = ch.clone();
        let handle = std::thread::spawn(move || {
            for i in 0..100 {
                ch_send.send(Buffer::from_f64_vec(vec![i as f64])).unwrap();
            }
        });

        let mut received = Vec::new();
        for _ in 0..100 {
            let buf = ch.recv().unwrap();
            received.push(buf.as_f64_slice()[0]);
        }
        handle.join().unwrap();
        received.sort_by(f64::total_cmp);
        assert_eq!(
            received,
            (0..100).map(|value| value as f64).collect::<Vec<_>>()
        );
    }

    #[test]
    fn multi_producer() {
        let ch = Channel::new(100);
        let mut handles = vec![];
        for t in 0..10 {
            let ch_clone = ch.clone();
            handles.push(std::thread::spawn(move || {
                for i in 0..10 {
                    ch_clone
                        .send(Buffer::from_f64_vec(vec![(t * 10 + i) as f64]))
                        .unwrap();
                }
            }));
        }

        let mut received = Vec::new();
        for _ in 0..100 {
            let buf = ch.recv().unwrap();
            received.push(buf.as_f64_slice()[0]);
        }
        for h in handles {
            h.join().unwrap();
        }
        received.sort_by(f64::total_cmp);
        assert_eq!(
            received,
            (0..100).map(|value| value as f64).collect::<Vec<_>>()
        );
    }

    #[test]
    fn channel_recv_reports_closed_why_internal_close_propagation_must_not_silently_break() {
        let ch = Channel::new(1);
        ch.sender.send(Msg::Closed).unwrap();

        let err = ch.recv().unwrap_err();

        assert_eq!(err, ParsecError::ChannelClosed);
    }

    #[test]
    fn try_recv_closed_msg() {
        let ch = Channel::new(1);
        ch.sender.send(Msg::Closed).unwrap();
        let err = ch.try_recv().unwrap_err();
        assert_eq!(err, ParsecError::ChannelClosed);
    }

    #[test]
    fn try_recv_disconnected() {
        // Drop the sender by creating a new channel and only keeping the receiver
        let (sender, receiver) = crossbeam_channel::bounded(1);
        drop(sender);
        let ch_disconnected = Channel {
            sender: {
                let (s, _) = crossbeam_channel::bounded(1);
                s
            },
            receiver,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let err = ch_disconnected.try_recv().unwrap_err();
        assert_eq!(err, ParsecError::ChannelClosed);
    }

    #[test]
    fn recv_disconnected() {
        let (sender, receiver) = crossbeam_channel::bounded::<Msg>(1);
        drop(sender);
        let ch = Channel {
            sender: {
                let (s, _) = crossbeam_channel::bounded(1);
                s
            },
            receiver,
            closed: Arc::new(AtomicBool::new(false)),
        };
        let err = ch.recv().unwrap_err();
        assert_eq!(err, ParsecError::ChannelClosed);
    }

    #[test]
    fn send_disconnected() {
        let (sender, receiver) = crossbeam_channel::bounded::<Msg>(1);
        drop(receiver);
        let ch = Channel {
            sender,
            receiver: {
                let (_, r) = crossbeam_channel::bounded(1);
                r
            },
            closed: Arc::new(AtomicBool::new(false)),
        };
        let err = ch.send(Buffer::from_f64_vec(vec![1.0])).unwrap_err();
        assert_eq!(err, ParsecError::ChannelClosed);
    }

    #[test]
    fn channel_receiver_accessor() {
        let ch = Channel::new(10);
        let _r = ch.receiver();
    }

    #[test]
    fn channel_sender_accessor() {
        let ch = Channel::new(10);
        let _s = ch.sender();
    }

    // --- Channel.close() ---

    #[test]
    fn close_prevents_send() {
        let ch = Channel::new(10);
        ch.close();
        assert!(ch.is_closed());
        let err = ch.send(Buffer::from_f64_vec(vec![1.0])).unwrap_err();
        assert_eq!(err, ParsecError::ChannelClosed);
    }

    #[test]
    fn close_drains_remaining_then_closed() {
        let ch = Channel::new(10);
        ch.send(Buffer::from_f64_vec(vec![1.0])).unwrap();
        ch.send(Buffer::from_f64_vec(vec![2.0])).unwrap();
        ch.close();

        // Drain existing items
        let b1 = ch.recv().unwrap();
        assert_eq!(b1.as_f64_slice(), &[1.0]);
        let b2 = ch.recv().unwrap();
        assert_eq!(b2.as_f64_slice(), &[2.0]);

        // Now should get ChannelClosed
        let err = ch.recv().unwrap_err();
        assert_eq!(err, ParsecError::ChannelClosed);
    }

    #[test]
    fn close_idempotent() {
        let ch = Channel::new(10);
        ch.close();
        ch.close(); // should not panic
        assert!(ch.is_closed());
    }

    #[test]
    fn close_wakes_blocking_receiver() {
        let ch = Channel::new(1);
        let ch_recv = ch.clone();
        let handle = std::thread::spawn(move || ch_recv.recv());

        std::thread::sleep(std::time::Duration::from_millis(10));
        ch.close();

        let result = handle.join().unwrap();
        assert_eq!(result.unwrap_err(), ParsecError::ChannelClosed);
    }

    #[test]
    fn close_shared_across_clones() {
        let ch1 = Channel::new(10);
        let ch2 = ch1.clone();
        ch1.close();
        assert!(ch2.is_closed());
        let err = ch2.send(Buffer::from_f64_vec(vec![1.0])).unwrap_err();
        assert_eq!(err, ParsecError::ChannelClosed);
    }
}

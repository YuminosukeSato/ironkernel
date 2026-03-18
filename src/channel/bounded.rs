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
#[derive(Clone)]
pub struct Channel {
    sender: Sender<Msg>,
    receiver: Receiver<Msg>,
}

impl Channel {
    /// Create a bounded channel with the given capacity.
    pub fn new(capacity: usize) -> Self {
        let (sender, receiver) = bounded(capacity);
        Channel { sender, receiver }
    }

    /// Send a buffer through the channel. Blocks if full.
    pub fn send(&self, buf: Buffer) -> ParsecResult<()> {
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
            received.push(buf.as_f64_slice()[0] as i32);
        }
        handle.join().unwrap();
        received.sort();
        assert_eq!(received, (0..100).collect::<Vec<_>>());
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
            received.push(buf.as_f64_slice()[0] as i32);
        }
        for h in handles {
            h.join().unwrap();
        }
        received.sort();
        assert_eq!(received, (0..100).collect::<Vec<_>>());
    }
}

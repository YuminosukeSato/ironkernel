use crossbeam_channel::Select;

use crate::buffer::inner::Buffer;
use crate::error::ParsecResult;

use super::bounded::{Channel, Msg};

/// Result of a select operation.
pub enum SelectResult {
    /// Received a buffer from the channel at the given index.
    Received(usize, Buffer),
    /// Default case was triggered (no channels ready).
    Default,
}

impl SelectResult {
    #[cfg(test)]
    fn into_received(self) -> Option<(usize, Buffer)> {
        match self {
            Self::Received(idx, buf) => Some((idx, buf)),
            Self::Default => None,
        }
    }
}

/// Perform a non-blocking select across multiple channels.
/// If `has_default` is true, returns Default when no channel is ready.
/// Otherwise blocks until one channel is ready.
pub fn select_channels(channels: &[&Channel], has_default: bool) -> ParsecResult<SelectResult> {
    let mut sel = Select::new();
    for ch in channels {
        sel.recv(ch.receiver());
    }

    if has_default {
        let oper = sel.try_select();
        match oper {
            Ok(oper) => {
                let idx = oper.index();
                let msg = oper
                    .recv(channels[idx].receiver())
                    .map_err(|_| crate::error::ParsecError::ChannelClosed)?;
                match msg {
                    Msg::Buffer(buf) => Ok(SelectResult::Received(idx, buf)),
                    Msg::Closed => Err(crate::error::ParsecError::ChannelClosed),
                }
            }
            Err(_) => Ok(SelectResult::Default),
        }
    } else {
        let oper = sel.select();
        let idx = oper.index();
        let msg = oper
            .recv(channels[idx].receiver())
            .map_err(|_| crate::error::ParsecError::ChannelClosed)?;
        match msg {
            Msg::Buffer(buf) => Ok(SelectResult::Received(idx, buf)),
            Msg::Closed => Err(crate::error::ParsecError::ChannelClosed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn select_default_no_data() {
        let ch_a = Channel::new(10);
        let ch_b = Channel::new(10);
        let result = select_channels(&[&ch_a, &ch_b], true).unwrap();
        assert!(result.into_received().is_none());
    }

    #[test]
    fn select_receives_from_ready() {
        let ch_a = Channel::new(10);
        let ch_b = Channel::new(10);
        ch_a.send(Buffer::from_f64_vec(vec![42.0])).unwrap();
        let result = select_channels(&[&ch_a, &ch_b], true).unwrap();
        let (idx, buf) = result
            .into_received()
            .expect("expected ready receive result");
        assert_eq!(idx, 0);
        assert_eq!(buf.as_f64_slice(), &[42.0]);
    }

    #[test]
    fn select_receives_from_second() {
        let ch_a = Channel::new(10);
        let ch_b = Channel::new(10);
        ch_b.send(Buffer::from_f64_vec(vec![99.0])).unwrap();
        let result = select_channels(&[&ch_a, &ch_b], true).unwrap();
        let (idx, buf) = result
            .into_received()
            .expect("expected ready receive result");
        assert_eq!(idx, 1);
        assert_eq!(buf.as_f64_slice(), &[99.0]);
    }

    #[test]
    fn select_blocking() {
        let ch_a = Channel::new(10);
        let ch_send = ch_a.clone();

        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(10));
            ch_send.send(Buffer::from_f64_vec(vec![7.0])).unwrap();
        });

        let result = select_channels(&[&ch_a], false).unwrap();
        let (idx, buf) = result
            .into_received()
            .expect("expected blocking receive result");
        assert_eq!(idx, 0);
        assert_eq!(buf.as_f64_slice(), &[7.0]);
    }

    #[test]
    fn select_reports_closed_channel_why_internal_close_propagation_must_not_hide_failures() {
        let ch_a = Channel::new(1);
        ch_a.sender().send(Msg::Closed).unwrap();

        assert!(matches!(
            select_channels(&[&ch_a], true),
            Err(crate::error::ParsecError::ChannelClosed)
        ));
    }

    #[test]
    fn select_blocking_closed_msg() {
        let ch_a = Channel::new(1);
        ch_a.sender().send(Msg::Closed).unwrap();

        assert!(matches!(
            select_channels(&[&ch_a], false),
            Err(crate::error::ParsecError::ChannelClosed)
        ));
    }
}

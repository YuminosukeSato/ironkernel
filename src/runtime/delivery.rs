use std::sync::OnceLock;
use std::thread;

use crossbeam_channel::{unbounded, Sender};

use crate::buffer::inner::Buffer;
use crate::channel::bounded::Channel;
use crate::runtime::task::TaskHandle;

/// A job for the delivery worker to send a buffer through a channel.
pub(crate) struct DeliveryJob {
    pub task: TaskHandle,
    pub channel: Channel,
    pub buffer: Buffer,
}

/// Global delivery sender, lazily initialized.
static DELIVERY_TX: OnceLock<Sender<DeliveryJob>> = OnceLock::new();

fn delivery_sender() -> &'static Sender<DeliveryJob> {
    DELIVERY_TX.get_or_init(|| {
        let (tx, rx) = unbounded::<DeliveryJob>();
        thread::Builder::new()
            .name("delivery-worker".into())
            .spawn(move || {
                for job in rx {
                    // Skip if task was already cancelled or failed.
                    if job.task.is_done() {
                        continue;
                    }
                    job.task.set_delivering();
                    match job.channel.send(job.buffer) {
                        Ok(()) => job.task.complete_delivery(),
                        Err(e) => job.task.fail(e),
                    }
                }
            })
            .expect("failed to spawn delivery worker thread");
        tx
    })
}

/// Submit a delivery job to the background worker.
pub(crate) fn submit_delivery(job: DeliveryJob) {
    delivery_sender()
        .send(job)
        .expect("delivery worker channel disconnected");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::task::TaskResult;
    use std::time::Duration;

    #[test]
    fn delivery_sends_buffer_to_channel() {
        let ch = Channel::new(10);
        let task = TaskHandle::new();
        task.set_running_compute();

        let buf = Buffer::from_f64_vec(vec![1.0, 2.0, 3.0]);
        task.set_delivery_queued(TaskResult::Buffer(buf.clone()));

        submit_delivery(DeliveryJob {
            task: task.clone(),
            channel: ch.clone(),
            buffer: buf,
        });

        let received = ch.recv().unwrap();
        assert_eq!(received.as_f64_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn delivery_completes_task() {
        let ch = Channel::new(10);
        let task = TaskHandle::new();
        task.set_running_compute();

        let buf = Buffer::from_f64_vec(vec![42.0]);
        task.set_delivery_queued(TaskResult::Buffer(buf.clone()));

        submit_delivery(DeliveryJob {
            task: task.clone(),
            channel: ch.clone(),
            buffer: buf,
        });

        // Wait for completion via condvar.
        let result = task.result().unwrap();
        assert_eq!(result.as_buffer().unwrap().as_f64_slice(), &[42.0]);
        assert!(task.is_done());
    }

    #[test]
    fn delivery_closed_channel_fails_task() {
        let ch = Channel::new(10);
        ch.close();

        let task = TaskHandle::new();
        task.set_running_compute();

        let buf = Buffer::from_f64_vec(vec![1.0]);
        task.set_delivery_queued(TaskResult::Buffer(buf.clone()));

        submit_delivery(DeliveryJob {
            task: task.clone(),
            channel: ch,
            buffer: buf,
        });

        let err = task.result().unwrap_err();
        assert_eq!(err, crate::error::ParsecError::ChannelClosed);
    }

    #[test]
    fn delivery_cancelled_task_skipped() {
        let ch = Channel::new(10);
        let task = TaskHandle::new();
        task.set_running_compute();

        let buf = Buffer::from_f64_vec(vec![1.0]);
        task.set_delivery_queued(TaskResult::Buffer(buf.clone()));
        task.cancel();

        submit_delivery(DeliveryJob {
            task: task.clone(),
            channel: ch.clone(),
            buffer: buf,
        });

        // Give worker time to process.
        thread::sleep(Duration::from_millis(50));

        // Channel should be empty (delivery was skipped).
        assert!(ch.try_recv().unwrap().is_none());
        // Task stays cancelled.
        assert_eq!(task.state(), crate::runtime::task::TaskState::Cancelled);
    }

    #[test]
    fn delivery_multiple_jobs_fifo() {
        let ch = Channel::new(10);

        let mut tasks = Vec::new();
        for i in 0..5 {
            let task = TaskHandle::new();
            task.set_running_compute();
            let buf = Buffer::from_f64_vec(vec![i as f64]);
            task.set_delivery_queued(TaskResult::Buffer(buf.clone()));
            submit_delivery(DeliveryJob {
                task: task.clone(),
                channel: ch.clone(),
                buffer: buf,
            });
            tasks.push(task);
        }

        // All should arrive in FIFO order.
        for i in 0..5 {
            let received = ch.recv().unwrap();
            assert_eq!(received.as_f64_slice(), &[i as f64]);
        }

        // All tasks should be completed.
        for task in &tasks {
            assert!(task.is_done());
        }
    }

    #[test]
    fn delivery_result_returns_staged_buffer() {
        let ch = Channel::new(10);
        let task = TaskHandle::new();
        task.set_running_compute();

        let buf = Buffer::from_f64_vec(vec![7.0, 8.0]);
        // Stage the result in the task.
        task.set_delivery_queued(TaskResult::Buffer(buf.clone()));

        submit_delivery(DeliveryJob {
            task: task.clone(),
            channel: ch.clone(),
            buffer: buf,
        });

        // task.result() should return the staged buffer.
        let result = task.result().unwrap();
        assert_eq!(result.as_buffer().unwrap().as_f64_slice(), &[7.0, 8.0]);

        // channel.recv() should also return the same data.
        let received = ch.recv().unwrap();
        assert_eq!(received.as_f64_slice(), &[7.0, 8.0]);
    }

    #[test]
    fn delivery_scalar_as_buffer() {
        let ch = Channel::new(10);
        let task = TaskHandle::new();
        task.set_running_compute();

        let scalar = 99.5_f64;
        let buf = Buffer::from_f64_vec(vec![scalar]);
        task.set_delivery_queued(TaskResult::Scalar(scalar));

        submit_delivery(DeliveryJob {
            task: task.clone(),
            channel: ch.clone(),
            buffer: buf,
        });

        // task.result() returns the Scalar.
        let result = task.result().unwrap();
        assert_eq!(result.as_scalar().unwrap(), 99.5);

        // channel.recv() returns the buffer wrapping the scalar.
        let received = ch.recv().unwrap();
        assert_eq!(received.as_f64_slice(), &[99.5]);
    }

    #[test]
    fn mutation_guard_delivery_submit_why_worker_sender_must_forward_payload_and_complete_task() {
        let ch = Channel::new(1);
        let task = TaskHandle::new();
        let buf = Buffer::from_f64_vec(vec![13.0, 21.0]);

        task.set_running_compute();
        task.set_delivery_queued(TaskResult::Buffer(buf.clone()));
        submit_delivery(DeliveryJob {
            task: task.clone(),
            channel: ch.clone(),
            buffer: buf,
        });

        let result = task.result().unwrap();

        assert!(task.is_done());
        let received = ch
            .try_recv()
            .unwrap()
            .expect("delivery should enqueue buffer");
        assert_eq!(result.as_buffer().unwrap().as_f64_slice(), &[13.0, 21.0]);

        assert_eq!(received.as_f64_slice(), &[13.0, 21.0]);
    }
}

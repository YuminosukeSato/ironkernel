use _ironkernel::buffer::inner::{Buffer, BufferInner, DType, Storage};
use _ironkernel::channel::bounded::Channel;
use _ironkernel::channel::select::{select_channels, SelectResult};
use _ironkernel::error::ParsecError;
use _ironkernel::ir::expr::Expr;
use _ironkernel::ir::kernel::{ArgSpec, KernelKind, KernelSpec, TensorSpec};
use _ironkernel::runtime::pool::init_pool;
use _ironkernel::runtime::task::{TaskHandle, TaskResult, TaskState};

#[test]
fn dtype_and_buffer_metadata_cover_public_getters_why_external_api_calls_should_count_toward_runtime_coverage(
) {
    let inner = BufferInner {
        dtype: DType::F64,
        shape: vec![2],
        storage: Storage::Owned(vec![0_u8; 16]),
    };
    let buffer = Buffer::from_f64_vec(vec![1.0, 2.0]);

    assert_eq!(DType::F32.byte_size(), 4);
    assert_eq!(DType::F64.byte_size(), 8);
    assert_eq!(DType::I32.byte_size(), 4);
    assert_eq!(DType::I64.byte_size(), 8);
    assert_eq!(DType::Bool.byte_size(), 1);
    assert_eq!(DType::F64.name(), "float64");
    assert_eq!(format!("{}", DType::Bool), "bool");
    assert_eq!(inner.byte_len(), 16);
    assert_eq!(buffer.dtype(), DType::F64);
    assert_eq!(buffer.shape(), &[2]);
    assert_eq!(buffer.len(), 2);
}

#[test]
fn parsec_error_display_covers_all_variants_why_release_surfaces_should_not_hide_error_messages() {
    assert_eq!(
        ParsecError::TypeError("x".into()).to_string(),
        "TypeError: x"
    );
    assert_eq!(
        ParsecError::ShapeError("x".into()).to_string(),
        "ShapeError: x"
    );
    assert_eq!(ParsecError::ArgError("x".into()).to_string(), "ArgError: x");
    assert_eq!(
        ParsecError::EmptyCollection("x".into()).to_string(),
        "EmptyCollection: x"
    );
    assert_eq!(ParsecError::Cancelled.to_string(), "Cancelled");
    assert_eq!(ParsecError::ChannelClosed.to_string(), "ChannelClosed");
    assert_eq!(ParsecError::Internal("x".into()).to_string(), "Internal: x");
}

#[test]
fn task_handle_public_methods_cover_terminal_paths_why_external_waiters_must_observe_each_state() {
    let scalar = TaskResult::Scalar(3.5);
    assert_eq!(scalar.as_scalar().unwrap(), 3.5);
    assert!(matches!(
        scalar.as_buffer(),
        Err(ParsecError::TypeError(message)) if message.contains("Buffer")
    ));

    let buffer = TaskResult::Buffer(Buffer::from_f64_vec(vec![1.0]));
    assert_eq!(buffer.as_buffer().unwrap().as_f64_slice(), &[1.0]);
    assert!(matches!(
        buffer.as_scalar(),
        Err(ParsecError::TypeError(message)) if message.contains("Scalar")
    ));

    let created = TaskHandle::new();
    assert_eq!(created.state(), TaskState::Created);
    assert!(!created.is_done());

    let completed = TaskHandle::new();
    completed.set_running_compute();
    completed.complete(TaskResult::Scalar(1.0));
    assert_eq!(completed.state(), TaskState::Completed);
    assert!(completed.is_done());
    assert_eq!(completed.result().unwrap().as_scalar().unwrap(), 1.0);

    let failed = TaskHandle::new();
    failed.fail(ParsecError::Internal("boom".into()));
    assert!(failed.is_done());
    assert_eq!(
        failed.result().unwrap_err(),
        ParsecError::Internal("boom".into())
    );

    let cancelled = TaskHandle::new();
    assert!(cancelled.cancel());
    assert!(!cancelled.cancel());
    assert_eq!(cancelled.result().unwrap_err(), ParsecError::Cancelled);
}

#[test]
fn channel_select_kernel_and_pool_cover_public_entrypoints_why_non_python_api_surfaces_should_still_be_measured(
) {
    let recv_closed_channel = Channel::new(1);
    recv_closed_channel.close();
    assert_eq!(
        recv_closed_channel.recv().unwrap_err(),
        ParsecError::ChannelClosed
    );
    assert!(recv_closed_channel.try_recv().unwrap().is_none());
    assert_eq!(recv_closed_channel.receiver().len(), 0);

    let try_recv_closed_channel = Channel::new(1);
    try_recv_closed_channel.close();
    assert_eq!(
        try_recv_closed_channel.try_recv().unwrap_err(),
        ParsecError::ChannelClosed
    );

    let default_channel = Channel::new(1);
    assert!(matches!(
        select_channels(&[&default_channel], true).unwrap(),
        SelectResult::Default
    ));

    let closed_select = Channel::new(1);
    closed_select.close();
    assert!(matches!(
        select_channels(&[&closed_select], true),
        Err(ParsecError::ChannelClosed)
    ));

    let spec = KernelSpec::elementwise(
        vec![ArgSpec {
            name: "x".to_string(),
            dtype: DType::F64,
            is_scalar: false,
        }],
        Expr::ArgRef(0),
    );
    assert_eq!(spec.kind, KernelKind::Elementwise);
    assert_eq!(spec.output, TensorSpec { dtype: DType::F64 });

    init_pool(0);
}

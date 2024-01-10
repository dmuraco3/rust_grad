use std::time::Instant;

use metal::{objc::rc::autoreleasepool, Device};

use crate::{
    devices::metal::MetalGPU,
    dtypes::Unit,
    shape::Shape,
    tensor::{
        tape::{Gradients, UniqueID},
        tensor_ops::add::AddKernel,
        Tensor,
    },
};

const LIB_DATA: &[u8] = include_bytes!("add.metallib");

const SHADER_NAME: &str = "add_matrices";

pub struct MetalState {
    pub queue: metal::CommandQueue,
    pub pipeline: metal::ComputePipelineState,
}

impl MetalState {
    pub fn new_with_shader(device: &Device, library_data: &[u8], shader_name: &str) -> Self {
        let queue = device.new_command_queue();
        let lib = device.new_library_with_data(library_data).unwrap();
        let function = lib.get_function(shader_name, None).unwrap();
        
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .unwrap();

        Self { queue, pipeline }
    }
}

impl<E: Unit> AddKernel<E> for MetalGPU {
    fn forward<S: Shape>(
        &self,
        lhs: &Tensor<S, E, Self>,
        rhs: &Tensor<S, E, Self>,
        out: &mut Tensor<S, E, Self>,
    ) -> Result<(), Self::Err> {
        let i = lhs.shape.concrete().into_iter().nth(0).unwrap();
        let j = rhs.shape.concrete().into_iter().nth(0).unwrap();

        let lhs_buffer = &lhs.data.read().unwrap().buf;
        let rhs_buffer = &rhs.data.read().unwrap().buf;
        let res_buffer = &out.data.write().unwrap().buf;

        autoreleasepool(|| {
            let state = MetalState::new_with_shader(&self.device, LIB_DATA, SHADER_NAME);

            let command_buffer = state.queue.new_command_buffer();
            let compute_encoder = command_buffer.new_compute_command_encoder();
            compute_encoder.set_compute_pipeline_state(&state.pipeline);
            compute_encoder.set_buffers(
                0,
                &[Some(lhs_buffer), Some(rhs_buffer), Some(res_buffer)],
                &[0; 3],
            );

            let i = i as u64;
            let j = j as u64;

            let w = state.pipeline.thread_execution_width();
            let h = state.pipeline.max_total_threads_per_threadgroup() / w;
            let grid_size = metal::MTLSize::new(i, j, 1);
            let threadgroup_size = metal::MTLSize::new(w, h, 1);

            compute_encoder.dispatch_threads(grid_size, threadgroup_size);

            // end encoding and execute commands
            compute_encoder.end_encoding();
            let start = Instant::now();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let elapsed = start.elapsed();
            println!("gpu time {:?}", elapsed);
        });

        return Ok(());
    }

    fn backward<S: crate::shape::Shape>(
        &self,
        grads: &mut Gradients<E, Self>,
        lhs_id: &UniqueID,
        rhs_id: &crate::tensor::tape::UniqueID,
        out_id: &UniqueID,
    ) -> Result<(), Self::Err> {
        todo!()
    }
}

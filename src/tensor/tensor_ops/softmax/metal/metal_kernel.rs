use std::{time::Instant, mem::size_of};

use metal::{objc::rc::autoreleasepool, ComputePipelineState, MTLResourceOptions};

use crate::{
    devices::metal::{MetalGPU, MetalVec, MetalState},
    dtypes::{FloatUnit, Unit},
    shape::{Rank0, Shape, Rank1},
    tensor::{tape::{self, UniqueID, Gradients}, tensor_ops::{cross_entropy::CrossEntropyKernel, softmax::SoftmaxKernel}, Tensor, ZerosTensor},
};

const LIB_DATA: &[u8] = include_bytes!("softmax.metallib");

const SHADER_NAME: &str = "softmax";


impl<E: FloatUnit> SoftmaxKernel<E> for MetalGPU {
    fn forward<S: Shape>(
        &self,
        src: &Tensor<S, E, Self>,
        out: &mut Tensor<S, E, Self>,
    ) -> Result<(), Self::Err> {
        let numel = src.shape.num_elements() as u64;

        let src_buffer = &src.data.read().unwrap().buf;
        let out_buffer = &out.data.write().unwrap().buf;

        let exp_sum_buffer = src.device.device.new_buffer(size_of::<E>() as u64, MTLResourceOptions::StorageModeShared);

        autoreleasepool(|| {
            let state = MetalState::new_with_shader(&self.device, LIB_DATA, SHADER_NAME);

            let command_buffer = state.queue.new_command_buffer();
            let compute_encoder = command_buffer.new_compute_command_encoder();
            compute_encoder.set_compute_pipeline_state(&state.pipeline);
            compute_encoder.set_buffer(0, Some(src_buffer), 0);
            compute_encoder.set_buffer(1, Some(out_buffer), 0);
            compute_encoder.set_buffer(2, Some(&exp_sum_buffer), 0);

            let grid_size = metal::MTLSize::new(numel, 1, 1);

            let mut thread_group_size = state.pipeline.max_total_threads_per_threadgroup();
            if thread_group_size > numel {
                thread_group_size = numel;
            }
            let threadgroup_size = metal::MTLSize::new(thread_group_size, 1, 1);

            compute_encoder.dispatch_threads(grid_size, threadgroup_size);

            compute_encoder.end_encoding();
            let start = Instant::now();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let elapsed = start.elapsed();

            println!("done doing softmax on da GPU! time taken: {:?}", elapsed);

            
        });

        Ok(())
    }

    fn backward<S: Shape>(
        &self,
        out_id: &UniqueID,
        grads: &mut Gradients<E, Self>,
    ) -> Result<(), Self::Err> {
        todo!()
    }
}
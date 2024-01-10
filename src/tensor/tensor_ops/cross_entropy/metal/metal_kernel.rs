use metal::{objc::rc::autoreleasepool, ComputePipelineState};

use crate::{
    devices::metal::MetalGPU,
    dtypes::{FloatUnit, Unit},
    shape::{Rank0, Shape},
    tensor::{tape, tensor_ops::{cross_entropy::CrossEntropyKernel, add::metal::metal_kernel::MetalState}, Tensor},
};

const LIB_DATA: &[u8] = include_bytes!("cross_entropy.metallib");

const SHADER_NAME: &str = "cross_entropy";


impl<E: FloatUnit> CrossEntropyKernel<E> for MetalGPU {
    fn forward<S: Shape>(
        &self,
        src: &Tensor<S, E, Self>,
        labels: &Tensor<S, E, Self>,
        out: &mut Tensor<Rank0, E, Self>,
    ) -> Result<(), Self::Err> {
        let numel = src.shape.num_elements();

        let src_buffer = &src.data.read().unwrap().buf;
        let lab_buffer = &labels.data.read().unwrap().buf;
        let out_buffer = &out.data.write().unwrap().buf;

        autoreleasepool(|| {
            let state = MetalState::new_with_shader(&self.device, LIB_DATA, SHADER_NAME);

            let command_buffer = state.queue.new_command_buffer();
            let compute_encoder = command_buffer.new_compute_command_encoder();
            compute_encoder.set_compute_pipeline_state(&state.pipeline);
            compute_encoder.set_buffers(
                0,
                &[Some(src_buffer), Some(lab_buffer), Some(out_buffer)],
                &[0; 3],
            );

            let (grid_size, threadgroup_size) = (|p: &ComputePipelineState| {
                let sz = (p.thread_execution_width(), p.max_total_threads_per_threadgroup());
                (
                    metal::MTLSize::new(numel as u64, 1, 1),
                    metal::MTLSize::new(sz.0, sz.1, 1),
                )
            })(&state.pipeline);

            compute_encoder.dispatch_threads(grid_size, threadgroup_size);
            compute_encoder.end_encoding();
            
            command_buffer.commit();
            command_buffer.wait_until_completed();
            println!("done doing cross_entropy on da GPU");
            
        });

        Ok(())
    }

    fn backward<S: Shape>(
        &self,
        src: &Tensor<S, E, Self>,
        labels: &Tensor<S, E, Self>,
        src_id: tape::UniqueID,
        out_id: tape::UniqueID,
        grads: &mut tape::Gradients<E, Self>,
    ) -> Result<(), Self::Err> {
        todo!()
    }
}

use std::time::Instant;

use metal::objc::rc::autoreleasepool;

use crate::{dtypes::Unit, shape::Dim, devices::metal::MetalGPU, tensor::{Tensor, tape::Gradients, ZerosTensor}};

use super::super::MatMatKernel;

const LIB_DATA: &[u8] = include_bytes!("matmul.metallib");

struct MetalState {
    pub queue: metal::CommandQueue,
    pub pipeline: metal::ComputePipelineState,
}

impl <E: Unit> MatMatKernel<E> for MetalGPU {
    fn forward<I: Dim, J: Dim, K: Dim>(
        &self, 
        lhs: &Tensor<(I,J), E, Self>,
        rhs: &Tensor<(J,K), E, Self>,
    ) -> Result<Tensor<(I,K), E, Self>, Self::Err> {
        let (i, j) = lhs.shape;
        let k = rhs.shape.1;

        let result: Tensor<(I, K), E, MetalGPU> = self.try_zeros_from(&(i, k)).unwrap();

        
        let lhs_buffer = &lhs.data.read().unwrap().buf;
        let rhs_buffer = &rhs.data.read().unwrap().buf;
        let res_buffer = &result.data.write().unwrap().buf;

        autoreleasepool(|| {
            let queue = self.device.new_command_queue();
    
            let lib = self.device.new_library_with_data(LIB_DATA).unwrap();
            
    
            let function = lib.get_function("mul_matrices", None).unwrap();
            
            let pipeline = self.device.new_compute_pipeline_state_with_function(&function).unwrap();
    
            let state = MetalState {
                queue,
                pipeline,
            };
    
            let command_buffer = state.queue.new_command_buffer();
            let compute_encoder = command_buffer.new_compute_command_encoder();
            compute_encoder.set_compute_pipeline_state(&state.pipeline);
            compute_encoder.set_buffers(
                0,
                &[Some(lhs_buffer), Some(rhs_buffer), Some(res_buffer)],
                &[0;3],
            );

            let i = i.size() as u64;
            let j = j.size() as u64;
            let k = k.size() as u64;

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
        
        return Ok(result.clone())
    }

    fn backward<I: Dim, J: Dim, K:Dim>(
        &self,
        lhs: &Tensor<(I,J), E, Self>,
        rhs: &Tensor<(J,K), E, Self>,
        grads: &mut Gradients<E, Self>,
        out: &Tensor<(I, K), E, Self>
    ) -> Result<(), Self::Err> {
        todo!()
    }
}
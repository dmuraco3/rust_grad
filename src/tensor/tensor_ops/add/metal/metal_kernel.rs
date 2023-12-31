use std::time::Instant;

use metal::objc::rc::autoreleasepool;

use crate::{tensor::{tensor_ops::add::AddKernel, Tensor, tape::{UniqueID, Gradients}}, devices::metal::MetalGPU, dtypes::Unit, shape::Shape};

const LIB_DATA: &[u8] = include_bytes!("add.metallib");

const SHADER_NAME: &str = "add_matrices";

struct MetalState {
    pub queue: metal::CommandQueue,
    pub pipeline: metal::ComputePipelineState,
}

impl <E: Unit> AddKernel<E> for MetalGPU {
    fn forward<S: Shape>(&self, lhs: &Tensor<S, E, Self>, rhs: &Tensor<S, E, Self>, out: &mut Tensor<S, E, Self>) -> Result<(), Self::Err> {
        let i = lhs.shape.concrete().into_iter().nth(0).unwrap();
        let j = rhs.shape.concrete().into_iter().nth(0).unwrap();

        
        let lhs_buffer = &lhs.data.read().unwrap().buf;
        let rhs_buffer = &rhs.data.read().unwrap().buf;
        let res_buffer = &out.data.write().unwrap().buf;

        autoreleasepool(|| {
            let queue = self.device.new_command_queue();
    
            let lib = self.device.new_library_with_data(LIB_DATA).unwrap();
            
    
            let function = lib.get_function(SHADER_NAME, None).unwrap();
            
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
        
        return Ok(())
    }

    fn backward<S: crate::shape::Shape>(&self, grads: &mut Gradients<E, Self>, lhs_id: &UniqueID, rhs_id: &crate::tensor::tape::UniqueID, out_id: &UniqueID) -> Result<(), Self::Err> {
        todo!()
    }
}
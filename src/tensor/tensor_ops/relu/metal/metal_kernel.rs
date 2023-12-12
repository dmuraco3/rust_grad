use std::time::Instant;

use metal::objc::rc::autoreleasepool;

use crate::{dtypes::Unit, tensor::{tensor_ops::relu::ReLUKernel, Tensor}, devices::metal::MetalGPU, shape::Shape};

const LIB_DATA: &[u8] = include_bytes!("relu.metallib");

const SHADER_NAME: &str = "relu";

struct MetalState {
    pub queue: metal::CommandQueue,
    pub pipeline: metal::ComputePipelineState,
}

impl <E: Unit> ReLUKernel<E> for MetalGPU {
    fn forward<S: Shape>(&self, src: &Tensor<S, E, Self>, out:&mut Tensor<S,E,Self>) -> Result<(), Self::Err> {
        let src_buffer = &src.data.read().unwrap().buf;
        let out_buffer = &out.data.write().unwrap().buf;

        let  i = src.shape.concrete().into_iter().nth(0).unwrap() as u64;

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
                &[Some(src_buffer), Some(out_buffer)],
                &[0;2],
            );

            let w = state.pipeline.thread_execution_width();
            let h = state.pipeline.max_total_threads_per_threadgroup() / w;
            let grid_size = metal::MTLSize::new(i, 1, 1);
            let threadgroup_size = metal::MTLSize::new(w, h, 1);
            
            compute_encoder.dispatch_threads(grid_size, threadgroup_size);

            compute_encoder.end_encoding();
            let start = Instant::now();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            let elapsed = start.elapsed();
            println!("gpu time {:?}", elapsed);
        });


        return Ok(())
    }

    fn backward<S: Shape>(&self, out: &Tensor<S, E, Self>, src_grad: &mut Self::Vec, out_grad: &Self::Vec) -> Result<(), Self::Err> {
        todo!()
    }
}
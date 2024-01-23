use std::{mem::size_of, sync::Arc, time::Instant};

use metal::{objc::rc::autoreleasepool, ComputePipelineState, MTLResourceOptions};

use crate::{
    devices::metal::{MetalGPU, MetalState},
    dtypes::{FloatUnit, Unit},
    shape::{Rank0, Shape},
    tensor::{
        tape,
        tensor_ops::cross_entropy::{self, CrossEntropyKernel},
        Tensor,
    },
};

const CROSS_ENTROPY_LIB_DATA: &[u8] = include_bytes!("cross_entropy.metallib");
const SOFTMAX_LIB_DATA: &[u8] = include_bytes!("../../softmax/metal/softmax.metallib");

const CROSS_ENTROPY_SHADER_NAME: &str = "cross_entropy";
const CROSS_ENTROPY_BACKWARD_SHADER_NAME: &str = "cross_entropy_backward";

const EXP_SUM_SHADER_NAME: &str = "exp_sum";
const SOFTMAX_SHADER_NAME: &str = "softmax";

impl<E: FloatUnit> CrossEntropyKernel<E> for MetalGPU {
    fn forward<S: Shape>(
        &self,
        src: &Tensor<S, E, Self>,
        labels: &Tensor<S, E, Self>,
        out: &mut Tensor<Rank0, E, Self>,
    ) -> Result<(), Self::Err> {
        let src_buffer = &src.data.read().unwrap().buf;
        let labels_buffer = &labels.data.read().unwrap().buf;
        let out_buffer = &out.data.write().unwrap().buf;

        autoreleasepool(|| {
            let softmax_buffer = &self.device.new_buffer(
                (size_of::<E>() * src.shape.num_elements()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let exp_sum_buffer = &self
                .device
                .new_buffer(size_of::<E>() as u64, MTLResourceOptions::StorageModeShared);

            let queue = self.device.new_command_queue();
            let softmax_lib = self.device.new_library_with_data(SOFTMAX_LIB_DATA).unwrap();
            let crossentropy_lib = self
                .device
                .new_library_with_data(CROSS_ENTROPY_LIB_DATA)
                .unwrap();

            let exp_sum_function = softmax_lib.get_function(EXP_SUM_SHADER_NAME, None).unwrap();
            let softmax_function = softmax_lib.get_function(SOFTMAX_SHADER_NAME, None).unwrap();
            let crossentropy_function = crossentropy_lib
                .get_function(CROSS_ENTROPY_SHADER_NAME, None)
                .unwrap();

            let exp_sum_pipeline = self
                .device
                .new_compute_pipeline_state_with_function(&exp_sum_function)
                .unwrap();
            let softmax_pipeline = self
                .device
                .new_compute_pipeline_state_with_function(&softmax_function)
                .unwrap();
            let crossentropy_pipeline = self
                .device
                .new_compute_pipeline_state_with_function(&crossentropy_function)
                .unwrap();

            let command_buffer = queue.new_command_buffer();

            // create & commit exp_sum compute_encoder
            {
                let compute_encoder = command_buffer.new_compute_command_encoder();
                compute_encoder.set_compute_pipeline_state(&exp_sum_pipeline);
                compute_encoder.set_buffer(0, Some(src_buffer), 0);
                compute_encoder.set_buffer(1, Some(exp_sum_buffer), 0);

                let w = exp_sum_pipeline.thread_execution_width();
                let grid_size = metal::MTLSize::new(src.shape.num_elements() as u64, 1, 1);
                let threadgroup_size = metal::MTLSize::new(w, 1, 1);
                compute_encoder.dispatch_threads(grid_size, threadgroup_size);
                compute_encoder.end_encoding();
            }

            // create & commit softmax compute_encoder
            {
                let compute_encoder = command_buffer.new_compute_command_encoder();
                compute_encoder.set_compute_pipeline_state(&softmax_pipeline);
                compute_encoder.set_buffer(0, Some(src_buffer), 0);
                compute_encoder.set_buffer(1, Some(exp_sum_buffer), 0);
                compute_encoder.set_buffer(2, Some(softmax_buffer), 0);

                let w = softmax_pipeline.thread_execution_width();
                let grid_size = metal::MTLSize::new(src.shape.num_elements() as u64, 1, 1);
                let threadgroup_size = metal::MTLSize::new(w, 1, 1);
                compute_encoder.dispatch_threads(grid_size, threadgroup_size);
                compute_encoder.end_encoding();
            }

            // create & commit crossentropy compute_encoder
            {
                let compute_encoder = command_buffer.new_compute_command_encoder();
                compute_encoder.set_compute_pipeline_state(&crossentropy_pipeline);
                compute_encoder.set_buffer(0, Some(softmax_buffer), 0);
                compute_encoder.set_buffer(1, Some(labels_buffer), 0);
                compute_encoder.set_buffer(2, Some(out_buffer), 0);

                let w = crossentropy_pipeline.thread_execution_width();
                let grid_size = metal::MTLSize::new(src.shape.num_elements() as u64, 1, 1);
                let threadgroup_size = metal::MTLSize::new(w, 1, 1);
                compute_encoder.dispatch_threads(grid_size, threadgroup_size);
                compute_encoder.end_encoding();
            }

            command_buffer.commit();
            let start = Instant::now();
            command_buffer.wait_until_completed();
            let elapsed = start.elapsed();
            println!("time to do cross_entropy on GPU: {:?}", elapsed);
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
        let src_arc = Arc::clone(&src.data);
        let src_data = src_arc.read().unwrap();

        let labels_arc = Arc::clone(&labels.data);
        let labels_data = labels_arc.read().unwrap();

        // let out_grad = grads.get_grad_ref(&out_id).clone();
        let src_grad = grads.get_grad_mut(&src_id);

        let src_buffer = &src_data.buf;
        let labels_buffer = &labels_data.buf;
        let src_grad_buffer = &src_grad.buf;
        // let out_grad_buffer = &out_grad.buf;

        autoreleasepool(|| {
            self.call_kernel(
                CROSS_ENTROPY_LIB_DATA,
                CROSS_ENTROPY_BACKWARD_SHADER_NAME,
                &[
                    src_buffer,
                    labels_buffer,
                    src_grad_buffer,
                ],
                src.shape.clone(),
            )
        })
    }
}

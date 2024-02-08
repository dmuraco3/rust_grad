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
const SHADER_BACKWARD_NAME: &str = "add_matrices_backward";

pub struct MetalState {
    pub queue: metal::CommandQueue,
    pub pipeline: metal::ComputePipelineState,
}

impl<E: Unit> AddKernel<E> for MetalGPU {
    fn forward<S: Shape>(
        &self,
        lhs: &Tensor<S, E, Self>,
        rhs: &Tensor<S, E, Self>,
        out: &mut Tensor<S, E, Self>,
    ) -> Result<(), Self::Err> {
        let shape = lhs.shape.clone();

        let mut shape_iter = shape.concrete().into_iter();

        let shape = (
            shape_iter.next().unwrap_or(1),
            shape_iter.next().unwrap_or(1),
            shape_iter.next().unwrap_or(1),
        );

        let buffers = &[
            &lhs.data.read().unwrap().buf,
            &rhs.data.read().unwrap().buf,
            &out.data.write().unwrap().buf,
        ];

        self.call_kernel(LIB_DATA, SHADER_NAME, buffers, shape)
    }

    fn backward<S: crate::shape::Shape>(
        &self,
        grads: &mut Gradients<E, Self>,
        lhs_id: &UniqueID,
        rhs_id: &UniqueID,
        out_id: &UniqueID,
    ) -> Result<(), Self::Err> {
        let out_grad = grads.get_grad_ref(&out_id).to_owned();
        let out_grad_buf = &out_grad.buf;

        let left_grad = grads.get_grad_mut(&lhs_id);
        let left_grad_buf = &left_grad.buf.clone();

        let right_grad = grads.get_grad_mut(&rhs_id);

        let right_grad_buf = &right_grad.buf.clone();

        let buffers = &[out_grad_buf, left_grad_buf, right_grad_buf];

        self.call_kernel(
            LIB_DATA,
            SHADER_BACKWARD_NAME,
            buffers,
            (out_grad.len, 1, 1),
        )
    }
}

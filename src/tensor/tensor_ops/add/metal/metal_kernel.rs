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



impl<E: Unit> AddKernel<E> for MetalGPU {
    fn forward<S: Shape>(
        &self,
        lhs: &Tensor<S, E, Self>,
        rhs: &Tensor<S, E, Self>,
        out: &mut Tensor<S, E, Self>,
    ) -> Result<(), Self::Err> {
        let buffers = &[
            &lhs.data.read().unwrap().buf,
            &rhs.data.read().unwrap().buf,
            &out.data.write().unwrap().buf,
        ];

        self.call_kernel(LIB_DATA, SHADER_NAME, buffers, lhs.shape)
    }

    fn backward<S: crate::shape::Shape>(
        &self,
        grads: &mut Gradients<E, Self>,
        lhs_id: &UniqueID,
        rhs_id: &UniqueID,
        out_id: &UniqueID,
    ) -> Result<(), Self::Err> {
        todo!()
    }
}

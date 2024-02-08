use crate::{
    devices::metal::MetalGPU,
    dtypes::Unit,
    shape::Shape,
    tensor::{tensor_ops::relu::ReLUKernel, Tensor},
};

const LIB_DATA: &[u8] = include_bytes!("relu.metallib");

const SHADER_NAME: &str = "relu";

impl<E: Unit> ReLUKernel<E> for MetalGPU {
    fn forward<S: Shape>(
        &self,
        src: &Tensor<S, E, Self>,
        out: &mut Tensor<S, E, Self>,
    ) -> Result<(), Self::Err> {
        let src_buffer = &src.data.read().unwrap().buf;
        let out_buffer = &out.data.write().unwrap().buf;

        let i = src.shape.num_elements();

        let buffers = &[src_buffer, out_buffer];

        self.call_kernel(LIB_DATA, SHADER_NAME, buffers, (i, 1, 1))?;

        return Ok(());
    }

    fn backward<S: Shape>(
        &self,
        out: &Tensor<S, E, Self>,
        src_grad: &mut Self::Vec,
        out_grad: &Self::Vec,
    ) -> Result<(), Self::Err> {
        let i = out.shape.num_elements();

        let out_data = out.data.read().unwrap();
        let out_buf = &out_data.buf;

        let out_grad = &out_grad.buf;
        let src_grad = &src_grad.buf;

        let buffers = &[out_buf, out_grad, src_grad];

        self.call_kernel(LIB_DATA, "relu_backward", buffers, (i, 1, 1))?;

        Ok(())
    }
}

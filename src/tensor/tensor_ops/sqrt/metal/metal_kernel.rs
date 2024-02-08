use crate::{
    devices::metal::MetalGPU,
    dtypes::FloatUnit,
    shape::Shape,
    tensor::{tensor_ops::sqrt::SqrtKernel, Tensor},
};

impl<E: FloatUnit> SqrtKernel<E> for MetalGPU {
    fn forward<S: Shape>(
        &self,
        src: &Tensor<S, E, Self>,
        out: &mut Tensor<S, E, Self>,
    ) -> Result<(), Self::Err> {
        let src_inner = src.data.read().unwrap();

        let mut out_inner = out.data.write().unwrap();

        for i in 0..src.shape.num_elements() {
            out_inner[i] = src_inner[i].sqrt()
        }

        Ok(())
    }

    #[allow(unused_variables)]
    fn backward<S: Shape>(
        &self,
        src: &Tensor<S, E, Self>,
        src_grad: &mut Self::Vec,
        out_grad: &Self::Vec,
    ) -> Result<(), Self::Err> {
        todo!()
    }
}

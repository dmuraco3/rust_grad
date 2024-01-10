use crate::{
    devices::cpu::CPU,
    dtypes::FloatUnit,
    shape::Shape,
    tensor::{tensor_ops::softmax::SoftmaxKernel, Tensor},
};

impl<E: FloatUnit> SoftmaxKernel<E> for CPU {
    fn forward<S: Shape>(
        &self,
        src: &Tensor<S, E, Self>,
        out: &mut Tensor<S, E, Self>,
    ) -> Result<(), Self::Err> {
        let src_inner = src.data.read().unwrap();
        let mut out_inner = out.data.write().unwrap();

        let divisor = src_inner.iter().fold(E::ZERO, |acc, &x| acc + x.exp());

        src_inner
            .iter()
            .zip(out_inner.iter_mut())
            .for_each(|(src, out)| {
                *out = src.exp() / divisor;
            });

        Ok(())
    }

    fn backward<S: Shape>(
        &self,
        out_id: &crate::tensor::tape::UniqueID,
        grads: &mut crate::tensor::tape::Gradients<E, Self>,
    ) -> Result<(), Self::Err> {
        todo!()
    }
}

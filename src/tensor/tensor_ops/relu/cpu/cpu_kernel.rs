use crate::{devices::cpu::CPU, dtypes::Unit, shape::Shape, tensor::Tensor};

use crate::tensor::tensor_ops::relu::ReLUKernel;

impl<E: Unit> ReLUKernel<E> for CPU {
    fn forward<S: Shape>(
        &self,
        src: &Tensor<S, E, Self>,
        out: &mut Tensor<S, E, Self>,
    ) -> Result<(), Self::Err> {
        let src_inner = src.data.read().unwrap();
        let mut out_inner = out.data.write().unwrap();

        src_inner
            .iter()
            .zip(out_inner.iter_mut())
            .for_each(|(lh, rh)| *rh = lh.max(*rh));

        Ok(())
    }

    fn backward<S: Shape>(
        &self,
        out: &Tensor<S, E, Self>,
        src_grad: &mut Self::Vec,
        out_grad: &Self::Vec,
    ) -> Result<(), Self::Err> {
        let out_data = out.data.read().unwrap();

        for i in 0..out.shape.num_elements() {
            if out_data[i] > E::ZERO {
                src_grad[i] = out_grad[i];
            }
        }

        Ok(())
    }
}

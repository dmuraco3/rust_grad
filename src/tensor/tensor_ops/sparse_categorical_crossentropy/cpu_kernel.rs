use crate::{dtypes::FloatUnit, devices::cpu::CPU, shape::Shape, tensor::{Tensor, tape::Gradients}};

use super::SCCEKernel;

impl <E: FloatUnit> SCCEKernel<E> for CPU {
    fn forward<S: Shape, OutShape: Shape>(src: &Tensor<S, E, Self>, actual: &Tensor<S, E, Self>, out: &mut Tensor<OutShape, E, Self>) -> Result<(), Self::Err> {


        let loss = src.data.read().unwrap().iter().zip(actual.data.read().unwrap().iter()).fold(E::ZERO, |acc, (&p, &t): (&E, &E)| acc + t * p.log_10());
        
        out.data.write().unwrap()[0] = loss;

        Ok(())
    }

    fn backward<S: Shape>(
        &self,
        src: &Tensor<S, E, Self>,
        actual: &Tensor<S, E, Self>,
        _out_grad: &Self::Vec,
        grads: &mut Gradients<E, Self>,
    ) -> Result<(), Self::Err> {
        // out = src - actual

        let src_grad = grads.get_grad_mut(&src.id);

        let src_data = src.data.read().unwrap();
        let actual_data = actual.data.read().unwrap();

        src_grad.iter_mut()
                .zip(src_data.iter())
                .zip(actual_data.iter())
                .for_each(|((grad, src), actual)| {
                    *grad = *src - *actual
                });


        Ok(())
    }
}
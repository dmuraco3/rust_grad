use crate::{dtypes::{Unit, FloatUnit}, devices::cpu::CPU, tensor::{Tensor, tape::{UniqueID, Gradients}}, shape::{Rank0, Shape}};

use super::CrossEntropyKernel;

impl <E: Unit + FloatUnit> CrossEntropyKernel<E> for CPU {
    fn forward<S: Shape>(
        src: &Tensor<S, E, Self>,
        labels: &Tensor<S, E, Self>,
        out: &mut Tensor<Rank0, E, Self>,
    ) -> Result<(), Self::Err> {
        let src_data = src.data.read().unwrap();
        let labels_data = labels.data.read().unwrap();
        let mut out_data = out.data.write().unwrap();
        
        out_data[0] = src_data.iter().zip(labels_data.iter()).fold(E::ZERO, |acc, (src_ele, label)| {
            acc + (*label * src_ele.ln())
        });

        out_data[0] *= -E::ONE;
        
        Ok(())
    }

    fn backward<S: crate::shape::Shape>(
        src: &Tensor<S, E, Self>,
        labels: &Tensor<S, E, Self>,
        src_id: UniqueID,
        out_id: UniqueID,
        grads: &mut Gradients<E, Self>
    ) -> Result<(), Self::Err> {
        let src_data = src.data.read().unwrap();
        let labels_data = labels.data.read().unwrap();

        let out_grad = grads.get_grad_ref(&out_id).to_owned();
        let src_grad = grads.get_grad_mut(&src_id);

        src_grad.iter_mut().zip(src_data.iter().zip(labels_data.iter())).for_each(|(grad, (src_ele, label))| {
            *grad = *src_ele - *label
        });

        Ok(())
    }
}
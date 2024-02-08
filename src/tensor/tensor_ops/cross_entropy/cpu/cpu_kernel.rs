use crate::{
    devices::cpu::CPU,
    dtypes::{FloatUnit, Unit},
    shape::{self, Rank0, Shape},
    tensor::{
        tape::{Gradients, UniqueID},
        Tensor,
        tensor_ops::cross_entropy::CrossEntropyKernel
    },
};

impl<E: Unit + FloatUnit> CrossEntropyKernel<E> for CPU {
    fn forward<S: Shape>(
        &self,
        src: &Tensor<S, E, Self>,
        labels: &Tensor<S, E, Self>,
        out: &mut Tensor<Rank0, E, Self>,
    ) -> Result<(), Self::Err> {
        let src_data = src.data.read().unwrap();
        let labels_data = labels.data.read().unwrap();
        let mut out_data = out.data.write().unwrap();

        out_data[0] = src_data
            .iter()
            .zip(labels_data.iter())
            .fold(E::ZERO, |acc, (src_ele, label)| {
                acc + (*label * (*src_ele).ln())
            });

        out_data[0] *= -E::ONE;

        Ok(())
    }

    fn backward<S: shape::Shape>(
        &self,
        src: &Tensor<S, E, Self>,
        labels: &Tensor<S, E, Self>,
        src_id: UniqueID,
        _out_id: UniqueID,
        grads: &mut Gradients<E, Self>,
    ) -> Result<(), Self::Err> {
        let src_data = src.data.read().unwrap();
        let labels_data = labels.data.read().unwrap();

        let src_grad = grads.get_grad_mut(&src_id);

        for i in 0..src_grad.len() {
            src_grad[i] = src_data[i] - labels_data[i];
        }
        // src_grad.iter_mut().zip(src_data.iter().zip(labels_data.iter())).for_each(|(grad, (src_ele, label))| {
        //     *grad = *src_ele - *label
        // });

        Ok(())
    }
}

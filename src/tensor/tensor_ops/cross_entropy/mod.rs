pub mod cpu_kernel;

use std::fmt::Debug;

use crate::{shape::{Storage, Rank0, Shape, ConstDim, Rank1}, dtypes::Unit, tensor::{Tensor, tape::{Gradients, UniqueID, Tape, Merge, SplitTape, PutTape}, ZerosTensor}};

use super::softmax::{TrySoftmax, SoftmaxKernel};

pub trait CrossEntropyKernel<E: Unit>: Storage<E> {
    
    /// # Arguments 
    /// 
    /// * `src`    - Tensor returned from softmax (saves operations)
    /// * `labels` - One-hot encoded vector 
    fn forward<S: Shape>(
        src: &Tensor<S, E, Self>,
        labels: &Tensor<S, E, Self>,
        out: &mut Tensor<Rank0, E, Self>,
    ) -> Result<(), Self::Err>;

    fn backward<S: Shape>(
        src: &Tensor<S, E, Self>,
        labels: &Tensor<S, E, Self>,
        src_id: UniqueID,
        out_id: UniqueID,
        grads: &mut Gradients<E, Self>
    ) -> Result<(), Self::Err>;
}

pub trait TryCrossEntropy<S: Shape, E: Unit, D: CrossEntropyKernel<E>, RTape: Tape<E, D>> {
    type Err: Debug;
    type Output;

    fn try_cross_entropy(self, labels: Tensor<S, E, D, RTape>) -> Result<Self::Output, Self::Err>;
}

impl <X, E, D, T, R> TryCrossEntropy<(X, ), E, D, R> for Tensor<(X, ), E, D, T>
where
    X: ConstDim,
    E: Unit,
    D: CrossEntropyKernel<E> + ZerosTensor<E> + SoftmaxKernel<E>,
    T: Tape<E, D> + Merge<R>,
    R: Tape<E, D>,
    Self: TrySoftmax<E>
{
    type Err = D::Err;

    type Output = Tensor<Rank0, E, D, T>;

    fn try_cross_entropy(self, labels: Tensor<(X, ), E, D, R>) -> Result<Self::Output, Self::Err> {
        let mut out = self.device.zeros();

        let (src, src_tape) = self.split_tape();
        let (labels, labels_tape) = labels.split_tape();

        let mut tape = src_tape.merge(labels_tape);

        let src_softmax = src.clone().try_softmax().unwrap();

        <D as CrossEntropyKernel<E>>::forward(&src_softmax, &labels, &mut out)?;

        let out_id = out.id.clone();

        tape.add_backward_op(move |grads| {
            grads.try_ones_for((&src.device, out_id, 1))?; // fill output grad
            grads.try_alloc_for((&src.device, src.id, src.shape.num_elements()))?;
            grads.try_alloc_for((&labels.device, labels.id, labels.shape.num_elements()))?;

            D::backward(&src_softmax, &labels, src.id.clone(), out_id, grads)?;

            Ok(())
        });

        Ok(out.put_tape(tape))
    }
}
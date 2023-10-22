use std::{fmt::Debug, process::Output};

use crate::{shape::{Shape, Storage, Dim, ConstDim, Const}, dtypes::{FloatUnit, Unit}, tensor::{Tensor, ZerosTensor, tape::{Tape, SplitTape, Merge, Gradients, PutTape}}};

pub mod cpu_kernel;

pub trait SCCEKernel<E: Unit>: Storage<E> {
    fn forward<S, OutShape>(
        src: &Tensor<S, E, Self>,
        labels: &Tensor<S, E, Self>,
        out: &mut Tensor<OutShape, E, Self>
    ) -> Result<(), Self::Err>
    where
        S: Shape,
        OutShape: Shape
    ;

    fn backward<S: Shape>(
        &self,
        src: &Tensor<S, E, Self>,
        actual: &Tensor<S, E, Self>,
        out_grad: &Self::Vec,
        grads: &mut Gradients<E, Self>,
    ) -> Result<(), Self::Err>;
}

pub trait TrySparseCategoricalCrossentropy<S: Shape, E: Unit, D: SCCEKernel<E>, R: Tape<E, D>> {
    type Error: Debug;

    type Output;

    fn try_sparse_categorical_crossentropy(self, actual: Tensor<S, E, D, R>) -> Result<Self::Output, Self::Error>;
}

impl <X, E, D, T, R> TrySparseCategoricalCrossentropy<(X,), E, D, R> for Tensor<(X,), E, D, T>
where
    X: ConstDim,
    E: Unit,
    D: SCCEKernel<E> + ZerosTensor<E>,
    T: Tape<E, D> + Merge<R>,
    R: Tape<E, D> 
{
    type Error = D::Err;

    type Output = Tensor<(Const<1>, ), E, D, T>;

    fn try_sparse_categorical_crossentropy(self, actual: Tensor<(X,), E, D, R>) -> Result<Self::Output, Self::Error> {
        let mut out = self.device.zeros::<(Const<1>, )>();

        let (lhs, lhs_tape) = self.split_tape();
        let (rhs, rhs_tape) = actual.split_tape();

        let mut tape = lhs_tape.merge(rhs_tape);

        D::forward(&lhs, &rhs, &mut out)?;

        let out_d = (out.id.clone(), out.shape.num_elements());

        tape.add_backward_op(move |grads| {
            grads.try_ones_for((&lhs.device, out_d.0, out_d.1))?;
            grads.try_alloc_for((&lhs.device, lhs.id, lhs.shape.num_elements()))?;
            grads.try_alloc_for((&rhs.device, rhs.id, rhs.shape.num_elements()))?;


            let out_grad = grads.get_grad_ref(&out.id).to_owned();

            lhs.device.backward(&lhs, &rhs, &out_grad, grads)?;
            Ok(())
        });

        Ok(out.put_tape(tape))
    }
}
pub mod cpu;
pub mod metal;

use crate::{
    dtypes::Unit,
    shape::Shape,
    storage::Storage,
    tensor::{
        tape::{Gradients, Merge, PutTape, SplitTape, Tape, UniqueID},
        HasErr, Tensor, ZerosTensor,
    },
};
use std::fmt::Debug;

pub trait AddKernel<E: Unit>: Storage<E> {
    fn forward<S: Shape>(
        &self,
        lhs: &Tensor<S, E, Self>,
        rhs: &Tensor<S, E, Self>,
        out: &mut Tensor<S, E, Self>,
    ) -> Result<(), Self::Err>;

    fn backward<S: Shape>(
        &self,
        grads: &mut Gradients<E, Self>,
        lhs_id: &UniqueID,
        rhs_id: &UniqueID,
        out_id: &UniqueID,
    ) -> Result<(), Self::Err>;
}

pub trait TryAdd<S: Shape, E: Unit, D: AddKernel<E>, R>: HasErr {
    type Error: Debug;
    type Output;

    fn try_add(self, rhs: Tensor<S, E, D, R>) -> Result<Self::Output, Self::Error>;

    fn add(self, rhs: Tensor<S, E, D, R>) -> Self::Output {
        self.try_add(rhs).unwrap()
    }
}

impl<S, E, D, T, R> TryAdd<S, E, D, R> for Tensor<S, E, D, T>
where
    S: Shape + 'static,
    E: Unit,
    D: AddKernel<E> + ZerosTensor<E>,
    T: Tape<E, D> + Merge<R>,
    R: Tape<E, D>,
{
    type Error = D::Err;

    type Output = Self;

    fn try_add(self, rhs: Tensor<S, E, D, R>) -> Result<Self::Output, Self::Error> {
        let mut out = self.device.try_zeros_from(&self.shape)?;

        let (lhs, lhs_tape) = self.split_tape();
        let (rhs, rhs_tape) = rhs.split_tape();

        lhs.device.forward(&lhs, &rhs, &mut out)?;

        let mut tape = lhs_tape.merge(rhs_tape);

        let out_id = out.id.clone();

        tape.add_backward_op(move |grads| {
            grads.try_alloc_raw(&lhs.device, &lhs.id, lhs.shape.num_elements())?;
            grads.try_alloc_raw(&rhs.device, &rhs.id, rhs.shape.num_elements())?;
            grads.try_ones_for((&lhs.device, out_id, lhs.shape.num_elements()))?;

            lhs.device.backward::<S>(grads, &lhs.id, &rhs.id, &out_id)?;

            Ok(())
        });

        Ok(out.put_tape(tape))
    }
}

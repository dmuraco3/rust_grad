pub mod cpu_kernel;

use std::fmt::Debug;

use crate::{dtypes::Unit, shape::{Storage, Shape, Rank0}, tensor::{Tensor, ZerosTensor, tape::{Tape, SplitTape, PutTape, OwnedTape}}};

pub trait SumKernel<E: Unit>: Storage<E> {
    fn forward<S: Shape>(
        &self,
        src: &Tensor<S, E, Self>,
        out: &mut Tensor<Rank0,E,Self>
    ) -> Result<(), Self::Err>;

    fn backward<S: Shape>(
        &self,
        src: &Tensor<S, E, Self>,
        src_grad: &mut Self::Vec,
        out_grad: &Self::Vec,
    ) -> Result<(), Self::Err>;
}

pub trait TrySum<S, E, D, T>
where
    S: Shape,
    E: Unit,
    D: SumKernel<E> 
{
    type Error: Debug;

    fn try_sum(self) -> Result<Tensor<Rank0, E, D, T>, Self::Error>;
}

impl <S, E, D, T> TrySum<S,E,D, T> for Tensor<S, E, D, T>
where
    S: Shape,
    E: Unit,
    D: SumKernel<E> + ZerosTensor<E>,
    T: Tape<E, D>
{
    type Error = D::Err;

    fn try_sum(self) -> Result<Tensor<Rank0, E, D, T>, Self::Error> {
        let mut out = self.device.zeros::<Rank0>();

        let (lhs, lhs_tape) = self.split_tape();

        lhs.device.forward::<S>(&lhs, &mut out)?;

        // lhs_tape.add_backward_op(move |grads| {

        // });

        Ok(out.put_tape(lhs_tape))  

    }
}
pub mod cpu_kernel;

use std::fmt::Debug;

use crate::{
    dtypes::Unit,
    shape::{Rank0, Shape},
    storage::Storage,
    tensor::{
        tape::{PutTape, SplitTape, Tape},
        Tensor, ZerosTensor,
    },
};

pub trait SumKernel<E: Unit>: Storage<E> {
    fn forward<S: Shape>(
        &self,
        src: &Tensor<S, E, Self>,
        out: &mut Tensor<Rank0, E, Self>,
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
    D: SumKernel<E>,
{
    type Error: Debug;

    fn try_sum(self) -> Result<Tensor<Rank0, E, D, T>, Self::Error>;
}

impl<S, E, D, T> TrySum<S, E, D, T> for Tensor<S, E, D, T>
where
    S: Shape + 'static,
    E: Unit,
    D: SumKernel<E> + ZerosTensor<E>,
    T: Tape<E, D>,
{
    type Error = D::Err;

    fn try_sum(self) -> Result<Tensor<Rank0, E, D, T>, Self::Error> {
        let mut out = self.device.zeros::<Rank0>();

        let (lhs, mut lhs_tape) = self.split_tape();

        lhs.device.forward::<S>(&lhs, &mut out)?;

        let out_d = (out.id.clone(), out.shape.num_elements());

        lhs_tape.add_backward_op(move |grads| {
            grads.try_alloc_for((&lhs.device, lhs.id, lhs.shape.num_elements()))?;
            grads.try_ones_for((&lhs.device, out_d.0, out_d.1))?;

            let out_grad = grads.get_grad_ref(&out_d.0).to_owned();

            let src_grad = grads.get_grad_mut(&lhs.id);

            lhs.device.backward(&lhs, src_grad, &out_grad)?;

            Ok(())
        });

        Ok(out.put_tape(lhs_tape))
    }
}

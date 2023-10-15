use std::collections::btree_map::Entry::Vacant;

use crate::{tensor::{HasErr, Tensor, self, tape::{Tape, Merge, SplitTape, PutTape, Gradients}, ZerosTensor}, shape::{Dim, Storage}, dtypes::Unit, devices::cpu::CPU};

use self::cpu_kernel::MatMatImpl;

pub mod cpu_kernel;

pub fn matmul<Lhs, Rhs>(lhs: Lhs, rhs: Rhs) -> Lhs::Output
where
    Lhs: TryMatMul<Rhs>
{
    lhs.matmul(rhs)
}

pub trait MatMatKernel<E: Unit>: Storage<E> {
    fn forward<I: Dim, J: Dim, K: Dim>(
        &self, 
        lhs: &Tensor<(I,J), E, Self>,
        rhs: &Tensor<(J,K), E, Self>,
    ) -> Result<Tensor<(I,K), E, Self>, Self::Err>;
    
    fn backward<I: Dim, J: Dim, K:Dim>(
        &self,
        lhs: &Tensor<(I,J), E, Self>,
        rhs: &Tensor<(J,K), E, Self>,
        grads: &mut Gradients<E, Self>,
        out: &Tensor<(I, K), E, Self>
        // lhs_grad: &mut Self::Vec,
        // rhs_grad: &mut Self::Vec,
        // out_grad: &Self::Vec,
    ) -> Result<(), Self::Err>;
}

pub trait MatVecKernel<E: Unit>: Storage<E> {
    fn forward<I: Dim, J: Dim>(
        lhs: &Tensor<(I, J), E, Self>,
        rhs: &Tensor<(J, ), E, Self>,
    ) -> Result<Tensor<(I,), E, Self>, Self::Err>;
}

pub trait TryMatMul<Rhs>: HasErr {
    type Output;
    fn matmul(self, rhs: Rhs) -> Self::Output {
        self.try_matmul(rhs).unwrap()
    }

    fn try_matmul(self, rhs: Rhs) -> Result<Self::Output, Self::Err>;

}

impl <I,J,K, E, D, T, R> TryMatMul<Tensor<(J, K), E, D, R,>> for Tensor<(I, J), E, D, T>
where 
    I: Dim,
    J: Dim,
    K: Dim,
    E: Unit,
    D: MatMatKernel<E> + MatMatImpl<E> + ZerosTensor<E>,
    T: Tape<E, D> + Merge<R>,
    R: Tape<E, D>
{
    type Output = Tensor<(I,K), E, D, T>;
    fn try_matmul(self, rhs: Tensor<(J, K), E, D, R>) -> Result<Self::Output, Self::Err> {
        assert_eq!(self.shape.1, rhs.shape.0);

        let (lhs, lhs_tape) = self.split_tape();
        let (rhs, rhs_tape) = rhs.split_tape();

        let mut tape = lhs_tape.merge(rhs_tape);

        let out = lhs.device.forward(&lhs, &rhs).unwrap();

        let out_clone = out.clone();
        // let lhs_d = (lhs.device.clone(), lhs.id.clone(), lhs.device.num_el(lhs.data.read().unwrap().to_owned()));
        // let rhs_d = (rhs.device.clone(), rhs.id.clone(), rhs.device.num_el(rhs.data.read().unwrap().to_owned()));
        // let out_d = (out.device.clone(), out.id.clone(), out.device.num_el(out.data.read().unwrap().to_owned()));


        tape.add_backward_op(move |grads| {
            // grads.try_alloc_for(lhs_d)?;
            // grads.try_alloc_for(rhs_d)?;
            // grads.try_alloc_for(out_d)?;

            // let mut lhs_grad = grads.get_grad_mut(&lhs.id);
            // let mut rhs_grad = grads.get_grad_mut(&rhs.id);
            // let out_grad = grads.get_grad_ref(&out.id);

            lhs.device.backward(&lhs, &rhs, grads, &out_clone)?;

            Ok(())
        });


        Ok(out.put_tape(tape))
    }


    
}

impl <I: Dim, J: Dim, E:Unit, D: MatVecKernel<E>> TryMatMul<Tensor<(J,), E, D>> for Tensor<(I,J), E, D> {
    type Output = Tensor<(I,), E, D>;

    fn try_matmul(self, rhs: Tensor<(J,), E, D>) -> Result<Self::Output, Self::Err> {
        D::forward(&self, &rhs)
    }
}
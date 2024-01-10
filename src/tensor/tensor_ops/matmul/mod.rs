use crate::{tensor::{HasErr, Tensor, tape::{Tape, Merge, SplitTape, PutTape, Gradients}, ZerosTensor}, shape::{Dim, Storage}, dtypes::Unit};

pub mod cpu;
pub mod metal;

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
    ) -> Result<(), Self::Err>;
}

pub trait MatVecKernel<E: Unit>: Storage<E> {
    fn forward<I: Dim, J: Dim>(
        lhs: &Tensor<(I, J), E, Self>,
        rhs: &Tensor<(J, ), E, Self>,
    ) -> Result<Tensor<(I,), E, Self>, Self::Err>;

    fn backward<I: Dim, J: Dim>(
        &self,
        lhs: &Tensor<(I, J), E, Self>,
        rhs: &Tensor<(J, ), E, Self>,
        grads: &mut Gradients<E, Self>,
        out: &Tensor<(I,), E, Self>,
    ) -> Result<(), Self::Err>;
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
    D: MatMatKernel<E> + ZerosTensor<E>,
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

impl <I, J, E, D, T, R> TryMatMul<Tensor<(J,), E, D, R>> for Tensor<(I,J), E, D, T>
where
    I: Dim,
    J: Dim,
    E: Unit,
    D: MatVecKernel<E>,
    T: Tape<E, D> + Merge<R>,
    R: Tape<E, D>
{
    type Output = Tensor<(I,), E, D, T>;

    fn try_matmul(self, rhs: Tensor<(J,), E, D, R>) -> Result<Self::Output, Self::Err> {

        let (lhs, lhs_tape) = self.split_tape();
        let (rhs, rhs_tape) = rhs.split_tape();

        let mut tape = lhs_tape.merge(rhs_tape);
        
        let out = D::forward(&lhs, &rhs)?;

        let out_clone = out.clone();

        tape.add_backward_op(move |grads| {
            lhs.device.backward(&lhs, &rhs, grads, &out_clone)?;

            Ok(())
        });

        Ok(out.put_tape(tape))

    }
}

impl <I, J, E, D, T, R> TryMatMul<Tensor<(I,J), E, D, R>> for Tensor<(J,), E, D, T>
where
    I: Dim,
    J: Dim,
    E: Unit,
    D: MatVecKernel<E>,
    T: Tape<E, D> + Merge<R>,
    R: Tape<E, D>
{
    type Output = Tensor<(I,), E, D, T>;

    fn try_matmul(self, rhs: Tensor<(I, J), E, D, R>) -> Result<Self::Output, Self::Err> {

        let (lhs, lhs_tape) = self.split_tape();
        let (rhs, rhs_tape) = rhs.split_tape();

        let mut tape = lhs_tape.merge(rhs_tape);
        
        let out = D::forward(&rhs, &lhs)?;

        let out_clone = out.clone();

        tape.add_backward_op(move |grads| {
            lhs.device.backward(&rhs, &lhs, grads, &out_clone)?;

            Ok(())
        });

        Ok(out.put_tape(tape))

    }
}

#[cfg(test)]
mod tests {
    use crate::{devices::cpu::CPU, tensor::{Tensor, Watch, tensor_ops::{softmax::TrySoftmax, utilities::backward::BackwardPropagate}}, shape::Const};

    use super::TryMatMul;

    #[test]
    fn test_mat_vec_mat() {
        let dev = CPU::default();

        let x: Tensor<(Const<5>,), f32, CPU> = dev.from_array([0.1, 0.2, 0.3, 0.4, 0.5]);

        let y: Tensor<(Const<5>, Const<5>), f32, CPU> = dev.from_2d_array([
            [0.1 , 0.2 , 0.3 , 0.4 , 0.5 ],
            [0.11, 0.21, 0.31, 0.41, 0.51],
            [0.12, 0.22, 0.32, 0.42, 0.52],
            [0.13, 0.23, 0.33, 0.43, 0.53],
            [0.14, 0.24, 0.34, 0.44, 0.54],
        ]);

        let res = y.watch_leaky().matmul(x.clone());
        let res_rev = x.watch_leaky().matmul(y.clone());

        assert!(res.allclose(&res_rev, None, None));

        let grad_res = res.softmax().backward();
        let grad_res_rev = res_rev.softmax().backward();

        // let grad_res = res.softmax().backward().get_grad_ref(&x.id);
        // let grad_res_rev = res_rev.softmax().backward().get_grad_ref(&x.id);

        assert_eq!(grad_res.get_grad_ref(&x.id), grad_res_rev.get_grad_ref(&x.id));


    }
}
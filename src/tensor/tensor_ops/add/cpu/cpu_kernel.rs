use crate::{tensor::{Tensor, tape::{Gradients, UniqueID}}, devices::cpu::CPU, dtypes::Unit};

use crate::tensor::tensor_ops::add::AddKernel;

impl <E: Unit> AddKernel<E> for CPU {
    fn forward<S: crate::shape::Shape>(&self, lhs: &Tensor<S, E, Self>, rhs: &Tensor<S, E, Self>, out: &mut Tensor<S, E, Self>) -> Result<(), Self::Err> {
        let lhs_data = lhs.data.read().unwrap();
        let rhs_data = rhs.data.read().unwrap();
        let mut out_data = out.data.write().unwrap();

        debug_assert_eq!(lhs_data.len(), rhs_data.len());
        debug_assert_eq!(lhs_data.len(), out_data.len());

        for i_i in 0..lhs_data.len() {
            out_data[i_i] = lhs_data[i_i] + rhs_data[i_i];
        }

        Ok(())
    }

    fn backward<S: crate::shape::Shape>(&self, grads: &mut Gradients<E, Self>, lhs_id: &UniqueID, rhs_id: &UniqueID, out_id: &UniqueID) -> Result<(), Self::Err> {
        let out_grad = grads.get_grad_ref(&out_id).to_owned();

        let left_grad = grads.get_grad_mut(&lhs_id);
        
        for ii in 0..out_grad.len() {
            left_grad[ii] = out_grad[ii];
        }
        
        let right_grad = grads.get_grad_mut(&rhs_id);

        for ii in 0..out_grad.len() {
            right_grad[ii] = out_grad[ii];
        }
        
        Ok(())
    }
}
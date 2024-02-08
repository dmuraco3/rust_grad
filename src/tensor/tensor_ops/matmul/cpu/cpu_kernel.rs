use std::sync::RwLockReadGuard;

use crate::{devices::cpu::CPU, dtypes::Unit, shape::{Dim, Shape}, storage::Storage, tensor::{tape::Gradients, Tensor, ZerosTensor}};

use crate::tensor::tensor_ops::matmul::{MatMatKernel, MatVecKernel};

pub trait MatMatImpl<E: Unit> {
    fn matmul<I:Dim, J:Dim, K:Dim>(
        dims     : (I,J,K),
        lh_data  : RwLockReadGuard<Vec<E>>,
        rh_data  : RwLockReadGuard<Vec<E>>,
        out_data : &mut Vec<E>,  
    );

    /// Used in backpropagation 
    /// output = lhs * rhs.T
    fn matmul_transpose<I:Dim, J:Dim, K:Dim>(
        dims     : (I,J,K),
        lh_data  : <CPU as Storage<E>>::Vec,
        rh_data  : <CPU as Storage<E>>::Vec,
        out_data : &mut <CPU as Storage<E>>::Vec,
        tranpose_sides: (bool, bool)
    );
}

pub trait MatVecImpl<E: Unit> {
    fn matmul<I: Dim, J: Dim>(
        dims: (I,J),
        lhs_data: RwLockReadGuard<Vec<E>>,
        rhs_data: RwLockReadGuard<Vec<E>>,
        out_data: &mut Vec<E>
    );
}


impl MatMatImpl<f32> for CPU {
    fn matmul<I:Dim, J:Dim, K:Dim>(
            dims      : (I,J,K),
            lh_data   : RwLockReadGuard<Vec<f32>>,
            rh_data   : RwLockReadGuard<Vec<f32>>,
            out_data  : &mut Vec<f32>,
    ){
        let (i,j,k) = dims;
        for i_i in 0..i.size() {
            for i_j in 0..j.size() {
                for i_k in 0..k.size() {
                    // lh_stride = (i, 1)
                    // rh_stride = (j, 1)
                    // out_stride = (i, 1)
                    let a = lh_data[j.size()*i_i + i_j];
                    let b = rh_data[k.size()*i_j + i_k];
                    out_data[i.size()*i_i + i_k] += a * b;
                }
            }
        }
    }

    fn matmul_transpose<I:Dim, J:Dim, K:Dim>(
        dims     : (I,J,K),
        lh_data  : Vec<f32>,
        rh_data  : Vec<f32>,
        out_data : &mut Vec<f32>,  
        transpose_sides: (bool, bool)
    ) {
        let (i,j,k) = dims;
        for i_i in 0..i.size() {
            for i_j in 0..j.size() {
                for i_k in 0..k.size() {
                    // lh_stride = (i, 1)
                    // rh_stride = (j, 1)
                    // out_stride = (i, 1)

                    let a = if transpose_sides.0 == false {
                        lh_data[j.size()*i_i + i_j]
                    } else {
                        lh_data[j.size()*i_j + i_i]
                    };

                    let b = if transpose_sides.1 == false {
                        rh_data[k.size()*i_j + i_k]
                    } else {
                        rh_data[k.size()*i_k + i_j]
                    };
                    
                    out_data[i.size()*i_i + i_k] += a * b;
                }
            }
        }
    }
    
}

impl MatVecImpl<f32> for CPU {
    fn matmul<I: Dim, J: Dim>(
        dims: (I,J),
        lhs_data: RwLockReadGuard<Vec<f32>>,
        rhs_data: RwLockReadGuard<Vec<f32>>,
        out_data: &mut Vec<f32>
    ) {
        let (i, j) = dims;
        for i_i in 0..i.size() {
            for i_j in 0..j.size() {
                let a = lhs_data[i_i * j.size() + i_j];
                let b = rhs_data[i_j];
                out_data[i_i] += a * b;
            }
        }
    }
}
impl MatVecImpl<f64> for CPU {
    fn matmul<I: Dim, J: Dim>(
        dims: (I,J),
        lhs_data: RwLockReadGuard<Vec<f64>>,
        rhs_data: RwLockReadGuard<Vec<f64>>,
        out_data: &mut Vec<f64>
    ) {
        let (i, j) = dims;
        for i_i in 0..i.size() {
            for i_j in 0..j.size() {
                let a = lhs_data[i_i * j.size() + i_j];
                let b = rhs_data[i_j];
                out_data[i_i] += a * b;
            }
        }
    }
}

impl<E: Unit> MatMatKernel<E> for CPU
where
    Self: MatMatImpl<E>
{
    fn forward<I: Dim, J: Dim, K: Dim>(
        &self,
        lhs: &crate::tensor::Tensor<(I,J), E, Self>,
        rhs: &crate::tensor::Tensor<(J,K), E, Self>,
    ) -> Result<crate::tensor::Tensor<(I,K), E, Self>, Self::Err> {
        let (i, j): (I,J) = lhs.shape;
        let k: K = rhs.shape.1;
        let output = self.try_zeros_from(&(i,k))?;

        <Self as MatMatImpl<E>>::matmul(
            (i, j, k),
            lhs.data.read().unwrap(),
            rhs.data.read().unwrap(),
            &mut output.data.write().unwrap(),
        );

        Ok(output)
    }

    fn backward<I: Dim, J: Dim, K:Dim>(
        &self,
        lhs: &Tensor<(I,J), E, Self>,
        rhs: &Tensor<(J,K), E, Self>,
        grads: &mut Gradients<E, Self>,
        out: &Tensor<(I, K), E, Self>,
    ) -> Result<(), Self::Err> {
        let (i, j): (I,J) = lhs.shape;
        let k: K = rhs.shape.1;

        grads.try_alloc_for((&lhs.device, lhs.id, lhs.device.num_el(lhs.data.read().unwrap().to_owned())))?;
        grads.try_alloc_for((&rhs.device, rhs.id, rhs.device.num_el(rhs.data.read().unwrap().to_owned())))?;
        grads.try_ones_for((&out.device, out.id, out.device.num_el(out.data.read().unwrap().to_owned())))?;

        <Self as MatMatImpl<E>>::matmul_transpose(
            (i, j, k),
            grads.get_grad_ref(&out.id).to_vec(),
            rhs.data.read().unwrap().to_vec(),
            &mut grads.get_grad_mut(&lhs.id),
            (false, true)
        );

        <Self as MatMatImpl<E>>::matmul_transpose(
            (i, j, k),
            lhs.data.read().unwrap().to_vec(),
            grads.get_grad_ref(&out.id).to_vec(),
            &mut grads.get_grad_mut(&rhs.id),
            (true, false)
        );

        Ok(())

    }
}

impl <E: Unit> MatVecKernel<E> for CPU 
where
    Self: MatVecImpl<E>
{
    fn forward<I: Dim, J: Dim>(
        &self,  
        lhs: &Tensor<(I, J), E, Self>,
        rhs: &Tensor<(J, ), E, Self>,
    ) -> Result<crate::tensor::Tensor<(I,), E, Self>, Self::Err> {
        let (i,_j) = lhs.shape;
        let output = lhs.device.try_zeros_from(&(i,)).unwrap();

        <CPU as MatVecImpl<E>>::matmul(lhs.shape, lhs.data.read().unwrap(), rhs.data.read().unwrap(), &mut output.data.write().unwrap());

        Ok(output)
    }

    fn backward<I: Dim, J: Dim>(
        &self,
        lhs: &Tensor<(I, J), E, Self>,
        rhs: &Tensor<(J, ), E, Self>,
        grads: &mut Gradients<E, Self>,
        out: &Tensor<(I,), E, Self>,
    ) -> Result<(), Self::Err> {
        let (i,j) = lhs.shape;
        
        grads.try_alloc_for((&lhs.device, lhs.id.clone(), lhs.shape.num_elements()))?;
        grads.try_alloc_for((&rhs.device, rhs.id.clone(), rhs.shape.num_elements()))?;
        grads.try_ones_for((&out.device, out.id.clone(), out.shape.num_elements()))?;

        let lhs_data = lhs.data.read().unwrap();
        let rhs_data = rhs.data.read().unwrap();

        let out_grad = grads.get_grad_ref(&out.id).to_vec();

        // dOut w.r.t. dLhs = (1->I).T * rhs

        // derive lhs
        let lhs_grad = grads.get_grad_mut(&lhs.id);

        for i_i in 0..i.size() {
            for i_j in 0..j.size() {
                let a = rhs_data[i_j];
                let b = out_grad[i_i];
                lhs_grad[i_i * j.size() + i_j] += a * b;
            }
        }

        // derive rhs
        let rhs_grad = grads.get_grad_mut(&rhs.id);
        
        for i_i in 0..i.size() {
            for i_j in 0..j.size() {
                let i_a = i_i * j.size() + i_j;
                let a = lhs_data[i_a];
                let b = out_grad[i_i];
                rhs_grad[i_j] += a * b;
            }
        }


        Ok(())
    }
}
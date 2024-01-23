use std::{borrow::BorrowMut, ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, SubAssign}, sync::Arc};

use crate::{tensor::{Tensor, ZerosTensor}, shape::{Shape, Storage}, dtypes::Unit};

impl <S: Shape, E: Unit, D: Storage<E> + ZerosTensor<E>> Add<E> for Tensor<S, E, D> {
    type Output = Self;

    fn add(self, rhs: E) -> Self::Output {
        let mut alloc = self.device.try_zeros_from(&self.shape).unwrap();
        let alloc_arc = Arc::clone(alloc.data.borrow_mut());
        let mut alloc_write = alloc_arc.write().unwrap();
       
        let self_read = self.data.read().unwrap();

        for ii in 0..self.shape.num_elements() {
            *alloc_write.index_mut(ii) = *self_read.index(ii) + rhs;
        }

        alloc
    }
}

impl <S: Shape, E: Unit, D: Storage<E> + ZerosTensor<E>> Div<E> for Tensor<S, E, D> {
    type Output = Self;

    fn div(self, rhs: E) -> Self::Output {
        let mut alloc = self.device.try_zeros_from(&self.shape).unwrap();
        let alloc_arc = Arc::clone(alloc.data.borrow_mut());
        let mut alloc_data = alloc_arc.write().unwrap();
        
        let self_data = self.data.read().unwrap();

        for ii in 0..self.shape.num_elements() {
            *alloc_data.index_mut(ii) = *self_data.index(ii) / rhs;
        }

        alloc
    }
}

impl <S: Shape, E: Unit, D: Storage<E> + ZerosTensor<E>> Div<E> for &mut Tensor<S, E, D> {
    type Output = Tensor<S, E, D>;

    fn div(self, rhs: E) -> Self::Output {
        let mut alloc = self.device.try_zeros_from(&self.shape).unwrap();
        let alloc_arc = Arc::clone(alloc.data.borrow_mut());
        let mut alloc_data = alloc_arc.write().unwrap();
        
        let self_data = self.data.read().unwrap();

        for ii in 0..self.shape.num_elements() {
            *alloc_data.index_mut(ii) = *self_data.index(ii) / rhs;
        }

        alloc
    }
}


impl <S: Shape, E: Unit, D: Storage<E> + ZerosTensor<E>> Mul<E> for Tensor<S, E, D> {
    type Output = Self;

    fn mul(self, rhs: E) -> Self::Output {
        let mut alloc = self.device.try_zeros_from(&self.shape).unwrap();
        let alloc_arc = Arc::clone(alloc.data.borrow_mut());
        let mut alloc_data = alloc_arc.write().unwrap();
        
        let self_data = self.data.read().unwrap();

        for ii in 0..self.shape.num_elements() {
            *alloc_data.index_mut(ii) = *self_data.index(ii) * rhs;
        }

        alloc
    }
}


impl <S: Shape, E: Unit, D: Storage<E> + ZerosTensor<E>> MulAssign<E> for Tensor<S, E, D> {
    fn mul_assign(&mut self, rhs: E) {
        let mut self_data = self.data.write().unwrap();

        for ii in 0..self.shape.num_elements() {
            *self_data.index_mut(ii) *= rhs;
        }
    }
}

impl <S: Shape, E: Unit, D: Storage<E> + ZerosTensor<E>> AddAssign<E> for Tensor<S, E, D> {
    fn add_assign(&mut self, rhs: E) {
        let mut self_data = self.data.write().unwrap();

        for ii in 0..self.shape.num_elements() {
            *self_data.index_mut(ii) += rhs;
        }
    }
}

impl <S: Shape, E: Unit, D: Storage<E> + ZerosTensor<E>> AddAssign<Tensor<S, E, D>> for Tensor<S, E, D> {
    fn add_assign(&mut self, rhs: Tensor<S, E, D>) {
        let mut self_data = self.data.write().unwrap();
        let rhs_data = rhs.data.read().unwrap();

        for ii in 0..self.shape.num_elements() {
            *self_data.index_mut(ii) += *rhs_data.index(ii);
        }
    }
}

impl <S: Shape, E: Unit, D: Storage<E> + ZerosTensor<E>> SubAssign<E> for Tensor<S, E, D> {
    fn sub_assign(&mut self, rhs: E) {
        let mut self_data = self.data.write().unwrap();

        for ii in 0..self.shape.num_elements() {
            *self_data.index_mut(ii) -= rhs;
        }
    }
}

impl <S: Shape, E: Unit, D: Storage<E> + ZerosTensor<E>> SubAssign<Tensor<S, E, D>> for Tensor<S, E, D> {
    fn sub_assign(&mut self, rhs: Tensor<S, E, D>) {
        let mut self_data = self.data.write().unwrap();
        let rhs_data = rhs.data.read().unwrap();

        for ii in 0..self.shape.num_elements() {
            *self_data.index_mut(ii) += *rhs_data.index(ii);
        }
    }
}


impl <S: Shape, E: Unit, D: Storage<E> + ZerosTensor<E>> DivAssign<Tensor<S, E, D>> for Tensor<S, E, D> {
    fn div_assign(&mut self, rhs: Tensor<S, E, D>) {
        todo!()
    }
}

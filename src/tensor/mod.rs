pub mod tensor_ops;
pub mod tape;

use std::{
    sync::{RwLock, Arc},
    fmt::{Debug, Display}, ops::{IndexMut, Range}
};

use rand::{distributions::Standard, prelude::Distribution};

use crate::{shape::{Shape, Storage, ConstShape, HasShape, Rank2, Rank1, Dim, ConstDim, Const}, dtypes::{Unit, FloatUnit}};

use self::tape::{UniqueID, Tape, NoneTape, OwnedTape, PutTape};

#[derive(Clone, Debug)]
pub struct Tensor<SHAPE: Shape, E: Unit, D: Storage<E>, T = NoneTape> {
    pub id: UniqueID,
    pub shape: SHAPE,
    pub data: Arc<RwLock<D::Vec>>,
    pub device: D,
    pub tape: T,
}

pub trait TensorLike<E: Unit, D: Storage<E>> {
    fn get_data_ref(&self) -> Arc<RwLock<D::Vec>>;
    fn get_num_el(&self) -> usize;
    fn get_id(&self) -> UniqueID;
}

impl <S: Shape,E: Unit, D:Storage<E>> TensorLike<E, D> for Tensor<S, E, D> {
    fn get_data_ref(&self) -> Arc<RwLock<D::Vec>> {
        self.data.clone()
    }
    fn get_num_el(&self) -> usize {
        self.shape.num_elements()
    }
    fn get_id(&self) -> UniqueID {
        self.id.to_owned()
    }
}

impl <S, E, D, T> Tensor<S, E, D, T>
where
    S: Shape,
    E: FloatUnit,
    D: Storage<E>, 
    T: Tape<E, D>
{
    pub fn allclose(&self, rhs: &Self, rtol: Option<E>, atol: Option<E>) -> bool {
        let rtol = rtol.unwrap_or(E::EPSILON);
        let atol = atol.unwrap_or(E::EPSILON*E::EPSILON);

        let lhs_data = self.data.read().unwrap().to_owned();
        let rhs_data = rhs.data.read().unwrap().to_owned();

        for (lhs, rhs) in lhs_data.into_iter().zip(rhs_data.into_iter()) {
            let abs: E = (lhs-rhs).abs();
            if !(abs <= (atol + rtol * rhs.abs())) {
                return false
            }
        };
        return true
    }
}

impl <S, E, D, T> PartialEq for Tensor<S, E, D, T>
where
    S: Shape,
    E: Unit,
    D: Storage<E>,
    T: Tape<E, D>,
{
    fn eq(&self, other: &Self) -> bool {
        let lhs_data = self.data.read().unwrap().to_owned();
        let rhs_data = other.data.read().unwrap().to_owned();


        for (lhs, rhs) in lhs_data.into_iter().zip(rhs_data.into_iter()) {
            if lhs != rhs {
                return false
            }
        }

        return true
    }
}


pub trait Watch<E, D: Storage<E>>: Clone {
    type Watched;

    fn watch_leaky(&self) -> Self::Watched {
        self.clone().watched_leaky()
    }

    fn watched_leaky(self) -> Self::Watched;
}

impl <E: Unit, D: Storage<E>, S: Shape> Watch<E, D> for Tensor<S, E, D> {
    type Watched = Tensor<S, E, D, OwnedTape<E, D>>;

    fn watched_leaky(self) -> Self::Watched {
        self.put_tape(OwnedTape::default())
    }

    
}

pub trait HasErr: Sized {
    type Err: Debug + Display;

    const Err: Self::Err;
}


pub trait ZerosTensor<E: Unit>: Storage<E> + HasErr {
    fn zeros<S: ConstShape>(&self) -> Tensor<S, E, Self> {
        Self::try_zeros_from::<S>(&self, &Default::default()).unwrap()
    }

    fn try_zeros_from<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err>;
}

pub trait RandTensor<E: Unit>: Storage<E> + HasErr {
    fn try_fill_rand<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err>
    where Standard: Distribution<E>;
    
    fn try_fill_rand_range<S: HasShape>(&self, src: &S, range: Range<E>) -> Result<Tensor<S::Shape, E, Self>, Self::Err>
    where Standard: Distribution<E>;
    
    fn fill_rand<S: ConstShape>(&self) -> Tensor<S, E, Self> 
    where Standard: Distribution<E>
    {
        Self::try_fill_rand::<S>(&self, &Default::default()).unwrap()
    }

    fn fill_rand_range<S: ConstShape>(&self, range: Range<E>) -> Tensor<S, E, Self>
    where Standard: Distribution<E>
    {
        Self::try_fill_rand_range::<S>(&self, &Default::default(), range).unwrap()
    }
}

pub trait Arange<E: Unit>: Storage<E> + HasErr {
    fn try_arange<S: Dim>(&self, end: &S) -> Result<Tensor<(S, ), E, Self>, Self::Err>;

    fn arange<const S: usize>(&self) -> Tensor<Rank1<S>, E, Self> {
        Self::try_arange::<Const::<S>>(&self, &Default::default()).unwrap()
    }
}

impl<const Y: usize, const X: usize, E: Unit + Copy, D: ZerosTensor<E>, T: Tape<E, D>> Tensor<Rank2<Y, X>, E, D, T> {
    pub fn copy_from_array(&mut self, src: [[E;X];Y]) {
        let mut self_inner = self.data.write().unwrap();
        for el_x in 0..X {
            for el_y in 0..Y {
                *self_inner.index_mut(X * el_y + el_x) = src[el_y][el_x];
            }
        }

    }
}

impl <const X: usize, E: Unit + Copy, D: ZerosTensor<E>> Tensor<Rank1<X>, E, D> {
    pub fn copy_from_array(&mut self, src: [E;X]){
        let mut self_inner = self.data.write().unwrap();
        for el_x in 0..X {
            *self_inner.index_mut(el_x) = src[el_x];
        }
    }

    pub fn copy_from_slice(&mut self, src: &[E;X]) {
        let mut self_inner = self.data.write().unwrap();
        for el_x in 0..X {
            *self_inner.index_mut(el_x) = src[el_x];
        }
    }
}

impl <S: Shape, E: Unit, D: Storage<E>, T: Tape<E, D>> HasErr for Tensor<S, E, D, T> {
    type Err = D::Err;

    const Err: Self::Err = D::Err;
}

impl <S: Shape, E: Unit, D: Storage<E>> HasShape for Tensor<S,E,D> {
    type WithShape<New: Shape> = Tensor<New,E,D>;

    type Shape = S;

    fn shape(&self) -> &Self::Shape {
        &self.shape
    }
}

#[cfg(test)]
mod tests {
    use crate::{shape::Rank2, devices::cpu::CPU, tensor::tape::NoneTape};

    use super::{Tensor, ZerosTensor};

    #[test]
    fn test_zeros() {
        let dev = CPU::default();
        let x: Tensor<Rank2<28,28>, f32, CPU, NoneTape> = dev.zeros();

        assert_eq!(*x.data.read().unwrap(), vec![0.0;28*28]);
    }
}

pub mod tensor_ops;
pub mod tape;

use std::{
    sync::{RwLock, Arc, Mutex, atomic::AtomicUsize},
    fmt::{Debug, Display}, ops::{IndexMut, Range}, collections::{BTreeMap, BTreeSet}, rc::Rc
};

use rand::{distributions::Standard, prelude::Distribution, Error};

use crate::{shape::{Shape, Storage, ConstShape, HasShape, Rank2, Rank1, Dim, Const}, dtypes::Unit, devices::cpu::CPU};

use self::tape::{UniqueID, Tape, NoneTape, OwnedTape, PutTape};

pub trait OP: Clone{
}

// pub struct MatMulMatOp<X: Dim, Y: Dim, Z: Dim, E: Unit, D: Storage<E>> {
//     pub lhs: Tensor<(Y, X), E, D>,
//     pub rhs: Tensor<(X, Z), E, D>,
// }

#[derive(Clone)]
pub struct Conv2dOp<KernelShape: Shape, LhsShape: Shape, E: Unit, D: Storage<E>> {
    lhs: Tensor<LhsShape, E, D>,
    weights: Tensor<KernelShape, E, D>,

}

impl <KernelShape: Shape, LhsShape: Shape, E: Unit, D: Storage<E>> OP for Conv2dOp<KernelShape, LhsShape, E, D> {}


#[derive(Clone)]
pub struct Tensor<SHAPE: Shape, E: Unit, D: Storage<E>, T = NoneTape> {
    pub id: UniqueID,
    pub shape: SHAPE,
    pub data: Arc<RwLock<D::Vec>>,
    pub device: D,
    pub tape: T,
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
}


pub trait ZerosTensor<E: Unit>: Storage<E> + HasErr {
    fn zeros<S: ConstShape>(&self) -> Tensor<S, E, Self> {
        Self::try_zeros_from::<S>(&self, &Default::default()).unwrap()
    }

    fn try_zeros_from<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err>;
}

pub trait RandTensor<E: Unit>: Storage<E> + HasErr {
    fn fill_rand<S: ConstShape>(&self) -> Tensor<S, E, Self> 
    where Standard: Distribution<E>
    {
        Self::try_fill_rand::<S>(&self, &Default::default()).unwrap()
    }

    fn try_fill_rand<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err>
    where Standard: Distribution<E>;
    
    fn fill_rand_range<S: ConstShape>(&self, range: Range<E>) -> Tensor<S, E, Self>
    where Standard: Distribution<E>
    {
        Self::try_fill_rand_range::<S>(&self, &Default::default(), range).unwrap()
    }

    fn try_fill_rand_range<S: HasShape>(&self, src: &S, range: Range<E>) -> Result<Tensor<S::Shape, E, Self>, Self::Err>
    where Standard: Distribution<E>;
    
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
}

impl <S: Shape, E: Unit, D: Storage<E>, T: Tape<E, D>> HasErr for Tensor<S, E, D, T> {
    type Err = D::Err;
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

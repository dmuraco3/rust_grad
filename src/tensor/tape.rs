use std::{sync::{atomic::AtomicUsize, Arc, RwLock}, collections::{BTreeMap, btree_map::Entry::Vacant}};

use crate::{shape::Shape, dtypes::Unit, storage::Storage};

use super::{Tensor, HasErr};

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, PartialOrd, Ord)]
pub struct UniqueID(pub usize);
pub fn unique_id() -> UniqueID {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    UniqueID(COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed))

}

#[derive(Clone)]
pub struct Gradients<E, D: Storage<E>> {
    pub gradient_by_id: BTreeMap<UniqueID, D::Vec>,
    // pub leaf_ids: Option<BTreeSet<UniqueID>>,
}

impl <E: Unit, D: Storage<E>> Gradients<E, D> {
    pub fn leaky() -> Self {
        Self {
            gradient_by_id: Default::default(),
            // leaf_ids: None
        }
    }

    pub fn try_alloc_for(&mut self, tensor_data: (&D, UniqueID, usize)) -> Result<(), D::Err> {
        if let Vacant(entry) = self.gradient_by_id.entry(tensor_data.1) {
            entry.insert(tensor_data.0.try_alloc_len(tensor_data.2)?);
        }

        Ok(())
    }

    pub fn try_alloc_raw(&mut self, device: &D, tensor_id: &UniqueID, num_ele: usize) -> Result<(), D::Err> {
        if let Vacant(entry) = self.gradient_by_id.entry(tensor_id.to_owned()) {
            entry.insert(device.try_alloc_len(num_ele)?);
        }
        Ok(())
    }

    pub fn try_ones_for(&mut self, tensor_data: (&D, UniqueID, usize)) -> Result<(), D::Err> {
        if let Vacant(entry) = self.gradient_by_id.entry(tensor_data.1) {
            entry.insert(tensor_data.0.try_alloc_ones(tensor_data.2)?);
        }

        Ok(())
    }

    pub fn get_grad_ref(&self, tensor_id: &UniqueID) -> &D::Vec {
        self.gradient_by_id.get(tensor_id).unwrap()
    }

    pub fn get_grad_mut(&mut self, tensor_id:&UniqueID) -> &mut D::Vec {
        self.gradient_by_id.get_mut(tensor_id).unwrap()
    }

    pub fn get<S: Shape, T>(&self, tensor: &Tensor<S, E, D, T>) -> Result<Tensor<S, E, D>, D::Err> {
        match self.gradient_by_id.get(&tensor.id) {
            None => Err(<D as HasErr>::ERR),
            Some(entry) => Ok(Tensor {
                id: tensor.id,
                shape: tensor.shape,
                data: Arc::new(RwLock::new(entry.to_owned())),
                device: tensor.device.clone(),
                tape: NoneTape,
            })
        }
    }

}


type BackwardOp<E, D, Err> = Box<dyn FnOnce(&mut Gradients<E, D>) -> Result<(), Err>>;

#[derive(Default, Debug, Clone, Copy)]
pub struct NoneTape;

pub struct OwnedTape<E, D: Storage<E>> {
    pub operations: Vec<(UniqueID, BackwardOp<E, D, D::Err>)>,
    pub gradients: Gradients<E, D>,
}

impl <E, D: Storage<E>> OwnedTape<E, D> {
    pub fn sort_ops_backprop(&mut self) {
        self.operations.sort_by(|a, b| b.0.cmp(&a.0));
    }
}

impl <E: Unit, D: Storage<E>> Default for OwnedTape<E, D> {
    fn default() -> Self {
        Self {
            operations: Default::default(),
            gradients: Gradients::leaky(),
        }
    }
}

pub trait Merge<T: ?Sized> {
    fn merge(self, other: T) -> Self;
}

impl Merge<Self> for NoneTape {
    fn merge(self, _other: Self) -> Self {
        self
    }
}

impl <E, D: Storage<E>> Merge<NoneTape> for OwnedTape<E, D> {
    fn merge(self, _: NoneTape) -> Self {
        self
    }
}

impl <E, D: Storage<E>> Merge<OwnedTape<E, D>> for OwnedTape<E, D> {
    fn merge(mut self, mut other: OwnedTape<E, D>) -> Self {
        self.gradients
            .gradient_by_id
            .extend(other.gradients.gradient_by_id);
        // if let Some(leafs) = other.gradients.leaf_ids {
        //     self.gradients
        //         .leaf_ids
        //         .get_or_insert_with(Default::default)
        //         .extend(leafs);
        // }
        self.operations.append(&mut other.operations);
        self
    }
}

pub trait Tape<E, D: Storage<E>>: Default + Merge<Self> + Merge<NoneTape> {
    const OWNS_TAPE: bool;
    fn add_backward_op<F>(&mut self, operation: F)
    where 
        F: 'static + FnOnce(&mut Gradients<E, D>) -> Result<(), D::Err>;
}

impl <E: Unit, D: Storage<E>> Tape<E,D> for OwnedTape<E, D> {
    const OWNS_TAPE: bool = true;

    fn add_backward_op<F>(&mut self, operation: F)
    where 
        F: 'static + FnOnce(&mut Gradients<E, D>) -> Result<(), <D>::Err> {
        self.operations.push((unique_id(), Box::new(operation)));
    }
}

impl <E, D: Storage<E>> Tape<E, D> for NoneTape {
    const OWNS_TAPE: bool = false;

    fn add_backward_op<F>(&mut self, _operation: F)
    where 
        F: 'static + FnOnce(&mut Gradients<E, D>) -> Result<(), <D>::Err> {
            #[cfg(debug_assertions)]
            {
                println!("Cannot Add Backward Op to NoneTape");
            }
    }
    
}

pub trait PutTape<T> {
    type Output;
    fn put_tape(self, tape: T) -> Self::Output;
}

impl <S: Shape, E: Unit, D: Storage<E>, T> PutTape<T> for Tensor<S, E, D> {
    type Output = Tensor<S, E, D, T>;

    fn put_tape(self, tape: T) -> Self::Output {
        Tensor {
            id: self.id,
            shape: self.shape,
            data: self.data,
            device: self.device,
            tape,
        }
    }
    
}

pub trait SplitTape {
    type Tape;
    type NoTape: PutTape<Self::Tape, Output = Self>;

    fn split_tape(self) -> (Self::NoTape, Self::Tape);

    
}

impl <S: Shape, E: Unit, D: Storage<E>, T> SplitTape for Tensor<S, E, D, T> {
    type Tape = T;
    type NoTape = Tensor<S, E, D>;

    fn split_tape(self) -> (Self::NoTape, Self::Tape) {
        (
            Tensor {
                id: self.id,
                shape: self.shape,
                data: self.data,
                device: self.device,
                tape: NoneTape,
            },
            self.tape
        )
    }
    
}
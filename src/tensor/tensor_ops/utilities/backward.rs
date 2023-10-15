use crate::{tensor::{tape::{Gradients, OwnedTape, SplitTape}, Tensor}, dtypes::Unit, shape::{Storage, Shape}};

pub trait BackwardPropagate<E: Unit, D: Storage<E>> {
    fn backward(self) -> Gradients<E, D>;
}

impl <E: Unit, D: Storage<E>, S: Shape> BackwardPropagate<E, D> for Tensor<S, E, D, OwnedTape<E, D>> {
    fn backward(self) -> Gradients<E, D> {
        // sort operations by newest
        let (end, mut end_tape) = self.split_tape();
        
        end_tape.operations.sort_by(|a, b| a.0.cmp(&b.0));

        for op in end_tape.operations {
            op.1(&mut end_tape.gradients).unwrap();
        }

        end_tape.gradients
    }
}
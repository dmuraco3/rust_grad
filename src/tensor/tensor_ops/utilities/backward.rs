use crate::{tensor::{tape::{Gradients, OwnedTape, SplitTape, UniqueID}, Tensor}, dtypes::Unit, shape::{Storage, Shape}};

pub trait BackwardPropagate<E: Unit, D: Storage<E>> {
    fn backward(self) -> Gradients<E, D>;

    fn backward_norm(self) -> Gradients<E, D>;
}

impl <E: Unit, D: Storage<E>, S: Shape> BackwardPropagate<E, D> for Tensor<S, E, D, OwnedTape<E, D>> {
    fn backward(mut self) -> Gradients<E, D> {
        // sort operations by newest
        // let (end, mut end_tape) = self.split_tape();
        
        // end_tape.operations.sort_by(|a, b| b.0.cmp(&a.0));

        self.tape.operations.sort_by(|a, b| a.0.cmp(&b.0));

        // println!("{:?}", self.tape.operations.iter().map(|(a, b)| a).collect::<Vec<&UniqueID>>());


        while !self.tape.operations.is_empty() {
            let op = self.tape.operations.pop().unwrap();
            op.1(&mut self.tape.gradients).unwrap();
        }

        self.tape.gradients
    }

    fn backward_norm(mut self) -> Gradients<E, D> {
        self.tape.operations.sort_by(|a, b| a.0.cmp(&b.0));

        // println!("{:?}", self.tape.operations.iter().map(|(a, b)| a).collect::<Vec<&UniqueID>>());

        // norm scaling the loss gradient
        {
            self.tape.operations.pop().unwrap().1(&mut self.tape.gradients).unwrap();
            
        }

        while !self.tape.operations.is_empty() {
            let op = self.tape.operations.pop().unwrap();
            op.1(&mut self.tape.gradients).unwrap();
        }

        self.tape.gradients
    }

    
}


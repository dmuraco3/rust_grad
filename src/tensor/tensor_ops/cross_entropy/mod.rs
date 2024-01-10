pub mod cpu;
pub mod metal;

use std::fmt::Debug;

use crate::{shape::{Storage, Rank0, Shape, ConstDim}, dtypes::Unit, tensor::{Tensor, tape::{Gradients, UniqueID, Tape, Merge, SplitTape, PutTape}, ZerosTensor, HasErr}, devices::metal::MetalGPU};

use super::softmax::{TrySoftmax, SoftmaxKernel};

pub trait CrossEntropyKernel<E: Unit>: Storage<E> {
    
    /// # Arguments 
    /// 
    /// * `src`    - Tensor returned from softmax (saves operations)
    /// * `labels` - One-hot encoded vector 
    fn forward<S: Shape>(
        &self,
        src: &Tensor<S, E, Self>,
        labels: &Tensor<S, E, Self>,
        out: &mut Tensor<Rank0, E, Self>,
    ) -> Result<(), Self::Err>;

    fn backward<S: Shape>(
        &self,
        src: &Tensor<S, E, Self>,
        labels: &Tensor<S, E, Self>,
        src_id: UniqueID,
        out_id: UniqueID,
        grads: &mut Gradients<E, Self>
    ) -> Result<(), Self::Err>;
}

pub trait TryCrossEntropy<S: Shape, E: Unit, D: CrossEntropyKernel<E>, RTape: Tape<E, D>> {
    type Err: Debug;
    type Output;

    fn try_cross_entropy(self, labels: Tensor<S, E, D, RTape>) -> Result<Self::Output, Self::Err>;
}

impl <X, E, D, T, R> TryCrossEntropy<(X, ), E, D, R> for Tensor<(X, ), E, D, T>
where
    X: ConstDim,
    E: Unit,
    D: CrossEntropyKernel<E> + ZerosTensor<E> + SoftmaxKernel<E>,
    T: Tape<E, D> + Merge<R>,
    R: Tape<E, D>,
    Self: TrySoftmax<E>
{
    type Err = D::Err;

    type Output = Tensor<Rank0, E, D, T>;

    fn try_cross_entropy(self, labels: Tensor<(X, ), E, D, R>) -> Result<Self::Output, Self::Err> {
        let mut out = self.device.zeros();

        let (src, src_tape) = self.split_tape();
        let (labels, labels_tape) = labels.split_tape();

        let mut tape = src_tape.merge(labels_tape);


        //
        // NOTE NOTE NOTE NOTE
        // UNCOMMENT THIS CODE IF TESTING FOR ANY DEVICE THAT ISNT THE METAL GPU
        // I'M GOING TO MERGE THE TWO KERNELS INTO ONE KERNEL FOR SOFTMAX CROSS ENTROPY ON METAL GPU
        // let src_softmax = src.clone().try_softmax().unwrap();

        <D as CrossEntropyKernel<E>>::forward(&src.device, &src, &labels, &mut out)?;

        let out_id = out.id.clone();

        tape.add_backward_op(move |grads| {
            grads.try_ones_for((&src.device, out_id, 1))?; // fill output grad
            grads.try_alloc_for((&src.device, src.id, src.shape.num_elements()))?;
            grads.try_alloc_for((&labels.device, labels.id, labels.shape.num_elements()))?;

            CrossEntropyKernel::backward(&src.device, &src, &labels, src.id.clone(), out_id, grads)?;

            Ok(())
        });

        Ok(out.put_tape(tape))
    }
}

#[cfg(test)]
mod tests {
    use crate::{devices::cpu::CPU, tensor::{Tensor, Watch}, shape::Rank1};

    use super::TryCrossEntropy;

    #[test]
    fn test_cross_entropy_forward() {
        let dev = CPU::default();

        let src: Tensor<Rank1<5>, f32, _> = dev.from_array([0.521809, 0.891292, 0.014712, 0.718290, 0.184821]);
        let labels: Tensor<Rank1<5>, f32, _> = dev.from_array([1.0, 0.0, 0.0, 0.0, 0.0]);

        let cross_entropy = src.try_cross_entropy(labels).unwrap();
        
        assert_eq!(cross_entropy.data.read().unwrap()[0], 1.6054815);
    }

    #[test]
    fn test_cross_entropy_backward() {
        let dev = CPU::default();

        let src: Tensor<Rank1<5>, f32, _> = dev.from_array([0.521809, 0.891292, 0.014712, 0.718290, 0.184821]);
        let labels: Tensor<Rank1<5>, f32, _> = dev.from_array([1.0, 0.0, 0.0, 0.0, 0.0]);

        let mut cross_entropy = src.watch_leaky().try_cross_entropy(labels).unwrap();

        let op = cross_entropy.tape.operations.pop().unwrap();
        op.1(&mut cross_entropy.tape.gradients).unwrap();

        let src_grad = cross_entropy.tape.gradients.get(&src).unwrap();

        let actual_grad: Tensor<Rank1<5>, f32, CPU> = dev.from_array([-0.79920715, 0.29054448, 0.12092575, 0.2443874, 0.14334951]);

        assert!(src_grad.allclose(&actual_grad, None, None));

        

    }
}

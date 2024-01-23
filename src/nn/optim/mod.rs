use std::{
    borrow::BorrowMut, default, ops::AddAssign, sync::{atomic::AtomicUsize, Arc, RwLock}, time::{Duration, Instant}
};

use crate::{
    devices::cpu::CPU,
    dtypes::FloatUnit,
    shape::{HasShape, Shape, Storage, TensorInnerLength},
    tensor::{
        tape::{Gradients, UniqueID},
        Tensor, TensorLike, ZerosTensor, tensor_ops::{pow::{TryPow, PowKernel}, sqrt::{SqrtKernel, TrySqrt}},
    },
};

pub struct ADAMConfig<E: FloatUnit> {
    step_size: E,
    beta1: E,
    beta2: E,
    epsilon: E,
}

impl<E: FloatUnit> Default for ADAMConfig<E> {
    fn default() -> Self {
        Self {
            step_size: E::from_f32(0.001),
            beta1: E::from_f32(0.9),
            beta2: E::from_f32(0.999),
            epsilon: E::from_f32(1e-08),
        }
    }
}

pub struct ADAM<'a, E: FloatUnit, D: Storage<E>> {
    pub moment1: Vec<Tensor<(usize,), E, D>>,
    pub moment2: Vec<Tensor<(usize,), E, D>>,
    pub trainable: Vec<&'a mut Tensor<(usize,), E, D>>,
    pub t: u32,
    pub config: ADAMConfig<E>,
}

// impl <E: FloatUnit, D: Storage<E>> Default for ADAM<E, D> {
//     fn default() -> Self {
//         Self {
//             moment1: Vec::new(),
//             moment2: Vec::new(),
//             config: ADAMConfig::default(),
//             trainable: Vec::default(),
//         }
//     }
// }

impl<'a, E: FloatUnit, D: Storage<E> + ZerosTensor<E>> ADAM<'a, E, D> {
    pub fn new(device: &D, trainable: &'a mut [Tensor<(usize,), E, D>]) -> Self {
        let mut moment1: Vec<Tensor<(usize,), E, D>> = Vec::with_capacity(trainable.len());
        let mut moment2: Vec<Tensor<(usize,), E, D>> = Vec::with_capacity(trainable.len());

        for tensor in trainable.iter() {
            moment1.push(device.try_zeros_from(&tensor.shape).unwrap());
            moment2.push(device.try_zeros_from(&tensor.shape).unwrap());
        }

        let trainable = trainable.iter_mut().collect::<Vec<_>>();

        Self {
            moment1,
            moment2,
            trainable,
            t: 0,
            config: Default::default(),
        }
    }
}
///
impl<E: FloatUnit, D: Storage<E> + ZerosTensor<E> + PowKernel<E> + SqrtKernel<E>> ADAM<'_, E, D> {
    pub fn step(&mut self, grads: Gradients<E, D>) {
        self.t += 1;

        let b1 = self.config.beta1;
        let b2 = self.config.beta2;
        let step_size = self.config.step_size;
        let epsilon = self.config.epsilon;

        for (weights, (mut moment1, mut moment2)) in self
            .trainable
            .iter_mut()
            .zip(self.moment1.iter_mut().zip(self.moment2.iter_mut()))
        {
            let grad = grads.get(weights).unwrap();

            *moment1 *= b1;
            *moment1 += grad.clone() * (E::ONE - b1);

            *moment2 *= b2;
            *moment2 += grad.clone().try_pow(2).unwrap() * (E::ONE - b2);

            let mut moment1_hat = moment1 / (E::ONE - b1.pow(self.t as u16));
            let mut moment2_hat = moment2 / (E::ONE - b2.pow(self.t as u16));

            let _ = moment2_hat.clone().try_sqrt();
            
            moment2_hat += epsilon;

            moment1_hat *= step_size;

            moment1_hat /= moment2_hat;

            **weights -= moment1_hat;
        }

        // self.trainable
        //     .iter()
        //     .zip(self.moment1.iter().zip(self.moment2.iter()))
        //     .for_each(
        //         |((trainable_id, trainable_vector), (m1_vector, m2_vector))| {
        //             let grad = grads.gradient_by_id.get(trainable_id).unwrap();
        //             let mut trainable_vector = trainable_vector.write().unwrap();

        //             let mut m1_vector_write = m1_vector.1.write().unwrap();
        //             let mut m2_vector_write = m2_vector.1.write().unwrap();

        //             grad.iter()
        //                 .zip(trainable_vector.iter_mut())
        //                 .zip(m1_vector_write.iter_mut().zip(m2_vector_write.iter_mut()))
        //                 .for_each(|((grad_ele, trainable_ele), (m1_ele, m2_ele))| {
        //                     // actually doing element-wise ADAM operations here

        //                     if *grad_ele < E::from_f32(-20.0) || *grad_ele > E::from_f32(20.0) {
        //                         println!("found a wild ass gradient: {}", grad_ele);
        //                     }
        //                     // clip gradient
        //                     let grad_ele = *grad_ele;
        //                     let grad_ele = grad_ele.max(E::from_f32(-20_f32));
        //                     let grad_ele = grad_ele.min(E::from_f32(20_f32));

        //                     // calculating moments for current timestep
        //                     *m1_ele = b1 * (*m1_ele) + (E::ONE - b1) * (grad_ele); // calculating m1_t
        //                     *m2_ele = b2 * (*m2_ele) + (E::ONE - b2) * (grad_ele.pow(2)); // calculating m2_t

        //                     // correcting bias
        //                     let m1_hat: E = *m1_ele / (E::ONE - b1.pow(self.t as u16));
        //                     let m2_hat: E = *m2_ele / (E::ONE - b2.pow(self.t as u16));

        //                     *trainable_ele =
        //                         *trainable_ele - step_size * (m1_hat / (m2_hat.sqrt() + epsilon));
        //                 })
        //         },
        //     );
    }
}

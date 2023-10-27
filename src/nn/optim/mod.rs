use std::{default, sync::{Arc, RwLock, atomic::AtomicUsize}};

use crate::{dtypes::FloatUnit, tensor::{tape::{Gradients, UniqueID}, Tensor, TensorLike}, shape::{Storage, TensorInnerLength}, devices::cpu::CPU};

pub struct ADAMConfig<E: FloatUnit> {
    step_size: E,
    beta1: E,
    beta2: E,
    epsilon: E,
}

impl <E:FloatUnit> Default for ADAMConfig<E> {
    fn default() -> Self {
        Self {
            step_size: E::from_f32(0.01),
            beta1: E::from_f32(0.9), 
            beta2: E::from_f32(0.999),
            epsilon: E::from_f32(1e-08),
        }
    }
}


pub struct ADAM<E: FloatUnit, D: Storage<E>> {
    moment1   : Vec<Arc<RwLock<D::Vec>>>             ,
    moment2   : Vec<Arc<RwLock<D::Vec>>>             ,
    trainable : Vec<(UniqueID, Arc<RwLock<D::Vec>>)> ,
    t         : u32                                  ,
    config    : ADAMConfig<E>                        ,
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

impl <'a, E: FloatUnit, D: Storage<E>> ADAM<E, D> {
    pub fn new(device: &D, trainable: &[&dyn TensorLike<E, D>]) -> Self {
        let mut moment1 = Vec::with_capacity(trainable.len());
        let mut moment2 = Vec::with_capacity(trainable.len());
        let trainable = trainable.iter().map(|tensor| {
            let moment_alloc = device.try_alloc_len(tensor.get_num_el()).unwrap();
            moment1.push(Arc::new(RwLock::new(moment_alloc.to_owned())));
            moment2.push(Arc::new(RwLock::new(moment_alloc)));

            (tensor.get_id(), tensor.get_data_ref())
        }).collect::<Vec<_>>();

        Self {
            moment1,
            moment2,
            trainable,
            t: 0,
            config: Default::default(),
        }
    }
}

impl <E: FloatUnit> ADAM<E, CPU> {
    pub fn step(&mut self,  grads: Gradients<E, CPU>) {

        self.t +=1; // increment time step at the start??

        // iterate through everything
        self.trainable.iter().zip(
            self.moment1.iter().zip(self.moment2.iter())
        ).for_each(|(
            (
                trainable_id,
                trainable_vector,
            ),(
                m1_vector,
                m2_vector,
            )
        )|{
            let grad = grads.gradient_by_id.get(trainable_id).unwrap();
            let mut trainable_vector = trainable_vector.write().unwrap();

            let mut m1_vector = m1_vector.write().unwrap();
            let mut m2_vector = m2_vector.write().unwrap();

            grad.iter().zip(trainable_vector.iter_mut())
            .zip(
                m1_vector.iter_mut().zip(m2_vector.iter_mut())
            ).for_each(|((
                grad_ele,
                trainable_ele
            ),(
                m1_ele,
                m2_ele
            ))| {
                // actually doing element-wise ADAM operations here
                let b1 = self.config.beta1;
                let b2 = self.config.beta2;
                let step_size = self.config.step_size;
                let epsilon = self.config.epsilon;

                // calculating moments for current timestep
                *m1_ele = b1*(*m1_ele) + (E::ONE-b1)*(*grad_ele); // calculating m1_t
                *m2_ele = b2*(*m2_ele) + (E::ONE-b2)*(grad_ele.pow(2)); // calculating m2_t

                
                // correcting bias
                let m1_hat: E = *m1_ele / (E::ONE - b1.pow(self.t));
                let m2_hat = *m2_ele / (E::ONE - b2.pow(self.t));
                
                // println!("{}", step_size * (m1_hat / (m2_hat.sqrt() + epsilon)));

                *trainable_ele = *trainable_ele - step_size * (m1_hat / (m2_hat.sqrt() + epsilon));

                // print!("{esc}c", esc = 27 as char);
            })
        });

    }
}
use std::{
    fmt::Display,
    ops::Range,
    sync::{Arc, RwLock},
};

use rand::{
    distributions::{uniform::SampleUniform, Standard, Uniform},
    prelude::Distribution,
    rngs::StdRng,
    Rng, SeedableRng,
};

use crate::{
    dtypes::Unit,
    shape::{Const, Dim, HasShape, Rank1, Rank2, Shape},
    storage::Storage,
    tensor::{
        tape::{unique_id, NoneTape, Tape},
        Arange, HasErr, RandTensor, Tensor, ZerosTensor,
    },
};

use super::metal::MetalGPU;

#[derive(Debug, Clone)]
pub struct CPU {}

impl Default for CPU {
    fn default() -> Self {
        Self {}
    }
}

impl<E: Unit> Storage<E> for CPU {
    type Vec = Vec<E>;

    fn try_alloc_len(&self, len: usize) -> Result<Self::Vec, Self::Err> {
        Ok(vec![E::ZERO; len])
    }

    fn try_alloc_ones(&self, len: usize) -> Result<Self::Vec, Self::Err> {
        Ok(vec![E::ONE; len])
    }

    fn num_el(&self, st: Self::Vec) -> usize {
        Vec::len(&st)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CpuError {
    SmallProblem,
}

impl Display for CpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self)
    }
}

impl HasErr for CPU {
    type Err = CpuError;
    const ERR: Self::Err = CpuError::SmallProblem;
}

impl<E: Unit> ZerosTensor<E> for CPU {
    fn try_zeros_from<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err> {
        let shape = *src.shape();
        let data = vec![E::ZERO; shape.num_elements()];
        let data: Arc<RwLock<Vec<E>>> = Arc::new(RwLock::new(data));
        Ok(Tensor {
            shape,
            data,
            device: CPU::default(),
            id: unique_id(),
            tape: NoneTape,
        })
    }
}

impl<E: Unit + SampleUniform> RandTensor<E> for CPU {
    fn try_fill_rand<S: HasShape>(&self, src: &S) -> Result<Tensor<S::Shape, E, Self>, Self::Err>
    where
        Standard: Distribution<E>,
    {
        let shape = *src.shape();
        let mut out_data = vec![E::ZERO; shape.num_elements()];

        {
            let mut rng = rand::thread_rng();
            for el in out_data.iter_mut() {
                let el_rand: E = rng.gen();
                *el = el_rand;
            }
        }
        let out_data = Arc::new(RwLock::new(out_data));
        Ok(Tensor {
            shape,
            data: out_data,
            device: Self::default(),
            id: unique_id(),
            tape: NoneTape,
        })
    }

    fn try_fill_rand_range<S: HasShape>(
        &self,
        src: &S,
        range: Range<E>,
    ) -> Result<Tensor<S::Shape, E, Self>, Self::Err>
    where
        Standard: Distribution<E>,
    {
        let shape = *src.shape();
        let mut out_data = vec![E::ZERO; shape.num_elements()];

        let id = unique_id();

        {
            let between = Uniform::from(range);
            // let mut rng = rand::thread_rng();
            let mut rng = StdRng::seed_from_u64(id.0 as u64);

            for el in out_data.iter_mut() {
                let el_rand: E = between.sample(&mut rng);
                *el = el_rand;
            }
        }

        let out_data = Arc::new(RwLock::new(out_data));
        Ok(Tensor {
            shape,
            id,
            data: out_data,
            device: Self::default(),
            tape: NoneTape,
        })
    }
}

impl<E: Unit> Arange<E> for CPU {
    fn try_arange<S: Dim>(&self, end: &S) -> Result<Tensor<(S,), E, Self>, Self::Err> {
        let seq = (0..end.size())
            .map(|e| E::from_usize(e))
            .collect::<Vec<E>>();
        Ok(Tensor {
            id: unique_id(),
            shape: (end.clone(),),
            data: Arc::new(RwLock::new(seq)),
            device: Self::default(),
            tape: NoneTape,
        })
    }
}

impl<Y: Dim, X: Dim, E: Unit> Display for Tensor<(Y, X), E, CPU> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.data.read().unwrap().clone();
        write!(f, "[")?;
        for i_y in 0..self.shape.0.size() {
            write!(f, "{}[", " ".repeat((i_y != 0) as usize))?;

            for i_x in 0..self.shape.1.size() {
                Display::fmt(&data[self.shape.1.size() * (i_y) + i_x], f)?;
                write!(
                    f,
                    "{}",
                    ", ".repeat((i_x != self.shape.1.size() - 1) as usize)
                )?;
            }
            write!(
                f,
                "]{}\n",
                [",", "]"][(i_y == self.shape.0.size() - 1) as usize]
            )?;
        }

        Ok(())
    }
}

impl<Y: Dim, X: Dim, E: Unit> Display for Tensor<(Y, X), E, MetalGPU> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.data.read().unwrap().clone();
        write!(f, "[")?;
        for i_y in 0..self.shape.0.size() {
            write!(f, "{}[", " ".repeat((i_y != 0) as usize))?;

            for i_x in 0..self.shape.1.size() {
                Display::fmt(&data[self.shape.1.size() * (i_y) + i_x], f)?;
                write!(
                    f,
                    "{}",
                    ", ".repeat((i_x != self.shape.1.size() - 1) as usize)
                )?;
            }
            write!(
                f,
                "]{}\n",
                [",", "]"][(i_y == self.shape.0.size() - 1) as usize]
            )?;
        }

        Ok(())
    }
}

// impl<X: Dim, E: Unit> Display for Tensor<(X,), E, CPU> {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         let data = self.data.read().unwrap().clone();
//         let mut data = data.iter();

//         if self.shape.0.size() == 1 {
//             writeln!(f, "[{:?}]", data.next().unwrap())?;
//         } else {
//             write!(f, "[{:?},", data.next().unwrap())?;
//             for _ in 0..self.shape.0.size()-2 {
//                 write!(f, "{:?},", data.next().unwrap())?;
//             }
//             write!(f, "{:?}]", data.next().unwrap())?;
//         }

//         Ok(())
//     }
// }

impl<X: Dim, E: Unit, D: Storage<E>, T: Tape<E, MetalGPU>> Display for Tensor<(X,), E, D, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.data.read().unwrap().clone();
        let mut data = data.into_iter();

        if self.shape.0.size() == 1 {
            writeln!(f, "[{:?}]", data.next().unwrap())?;
        } else {
            write!(f, "[{:?},", data.next().unwrap())?;
            for _ in 0..self.shape.0.size() - 2 {
                write!(f, "{:?},", data.next().unwrap())?;
            }
            write!(f, "{:?}]", data.next().unwrap())?;
        }

        Ok(())
    }
}

impl<E: Unit> Display for Tensor<(), E, CPU> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.data.read().unwrap()[0])?;
        Ok(())
    }
}

#[allow(dead_code)]
impl CPU {
    pub fn try_from_2d_array<const I: usize, const J: usize, E: Unit>(
        &self,
        src: [[E; I]; J],
    ) -> Result<Tensor<Rank2<I, J>, E, Self>, CpuError> {
        let mut out_data = vec![E::ZERO; I * J];

        for i in 0..I {
            for j in 0..J {
                out_data[J * i + j] = src[i][j]
            }
        }
        let out_data = Arc::new(RwLock::new(out_data));

        Ok(Tensor {
            shape: (Const::<I>, Const::<J>),
            data: out_data,
            device: Self::default(),
            id: unique_id(),
            tape: NoneTape,
        })
    }

    pub fn try_from_array<const I: usize, E: Unit>(
        &self,
        src: [E; I],
    ) -> Result<Tensor<Rank1<I>, E, Self>, CpuError> {
        let mut out_data = vec![E::ZERO; I];

        for i_i in 0..I {
            out_data[i_i] = src[i_i];
        }

        let out_data = Arc::new(RwLock::new(out_data));

        Ok(Tensor {
            id: unique_id(),
            shape: (Const::<I>,),
            data: out_data,
            device: self.clone(),
            tape: NoneTape,
        })
    }

    pub fn from_2d_array<const I: usize, const J: usize, E: Unit>(
        &self,
        src: [[E; I]; J],
    ) -> Tensor<Rank2<I, J>, E, Self> {
        Self::try_from_2d_array(&self, src).unwrap()
    }

    pub fn from_array<const I: usize, E: Unit>(&self, src: [E; I]) -> Tensor<Rank1<I>, E, Self> {
        Self::try_from_array(&self, src).unwrap()
    }

    pub fn try_from_slice<E: Unit, S: Shape>(
        &self,
        src: &[E],
        shape: S,
    ) -> Result<Tensor<S, E, Self>, CpuError> {
        let out_data = Arc::new(RwLock::new(src.to_vec()));

        Ok(Tensor {
            shape,
            id: unique_id(),
            data: out_data,
            device: Self::default(),
            tape: NoneTape,
        })
    }

    pub fn from_slice<E: Unit, S: Shape>(&self, src: &[E], shape: S) -> Tensor<S, E, Self> {
        Self::try_from_slice(&self, src, shape).unwrap()
    }
}

use core::num;
use std::{fmt::{Debug, Display}, ops::{Add, AddAssign, Sub, Mul, Div, Neg, MulAssign}};

pub trait Unit: 
'static
+ Debug
+ Default
+ Copy
+ Clone
+ Add<Self, Output = Self>
+ AddAssign<Self>
+ Sub<Output = Self>
+ Mul<Output = Self>
+ MulAssign<Self>
+ Div<Output = Self>
+ PartialOrd<Self>
+ PartialEq<Self>
+ Neg<Output = Self>
+ Display
{
    const ONE: Self;
    const ZERO: Self;

    fn max(self, rhs: Self) -> Self {
        if self > rhs {
            return self
        } else {
            return rhs
        }
    }

    fn pow(self, exponent: u32) -> Self {
        let mut x = Self::ONE;
        for _ in 0..exponent {
            x = x * self;
        }
        x
    }

    fn abs(self) -> Self;

    fn from_u32(src: u32) -> Self;

    fn from_usize(src: usize) -> Self;
}

pub trait FloatUnit: Unit {
    const EPSILON: Self;

    fn exp(self) -> Self;
    fn ln(self) -> Self;
    
    fn log_10(self) -> Self;

    fn sqrt(self) -> Self;

    fn from_f32(src: f32) -> Self;
    fn from_f64(src: f64) -> Self;

}

macro_rules! craft_unit {
    ($rust_type: ty, $one: expr, $zero: expr) => {

        impl Unit for $rust_type {
            const ONE: Self = $one;
            const ZERO: Self = $zero;

            fn from_u32(src:u32) -> $rust_type {
                src as $rust_type
            }
            fn from_usize(src: usize) -> $rust_type {
                src as $rust_type
            }

            fn abs(self) -> Self {
                self.abs()
            }
        }

    };
}


craft_unit!(f32, 1.0_f32, 0.0_f32);
craft_unit!(f64, 1.0_f64, 0.0_f64);
craft_unit!(i32, 1_i32, 0_i32);

impl FloatUnit for f32 {
    const EPSILON: Self = 1e-04;
    fn exp(self) -> Self {
        self.exp()
    }
    fn ln(self) -> Self {
        self.ln()
    }
    
    fn log_10(self) -> Self {
        self.log10()
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn from_f32(src: f32) -> Self {
        return src as Self
    }
    fn from_f64(src: f64) -> Self {
        return src as Self
    }
    
}

impl FloatUnit for f64 {
    const EPSILON: Self = 1e-04;
    fn exp(self) -> Self {
        self.exp()
    }
    fn ln(self) -> Self {
        self.ln()
    }

    fn log_10(self) -> Self {
        self.log10()
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn from_f32(src: f32) -> Self {
        return src as Self
    }
    fn from_f64(src: f64) -> Self {
        return src as Self
    }

}

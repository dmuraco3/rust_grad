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
}

pub trait FloatUnit: Unit + From<f32> {
    fn exp(self) -> Self;

    fn ln(self) -> Self;

    fn log_10(self) -> Self;

}

macro_rules! craft_unit {
    ($rust_type: ty, $one: expr, $zero: expr) => {

        impl Unit for $rust_type {
            const ONE: Self = $one;
            const ZERO: Self = $zero;

            fn from_u32(src:u32) -> $rust_type {
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
    fn exp(self) -> Self {
        self.exp()
    }

    fn ln(self) -> Self {
        self.ln()
    }
    
    fn log_10(self) -> Self {
        self.log10()
    }
}

impl FloatUnit for f64 {
    fn exp(self) -> Self {
        self.exp()
    }

    fn ln(self) -> Self {
        self.ln()
    }

    fn log_10(self) -> Self {
        self.log10()
    }

}

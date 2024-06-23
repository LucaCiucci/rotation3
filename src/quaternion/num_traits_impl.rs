
use std::ops::Div;

use num_traits::Inv;

use super::*;

impl<T> Inv for Quaternion<T>
where
    T: Copy + Neg<Output = T> + Div<Output = T> + Add<Output = T> + Mul<Output = T>,
{
    type Output = Self;

    fn inv(self) -> Self {
        let conj = self.conjugate();
        let norm2 = self.norm_squared();
        conj / norm2
    }
}
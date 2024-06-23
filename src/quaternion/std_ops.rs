
use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, MulAssign, Div, DivAssign};

use num_traits::Inv;

use super::*;

impl<T> Add for Quaternion<T>
where
    T: Add<Output = T> + Copy,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(std::array::from_fn(|i| self.0[i] + rhs.0[i]))
    }
}

impl<T> AddAssign for Quaternion<T>
where
    T: AddAssign + Copy,
{
    fn add_assign(&mut self, rhs: Self) {
        self.0.iter_mut().zip(rhs.0.iter()).for_each(|(a, b)| *a += *b);
    }
}

impl<T> Sub for Quaternion<T>
where
    T: Sub<Output = T> + Copy,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(std::array::from_fn(|i| self.0[i] - rhs.0[i]))
    }
}

impl<T> SubAssign for Quaternion<T>
where
    T: SubAssign + Copy,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.0.iter_mut().zip(rhs.0.iter()).for_each(|(a, b)| *a -= *b);
    }
}

impl<T> Mul for Quaternion<T>
where
    T: Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Copy,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self([
            self.0[0] * rhs.0[0] - self.0[1] * rhs.0[1] - self.0[2] * rhs.0[2] - self.0[3] * rhs.0[3],
            self.0[0] * rhs.0[1] + self.0[1] * rhs.0[0] + self.0[2] * rhs.0[3] - self.0[3] * rhs.0[2],
            self.0[0] * rhs.0[2] - self.0[1] * rhs.0[3] + self.0[2] * rhs.0[0] + self.0[3] * rhs.0[1],
            self.0[0] * rhs.0[3] + self.0[1] * rhs.0[2] - self.0[2] * rhs.0[1] + self.0[3] * rhs.0[0],
        ])
    }
}

impl<T> MulAssign for Quaternion<T>
where
    T: Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Copy,
{
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<T> Mul<T> for Quaternion<T>
where
    T: Mul<Output = T> + Copy,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self(std::array::from_fn(|i| self.0[i] * rhs))
    }
}

impl<T> MulAssign<T> for Quaternion<T>
where
    T: MulAssign + Copy,
{
    fn mul_assign(&mut self, rhs: T) {
        self.0.iter_mut().for_each(|a| *a *= rhs);
    }
}

impl<T> Div for Quaternion<T>
where
    T: Copy,
    T: Neg<Output = T> + Div<Output = T> + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        self * rhs.inv()
    }
}

impl<T> DivAssign for Quaternion<T>
where
    T: Copy,
    T: Neg<Output = T> + Div<Output = T> + Add<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    #[allow(clippy::suspicious_op_assign_impl)]
    fn div_assign(&mut self, rhs: Self) {
        *self *= rhs.inv();
    }
}

impl<T> Div<T> for Quaternion<T>
where
    T: Div<Output = T> + Copy,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Self(std::array::from_fn(|i| self.0[i] / rhs))
    }
}

impl<T> DivAssign<T> for Quaternion<T>
where
    T: DivAssign + Copy,
{
    fn div_assign(&mut self, rhs: T) {
        self.0.iter_mut().for_each(|a| *a /= rhs);
    }
}
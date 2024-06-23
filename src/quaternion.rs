use std::{ops::{Neg, Add, Mul, Index}, fmt::Debug};

use nalgebra::Vector3;
use num_traits::{Zero, One, real::Real};

/// A quaternion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Quaternion<T = f64>([T; 4]);

mod std_ops;
mod num_traits_impl;

impl<T> Quaternion<T> {
    /// Creates a new quaternion from the given components.
    ///
    /// # Arguments
    ///  * `a` - The **real** component.
    ///  * `b` - The imaginary component along the **i** axis.
    ///  * `c` - The imaginary component along the **j** axis.
    ///  * `d` - The imaginary component along the **k** axis.
    pub fn new(a: T, b: T, c: T, d: T) -> Self {
        Quaternion([a, b, c, d])
    }

    /// Creates a new quaternion from the given array of components.
    pub fn from_components_array(arr: [T; 4]) -> Self {
        Quaternion(arr)
    }

    /// Creates a new quaternion from the given vector and scalar components.
    pub fn from_scalar_and_vector(s: T, v: Vector3<T>) -> Self
    where
        T: Copy,
    {
        Quaternion([s, v[0], v[1], v[2]])
    }

    /// Creates a new quaternion from the given vector and scalar components.
    /// 
    /// TODO explain why this is useful. (Differentiability at the origin)
    ///
    /// # Notes
    ///  * This function is differentiable at the origin for the first order (C1).
    pub fn from_rotation_vector_differentiable(rot: Vector3<T>) -> Self
    where
        T: Real
    {
        let norm_squared = rot[0].powi(2) + rot[1].powi(2) + rot[2].powi(2);

        if !norm_squared.is_zero() {
            let norm = norm_squared.sqrt();
            let half_angle = norm / T::from(2.0).unwrap();
            let (sin_half_angle, cos_half_angle) = half_angle.sin_cos();
            
            let scale = sin_half_angle / norm;
            Quaternion::new(
                cos_half_angle,
                rot[0] * scale,
                rot[1] * scale,
                rot[2] * scale,
            )
        } else {
            // We want to keep differentiability at the origin. To do this, we
            // call "x" the half angle (theta / 2). The imaginary components
            // will be:
            //   = (rot / norm) * sin(x)
            //   = rot * sin(x) / norm
            //   = rot * sin(x) / (x * 2)
            //   = rot * (sin(x) / x) / 2
            // But, at the origin, sin(x) / x = 1 at the first order. So, we
            // have:
            //   = rot / 2
            Quaternion::new(
                T::one(),
                rot[0] / T::from(2.0).unwrap(),
                rot[1] / T::from(2.0).unwrap(),
                rot[2] / T::from(2.0).unwrap(),
            )
        }
    }

    /// Returns the components of the quaternion.
    pub fn components(&self) -> &[T; 4] {
        &self.0
    }

    /// Returns the components of the quaternion.
    pub fn components_mut(&mut self) -> &mut [T; 4] {
        &mut self.0
    }

    /// Returns the real component of the quaternion.
    pub fn real(&self) -> &T {
        &self.0[0]
    }

    /// Returns the real component of the quaternion.
    pub fn real_mut(&mut self) -> &mut T {
        &mut self.0[0]
    }

    /// Returns the imaginary components of the quaternion.
    pub fn imag(&self) -> &[T] {
        &self.0[1..]
    }

    /// Returns the imaginary components of the quaternion.
    pub fn imag_mut(&mut self) -> &mut [T] {
        &mut self.0[1..]
    }

    /// Returns the imaginary components of the quaternion as a vector.
    pub fn imag_vec(&self) -> Vector3<T>
    where
        T: Copy,
    {
        Vector3::new(self.0[1], self.0[2], self.0[3])
    }

    /// Same as [`real`](Quaternion::real).
    pub fn scalar(&self) -> &T {
        &self.0[0]
    }

    /// Same as [`real_mut`](Quaternion::real_mut).
    pub fn scalar_mut(&mut self) -> &mut T {
        &mut self.0[0]
    }

    /// Returns the real unit quaternion.
    pub fn r() -> Self
    where
        T: Zero + One,
    {
        Quaternion([T::one(), T::zero(), T::zero(), T::zero()])
    }

    /// returns the **i** imaginary unit quaternion.
    pub fn i() -> Self
    where
        T: Zero + One,
    {
        Quaternion([T::zero(), T::one(), T::zero(), T::zero()])
    }

    /// Returns the **j** imaginary unit quaternion.
    pub fn j() -> Self
    where
        T: Zero + One,
    {
        Quaternion([T::zero(), T::zero(), T::one(), T::zero()])
    }

    /// Returns the **k** imaginary unit quaternion.
    pub fn k() -> Self
    where
        T: Zero + One,
    {
        Quaternion([T::zero(), T::zero(), T::zero(), T::one()])
    }

    /// Returns the conjugate of the quaternion.
    pub fn conjugate(&self) -> Self
    where
        T: Copy + Neg<Output = T>,
    {
        Quaternion([self.0[0], -self.0[1], -self.0[2], -self.0[3]])
    }

    /// Returns the squared norm of the quaternion.
    ///
    /// Returns `self * self.conjugate() = a^2 + b^2 + c^2 + d^2`.
    pub fn norm_squared(&self) -> T
    where
        T: Copy + Add<Output = T> + Mul<Output = T>,
    {
        self.0[0] * self.0[0] + self.0[1] * self.0[1] + self.0[2] * self.0[2] + self.0[3] * self.0[3]
    }

    /// Rotates a vector by the quaternion.
    pub fn rotate_vector(&self, v: Vector3<T>) -> Vector3<T>
    where
        T: Real,
    {
        let qv = Quaternion::from_scalar_and_vector(T::zero(), v);
        let qv = *self * qv * self.conjugate();
        qv.imag_vec()
    }
}

impl<T> Index<usize> for Quaternion<T> {
    type Output = T;

    fn index(&self, i: usize) -> &Self::Output {
        &self.0[i]
    }
}

impl<T> std::fmt::Display for Quaternion<T>
where
    T: std::fmt::Display,
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if f.alternate() {
            write!(f, "({} + {}i + {}j + {}k)", self.0[0], self.0[1], self.0[2], self.0[3])
        } else {
            write!(f, "{:?}", self.0)
        }
    }
}
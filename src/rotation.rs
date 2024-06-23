use std::{ops::{Mul, Index, IndexMut, Neg}, fmt::Debug};

use nalgebra::Vector3;
use num_traits::{real::Real, NumAssignOps};
use serde::{Serialize, Deserialize};
use nalgebra::Matrix3;

use crate::Quaternion;

/// A 3D rotation.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Rotation<T = f64>([T; 3]);

impl<T> Rotation<T> {
    /// Creates a new rotation.
    ///
    /// The rotation is specified by the three components of the rotation vector.
    /// The norm of the rotation vector is the angle of rotation in radians and
    /// the direction of the rotation vector is the axis of rotation.
    pub fn new(x: T, y: T, z: T) -> Self {
        Self([x, y, z])
    }

    /// Creates a new rotation from the given array of components.
    pub fn from_components_array(arr: [T; 3]) -> Self {
        Self(arr)
    }

    /// Creates a new rotation from the given vector.
    pub fn from_vector(v: Vector3<T>) -> Self
    where
        T: Copy,
    {
        Self([v[0], v[1], v[2]])
    }

    /// Creates a new rotation from the given quaternion.
    ///
    /// TODO explain why this is useful. (Differentiability at the origin)
    pub fn from_quaternion_differentiable(q: Quaternion<T>) -> Self
    where
        T: Real
    {
        let sin_theta_2_squared = {
            let imag = q.imag();
            imag[0] * imag[0] + imag[1] * imag[1] + imag[2] * imag[2]
        };

        if !sin_theta_2_squared.is_zero() {
            let sin_theta_2 = sin_theta_2_squared.sqrt();
            let cos_theta_2 = q[0]; // TODO maybe abs??

            let theta_2 = sin_theta_2.atan2(cos_theta_2);
            let theta = theta_2 * T::from(2.0).unwrap();

            let scale = theta / sin_theta_2;

            Self::new(q[1] * scale, q[2] * scale, q[3] * scale)
        } else {
            Self::new(
                q[1] * T::from(2.0).unwrap(),
                q[2] * T::from(2.0).unwrap(),
                q[3] * T::from(2.0).unwrap(),
            )
        }
    }

    /// Creates a new rotation from the euler angles.
    pub fn from_euler_angles(yaw: T, pitch: T, roll: T) -> Self
    where
        T: Real,
    {
        Self::from_components_array([T::zero(), yaw, T::zero()]) *
        Self::from_components_array([pitch, T::zero(), T::zero()]) *
        Self::from_components_array([T::zero(), T::zero(), roll])
    }

    /// Finds the rotation between two vectors.
    pub fn between(from: Vector3<T>, to: Vector3<T>) -> Self
    where
        T: Real + NumAssignOps + Debug + 'static,
    {
        if from == to {
            // TODO differentiability at the origin
            return Self::new(T::zero(), T::zero(), T::zero());
        }

        // TODO use nalgebra norm
        let from_norm_squared = from.x.powi(2) + from.y.powi(2) + from.z.powi(2);
        let to_norm_squared = to.x.powi(2) + to.y.powi(2) + to.z.powi(2);

        if from_norm_squared.is_zero() || to_norm_squared.is_zero() {
            return Self::new(T::zero(), T::zero(), T::zero());
        }

        let from_norm = from_norm_squared.sqrt();
        let to_norm = to_norm_squared.sqrt();

        // cross product
        let cross = from.cross(&to) / (from_norm * to_norm);
        let dot = from.dot(&to) / (from_norm * to_norm);

        // TODO use nalgebra norm
        let cross_norm_squared = cross.x.powi(2) + cross.y.powi(2) + cross.z.powi(2);
        let cross_norm = cross_norm_squared.sqrt();

        let angle = cross_norm.atan2(dot);

        let dir = cross / cross_norm;
        Self::new(dir.x * angle, dir.y * angle, dir.z * angle)
    }

    /// Returns the rotation vector.
    pub fn norm_squared(&self) -> T
    where
        T: Real
    {
        self.0[0] * self.0[0] + self.0[1] * self.0[1] + self.0[2] * self.0[2]
    }

    /// Returns the rotation vector.
    pub fn norm(&self) -> T
    where
        T: Real
    {
        self.norm_squared().sqrt()
    }

    /// Returns the rotation vector.
    pub fn angle(&self) -> T
    where
        T: Real
    {
        self.norm()
    }

    /// Returns the rotation axis.
    pub fn axis(&self) -> Vector3<T>
    where
        T: Real
    {
        let n = self.norm();
        if n == T::zero() {
            Vector3::new(T::zero(), T::zero(), T::zero())
        } else {
            Vector3::new(self.0[0] / n, self.0[1] / n, self.0[2] / n)
        }
    }

    /// Returns this as a vector
    pub fn to_vector(&self) -> Vector3<T>
    where
        T: Copy,
    {
        Vector3::new(self.0[0], self.0[1], self.0[2])
    }

    /// Returns this as a quaternion
    pub fn to_quaternion(&self) -> Quaternion<T>
    where
        T: Real
    {
        Quaternion::from_rotation_vector_differentiable(self.to_vector())
    }

    /// Rotates the given vector by this rotation.
    pub fn rotate_vector(&self, vec: Vector3<T>) -> Vector3<T>
    where
        T: Real
    {
        self.to_quaternion().rotate_vector(vec)
    }

    /// Returns the inverse of this rotation.
    pub fn inverse(&self) -> Self
    where
        T: Real
    {
        // the inverse is just a rotation with opposite angle and same axis
        Self::from_components_array([
            -self.0[0],
            -self.0[1],
            -self.0[2],
        ])
    }

    /// Returns the rotation matrix.
    pub fn to_matrix(&self) -> Matrix3<T>
    where
        T: Real // TODO less restrictive bound?
    {
        let v1 = self.rotate_vector(Vector3::new(T::one(), T::zero(), T::zero()));
        let v2 = self.rotate_vector(Vector3::new(T::zero(), T::one(), T::zero()));
        let v3 = self.rotate_vector(Vector3::new(T::zero(), T::zero(), T::one()));
        Matrix3::new(
            v1[0], v2[0], v3[0],
            v1[1], v2[1], v3[1],
            v1[2], v2[2], v3[2],
        )
    }

    // TODO to quaternion and to nalgebra quaternion
}

impl<T> Default for Rotation<T>
where
    T: Real
{
    fn default() -> Self {
        Self::new(T::zero(), T::zero(), T::zero())
    }
}

impl<T> Mul<Vector3<T>> for Rotation<T>
where
    T: Real
{
    type Output = Vector3<T>;

    fn mul(self, rhs: Vector3<T>) -> Self::Output {
        self.rotate_vector(rhs)
    }
}

impl<T> Mul for Rotation<T>
where
    T: Real
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::from_quaternion_differentiable(self.to_quaternion() * rhs.to_quaternion())
    }
}

impl<T> Mul<T> for Rotation<T>
where
    T: Real
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self::from_components_array([
            self.0[0] * rhs,
            self.0[1] * rhs,
            self.0[2] * rhs,
        ])
    }
}

impl<T> Neg for Rotation<T>
where
    T: Real
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.inverse()
    }
}

impl<T> Index<usize> for Rotation<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T> IndexMut<usize> for Rotation<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

#[cfg(test)]
mod tests {

    use approx::assert_ulps_eq;

    use super::*;

    #[test]
    fn angle_between_dirs() {
        let dir1 = Vector3::new(1.0, 0.0, 0.0);
        let dir2 = Vector3::new(0.0, 2.0, 0.0);
        let expected = Rotation::new(0.0, 0.0, 1.0) * std::f64::consts::FRAC_PI_2;
        let actual = Rotation::between(dir1, dir2);
        assert_ulps_eq!(actual[0], expected[0]);
        assert_ulps_eq!(actual[1], expected[1]);
        assert_ulps_eq!(actual[2], expected[2]);

        let dir1 = Vector3::new(1.0, 0.0, 0.0);
        let dir2 = Vector3::new(1.0, 1.0, 0.0);
        let expected = Rotation::new(0.0, 0.0, 1.0) * std::f64::consts::FRAC_PI_4;
        let actual = Rotation::between(dir1, dir2);

        assert_ulps_eq!(actual[0], expected[0]);
        assert_ulps_eq!(actual[1], expected[1]);
        assert_ulps_eq!(actual[2], expected[2]);
    }
}
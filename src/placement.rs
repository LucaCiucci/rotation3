/*!
Math utils to work with 3D placements (position and rotation) and differential forms.
 */
use std::{fmt::Debug, ops::Mul};

use nalgebra::{Vector3, Point3, Matrix3};
use num_traits::{real::Real, NumOps, NumAssignOps};
use serde::{Serialize, Deserialize};

use super::rotation::Rotation;

// TODO move under
/// A precomputed placement (precomputed rotation matrix)
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct PrecomputedPlacement3<T = f64>
where
    T: Real + Debug + 'static,
{
    /// The rotation matrix
    pub rotation: Matrix3<T>,
    /// The translation vector
    pub translation: Vector3<T>,
}

impl<T> Mul<Point3<T>> for PrecomputedPlacement3<T>
where
    T: Real + Debug + 'static + NumOps + NumAssignOps,
{
    type Output = Point3<T>;
    fn mul(self, rhs: Point3<T>) -> Self::Output {
        self.rotation * rhs + self.translation
    }
}

impl<T> Mul<&Point3<T>> for &PrecomputedPlacement3<T>
where
    T: Real + Debug + 'static + NumOps + NumAssignOps,
{
    type Output = Point3<T>;
    fn mul(self, rhs: &Point3<T>) -> Self::Output {
        self.rotation * rhs + self.translation
    }
}

impl<T> Mul<Vector3<T>> for PrecomputedPlacement3<T>
where
    T: Real + Debug + 'static + NumOps + NumAssignOps,
{
    type Output = Vector3<T>;
    fn mul(self, rhs: Vector3<T>) -> Self::Output {
        self.rotation * rhs
    }
}

impl<T> Mul<&Vector3<T>> for &PrecomputedPlacement3<T>
where
    T: Real + Debug + 'static + NumOps + NumAssignOps,
{
    type Output = Vector3<T>;
    fn mul(self, rhs: &Vector3<T>) -> Self::Output {
        self.rotation * rhs
    }
}

/// Position and rotation of in a 3D space
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Placement3<T = f64>
where
    T: Real + Debug + 'static,
{
    /// Position of the object
    pub position: Vector3<T>,

    /// Rotation of the object
    pub rotation: Rotation<T>,
}

impl<T> Default for Placement3<T>
where
    T: Real + Debug + 'static,
{
    fn default() -> Self {
        Self {
            position: Vector3::new(T::zero(), T::zero(), T::zero()),
            rotation: Rotation::default(),
        }
    }
}

impl<T> Placement3<T>
where
    T: Real + Debug + 'static + NumOps + NumAssignOps,
{
    /// Precomputes the placement
    pub fn precompute(&self) -> PrecomputedPlacement3<T> {
        PrecomputedPlacement3 {
            rotation: self.rotation.to_matrix(),
            translation: self.position,
        }
    }

    /// Transforms a point from the local coordinate system to the global coordinate system.
    pub fn transform_point(&self, point: &Point3<T>) -> Point3<T> {
        Point3 {
            coords: self.transform_vector_as_point(&point.coords)
        }
    }

    /// Transforms a vector from the local coordinate system to the global coordinate system,
    /// assuming that the vector is a point,
    pub fn transform_vector_as_point(&self, point: &Vector3<T>) -> Vector3<T> {
        self.rotation.rotate_vector(*point) + self.position
    }

    /// Transforms a vector from the local coordinate system to the global coordinate system,
    /// assuming that the vector is a direction (i.e. it has **no translation** component).
    pub fn transform_vector(&self, vector: &Vector3<T>) -> Vector3<T> {
        self.rotation.rotate_vector(*vector)
    }

    /// Inverse of the placement
    pub fn inverse(&self) -> Self {
        /*
        The placement is defined as:
        p' = R * p + t
        where p' is the point in the global coordinate system, p is the point in the local coordinate system,
        R is the rotation matrix and t is the translation vector.

        By subtracting t from both sides, we get:
        p' - t = R * p
        thus
        p = R^T * (p' - t) = R^T * p' - R^T * t
        */

        let mut inv = *self;
        inv.rotation = self.rotation.inverse();
        inv.position = -inv.rotation.rotate_vector(self.position);
        inv
    }
}

impl<T: Real> Mul for Placement3<T>
where
    T: Debug + 'static + NumOps + NumAssignOps,
{
    type Output = Placement3<T>;
    fn mul(self, other: Placement3<T>) -> Placement3<T> {
        Placement3 {
            position: self.position + self.rotation.rotate_vector(other.position),
            rotation: self.rotation * other.rotation,
        }
    }
}
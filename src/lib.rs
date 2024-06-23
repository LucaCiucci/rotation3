
pub mod external {
    pub use nalgebra;
}

mod quaternion;
mod rotation;

pub use quaternion::*;
pub use rotation::*;
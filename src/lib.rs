
pub mod external {
    pub use nalgebra;
}

mod quaternion;
mod rotation;
mod placement;

pub use quaternion::*;
pub use rotation::*;
pub use placement::*;
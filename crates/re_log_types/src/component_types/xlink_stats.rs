use crate::{Component, EntityPath};
use arrow2::array::Int128Array;
use arrow2_convert::{field::I128, ArrowDeserialize, ArrowField, ArrowSerialize};

// TODO(filip): Convert to use i128
#[derive(Clone, Debug, PartialEq, ArrowField, ArrowSerialize, ArrowDeserialize)]
pub struct XlinkStats {
    pub bytes_read: i64,
    pub bytes_written: i64,
}

impl XlinkStats {
    pub fn entity_path() -> EntityPath {
        "xlink_stats".into()
    }
}

impl Component for XlinkStats {
    #[inline]
    fn name() -> crate::ComponentName {
        "rerun.xlink_stats".into()
    }
}

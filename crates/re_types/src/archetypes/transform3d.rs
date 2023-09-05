// DO NOT EDIT!: This file was auto-generated by crates/re_types_builder/src/codegen/rust/api.rs:165.

#![allow(trivial_numeric_casts)]
#![allow(unused_parens)]
#![allow(clippy::clone_on_copy)]
#![allow(clippy::iter_on_single_items)]
#![allow(clippy::map_flatten)]
#![allow(clippy::match_wildcard_for_single_variants)]
#![allow(clippy::needless_question_mark)]
#![allow(clippy::redundant_closure)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::unnecessary_cast)]

/// A 3D transform.
///
/// ## Example
///
/// ```ignore
/// //! Log some transforms.
///
/// use rerun::{
///    archetypes::{Arrows3D, Transform3D},
///    datatypes::{
///        Angle, Mat3x3, RotationAxisAngle, Scale3D, TranslationAndMat3x3, TranslationRotationScale3D,
///    },
///    RecordingStreamBuilder,
/// };
/// use std::f32::consts::PI;
///
/// fn main() -> Result<(), Box<dyn std::error::Error>> {
///    let (rec, storage) = RecordingStreamBuilder::new("rerun_example_transform3d").memory()?;
///
///    rec.log("base", &Arrows3D::new([(0.0, 1.0, 0.0)]))?;
///
///    rec.log(
///        "base/translated",
///        &Transform3D::new(TranslationAndMat3x3::new([1.0, 0.0, 0.0], Mat3x3::IDENTITY)),
///    )?;
///
///    rec.log("base/translated", &Arrows3D::new([(0.0, 1.0, 0.0)]))?;
///
///    rec.log(
///        "base/rotated_scaled",
///        &Transform3D::new(TranslationRotationScale3D {
///            rotation: Some(RotationAxisAngle::new([0.0, 0.0, 1.0], Angle::Radians(PI / 4.)).into()),
///            scale: Some(Scale3D::from(2.0)),
///            ..Default::default()
///        }),
///    )?;
///
///    rec.log("base/rotated_scaled", &Arrows3D::new([(0.0, 1.0, 0.0)]))?;
///
///    rerun::native_viewer::show(storage.take())?;
///    Ok(())
/// }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Transform3D {
    /// The transform
    pub transform: crate::components::Transform3D,
}

static REQUIRED_COMPONENTS: once_cell::sync::Lazy<[crate::ComponentName; 1usize]> =
    once_cell::sync::Lazy::new(|| ["rerun.transform3d".into()]);

static RECOMMENDED_COMPONENTS: once_cell::sync::Lazy<[crate::ComponentName; 0usize]> =
    once_cell::sync::Lazy::new(|| []);

static OPTIONAL_COMPONENTS: once_cell::sync::Lazy<[crate::ComponentName; 0usize]> =
    once_cell::sync::Lazy::new(|| []);

static ALL_COMPONENTS: once_cell::sync::Lazy<[crate::ComponentName; 1usize]> =
    once_cell::sync::Lazy::new(|| ["rerun.transform3d".into()]);

impl Transform3D {
    pub const NUM_COMPONENTS: usize = 1usize;
}

impl crate::Archetype for Transform3D {
    #[inline]
    fn name() -> crate::ArchetypeName {
        "rerun.archetypes.Transform3D".into()
    }

    #[inline]
    fn required_components() -> ::std::borrow::Cow<'static, [crate::ComponentName]> {
        REQUIRED_COMPONENTS.as_slice().into()
    }

    #[inline]
    fn recommended_components() -> ::std::borrow::Cow<'static, [crate::ComponentName]> {
        RECOMMENDED_COMPONENTS.as_slice().into()
    }

    #[inline]
    fn optional_components() -> ::std::borrow::Cow<'static, [crate::ComponentName]> {
        OPTIONAL_COMPONENTS.as_slice().into()
    }

    #[inline]
    fn all_components() -> ::std::borrow::Cow<'static, [crate::ComponentName]> {
        ALL_COMPONENTS.as_slice().into()
    }

    #[inline]
    fn indicator_component() -> crate::ComponentName {
        "rerun.components.Transform3DIndicator".into()
    }

    #[inline]
    fn num_instances(&self) -> usize {
        1
    }

    fn as_component_lists(&self) -> Vec<&dyn crate::ComponentList> {
        [Some(&self.transform as &dyn crate::ComponentList)]
            .into_iter()
            .flatten()
            .collect()
    }

    #[inline]
    fn try_to_arrow(
        &self,
    ) -> crate::SerializationResult<
        Vec<(::arrow2::datatypes::Field, Box<dyn ::arrow2::array::Array>)>,
    > {
        use crate::{Loggable as _, ResultExt as _};
        Ok([
            {
                Some({
                    let array = <crate::components::Transform3D>::try_to_arrow([&self.transform]);
                    array.map(|array| {
                        let datatype = ::arrow2::datatypes::DataType::Extension(
                            "rerun.components.Transform3D".into(),
                            Box::new(array.data_type().clone()),
                            Some("rerun.transform3d".into()),
                        );
                        (
                            ::arrow2::datatypes::Field::new("transform", datatype, false),
                            array,
                        )
                    })
                })
                .transpose()
                .with_context("rerun.archetypes.Transform3D#transform")?
            },
            {
                let datatype = ::arrow2::datatypes::DataType::Extension(
                    "rerun.components.Transform3DIndicator".to_owned(),
                    Box::new(::arrow2::datatypes::DataType::Null),
                    Some("rerun.components.Transform3DIndicator".to_owned()),
                );
                let array = ::arrow2::array::NullArray::new(
                    datatype.to_logical_type().clone(),
                    self.num_instances(),
                )
                .boxed();
                Some((
                    ::arrow2::datatypes::Field::new(
                        "rerun.components.Transform3DIndicator",
                        datatype,
                        false,
                    ),
                    array,
                ))
            },
        ]
        .into_iter()
        .flatten()
        .collect())
    }

    #[inline]
    fn try_from_arrow(
        arrow_data: impl IntoIterator<
            Item = (::arrow2::datatypes::Field, Box<dyn ::arrow2::array::Array>),
        >,
    ) -> crate::DeserializationResult<Self> {
        use crate::{Loggable as _, ResultExt as _};
        let arrays_by_name: ::std::collections::HashMap<_, _> = arrow_data
            .into_iter()
            .map(|(field, array)| (field.name, array))
            .collect();
        let transform = {
            let array = arrays_by_name
                .get("transform")
                .ok_or_else(crate::DeserializationError::missing_data)
                .with_context("rerun.archetypes.Transform3D#transform")?;
            <crate::components::Transform3D>::try_from_arrow_opt(&**array)
                .with_context("rerun.archetypes.Transform3D#transform")?
                .into_iter()
                .next()
                .flatten()
                .ok_or_else(crate::DeserializationError::missing_data)
                .with_context("rerun.archetypes.Transform3D#transform")?
        };
        Ok(Self { transform })
    }
}

impl Transform3D {
    pub fn new(transform: impl Into<crate::components::Transform3D>) -> Self {
        Self {
            transform: transform.into(),
        }
    }
}

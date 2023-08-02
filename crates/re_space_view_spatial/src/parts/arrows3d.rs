use re_data_store::{EntityPath, InstancePathHash};
use re_query::{ArchetypeView, QueryError};
use re_renderer::renderer::LineStripFlags;
use re_types::{
    archetypes::Arrows3D,
    components::{Arrow3D, Label},
    Archetype as _,
};
use re_viewer_context::{
    ArchetypeDefinition, ResolvedAnnotationInfo, SpaceViewSystemExecutionError,
    ViewContextCollection, ViewPartSystem, ViewQuery, ViewerContext,
};

use super::{picking_id_from_instance_key, SpatialViewPartData};
use crate::{
    contexts::{EntityDepthOffsets, SpatialSceneEntityContext},
    parts::{
        entity_iterator::process_archetype_views, process_annotations_and_keypoints,
        process_colors, process_radii, UiLabel, UiLabelTarget,
    },
    view_kind::SpatialSpaceViewKind,
};

pub struct Arrows3DPart {
    /// If the number of arrows in the batch is > max_labels, don't render point labels.
    pub max_labels: usize,
    pub data: SpatialViewPartData,
}

impl Default for Arrows3DPart {
    fn default() -> Self {
        Self {
            max_labels: 10,
            data: SpatialViewPartData::new(Some(SpatialSpaceViewKind::ThreeD)),
        }
    }
}

impl Arrows3DPart {
    fn process_labels<'a>(
        arch_view: &'a ArchetypeView<Arrows3D>,
        instance_path_hashes: &'a [InstancePathHash],
        colors: &'a [egui::Color32],
        annotation_infos: &'a [ResolvedAnnotationInfo],
        world_from_obj: glam::Affine3A,
    ) -> Result<impl Iterator<Item = UiLabel> + 'a, QueryError> {
        let labels = itertools::izip!(
            annotation_infos.iter(),
            arch_view.iter_required_component::<Arrow3D>()?,
            arch_view.iter_optional_component::<Label>()?,
            colors,
            instance_path_hashes,
        )
        .filter_map(
            move |(annotation_info, arrow, label, color, labeled_instance)| {
                let label = annotation_info.label(label.map(|l| l.0).as_ref());
                match (arrow, label) {
                    (arrow, Some(label)) => {
                        let midpoint = (glam::Vec3::from(arrow.origin())
                            + glam::Vec3::from(arrow.vector()))
                            * 0.45; // `0.45` rather than `0.5` to account for cap and such
                        Some(UiLabel {
                            text: label,
                            color: *color,
                            target: UiLabelTarget::Position3D(
                                world_from_obj.transform_point3(midpoint),
                            ),
                            labeled_instance: *labeled_instance,
                        })
                    }
                    _ => None,
                }
            },
        );
        Ok(labels)
    }

    fn process_arch_view(
        &mut self,
        query: &ViewQuery<'_>,
        arch_view: &ArchetypeView<Arrows3D>,
        ent_path: &EntityPath,
        ent_context: &SpatialSceneEntityContext<'_>,
    ) -> Result<(), QueryError> {
        let (annotation_infos, _) = process_annotations_and_keypoints::<Arrow3D, Arrows3D>(
            query,
            arch_view,
            &ent_context.annotations,
            |arrow| arrow.origin().into(),
        )?;

        let colors = process_colors(arch_view, ent_path, &annotation_infos)?;
        let radii = process_radii(arch_view, ent_path)?;

        if arch_view.num_instances() <= self.max_labels {
            // Max labels is small enough that we can afford iterating on the colors again.
            let colors =
                process_colors(arch_view, ent_path, &annotation_infos)?.collect::<Vec<_>>();

            let instance_path_hashes_for_picking = {
                re_tracing::profile_scope!("instance_hashes");
                arch_view
                    .iter_instance_keys()
                    .map(|instance_key| InstancePathHash::instance(ent_path, instance_key))
                    .collect::<Vec<_>>()
            };

            self.data.ui_labels.extend(Self::process_labels(
                arch_view,
                &instance_path_hashes_for_picking,
                &colors,
                &annotation_infos,
                ent_context.world_from_obj,
            )?);
        }

        let mut line_builder = ent_context.shared_render_builders.lines();
        let mut line_batch = line_builder
            .batch("arrows")
            .world_from_obj(ent_context.world_from_obj)
            .outline_mask_ids(ent_context.highlight.overall)
            .picking_object_id(re_renderer::PickingLayerObjectId(ent_path.hash64()));

        let instance_keys = arch_view.iter_instance_keys();
        let pick_ids = arch_view
            .iter_instance_keys()
            .map(picking_id_from_instance_key);
        let arrows = arch_view.iter_required_component::<Arrow3D>()?;

        let mut bounding_box = macaw::BoundingBox::nothing();

        for (instance_key, arrow, radius, color, pick_id) in
            itertools::izip!(instance_keys, arrows, radii, colors, pick_ids)
        {
            let origin: glam::Vec3 = arrow.origin().into();
            let vector: glam::Vec3 = arrow.vector().into();
            let end = origin + vector;

            let segment = line_batch
                .add_segment(origin, end)
                .radius(radius)
                .color(color)
                .flags(
                    LineStripFlags::FLAG_COLOR_GRADIENT
                        | LineStripFlags::FLAG_CAP_END_TRIANGLE
                        | LineStripFlags::FLAG_CAP_START_ROUND
                        | LineStripFlags::FLAG_CAP_START_EXTEND_OUTWARDS,
                )
                .picking_instance_id(pick_id);

            if let Some(outline_mask_ids) = ent_context.highlight.instances.get(&instance_key) {
                segment.outline_mask_ids(*outline_mask_ids);
            }

            bounding_box.extend(origin);
            bounding_box.extend(end);
        }

        self.data
            .extend_bounding_box(bounding_box, ent_context.world_from_obj);

        Ok(())
    }
}

impl ViewPartSystem for Arrows3DPart {
    fn archetype(&self) -> ArchetypeDefinition {
        Arrows3D::all_components().try_into().unwrap()
    }

    fn execute(
        &mut self,
        ctx: &mut ViewerContext<'_>,
        query: &ViewQuery<'_>,
        view_ctx: &ViewContextCollection,
    ) -> Result<Vec<re_renderer::QueueableDrawData>, SpaceViewSystemExecutionError> {
        re_tracing::profile_scope!("Arrows3DPart");

        process_archetype_views::<Arrows3D, { Arrows3D::NUM_COMPONENTS }, _>(
            ctx,
            query,
            view_ctx,
            view_ctx.get::<EntityDepthOffsets>()?.points,
            |_ctx, ent_path, arch_view, ent_context| {
                self.process_arch_view(query, &arch_view, ent_path, ent_context)
            },
        )?;

        Ok(Vec::new()) // TODO(andreas): Optionally return point & line draw data once SharedRenderBuilders is gone.
    }

    fn data(&self) -> Option<&dyn std::any::Any> {
        Some(self.data.as_any())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
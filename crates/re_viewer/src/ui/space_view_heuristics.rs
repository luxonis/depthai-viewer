use std::{borrow::BorrowMut, collections::BTreeMap};

use ahash::HashMap;
use itertools::Itertools;
use nohash_hasher::IntSet;
use re_arrow_store::{DataStore, LatestAtQuery, Timeline};
use re_data_store::{log_db::EntityDb, query_latest_single, ComponentName, EntityPath, EntityTree};
use re_log_types::{
    component_types::{Tensor, TensorDataMeaning},
    Component, EntityPathPart,
};

use crate::{
    misc::{space_info::SpaceInfoCollection, ViewerContext},
    ui::{view_category::categorize_entity_path, ViewCategory},
};

use super::{view_category::ViewCategorySet, view_spatial::SpatialNavigationMode, SpaceView};

/// List out all space views we allow the user to create.
pub fn all_possible_space_views(
    ctx: &ViewerContext<'_>,
    spaces_info: &SpaceInfoCollection,
) -> Vec<SpaceView> {
    crate::profile_function!();

    // Everything with a SpaceInfo is a candidate (that is root + whenever there is a transform),
    // as well as all direct descendants of the root that have some messages.
    let root_children = &ctx
        .log_db
        .entity_db
        .tree
        .children
        .iter()
        .filter(|(_k, v)| {
            let timelines = v.prefix_times.timelines();
            let mut total_msgs = 0;
            for timeline in timelines {
                if let Some(hist) = v.prefix_times.get(timeline) {
                    total_msgs += hist.total_count();
                }
            }
            total_msgs != 0
        })
        .collect::<BTreeMap<&EntityPathPart, &EntityTree>>();

    let candidate_space_paths = spaces_info
        .iter()
        .map(|info| &info.path)
        .chain(root_children.values().map(|sub_tree| &sub_tree.path))
        .unique();

    // For each candidate, create space views for all possible categories.
    candidate_space_paths
        .flat_map(|candidate_space_path| {
            default_queried_entities_by_category(ctx, candidate_space_path, spaces_info)
                .iter()
                .map(|(category, entity_paths)| {
                    SpaceView::new(ctx, *category, candidate_space_path, entity_paths)
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn contains_any_image(
    entity_path: &EntityPath,
    entity_db: &EntityDb,
    query: &LatestAtQuery,
) -> bool {
    re_query::query_entity_with_primary::<Tensor>(&entity_db.data_store, query, entity_path, &[])
        .map_or(false, |entity_view| {
            entity_view
                .iter_primary_flattened()
                .any(|tensor| tensor.is_shaped_like_an_image())
        })
}

fn is_interesting_space_view_at_root(
    entity_db: &EntityDb,
    candidate: &SpaceView,
    query: &LatestAtQuery,
) -> bool {
    // Not interesting if it has only data blueprint groups and no direct entities.
    // -> If there In that case we want spaceviews at those groups.
    if candidate.data_blueprint.root_group().entities.is_empty() {
        return false;
    }

    // If there are any images directly under the root, don't create root space either.
    // -> For images we want more fine grained control and resort to child-of-root spaces only.
    for entity_path in &candidate.data_blueprint.root_group().entities {
        if contains_any_image(entity_path, entity_db, query) {
            return false;
        }
    }

    true
}

fn is_interesting_space_view_not_at_root(
    entity_db: &EntityDb,
    candidate: &SpaceView,
    categories_with_interesting_roots: &ViewCategorySet,
    query: &LatestAtQuery,
) -> bool {
    // Consider children of the root interesting, *unless* a root with the same category was already considered interesting!
    if candidate.space_path.len() == 1
        && !categories_with_interesting_roots.contains(candidate.category)
    {
        return true;
    }

    // .. otherwise, spatial views are considered only interesting if they have an interesting transform.
    // -> If there is no transform or just a rigid transform, it is trivial to display in a root/child-of-root space view.
    //    If however there is ..
    //       .. an unknown transform, the children can't be shown otherwise
    //       .. an pinhole transform, we'd like to see the world from this camera's pov as well!
    if candidate.category == ViewCategory::Spatial {
        if let Some(transform) = query_latest_single(entity_db, &candidate.space_path, query) {
            match transform {
                re_log_types::Transform::Rigid3(_) => {}
                re_log_types::Transform::Pinhole(_) | re_log_types::Transform::Unknown => {
                    return true;
                }
            }
        }
    }

    // Not interesting!
    false
}

/// Function to create the non data space views. Configuration panel and blueprint panel.
fn create_blueprint_and_selection_space_view() {
    // let mut config_viewport =
}

/// List out all space views we generate by default for the available data.
pub fn default_created_space_views(
    ctx: &ViewerContext<'_>,
    spaces_info: &SpaceInfoCollection,
) -> Vec<SpaceView> {
    let candidates = all_possible_space_views(ctx, spaces_info);
    let default_space_views =
        default_created_space_views_from_candidates(ctx, &ctx.log_db.entity_db, candidates);
    if !ctx.depthai_state.selected_device.id.is_empty() {
        return default_depthai_space_views(ctx, default_space_views);
    }
    default_space_views
}

fn default_depthai_space_views(
    ctx: &ViewerContext<'_>,
    default_created_views: Vec<SpaceView>,
) -> Vec<SpaceView> {
    // Remove 3D views that don't have a Depth tensor in them,
    let mut cam_with_depth = None;
    let mut space_views = default_created_views
        .into_iter()
        .filter_map(|mut space_view| {
            // filter_map, so space_view is moved into the closure.
            if space_view.space_path.len() == 1 {
                // These are CAM_A, CAM_B, ..., basically the root spaces.
                // Check if there is a depth tensor in the space view.
                let query = LatestAtQuery::new(Timeline::log_time(), re_arrow_store::TimeInt::MAX);
                let entity_db = &ctx.log_db.entity_db;
                for entity_path in space_view.data_blueprint.entity_paths() {
                    if let Ok(entity_view) = re_query::query_entity_with_primary::<Tensor>(
                        &entity_db.data_store,
                        &query,
                        entity_path,
                        &[],
                    ) {
                        for tensor in entity_view.iter_primary_flattened() {
                            if tensor.meaning() == TensorDataMeaning::Depth {
                                cam_with_depth = space_view.space_path.iter().last().cloned();
                                // We also wan't to remove the Image and Detections from the 3D space view!
                                let entity_paths = space_view.data_blueprint.entity_paths().clone();
                                let entities_to_remove = entity_paths.iter().filter(|ep| {
                                    ep.last() == Some(&EntityPathPart::Name("Image".into()))
                                        || ep.last()
                                            == Some(&EntityPathPart::Name("Detections".into()))
                                        || ep.last()
                                            == Some(&EntityPathPart::Name("Detection".into()))
                                });
                                for entity in entities_to_remove {
                                    let mut props = space_view
                                        .data_blueprint
                                        .data_blueprints_individual()
                                        .get(&entity);
                                    props.visible = false;
                                    space_view
                                        .data_blueprint
                                        .data_blueprints_individual()
                                        .set(entity.clone(), props);
                                }
                                return Some(space_view);
                            }
                        }
                    }
                }
                return None; // No depth tensor was found inside of this 3D view.
            }
            Some(space_view)
        })
        .collect::<Vec<_>>();

    // If a depth tensor is found, we want to find the 2D space view that has the Image + Depth tensor.
    // We then wan't to create two separate 2D space views, one for the image and one for the depth.
    // But we only want to hide the depth (or image), not remove it from the space view.
    if let Some(depth_2d) = space_views
        .iter_mut()
        .find(|space_view| space_view.space_path.as_slice().first() == cam_with_depth.as_ref())
    {
        if let Some(image_entity) = depth_2d
            .data_blueprint
            .entity_paths()
            .iter()
            .find(|entity_path| entity_path.last() == Some(&EntityPathPart::Name("Image".into())))
            .cloned()
        {
            let mut duplicate = depth_2d.duplicate(ctx);
            // Change depth 2d to only show depth.
            let mut depth_2d_props = depth_2d
                .data_blueprint
                .data_blueprints_individual()
                .get(&image_entity);
            depth_2d_props.visible = false;
            depth_2d
                .data_blueprint
                .data_blueprints_individual()
                .set(image_entity, depth_2d_props);

            // Change duplicate to only show image.
            if let Some(depth_entity) = depth_2d
                .data_blueprint
                .entity_paths()
                .iter()
                .find(|entity_path| {
                    entity_path.last() == Some(&EntityPathPart::Name("Depth".into()))
                })
                .cloned()
            {
                let mut duplicate_props = duplicate
                    .data_blueprint
                    .data_blueprints_individual()
                    .get(&depth_entity);
                duplicate_props.visible = false;
                duplicate
                    .data_blueprint
                    .data_blueprints_individual()
                    .set(depth_entity, duplicate_props);
            }
            space_views.push(duplicate);
        }
    }
    space_views
}

fn default_created_space_views_from_candidates(
    ctx: &ViewerContext<'_>,
    entity_db: &EntityDb,
    candidates: Vec<SpaceView>,
) -> Vec<SpaceView> {
    crate::profile_function!();

    // All queries are "right most" on the log timeline.
    let query = LatestAtQuery::new(Timeline::log_time(), re_arrow_store::TimeInt::MAX);

    // First pass to look for interesting roots, as their existence influences the heuristic for non-roots!
    let categories_with_interesting_roots = candidates
        .iter()
        .filter_map(|space_view_candidate| {
            (space_view_candidate.space_path.is_root()
                && is_interesting_space_view_at_root(entity_db, space_view_candidate, &query))
            .then_some(space_view_candidate.category)
        })
        .collect::<ViewCategorySet>();

    let mut space_views = Vec::new();

    // Main pass through all candidates.
    // We first check if a candidate is "interesting" and then split it up/modify it further if required.
    for candidate in candidates {
        if candidate.space_path.is_root() {
            if !categories_with_interesting_roots.contains(candidate.category) {
                continue;
            }
        } else if !is_interesting_space_view_not_at_root(
            entity_db,
            &candidate,
            &categories_with_interesting_roots,
            &query,
        ) {
            continue;
        }

        // For tensors create one space view for each tensor (even though we're able to stack them in one view)
        if candidate.category == ViewCategory::Tensor {
            for entity_path in candidate.data_blueprint.entity_paths() {
                let mut space_view = SpaceView::new(
                    ctx,
                    ViewCategory::Tensor,
                    entity_path,
                    &[entity_path.clone()],
                );
                space_view.entities_determined_by_user = true; // Suppress auto adding of entities.
                space_views.push(space_view);
            }
            continue;
        }

        // Spatial views with images get extra treatment as well.
        if candidate.category == ViewCategory::Spatial {
            let mut images_by_size: HashMap<(u64, u64), Vec<EntityPath>> = HashMap::default();

            // For this we're only interested in the direct children.
            for entity_path in &candidate.data_blueprint.root_group().entities {
                if let Ok(entity_view) = re_query::query_entity_with_primary::<Tensor>(
                    &entity_db.data_store,
                    &query,
                    entity_path,
                    &[],
                ) {
                    for tensor in entity_view.iter_primary_flattened() {
                        if tensor.is_shaped_like_an_image() {
                            debug_assert!(matches!(tensor.real_shape().len(), 2 | 3));
                            let dim = (
                                tensor.real_shape().as_slice()[0].size,
                                tensor.real_shape().as_slice()[1].size,
                            );
                            images_by_size
                                .entry(dim)
                                .or_default()
                                .push(entity_path.clone());
                        }
                    }
                }
            }

            // If all images are the same size, proceed with the candidate as is. Otherwise...
            if images_by_size.len() > 1 {
                // ...stack images of the same size, but no others.
                for dim in images_by_size.keys() {
                    // Ignore every image that has a different size.
                    let images_of_different_size = images_by_size
                        .iter()
                        .filter_map(|(other_dim, images)| (dim != other_dim).then_some(images))
                        .flatten()
                        .cloned()
                        .collect::<IntSet<_>>();
                    let entities = candidate
                        .data_blueprint
                        .entity_paths()
                        .iter()
                        .filter(|path| !images_of_different_size.contains(path))
                        .cloned()
                        .collect_vec();

                    let mut space_view =
                        SpaceView::new(ctx, candidate.category, &candidate.space_path, &entities);
                    space_view.entities_determined_by_user = true; // Suppress auto adding of entities.
                    space_views.push(space_view);
                }
                continue;
            }
        }

        // Take the candidate as is.
        space_views.push(candidate);
    }

    space_views
}

fn has_any_component_except(
    entity_path: &EntityPath,
    data_store: &DataStore,
    timeline: Timeline,
    excluded_components: &[ComponentName],
) -> bool {
    data_store
        .all_components(&timeline, entity_path)
        .map_or(false, |all_components| {
            all_components
                .iter()
                .any(|comp| !excluded_components.contains(comp))
        })
}

/// Whether an entity should be added to a space view at a given path (independent of its category!)
fn is_default_added_to_space_view(
    entity_path: &EntityPath,
    space_path: &EntityPath,
    data_store: &DataStore,
    timeline: Timeline,
) -> bool {
    let ignored_components = [
        re_log_types::Transform::name(),
        re_log_types::ViewCoordinates::name(),
        re_log_types::component_types::InstanceKey::name(),
        re_log_types::component_types::KeypointId::name(),
        DataStore::insert_id_key(),
        re_log_types::ImuData::name(), // Separate plotting view for IMU data.
        re_log_types::XlinkStats::name(), // Separate plotting view for XLink stats.
    ];

    entity_path.is_descendant_of(space_path)
        || (entity_path == space_path
            && has_any_component_except(entity_path, data_store, timeline, &ignored_components))
}

/// List of entities a space view queries by default for a given category.
///
/// These are all entities in the given space which have the requested category and are reachable by a transform.
pub fn default_queried_entities(
    ctx: &ViewerContext<'_>,
    space_path: &EntityPath,
    spaces_info: &SpaceInfoCollection,
    category: ViewCategory,
) -> Vec<EntityPath> {
    crate::profile_function!();

    let timeline = Timeline::log_time();
    let log_db = &ctx.log_db;
    let data_store = &log_db.entity_db.data_store;

    let mut entities = Vec::new();
    let space_info = spaces_info.get_first_parent_with_info(space_path);

    space_info.visit_descendants_with_reachable_transform(spaces_info, &mut |space_info| {
        entities.extend(
            space_info
                .descendants_without_transform
                .iter()
                .filter(|entity_path| {
                    is_default_added_to_space_view(entity_path, space_path, data_store, timeline)
                        && categorize_entity_path(timeline, log_db, entity_path).contains(category)
                })
                .cloned(),
        );
    });
    entities
}

/// List of entities a space view queries by default for all possible category.
fn default_queried_entities_by_category(
    ctx: &ViewerContext<'_>,
    space_path: &EntityPath,
    space_info_collection: &SpaceInfoCollection,
) -> BTreeMap<ViewCategory, Vec<EntityPath>> {
    crate::profile_function!();

    let timeline = Timeline::log_time();
    let log_db = &ctx.log_db;
    let data_store = &log_db.entity_db.data_store;

    let mut groups: BTreeMap<ViewCategory, Vec<EntityPath>> = BTreeMap::default();
    let space_info = space_info_collection.get_first_parent_with_info(space_path);

    space_info.visit_descendants_with_reachable_transform(
        space_info_collection,
        &mut |space_info| {
            for entity_path in &space_info.descendants_without_transform {
                if is_default_added_to_space_view(entity_path, space_path, data_store, timeline) {
                    for category in categorize_entity_path(timeline, log_db, entity_path) {
                        groups
                            .entry(category)
                            .or_default()
                            .push(entity_path.clone());
                    }
                }
            }
        },
    );

    groups
}

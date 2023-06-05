//! The viewport panel.
//!
//! Contains all space views.

use ahash::HashMap;
use itertools::Itertools as _;

use re_arrow_store::LatestAtQuery;
use re_data_store::{query_latest_single, EntityPath};
use re_log_types::{
    component_types::TensorDataMeaning, EntityPathPart, Tensor, Time, TimeInt, Timeline,
};

use crate::{
    depthai::depthai,
    misc::{space_info::SpaceInfoCollection, Item, SpaceViewHighlights, ViewerContext},
    ui::{space_view_heuristics::default_created_space_views, stats_panel::StatsPanel},
};

use super::{
    device_settings_panel::DeviceSettingsPanel, selection_panel::SelectionPanel,
    space_view_entity_picker::SpaceViewEntityPicker,
    space_view_heuristics::all_possible_space_views, stats_panel::StatsPanelState,
    view_category::ViewCategory, view_spatial::SpatialNavigationMode, SpaceView, SpaceViewId,
    SpaceViewKind,
};

// ----------------------------------------------------------------------------

/// What views are visible?
type VisibilitySet = std::collections::BTreeSet<SpaceViewId>;

/// Describes the layout and contents of the Viewport Panel.
#[derive(Default, serde::Deserialize, serde::Serialize)]
#[serde(default)]
pub struct Viewport {
    /// Where the space views are stored.
    space_views: HashMap<SpaceViewId, SpaceView>,

    /// Which views are visible.
    visible: VisibilitySet,

    /// The layouts of all the space views.
    ///
    /// One for each combination of what views are visible.
    /// So if a user toggles the visibility of one SpaceView, we
    /// switch which layout we are using. This is somewhat hacky.
    trees: HashMap<VisibilitySet, egui_dock::Tree<Tab>>,

    /// Show one tab as maximized?
    maximized: Option<SpaceViewId>,

    /// Store for each space view if the user has edited it (eg. removed).
    /// Is reset when a space view get's automatically removed.
    has_been_user_edited: HashMap<EntityPath, bool>,

    #[serde(skip)]
    space_view_entity_window: Option<SpaceViewEntityPicker>,
    device_settings_panel: DeviceSettingsPanel,

    #[serde(skip)]
    stats_panel_state: StatsPanelState,

    previous_frame_tree: Option<egui_dock::Tree<Tab>>,
}

impl Viewport {
    /// Create a default suggested blueprint using some heuristics.
    pub fn new(ctx: &mut ViewerContext<'_>, spaces_info: &SpaceInfoCollection) -> Self {
        crate::profile_function!();

        let mut blueprint = Self::default();
        for space_view in default_created_space_views(ctx, spaces_info) {
            blueprint.add_space_view(space_view);
        }
        blueprint
    }

    pub(crate) fn space_view(&self, space_view: &SpaceViewId) -> Option<&SpaceView> {
        self.space_views.get(space_view)
    }

    pub(crate) fn space_view_mut(&mut self, space_view_id: &SpaceViewId) -> Option<&mut SpaceView> {
        self.space_views.get_mut(space_view_id)
    }

    pub(crate) fn remove(&mut self, space_view_id: &SpaceViewId) -> Option<SpaceView> {
        let Self {
            space_views,
            visible,
            trees,
            maximized,
            has_been_user_edited: _,
            space_view_entity_window,
            device_settings_panel: _,
            stats_panel_state: _,
            previous_frame_tree: _,
        } = self;

        if let Some(window) = space_view_entity_window {
            if window.space_view_id == *space_view_id {
                *space_view_entity_window = None;
            }
        }

        trees.retain(|vis_set, _| !vis_set.contains(space_view_id));

        if *maximized == Some(*space_view_id) {
            *maximized = None;
        }

        visible.remove(space_view_id);
        space_views.remove(space_view_id)
    }

    pub fn add_or_remove_space_views_ui(
        &mut self,
        ctx: &mut ViewerContext<'_>,
        ui: &mut egui::Ui,
        spaces_info: &SpaceInfoCollection,
    ) {
        crate::profile_function!();

        egui::ScrollArea::vertical()
            .auto_shrink([false; 2])
            .show(ui, |ui| {
                ui.style_mut().wrap = Some(false);

                // Only show "logical" space views like Color camera, Mono Camera etc, don't go into
                // details like rerun does in tree_ui, depthai-viewer users don't care about that
                // as they didn't create the blueprint by logging the data
                for space_view in all_possible_space_views(ctx, spaces_info)
                    .into_iter()
                    .filter(|sv| sv.is_depthai_spaceview)
                {
                    self.available_space_view_row_ui(ctx, ui, space_view);
                }
            });
    }

    fn available_space_view_row_ui(
        &mut self,
        ctx: &mut ViewerContext<'_>,
        ui: &mut egui::Ui,
        space_view: SpaceView,
    ) {
        let space_path = space_view.space_path.clone(); // to avoid borrowing issue in .body() of collapsing state
        let collapsing_header_id = ui.id().with(space_view.display_name.clone());
        egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            collapsing_header_id,
            true,
        )
        .show_header(ui, |ui| {
            ui.label(space_view.display_name.clone());
            let mut ui = ui.child_ui(
                ui.max_rect(),
                egui::Layout::right_to_left(egui::Align::Center),
            );
            if ctx
                .re_ui
                .small_icon_button(&mut ui, &re_ui::icons::ADD)
                .clicked()
            {
                self.add_space_view(space_view);
            }
        })
        .body(|ui| {
            let instances_of_space_view = self.visible.iter().filter(|id| {
                if let Some(sv) = self.space_views.get(*id) {
                    sv.space_path == space_path
                } else {
                    false
                }
            });
            let mut space_views_to_remove = Vec::new();
            for sv_id in instances_of_space_view {
                ui.horizontal(|ui| {
                    let label = format!("🔹 {}", self.space_views[sv_id].display_name);
                    ui.label(label);
                    let mut ui = ui.child_ui(
                        ui.max_rect(),
                        egui::Layout::right_to_left(egui::Align::Center),
                    );
                    if ctx
                        .re_ui
                        .small_icon_button(&mut ui, &re_ui::icons::REMOVE)
                        .clicked()
                    {
                        space_views_to_remove.push(*sv_id);
                        self.has_been_user_edited
                            .insert(self.space_views[sv_id].space_path.clone(), true);
                    }
                });
            }
            for sv_id in &space_views_to_remove {
                self.remove(sv_id);
            }
        });
    }

    pub(crate) fn mark_user_interaction(&mut self) {}

    pub(crate) fn add_space_view(&mut self, mut space_view: SpaceView) -> SpaceViewId {
        let id = space_view.id;

        // Find a unique name for the space view
        let mut candidate_name = space_view.display_name.clone();
        let mut append_count = 1;
        let unique_name = 'outer: loop {
            for view in &self.space_views {
                if candidate_name == view.1.display_name {
                    append_count += 1;
                    candidate_name = format!("{} ({})", space_view.display_name, append_count);

                    continue 'outer;
                }
            }
            break candidate_name;
        };

        space_view.display_name = unique_name;

        self.space_views.insert(id, space_view);
        self.visible.insert(id);
        self.trees.clear(); // Reset them
        id
    }

    pub fn show_add_remove_entities_window(&mut self, space_view_id: SpaceViewId) {
        self.space_view_entity_window = Some(SpaceViewEntityPicker { space_view_id });
    }

    /// Gets the space views that don't actually have an instance associated with them
    fn get_space_views_to_delete(
        &mut self,
        ctx: &mut ViewerContext<'_>,
        spaces_info: &SpaceInfoCollection,
    ) -> Vec<SpaceViewId> {
        let mut space_views_to_delete = Vec::new();
        let binding = all_possible_space_views(ctx, spaces_info);
        let possible_space_views = binding
            .iter()
            .cloned()
            .filter(|sv| sv.is_depthai_spaceview)
            .collect_vec();
        for space_view in &possible_space_views {
            let mut found = false;
            for existing_view in self.space_views.values() {
                if existing_view.space_path == space_view.space_path {
                    found = true;
                    break;
                }
            }
            if !found {
                space_views_to_delete.push(space_view.id);
            }
        }
        for space_view in self.space_views.values() {
            if !possible_space_views
                .iter()
                .map(|sv| sv.space_path.clone())
                .contains(&space_view.space_path)
            {
                self.has_been_user_edited
                    .insert(space_view.space_path.clone(), false);
                space_views_to_delete.push(space_view.id);
            }
        }
        space_views_to_delete
    }

    pub fn on_frame_start(
        &mut self,
        ctx: &mut ViewerContext<'_>,
        spaces_info: &SpaceInfoCollection,
    ) {
        crate::profile_function!();

        for space_view_id in &self.get_space_views_to_delete(ctx, spaces_info) {
            self.remove(space_view_id);
        }

        self.stats_panel_state.update(ctx);

        for space_view in self.space_views.values_mut() {
            space_view.on_frame_start(ctx, spaces_info);
        }
        for space_view_candidate in default_created_space_views(ctx, spaces_info) {
            if !self
                .has_been_user_edited
                .get(&space_view_candidate.space_path)
                .unwrap_or(&false)
                && self.should_auto_add_space_view(&space_view_candidate)
            {
                self.add_space_view(space_view_candidate);
            }
        }
    }

    fn should_auto_add_space_view(&self, space_view_candidate: &SpaceView) -> bool {
        for existing_view in self.space_views.values() {
            if existing_view.space_path == space_view_candidate.space_path {
                if existing_view.entities_determined_by_user {
                    // Since the user edited a space view with the same space path, we can't be sure our new one isn't redundant.
                    // So let's skip that.
                    return false;
                }

                if space_view_candidate
                    .data_blueprint
                    .entity_paths()
                    .is_subset(existing_view.data_blueprint.entity_paths())
                {
                    // This space view wouldn't add anything we haven't already
                    return false;
                }
            }
        }

        true
    }

    pub fn viewport_ui(&mut self, ui: &mut egui::Ui, ctx: &mut ViewerContext<'_>) {
        if let Some(window) = &mut self.space_view_entity_window {
            if let Some(space_view) = self.space_views.get_mut(&window.space_view_id) {
                if !window.ui(ctx, ui, space_view) {
                    self.space_view_entity_window = None;
                }
            } else {
                // The space view no longer exist, close the window!
                self.space_view_entity_window = None;
            }
        }

        let is_zero_sized_viewport = ui.available_size().min_elem() <= 0.0;
        if is_zero_sized_viewport {
            return;
        }

        self.trees.retain(|_, tree| is_tree_valid(tree));

        let visible_space_views = if let Some(space_view_id) = self.maximized {
            std::iter::once(space_view_id).collect()
        } else {
            self.visible.clone()
        };

        // Lazily create a layout tree based on which SpaceViews should be visible:
        let mut tree = self
            .trees
            .entry(visible_space_views.clone())
            .or_insert_with(|| {
                super::auto_layout::default_tree_from_space_views(
                    ui.available_size(),
                    &visible_space_views,
                    &self.space_views,
                    self.maximized.is_some(),
                )
            })
            .clone();
        self.previous_frame_tree = Some(tree.clone());

        let num_space_views = tree.num_tabs();
        if num_space_views == 0 {
            return;
        }

        let mut tab_viewer = TabViewer {
            ctx,
            viewport: self,
        };

        ui.scope(|ui| {
            // we need a scope, because egui_dock unfortunately messes with the ui clip rect

            ui.spacing_mut().item_spacing.x = re_ui::ReUi::view_padding();

            egui_dock::DockArea::new(&mut tree)
                .id(egui::Id::new("space_view_dock"))
                .style(re_ui::egui_dock_style(ui.style()))
                .show_inside(ui, &mut tab_viewer);
        });

        // Two passes so we avoid borrowing issues:
        let tab_bars = tree
            .iter()
            .filter_map(|node| {
                let egui_dock::Node::Leaf { rect, viewport, tabs, active } = node else {
                    return None;
                };

                let space_view_tab = tabs.get(active.0)?;

                // `rect` includes the tab area, while `viewport` is just the tab body.
                // so the tab bar rect is:
                let tab_bar_rect =
                    egui::Rect::from_x_y_ranges(rect.x_range(), rect.top()..=viewport.top());

                // rect/viewport can be invalid for the first frame
                tab_bar_rect
                    .is_finite()
                    .then_some((space_view_tab.clone(), tab_bar_rect))
            })
            .collect_vec();
        self.trees.insert(visible_space_views, tree);

        for (
            Tab {
                space_view_id,
                space_view_kind,
                ..
            },
            tab_bar_rect,
        ) in tab_bars
        {
            match space_view_kind {
                SpaceViewKind::Data | SpaceViewKind::Stats => {
                    let mut entities_to_skip = Vec::new();
                    if let Some(space_view) = self.space_views.get_mut(&space_view_id) {
                        let mut is3d = false;
                        let mut has_depth = false;
                        let mut image_to_hide = None;
                        space_view.data_blueprint.visit_group_entities_recursively(
                            space_view.data_blueprint.root_handle(),
                            &mut |entity_path| {
                                if is3d && has_depth {
                                    if let Some(last_part) = entity_path.iter().last() {
                                        if last_part == &EntityPathPart::from("Image") {
                                            image_to_hide = Some(entity_path.clone());
                                        }
                                    }
                                }
                                is3d |= entity_path.len() == 2
                                    && entity_path.iter().last().unwrap()
                                        == &EntityPathPart::from("transform");
                                if let Some(last_part) = entity_path.iter().last() {
                                    has_depth |= last_part == &EntityPathPart::from("Depth");
                                }
                            },
                        );
                        if let Some(image_to_hide) = image_to_hide {
                            entities_to_skip.push(image_to_hide.clone());
                            let mut props = space_view
                                .data_blueprint
                                .data_blueprints_individual()
                                .get(&image_to_hide);
                            props.visible = false;
                            space_view
                                .data_blueprint
                                .data_blueprints_individual()
                                .set(image_to_hide.clone(), props);
                        }
                    }
                    space_view_options_ui(
                        ctx,
                        ui,
                        self,
                        tab_bar_rect,
                        space_view_id,
                        num_space_views,
                        entities_to_skip.as_slice(),
                    );
                }
                SpaceViewKind::Selection => {
                    SelectionPanel::selection_panel_options_ui(ctx, ui, self, tab_bar_rect);
                }
                _ => {}
            }
        }
    }

    pub fn space_views_containing_entity_path(&self, path: &EntityPath) -> Vec<SpaceViewId> {
        self.space_views
            .iter()
            .filter_map(|(space_view_id, space_view)| {
                if space_view.data_blueprint.contains_entity(path) {
                    Some(*space_view_id)
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Show a single button (`add_content`), justified,
/// and show a visibility button if the row is hovered.
///
/// Returns true if visibility changed.
fn blueprint_row_with_buttons(
    re_ui: &re_ui::ReUi,
    ui: &mut egui::Ui,
    enabled: bool,
    visible: bool,
    add_content: impl FnOnce(&mut egui::Ui) -> egui::Response,
    add_on_hover_buttons: impl FnOnce(&re_ui::ReUi, &mut egui::Ui),
) {
    let where_to_add_hover_rect = ui.painter().add(egui::Shape::Noop);

    // Make the main button span the whole width to make it easier to click:
    let main_button_response = ui
        .with_layout(egui::Layout::top_down_justified(egui::Align::LEFT), |ui| {
            ui.style_mut().wrap = Some(false);

            // Turn off the background color of hovered buttons.
            // Why? Because we add a manual hover-effect later.
            // Why? Because we want that hover-effect even when only the visibility button is hovered.
            let visuals = ui.visuals_mut();
            visuals.widgets.hovered.weak_bg_fill = egui::Color32::TRANSPARENT;
            visuals.widgets.hovered.bg_fill = egui::Color32::TRANSPARENT;
            visuals.widgets.active.weak_bg_fill = egui::Color32::TRANSPARENT;
            visuals.widgets.active.bg_fill = egui::Color32::TRANSPARENT;
            visuals.widgets.open.weak_bg_fill = egui::Color32::TRANSPARENT;
            visuals.widgets.open.bg_fill = egui::Color32::TRANSPARENT;

            if ui
                .interact(ui.max_rect(), ui.id(), egui::Sense::hover())
                .hovered()
            {
                // Clip the main button so that the on-hover buttons have room to cover it.
                // Ideally we would only clip the button _text_, not the button background, but that's not possible.
                let mut clip_rect = ui.max_rect();
                let on_hover_buttons_width = 36.0;
                clip_rect.max.x -= on_hover_buttons_width;
                ui.set_clip_rect(clip_rect);
            }

            if !visible || !enabled {
                // Dim the appearance of things added by `add_content`:
                let widget_visuals = &mut ui.visuals_mut().widgets;

                fn dim_color(color: &mut egui::Color32) {
                    *color = color.gamma_multiply(0.5);
                }
                dim_color(&mut widget_visuals.noninteractive.fg_stroke.color);
                dim_color(&mut widget_visuals.inactive.fg_stroke.color);
            }

            add_content(ui)
        })
        .inner;

    let main_button_rect = main_button_response.rect;

    // We check the same rectangle as the main button,
    // but we will also catch hovers on the visibility button (if any).
    let button_hovered = ui
        .interact(main_button_rect, ui.id(), egui::Sense::hover())
        .hovered();
    if button_hovered {
        // Just put the buttons on top of the existing ui:
        let mut ui = ui.child_ui(
            ui.max_rect(),
            egui::Layout::right_to_left(egui::Align::Center),
        );
        add_on_hover_buttons(re_ui, &mut ui);
    }

    // The main button might have been highlighted because what it was referring
    // to was hovered somewhere else, and then we also want it highlighted here.
    if button_hovered || main_button_response.highlighted() {
        // Highlight the row:
        let visuals = ui.visuals().widgets.hovered;
        let hover_rect = main_button_rect.expand(visuals.expansion);
        ui.painter().set(
            where_to_add_hover_rect,
            egui::Shape::rect_filled(hover_rect, visuals.rounding, visuals.bg_fill),
        );
    }
}

fn visibility_button_ui(
    re_ui: &re_ui::ReUi,
    ui: &mut egui::Ui,
    enabled: bool,
    visible: &mut bool,
) -> egui::Response {
    ui.set_enabled(enabled);
    re_ui
        .visibility_toggle_button(ui, visible)
        .on_hover_text("Toggle visibility")
        .on_disabled_hover_text("A parent is invisible")
}

// ----------------------------------------------------------------------------

struct ViewportView {}

struct TabViewer<'a, 'b> {
    ctx: &'a mut ViewerContext<'b>,
    viewport: &'a mut Viewport,
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Tab {
    pub space_view_id: SpaceViewId,

    /// Rerun only displayed logged data in the viewport.
    /// Depthai Viewer also has fixed ui panels that are part of the tab system.
    /// This is used to distinguish between the two.
    pub space_view_kind: SpaceViewKind,

    pub space_path: Option<EntityPath>,
}

impl From<&SpaceView> for Tab {
    fn from(space_view: &SpaceView) -> Self {
        Self {
            space_view_id: space_view.id,
            space_view_kind: SpaceViewKind::Data,
            space_path: Some(space_view.space_path.clone()),
        }
    }
}

impl From<SpaceView> for Tab {
    fn from(space_view: SpaceView) -> Self {
        Self::from(&space_view)
    }
}

impl<'a, 'b> egui_dock::TabViewer for TabViewer<'a, 'b> {
    type Tab = Tab;

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        let Tab {
            space_view_id,
            space_view_kind,
            space_path,
        } = tab;
        crate::profile_function!();

        match space_view_kind {
            SpaceViewKind::Data => {
                let highlights = self
                    .ctx
                    .selection_state()
                    .highlights_for_space_view(*space_view_id, &self.viewport.space_views);
                let space_view = self
                    .viewport
                    .space_views
                    .get_mut(space_view_id)
                    .expect("Should have been populated beforehand");

                space_view_ui(self.ctx, ui, space_view, &highlights);
            }
            SpaceViewKind::Selection => SelectionPanel::show_panel(self.ctx, ui, self.viewport),
            SpaceViewKind::Config => self.viewport.device_settings_panel.show_panel(self.ctx, ui),
            SpaceViewKind::Stats => {
                StatsPanel::show_panel(self.ctx, ui, &mut self.viewport.stats_panel_state);
            }
            _ => {}
        }
    }

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        match tab.space_view_kind {
            SpaceViewKind::Data => {
                let space_view = self
                    .viewport
                    .space_views
                    .get_mut(&tab.space_view_id)
                    .expect("Should have been populated beforehand");

                let mut text = egui::WidgetText::RichText(egui::RichText::new(
                    space_view.display_name.clone(),
                ));

                if self
                    .ctx
                    .selection()
                    .contains(&Item::SpaceView(tab.space_view_id))
                {
                    // Show that it is selected:
                    let egui_ctx = &self.ctx.re_ui.egui_ctx;
                    let selection_bg_color = egui_ctx.style().visuals.selection.bg_fill;
                    text = text.background_color(selection_bg_color);
                }

                text
            }
            SpaceViewKind::Stats => "Stats".into(),
            SpaceViewKind::Config => "Device Settings".into(),
            SpaceViewKind::Selection => "Selection".into(),
        }
    }

    fn on_tab_button(&mut self, tab: &mut Self::Tab, response: &egui::Response) {
        if response.clicked() {
            self.ctx
                .set_single_selection(Item::SpaceView(tab.space_view_id));
        }
    }
}

fn help_text_ui(ui: &mut egui::Ui, space_view: &SpaceView) {
    let help_text = match space_view.category {
        ViewCategory::TimeSeries => Some(crate::ui::view_time_series::HELP_TEXT),
        ViewCategory::BarChart => Some(crate::ui::view_bar_chart::HELP_TEXT),
        ViewCategory::Spatial => Some(space_view.view_state.state_spatial.help_text()),
        ViewCategory::Text | ViewCategory::Tensor => None,
        ViewCategory::NodeGraph => None,
    };

    if let Some(help_text) = help_text {
        crate::misc::help_hover_button(ui).on_hover_text(help_text);
    }
}

/// Shown in the right of the tab panel
fn space_view_options_ui(
    ctx: &mut ViewerContext<'_>,
    ui: &mut egui::Ui,
    viewport: &mut Viewport,
    tab_bar_rect: egui::Rect,
    space_view_id: SpaceViewId,
    num_space_views: usize,
    entities_to_skip: &[EntityPath],
) {
    let tab_bar_rect = tab_bar_rect.shrink2(egui::vec2(4.0, 0.0)); // Add some side margin outside the frame

    ui.allocate_ui_at_rect(tab_bar_rect, |ui| {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let where_to_put_background = ui.painter().add(egui::Shape::Noop);

            ui.add_space(4.0); // margin within the frame

            if viewport.maximized == Some(space_view_id) {
                // Show minimize-button:
                if ctx
                    .re_ui
                    .small_icon_button(ui, &re_ui::icons::MINIMIZE)
                    .on_hover_text("Restore - show all spaces")
                    .clicked()
                {
                    viewport.maximized = None;
                }
            } else if num_space_views > 1 {
                // Show maximize-button:
                if ctx
                    .re_ui
                    .small_icon_button(ui, &re_ui::icons::MAXIMIZE)
                    .on_hover_text("Maximize Space View")
                    .clicked()
                {
                    viewport.maximized = Some(space_view_id);
                    ctx.set_single_selection(Item::SpaceView(space_view_id));
                }
            }
            let Some(space_view) = viewport.space_views.get_mut(&space_view_id) else {
                return;
            };

            let icon_image = ctx.re_ui.icon_image(&re_ui::icons::GEAR);
            let texture_id = icon_image.texture_id(ui.ctx());
            ui.menu_image_button(texture_id, re_ui::ReUi::small_icon_size(), |ui| {
                ui.style_mut().wrap = Some(false);
                let entities = space_view.data_blueprint.entity_paths().clone();
                let entities = entities.iter().filter(|ep| {
                    if let Some(last_part) = ep.iter().last() {
                        last_part != &EntityPathPart::from("transform")
                            && last_part != &EntityPathPart::from("camera")
                    } else {
                        false
                    }
                });
                for entity_path in entities {
                    if entities_to_skip.contains(entity_path) {
                        continue;
                    }
                    ui.horizontal(|ui| {
                        let mut properties = space_view
                            .data_blueprint
                            .data_blueprints_individual()
                            .get(entity_path);
                        blueprint_row_with_buttons(
                            ctx.re_ui,
                            ui,
                            true,
                            properties.visible,
                            |ui| {
                                let name = entity_path.iter().last().unwrap().to_string();
                                let label = format!("🔹 {name}");
                                ctx.data_blueprint_button_to(ui, label, space_view.id, entity_path)
                            },
                            |re_ui, ui| {
                                if visibility_button_ui(re_ui, ui, true, &mut properties.visible)
                                    .changed()
                                {
                                    space_view
                                        .data_blueprint
                                        .data_blueprints_individual()
                                        .set(entity_path.clone(), properties);
                                }
                            },
                        );
                    });
                }
            });

            // Show help last, since not all space views have help text
            help_text_ui(ui, space_view);

            // Put a frame so that the buttons cover any labels they intersect with:
            let rect = ui.min_rect().expand2(egui::vec2(1.0, -2.0));
            ui.painter().set(
                where_to_put_background,
                egui::Shape::rect_filled(
                    rect,
                    0.0,
                    re_ui::egui_dock_style(ui.style()).tab_bar_background_color,
                ),
            );
        });
    });
}

fn space_view_ui(
    ctx: &mut ViewerContext<'_>,
    ui: &mut egui::Ui,
    space_view: &mut SpaceView,
    space_view_highlights: &SpaceViewHighlights,
) {
    let Some(latest_at) = ctx.rec_cfg.time_ctrl.time_int() else {
        ui.centered_and_justified(|ui| {
            ui.label(ctx.re_ui.warning_text("No time selected"));
        });
        return;
    };

    space_view.scene_ui(ctx, ui, latest_at, space_view_highlights);
}

// ----------------------------------------------------------------------------

fn focus_tab(tree: &mut egui_dock::Tree<Tab>, tab: &Tab) {
    if let Some((node_index, tab_index)) = tree.find_tab(tab) {
        tree.set_focused_node(node_index);
        tree.set_active_tab(node_index, tab_index);
    }
}

fn is_tree_valid(tree: &egui_dock::Tree<Tab>) -> bool {
    tree.iter().all(|node| match node {
        egui_dock::Node::Vertical { rect: _, fraction }
        | egui_dock::Node::Horizontal { rect: _, fraction } => fraction.is_finite(),
        _ => true,
    })
}

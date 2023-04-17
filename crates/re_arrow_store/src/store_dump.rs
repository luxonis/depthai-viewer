use ahash::HashMapExt;
use arrow2::Either;
use nohash_hasher::IntMap;
use re_log_types::{
    DataCellColumn, DataTable, ErasedTimeVec, RowIdVec, TableId, TimeRange, Timeline,
};

use crate::{
    store::{IndexedBucketInner, PersistentIndexedTable},
    DataStore, IndexedBucket,
};

// ---

impl DataStore {
    /// Serializes the entire datastore into an iterator of [`DataTable`]s.
    // TODO(#1793): This shouldn't dump cluster keys that were autogenerated.
    // TODO(#1794): Implement simple recompaction.
    pub fn to_data_tables(
        &self,
        time_filter: Option<(Timeline, TimeRange)>,
    ) -> impl Iterator<Item = DataTable> + '_ {
        let timeless = self.dump_timeless_tables();
        let temporal = if let Some(time_filter) = time_filter {
            Either::Left(self.dump_temporal_tables_filtered(time_filter))
        } else {
            Either::Right(self.dump_temporal_tables())
        };

        timeless.chain(temporal)
    }

    fn dump_timeless_tables(&self) -> impl Iterator<Item = DataTable> + '_ {
        self.timeless_tables.values().map(|table| {
            crate::profile_scope!("timeless_table");

            let PersistentIndexedTable {
                ent_path,
                cluster_key: _,
                col_insert_id: _,
                col_row_id,
                col_num_instances,
                columns,
            } = table;

            DataTable {
                table_id: TableId::random(),
                col_row_id: col_row_id.clone(),
                col_timelines: Default::default(),
                col_entity_path: std::iter::repeat_with(|| ent_path.clone())
                    .take(table.num_rows() as _)
                    .collect(),
                col_num_instances: col_num_instances.clone(),
                columns: columns.clone(), // shallow
            }
        })
    }

    fn dump_temporal_tables(&self) -> impl Iterator<Item = DataTable> + '_ {
        self.tables.values().flat_map(|table| {
            crate::profile_scope!("temporal_table");

            table.buckets.values().map(move |bucket| {
                crate::profile_scope!("temporal_bucket");

                bucket.sort_indices_if_needed();

                let IndexedBucket {
                    timeline,
                    cluster_key: _,
                    inner,
                } = bucket;

                let IndexedBucketInner {
                    is_sorted,
                    time_range: _,
                    col_time,
                    col_insert_id: _,
                    col_row_id,
                    col_num_instances,
                    columns,
                    size_bytes: _,
                } = &*inner.read();
                debug_assert!(is_sorted);

                DataTable {
                    table_id: TableId::random(),
                    col_row_id: col_row_id.clone(),
                    col_timelines: [(*timeline, col_time.iter().copied().map(Some).collect())]
                        .into(),
                    col_entity_path: std::iter::repeat_with(|| table.ent_path.clone())
                        .take(table.num_rows() as _)
                        .collect(),
                    col_num_instances: col_num_instances.clone(),
                    columns: columns.clone(), // shallow
                }
            })
        })
    }

    fn dump_temporal_tables_filtered(
        &self,
        (timeline_filter, time_filter): (Timeline, TimeRange),
    ) -> impl Iterator<Item = DataTable> + '_ {
        self.tables
            .values()
            .filter_map(move |table| {
                crate::profile_scope!("temporal_table_filtered");

                if table.timeline != timeline_filter {
                    return None;
                }

                Some(table.buckets.values().filter_map(move |bucket| {
                    crate::profile_scope!("temporal_bucket_filtered");

                    bucket.sort_indices_if_needed();

                    let IndexedBucket {
                        timeline,
                        cluster_key: _,
                        inner,
                    } = bucket;

                    let IndexedBucketInner {
                        is_sorted,
                        time_range,
                        col_time,
                        col_insert_id: _,
                        col_row_id,
                        col_num_instances,
                        columns,
                        size_bytes: _,
                    } = &*inner.read();
                    debug_assert!(is_sorted);

                    if !time_range.intersects(time_filter) {
                        return None;
                    }

                    let col_row_id: RowIdVec =
                        filter_column(col_time, col_row_id.iter(), time_filter).collect();

                    // NOTE: Shouldn't ever happen due to check above, but better safe than
                    // sorry...
                    debug_assert!(!col_row_id.is_empty());
                    if col_row_id.is_empty() {
                        return None;
                    }

                    let col_timelines = [(
                        *timeline,
                        filter_column(col_time, col_time.iter(), time_filter)
                            .map(Some)
                            .collect(),
                    )]
                    .into();

                    let col_entity_path = std::iter::repeat_with(|| table.ent_path.clone())
                        .take(col_row_id.len())
                        .collect();

                    let col_num_instances =
                        filter_column(col_time, col_num_instances.iter(), time_filter).collect();

                    let mut columns2 = IntMap::with_capacity(columns.len());
                    for (component, column) in columns {
                        let column = filter_column(col_time, column.iter(), time_filter).collect();
                        columns2.insert(*component, DataCellColumn(column));
                    }

                    Some(DataTable {
                        table_id: TableId::random(),
                        col_row_id,
                        col_timelines,
                        col_entity_path,
                        col_num_instances,
                        columns: columns2,
                    })
                }))
            })
            .flatten()
    }
}

fn filter_column<'a, T: 'a + Clone>(
    col_time: &'a ErasedTimeVec,
    column: impl Iterator<Item = &'a T> + 'a,
    time_filter: TimeRange,
) -> impl Iterator<Item = T> + 'a {
    col_time
        .iter()
        .zip(column)
        .filter_map(move |(time, v)| time_filter.contains((*time).into()).then(|| v.clone()))
}

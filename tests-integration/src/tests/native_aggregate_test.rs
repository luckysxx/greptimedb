// Copyright 2023 Greptime Team
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! A complete aggregate over one ordinary Mito table, executed by the real standalone frontend and
//! the real datanode in a single process.
//!
//! The table is ordinary - one region, no `PARTITION` clause - and its five series sit in four SST
//! files and one memtable range. With the scan asked for four partitions, each of them ends up with
//! at least one series; the per-partition metric assertion below checks that instead of assuming it.
//! Every query runs twice: once with the native route asked for per request, once on the plan the
//! existing planner builds. Both are compared with expectations worked out from the fixture, so
//! neither run is the other's oracle.
//!
//! In the plans the engines really executed, the native run must hold one fused producer reading the
//! raw series scan directly - the scan itself, or the one transparent cooperative-scheduling
//! wrapper the engine may put above it, but never an exchange - the frontend-side merge of that
//! producer's state (so the answer is not a fall back), and `native.root.output_values` for every
//! one of the four scan partitions. The metric is read after the result stream was drained, the
//! only point at which it is complete. Plans are observed through [`PhysicalPlanWrapper`], which
//! both query engines install, so what is checked is the tree that ran rather than a re-plan made
//! for an `EXPLAIN`.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use common_base::Plugins;
use common_query::OutputData;
use common_recordbatch::RecordBatches;
use common_recordbatch::util::collect_batches;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::coop::CooperativeExec;
use datafusion::physical_plan::displayable;
use datafusion::physical_plan::metrics::MetricValue;
use datatypes::arrow::array::{Array, ArrayRef, Float64Array, Int64Array, StringArray};
use datatypes::arrow::compute::cast;
use datatypes::arrow::datatypes::DataType;
use frontend::instance::Instance;
use query::datafusion::QUERY_PARALLELISM_HINT;
use query::options::QueryOptions;
use query::physical_wrapper::{PhysicalPlanWrapper, PhysicalPlanWrapperRef};
use servers::query_handler::sql::SqlQueryHandler;
use session::context::{QueryContext, QueryContextRef};
use table::table::scan::SeriesScanExec;

use crate::standalone::GreptimeDbStandaloneBuilder;

/// The per-request key that turns the native route on for one query. The planner's own constant is
/// crate-private, so it is repeated here.
const NATIVE_PROMQL_EXTENSION_KEY: &str = "query.experimental_native_promql";
/// The physical name of the fused producer.
const FUSED_EXEC_NAME: &str = "FusedSeriesExec";
/// What every name of the frontend-side merge node contains (`AggregateFinalize`, `...Exec`).
const FINALIZE_EXEC_NAME: &str = "AggregateFinalize";
/// The metric a fused producer registers for the values its own root node produced, per partition.
const ROOT_OUTPUT_VALUES_METRIC: &str = "native.root.output_values";
/// The parallelism the scan is asked to split its ranges into.
const SCAN_PARTITIONS: usize = 4;
/// The first sample of the fixture, in seconds since the epoch.
const BASE_SECS: i64 = 1_700_000_000;
/// The instant every query is evaluated at: `BASE_SECS + 240`, so the oldest sample is 240s old and
/// every sample is inside the default 5m lookback.
const EVAL_SECS: i64 = BASE_SECS + 240;
/// [`EVAL_SECS`] in milliseconds, the `ts` the results are reported at.
const EVAL_MILLIS: i64 = EVAL_SECS * 1000;
/// The table the aggregate queries read.
const GAUGE_TABLE: &str = "native_agg_gauge";
/// The table the range-window smoke test reads.
const RANGE_TABLE: &str = "native_agg_range";
/// The tolerance of the value comparisons: relative, with an absolute floor.
const FLOAT_TOLERANCE: f64 = 1e-10;

/// The five series of the gauge fixture in insert order: `job = a` holds 2, 6 and 10; `job = b`
/// holds 4 and 8, so every aggregate below has an answer that can be worked out by hand.
const GAUGE_SERIES: [(&str, &str, f64); 5] = [
    ("a", "sh1", 2.0),
    ("a", "h2", 6.0),
    ("a", "h3", 10.0),
    ("b", "h4", 4.0),
    ("b", "h5", 8.0),
];

/// Observes the physical plan each query engine is about to execute, and keeps it.
///
/// The frontend and the datanode it reads its regions from install the same plugin instance, so both
/// trees of one query arrive here. They are kept in a list because one query produces one entry per
/// engine, and keeping a single plan would lose the region's.
#[derive(Default)]
struct NativePlanObserver {
    seen: Mutex<Vec<Arc<dyn ExecutionPlan>>>,
}

impl NativePlanObserver {
    /// Marks the current end of the list, for [`Self::since`].
    fn mark(&self) -> usize {
        self.seen.lock().unwrap().len()
    }

    /// The plans handed over since `mark`.
    fn since(&self, mark: usize) -> Vec<Arc<dyn ExecutionPlan>> {
        self.seen.lock().unwrap()[mark..].to_vec()
    }
}

impl PhysicalPlanWrapper for NativePlanObserver {
    fn wrap(
        &self,
        origin: Arc<dyn ExecutionPlan>,
        _ctx: QueryContextRef,
    ) -> Arc<dyn ExecutionPlan> {
        self.seen.lock().unwrap().push(Arc::clone(&origin));
        origin
    }
}

/// The native nodes one query's plans hold: the partition-local producers, one per region the query
/// read, and the frontend-side merges of the state they export.
#[derive(Default)]
struct NativeCapture {
    fused: Vec<Arc<dyn ExecutionPlan>>,
    finalize: Vec<Arc<dyn ExecutionPlan>>,
}

impl NativeCapture {
    /// Picks the nodes of interest out of the plans one query handed over.
    fn from_plans(plans: &[Arc<dyn ExecutionPlan>]) -> Self {
        let mut capture = Self::default();
        let mut stack = plans.to_vec();
        while let Some(node) = stack.pop() {
            if node.name() == FUSED_EXEC_NAME {
                capture.fused.push(Arc::clone(&node));
            }
            if node.name().contains(FINALIZE_EXEC_NAME) {
                capture.finalize.push(Arc::clone(&node));
            }
            stack.extend(node.children().into_iter().cloned());
        }
        capture
    }
}

/// One result row, read by column role rather than by column name.
#[derive(Debug)]
struct ResultRow {
    /// The evaluation time, in milliseconds.
    ts_millis: i64,
    /// The string label columns by name; a null label is an absent entry.
    labels: BTreeMap<String, String>,
    /// The value column, [`None`] when the row carries no value for it.
    value: Option<f64>,
}

/// The rows of one PromQL query and the columns they were reported in.
struct QueryResult {
    /// Every column name, in schema order.
    columns: Vec<String>,
    /// The rows, in a canonical order.
    rows: Vec<ResultRow>,
}

impl QueryResult {
    /// Reads the batches of one PromQL result. The value column is found by type: the frontend names
    /// it after the expression it computes (`sum(v)` and the like), so that name is checked against
    /// the legacy run instead of being assumed here. `count` reports an integer, so every numeric
    /// type is accepted; the timestamp column was taken first and is therefore not a candidate.
    fn from_batches(batches: &RecordBatches, query: &str) -> Self {
        let schema = batches.schema();
        let fields = schema.arrow_schema().fields();
        let columns = fields
            .iter()
            .map(|field| field.name().to_string())
            .collect::<Vec<_>>();
        let time_index = fields
            .iter()
            .position(|field| matches!(field.data_type(), DataType::Timestamp(_, _)))
            .unwrap_or_else(|| {
                panic!("`{query}`: the result carries no timestamp column: {columns:?}")
            });
        let label_indices = fields
            .iter()
            .enumerate()
            .filter(|(_, field)| {
                matches!(
                    field.data_type(),
                    DataType::Utf8
                        | DataType::LargeUtf8
                        | DataType::Utf8View
                        | DataType::Dictionary(_, _)
                )
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let value_indices = fields
            .iter()
            .enumerate()
            .filter(|(_, field)| field.data_type().is_numeric())
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        assert_eq!(
            1,
            value_indices.len(),
            "`{query}`: expected exactly one value column in {columns:?}"
        );

        let mut rows = Vec::new();
        for batch in batches.iter() {
            let ts = cast(batch.column(time_index), &DataType::Int64).unwrap();
            let ts = ts.as_any().downcast_ref::<Int64Array>().unwrap();
            let labels = label_indices
                .iter()
                .map(|index| {
                    let array = cast(batch.column(*index), &DataType::Utf8).unwrap();
                    (columns[*index].clone(), array)
                })
                .collect::<Vec<(String, ArrayRef)>>();
            let value = cast(batch.column(value_indices[0]), &DataType::Float64).unwrap();
            let value = value.as_any().downcast_ref::<Float64Array>().unwrap();
            for row in 0..batch.num_rows() {
                let mut row_labels = BTreeMap::new();
                for (name, array) in &labels {
                    let array = array.as_any().downcast_ref::<StringArray>().unwrap();
                    if !array.is_null(row) {
                        row_labels.insert(name.clone(), array.value(row).to_string());
                    }
                }
                rows.push(ResultRow {
                    ts_millis: ts.value(row),
                    labels: row_labels,
                    value: (!value.is_null(row)).then_some(value.value(row)),
                });
            }
        }
        // Row order across partitions is not part of the answer, so both sides are ordered the same
        // way first. The sort is stable, so duplicate rows keep their multiplicity.
        rows.sort_by(|left, right| {
            (left.ts_millis, &left.labels).cmp(&(right.ts_millis, &right.labels))
        });

        Self { columns, rows }
    }
}

/// One query, and what it must answer.
struct AggregateCase {
    promql: String,
    expected: Vec<ResultRow>,
    /// Whether every one of the four scan partitions must report a folded value.
    per_partition: bool,
}

/// Builds one case from a PromQL expression and its `(labels, value)` rows.
fn case(promql: String, expected: &[(&[(&str, &str)], f64)], per_partition: bool) -> AggregateCase {
    AggregateCase {
        promql,
        expected: expected
            .iter()
            .map(|(labels, value)| ResultRow {
                ts_millis: EVAL_MILLIS,
                labels: labels
                    .iter()
                    .map(|(name, value)| ((*name).to_string(), (*value).to_string()))
                    .collect(),
                value: Some(*value),
            })
            .collect(),
        per_partition,
    }
}

/// The complete statement one case runs, reported in every assertion message.
fn tql_eval(promql: &str) -> String {
    format!("TQL EVAL ({EVAL_SECS}, {EVAL_SECS}, '1m') {promql}")
}

/// Runs `promql` at [`EVAL_SECS`] and returns its rows with the native nodes its plans held.
async fn run_query(
    instance: &Arc<Instance>,
    promql: &str,
    native: bool,
    observer: &NativePlanObserver,
) -> (QueryResult, NativeCapture) {
    let sql = tql_eval(promql);
    let mark = observer.mark();
    let mut query_ctx = QueryContext::with_db_name(None);
    query_ctx.set_extension(QUERY_PARALLELISM_HINT, SCAN_PARTITIONS.to_string());
    query_ctx.set_extension(NATIVE_PROMQL_EXTENSION_KEY, native.to_string());
    let output = instance
        .do_query(&sql, Arc::new(query_ctx))
        .await
        .remove(0)
        .unwrap_or_else(|err| panic!("`{sql}` failed: {err}"));
    let batches = match output.data {
        OutputData::Stream(stream) => collect_batches(stream)
            .await
            .unwrap_or_else(|err| panic!("`{sql}` produced no batches: {err}")),
        OutputData::RecordBatches(batches) => batches,
        other => panic!("`{sql}` produced no batches: {other:?}"),
    };
    // Read the plans only now: a producer starts reporting a partition's metrics when that partition
    // runs and stops updating them when it ends.
    (
        QueryResult::from_batches(&batches, &sql),
        NativeCapture::from_plans(&observer.since(mark)),
    )
}

/// Executes one statement and drops its result.
async fn execute_sql(instance: &Arc<Instance>, sql: &str) {
    for result in instance.do_query(sql, QueryContext::arc()).await {
        result.unwrap_or_else(|err| panic!("`{sql}` failed: {err}"));
    }
}

/// The DDL of both fixture tables: an ordinary table, no `PARTITION` clause, one region.
///
/// TWCS keeps its files apart until a hundred exist, so flushing one file per series leaves the
/// fixture with four files instead of a compacted one.
fn create_table_sql(table: &str) -> String {
    format!(
        "CREATE TABLE {table} (ts TIMESTAMP(3) TIME INDEX, job STRING, instance STRING, v DOUBLE, \
         PRIMARY KEY (job, instance)) WITH ('compaction.type' = 'twcs', \
         'compaction.twcs.trigger_file_num' = '100')"
    )
}

/// Fills both fixture tables.
///
/// The gauge table gets one flush per series but the last, so four files and one memtable range
/// reach the scan; the two samples of series `index` are 10s apart and 40s after the previous
/// series'. The range table gets one series whose samples are 390s apart: both are inside a 10m range
/// at [`EVAL_SECS`] and the older one is outside the default 5m lookback, so a query that reads its
/// own range counts two samples where one that used the lookback would count one.
async fn setup_fixture(instance: &Arc<Instance>) {
    execute_sql(instance, &create_table_sql(GAUGE_TABLE)).await;
    for (index, (job, host, v)) in GAUGE_SERIES.iter().enumerate() {
        let first = (BASE_SECS + 40 * index as i64) * 1000;
        let second = first + 10_000;
        execute_sql(
            instance,
            &format!(
                "INSERT INTO {GAUGE_TABLE} (ts, job, instance, v) VALUES \
                 ({first}, '{job}', '{host}', {v:?}), ({second}, '{job}', '{host}', {v:?})"
            ),
        )
        .await;
        if index + 1 < GAUGE_SERIES.len() {
            execute_sql(instance, &format!("ADMIN FLUSH_TABLE('{GAUGE_TABLE}')")).await;
        }
    }

    execute_sql(instance, &create_table_sql(RANGE_TABLE)).await;
    execute_sql(
        instance,
        &format!(
            "INSERT INTO {RANGE_TABLE} (ts, job, instance, v) VALUES \
             ({}, 'r', 'r1', 1.0), ({}, 'r', 'r1', 1.0)",
            (EVAL_SECS - 400) * 1000,
            (EVAL_SECS - 10) * 1000
        ),
    )
    .await;
}

/// Whether two values agree within [`FLOAT_TOLERANCE`].
fn close_enough(left: f64, right: f64) -> bool {
    if left.is_nan() || right.is_nan() {
        return left.is_nan() && right.is_nan();
    }
    (left - right).abs() <= FLOAT_TOLERANCE * left.abs().max(right.abs()).max(1.0)
}

/// Asserts `actual` holds exactly `expected`, row order aside but multiplicity included.
fn assert_rows(actual: &[ResultRow], expected: &[ResultRow], context: &str) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "{context}: expected {} rows, got {}:\n{actual:#?}",
        expected.len(),
        actual.len()
    );
    for (want, got) in expected.iter().zip(actual.iter()) {
        assert_eq!(want.ts_millis, got.ts_millis, "{context}: wrong `ts`");
        assert_eq!(
            want.labels, got.labels,
            "{context}: wrong labels in the row at {}",
            want.ts_millis
        );
        match (want.value, got.value) {
            (Some(want_value), Some(got_value)) => assert!(
                close_enough(want_value, got_value),
                "{context}: expected {want_value}, got {got_value} for labels {:?}",
                want.labels
            ),
            (want_value, got_value) => panic!(
                "{context}: expected the value {want_value:?}, got {got_value:?} for labels {:?}",
                want.labels
            ),
        }
    }
}

/// Asserts both routes report the answer with the same columns.
///
/// The value column is named after the expression the frontend computes, which is the legacy naming:
/// a native run that renamed it, or dropped, added or reordered a column, fails here rather than
/// passing on values alone. Column order is not compared, only the answer's shape.
fn assert_same_columns(native: &QueryResult, legacy: &QueryResult, query: &str) {
    let sorted = |result: &QueryResult| {
        let mut columns = result.columns.clone();
        columns.sort();
        columns
    };
    assert_eq!(
        sorted(legacy),
        sorted(native),
        "`{query}`: the native route must report the columns the legacy plan reports"
    );
}

/// The `native.root.output_values` counts of the fused producers in `capture`, per partition.
///
/// An entry stays [`None`] when the producer never registered the metric for a partition, which is a
/// different failure from a partition that produced nothing; both are reported as they are.
fn fused_output_values_per_partition(capture: &NativeCapture, context: &str) -> Vec<Option<usize>> {
    let mut per_partition = vec![None; SCAN_PARTITIONS];
    for fused in &capture.fused {
        let plan = displayable(fused.as_ref()).indent(false).to_string();
        let metrics = fused
            .metrics()
            .unwrap_or_else(|| panic!("{context}: the fused producer reports no metrics:\n{plan}"));
        for metric in metrics.iter() {
            let MetricValue::Count { count, .. } = metric.value() else {
                continue;
            };
            if metric.value().name() != ROOT_OUTPUT_VALUES_METRIC {
                continue;
            }
            let partition = metric.partition().unwrap_or_else(|| {
                panic!(
                    "{context}: `{ROOT_OUTPUT_VALUES_METRIC}` is not labelled with its partition"
                )
            });
            assert!(
                partition < SCAN_PARTITIONS,
                "{context}: partition {partition} is beyond the {SCAN_PARTITIONS} the scan was \
                 asked for"
            );
            per_partition[partition] = Some(per_partition[partition].unwrap_or(0) + count.value());
        }
    }
    per_partition
}

/// Asserts the fused producer reads the raw series scan itself, with no exchange between them.
///
/// One node may sit in between: the cooperative-scheduling decorator clones its input's properties,
/// changes only the scheduling type, and forwards `execute` unchanged, so it neither re-partitions
/// nor re-orders the frames. It is accepted only as a single typed wrapper above the scan, never as
/// a wildcard. An exchange above the scan would re-partition the frames, and the producer's
/// per-partition fold would then no longer be one series at a time - so the plan would be wrong even
/// if a result appeared.
fn assert_fused_reads_scan_directly(fused: &Arc<dyn ExecutionPlan>, context: &str) {
    let plan = displayable(fused.as_ref()).indent(false).to_string();
    assert_eq!(
        SCAN_PARTITIONS,
        fused.properties().output_partitioning().partition_count(),
        "{context}: the fused producer must keep the {SCAN_PARTITIONS} scan partitions:\n{plan}"
    );
    let children = fused.children();
    assert_eq!(
        1,
        children.len(),
        "{context}: the fused producer must take exactly one input:\n{plan}"
    );
    let child = children[0];
    // At most one transparent wrapper is allowed between the producer and the scan.
    let scan = if child.is::<CooperativeExec>() {
        let wrapper_children = child.children();
        assert_eq!(
            1,
            wrapper_children.len(),
            "{context}: the cooperative wrapper must wrap exactly one input:\n{plan}"
        );
        assert_eq!(
            SCAN_PARTITIONS,
            child.properties().output_partitioning().partition_count(),
            "{context}: the cooperative wrapper must not change the {SCAN_PARTITIONS} scan \
             partitions:\n{plan}"
        );
        wrapper_children[0]
    } else {
        child
    };
    assert!(
        scan.is::<SeriesScanExec>(),
        "{context}: the fused producer must read the raw series scan directly, with no \
         `CoalesceExec` or `RepartitionExec` between them, but its input is `{}`:\n{plan}",
        scan.name()
    );
    assert_eq!(
        SCAN_PARTITIONS,
        scan.properties().output_partitioning().partition_count(),
        "{context}: `SeriesScanExec` must expose the {SCAN_PARTITIONS} scan partitions the fused \
         producer reads:\n{plan}"
    );
}

/// Checks one case on both routes: the legacy run first, then the native one.
async fn check_case(instance: &Arc<Instance>, observer: &NativePlanObserver, case: &AggregateCase) {
    let promql = case.promql.as_str();
    let query = tql_eval(promql);
    let (legacy, capture) = run_query(instance, promql, false, observer).await;
    assert!(
        capture.fused.is_empty() && capture.finalize.is_empty(),
        "`{query}`: the legacy run must not plan a native node"
    );
    assert_rows(&legacy.rows, &case.expected, &format!("legacy `{query}`"));

    let (native, capture) = run_query(instance, promql, true, observer).await;
    assert_same_columns(&native, &legacy, &query);
    assert_rows(&native.rows, &case.expected, &format!("native `{query}`"));

    assert_eq!(
        1,
        capture.fused.len(),
        "`{query}`: one region is read, so exactly one fused producer must run, but the plans held \
         {} of them",
        capture.fused.len()
    );
    for fused in &capture.fused {
        assert_fused_reads_scan_directly(fused, &format!("native `{query}`"));
    }
    assert!(
        !capture.finalize.is_empty(),
        "`{query}`: the native run must plan the frontend-side `{FINALIZE_EXEC_NAME}` merge, \
         otherwise the answer came from a fall back to the legacy plan"
    );

    let per_partition = fused_output_values_per_partition(&capture, &format!("native `{query}`"));
    eprintln!(
        "`{query}` native per-partition `{ROOT_OUTPUT_VALUES_METRIC}` counts: {per_partition:?}"
    );
    if case.per_partition {
        for (partition, values) in per_partition.iter().enumerate() {
            assert!(
                values.is_some_and(|values| values > 0),
                "`{query}`: every one of the {SCAN_PARTITIONS} scan partitions must fold at least \
                 one value, but partition {partition} reported {values:?}; all partitions: \
                 {per_partition:?}"
            );
        }
    } else {
        assert!(
            per_partition.iter().flatten().sum::<usize>() > 0,
            "`{query}`: the fused producer must report folded values, got {per_partition:?}"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn native_aggregate_complete_over_four_nonempty_mito_partitions() {
    common_telemetry::init_default_ut_logging();

    let observer = Arc::new(NativePlanObserver::default());
    let plugins = Plugins::new();
    plugins.insert(Arc::clone(&observer) as PhysicalPlanWrapperRef);
    // The engine-wide switch alone leaves the route off: a query also has to ask for it, which every
    // native run below does and every legacy run does not.
    let standalone = GreptimeDbStandaloneBuilder::new("native_aggregate_complete")
        .with_query_options(QueryOptions {
            experimental_native_promql: true,
            ..Default::default()
        })
        .with_plugin(plugins)
        .build()
        .await;
    let instance = Arc::clone(standalone.fe_instance());
    setup_fixture(&instance).await;

    /// The two expected rows of a query grouped by `job`.
    fn by_job(a: f64, b: f64) -> [(&'static [(&'static str, &'static str)], f64); 2] {
        const GROUP_A: &[(&str, &str)] = &[("job", "a")];
        const GROUP_B: &[(&str, &str)] = &[("job", "b")];
        [(GROUP_A, a), (GROUP_B, b)]
    }
    // Both tables have exactly one value field (`v`), so a bare table selector reads it. A
    // `__field__` matcher would narrow the field set, and the native route declines such a query.
    let cases = [
        case(
            format!("sum by (job) ({GAUGE_TABLE})"),
            &by_job(18.0, 12.0),
            true,
        ),
        case(
            format!("avg by (job) ({GAUGE_TABLE})"),
            &by_job(6.0, 6.0),
            true,
        ),
        case(
            format!("count by (job) ({GAUGE_TABLE})"),
            &by_job(3.0, 2.0),
            true,
        ),
        // `a` deviates by sqrt(32/3) and `b` by 2.
        case(
            format!("stddev by (job) ({GAUGE_TABLE})"),
            &by_job((32.0f64 / 3.0).sqrt(), 2.0),
            true,
        ),
        case(
            format!("sum without (instance) ({GAUGE_TABLE})"),
            &by_job(18.0, 12.0),
            true,
        ),
        case(
            format!("abs(sum by (job) ({GAUGE_TABLE}))"),
            &by_job(18.0, 12.0),
            true,
        ),
        // The inner average is one value per group and the outer sum over the ungrouped result is a
        // single 12, so the merge's suffix really runs on merged values.
        case(
            format!("sum(avg({GAUGE_TABLE}) by (job))"),
            &[(&[], 12.0)],
            true,
        ),
        // A gauge over a constant value has a rate of exactly zero, so these two only cover the
        // zero-rate case: every series has both samples inside the 5m window, hence every partition
        // must fold a value. They do not cross a counter's partial/final state bridge, because
        // nothing in the fixture is a counter.
        case(format!("sum(rate({GAUGE_TABLE}[5m]))"), &[(&[], 0.0)], true),
        case(
            format!("sum by (job) (rate({GAUGE_TABLE}[5m]))"),
            &by_job(0.0, 0.0),
            true,
        ),
        // The range table holds one series, so this is a correctness smoke test of one populated
        // partition rather than a parallel acceptance.
        case(
            format!("sum(count_over_time({RANGE_TABLE}[10m]))"),
            &[(&[], 2.0)],
            false,
        ),
    ];
    for case in &cases {
        check_case(&instance, &observer, case).await;
    }
}

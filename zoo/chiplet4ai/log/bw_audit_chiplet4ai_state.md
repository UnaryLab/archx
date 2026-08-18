# chiplet4ai bandwidth model: current state (T2)

> **SUPERSEDED - this is the PRE-fix state of 2026-08-17.** Fixes T4 through T9 landed the
> same day, so every measurement below describes the model *before* those fixes. Apart from
> bracketed cross-reference notes, the body has not been revised; the numbers are kept as a
> dated snapshot.
> **`zoo/chiplet4ai/scalesim_audit.md` is the source of truth for current state.**
>
> **Known-wrong claims below, not merely superseded numbers.** The blanket above covers
> measurements overtaken by the fixes. These three were **wrong when written** - reasoning
> errors, not stale numbers - and are corrected here rather than in the body:
>
> 1. **D.12's "Latent" heading and its "Not triggered today" claim are false.** The
>    `_component_event_counts` diamond bug was **live**, not latent: old versus new differ on
>    **40/40** sampled root queries, by up to **2.37x** (`pe` 1.84e19 vs 7.76e18). It never
>    reached published numbers only because the three buckets filter on `_arr`/`_sram`/`_dram`
>    suffixes sitting **above** the diamond, so the frozen CSVs were always safe. That is a
>    materially different reason from "not triggered", and the fix was real.
> 2. **D.11's misnomer is no longer live.** The metric has since been renamed
>    `memory_stall_cycle_count` -> **`dram_stall_cycle_count`**, a name that appears nowhere
>    else in this document. D.11's diagnosis was right; its heading reads as a present-tense
>    defect and is not one.
> 3. **The flat "`unattributed_cycle_count == 0.0`" phrasing in D.12's prose is wrong as
>    stated**, and it is what makes (1) look safe. The correct claim is **residual <= 2e-16
>    relative**, ULP-nonzero on 11 of 40 sampled runs. (The dated result dicts in Part A are a
>    snapshot of one run and are correct as printed; only the load-bearing prose in D.12 is at
>    issue.)
>
> **Stall events renamed since this was written.** The `sram_*` old names occur in the body;
> the three `dram_*` old names do **not** - the body only ever writes `dram_*_stall`:
>
> | old | new | occurs in body |
> |---|---|---|
> | `dram_weight_read_stall` | `dram_fill_stall` | no |
> | `dram_input_read_stall` | `dram_steady_stall` | no |
> | `dram_output_write_stall` | `dram_tail_stall` | no |
> | `sram_weight_fill_stall` | `sram_fill_stall` | yes |
> | `sram_steady_state_stall` | `sram_steady_stall` | yes |
> | `sram_output_tail_stall` | `sram_tail_stall` | yes |
>
> **Model identity, all three clauses (never quote the rounding bound alone):**
> (i) byte conservation is **exact** (max relative residual 1.3e-16 across the shape sweep);
> (ii) rounding error is **bounded and independent of the number of prorated streams** -
> `E` in `{0..P-1}` over `P` nonzero-window phases, and the supremum `P-1` **is attained**,
> not merely approached; stream count drops out because residues from different streams add
> **before** the per-phase ceil, so a tenth prorated stream still cannot push `E` past `P-1`
> (at most a couple of cycles here); (iii) slack error is **unbounded, and the one that
> matters** - with slack in any phase the overstatement is bounded only by the roofline
> itself, up to 2x runtime; measured, that is **12.1%** on the reference PE at bw=1024,
> **more than 2 cycles in 18.5%** of a 57,820-point sweep, and **74.1%** worst relative. It
> is being fixed under T12. Quoting the bounded rounding clause on its own covers a genuinely
> small artifact while licensing a 3x10^11-cycle error under a 2-cycle banner.

Analysis only - no code changed. Scope: `zoo/chiplet4ai/common/performance/{mapping,memory,node,utils}.py`
and `zoo/chiplet4ai/results/query/{fig_2_query,utils}.py`, checked against
`zoo/chiplet4ai/scalesim_audit.md` (the prior audit).

Unqualified line cites below are `mapping.py:<line>`. All numbers labelled "measured" come
from live calls against the audit's reference config: `pe=[512,512]`, isram/wsram/osram each
10 MiB (`width 16, bank 1024, depth 5120`), `dram bandwidth 256, frequency 1000`,
GEMM `batch=1, M=2048, K=N=8192, step_dim=None`.

---

## Part 0 - Executive summary

- **All five `[FIXED]` / `[MOSTLY FIXED]` audit items (1, 2, 3, 4, 6) hold in today's code.
  No regressions.** Every quantitative claim in the audit reproduces, with two wording
  corrections (§B).
- The two open items (5, and the residual of 6) are confirmed and characterised precisely
  in §C, with worked numbers: item 5 causes a **41 % under-count of DRAM stall in the
  reference config, and a 2.29× optimistic runtime under `k_outer`**.
- §D lists 14 bandwidth-relevant behaviours the audit does not cover. The three that most
  affect published numbers: **DRAM output reads generate no stall at all**; **the SRAM
  bandwidth columns in `fig_2_query` are dimensionally wrong** (fill counts multiplied by
  bytes-per-word); and **the weight panel of fig_2 has no real signal** - its slope is a
  heuristic startup-span artifact, not reuse.

---

## Part A - The formula chain, end to end

### A.0 Rate and unit primitives

| quantity | expression | line |
|---|---|---|
| `_frequency_mhz` | `float(arch['dram']['query']['frequency'])`, default `1000.0` | :33-37 |
| `_dram_bytes_per_cycle` | `max(1e-12, bandwidth_gbs * 2**30 / frequency_hz)` | :39-42 |
| `_bandwidth_gib_per_second` | `bytes_count / cycles * frequency_hz / 2**30` | :47-51 |
| `_active_fraction` | `query.get('active_fraction', query.get('active_buffer_fraction', 0.5))`, clamped | :53-56 |
| `_sram_elements` | `max(1, width*bank*depth // width)` = `bank*depth` | :58-60, utils.py:44-46 |
| `_buffer_elements` | `active = max(1, min(total, floor(total*frac)))`; `prefetch = total - active` | :62-66 |
| `_sram_bytes_per_cycle` | `max(1e-12, max(1.0, bank/2) * (width/8))` | :68-72 |
| `_service_cycles` | `ceil(bytes / max(1e-12, bytes_per_cycle))` | :74-77 |
| `_fit_fraction` | `clamp(capacity / working_set, 0, 1)` - linear, uses the **active half** | :291-294 |
| `_resident_refetch_elements` | `unique + max(0, streamed - unique) * (1 - fit)` | :296-298 |

Measured: `dram_bpc = 274.877906944` B/cy; each SRAM `sram_bpc = 1024.0` B/cy;
`active/prefetch = (2621440, 2621440)` elements per SRAM.

### A.1 Fold structure and compute span (`_ws_schedule`, :316-537)

No-step branch (:387-402):

```
k_folds = ceil(K / array_rows)                       :388   = 16
n_folds = ceil(N / array_cols)                       :389   = 16
fold_count = k_folds * n_folds                       :395   = 256
compute_cycles = batch * fold_count * (M + R + C - 2)   :396 = 256 * 3070 = 785,920
```

Stepped branches (`step_dim` in `'m'`/`'k'`/`'n'`, :345-386) replace the scalars with
closed-form sums (`_range_sum` :271-274, `_ceil_sum` :276-285); `compute_cycles` at
:352 / :366 / :380.

### A.2 Element counts (:397-401, :416-417)

```
input_unique_elements  = batch*M*K                        :397  = 16,777,216
input_sram_elements    = input_unique_elements * n_folds  :398  = 268,435,456
weight_elements        = batch*K*N                        :399  = 67,108,864
output_write_elements  = batch*M*N*k_folds                :400  = 268,435,456
output_read_elements   = batch*M*N*max(0, k_folds-1)      :401  = 251,658,240
output_final_elements  = output_write_elements - output_read_elements  :416 = 16,777,216
output_accum_elements  = output_read_elements                          :417 = 251,658,240
```

### A.3 Loop order → DRAM traffic gating (:404-438)

Candidate working sets (:409-413):

```
max_input_strip   = batch*max_M*min(array_rows, max_K)    :409
max_input_matrix  = batch*max_M*max_K                     :410
max_output_strip  = batch*max_M*min(array_cols, max_N)    :411
max_output_matrix = batch*max_M*max_N                     :412
```

`_order_traffic` (:419-426) derives **both** operands from one `(input_tile, output_tile)` pair:

```python
input_fit  = _fit_fraction(isram_active_elements, input_tile)                  :420
output_fit = _fit_fraction(osram_active_elements, output_tile)                 :421
input_read = _resident_refetch_elements(input_unique_elements,
                                        input_sram_elements, input_fit)        :422
dram_write = output_final_elements + output_accum_elements * (1.0 - output_fit) :423
dram_read  = output_accum_elements * (1.0 - output_fit)                        :424
dram_bytes = input_read*input_bytes + (dram_write + dram_read)*output_bytes    :425
```

```python
order_traffic = {'k_outer': _order_traffic(max_input_strip,  max_output_matrix),   :429
                 'n_outer': _order_traffic(max_input_matrix, max_output_strip)}    :430
if loop_order == 'auto':
    loop_order = min(order_traffic, key=lambda o: order_traffic[o][-1])            :434
assert loop_order in order_traffic, ...                                            :435
```

`'auto'` = argmin over `dram_bytes`; weights excluded (they move once either way). Ties
resolve to `k_outer` by dict insertion order.

**Reference-config arithmetic (measured, verbatim):**

```
k_outer input_tile 1048576  ifit 1.0      output_tile 16777216 ofit 0.15625
        input_read_el 16777216.0   dram_w 229113856.0  dram_r 212336640.0  dram_bytes 916455424.0
n_outer input_tile 16777216 ifit 0.15625  output_tile 1048576  ofit 1.0
        input_read_el 229113856.0  dram_w 16777216.0   dram_r 0.0          dram_bytes 491782144.0
```

`491,782,144 < 916,455,424` ⇒ **`'auto'` picks `n_outer`**. It trades 425 MB of DRAM output
partial-sum traffic for 425 MB of input refetch and wins because output partials are counted
twice (:423 and :424 both carry `output_accum*(1−ofit)`) while input elements are counted once.

### A.4 Byte counts (:442-451)

| quantity | expression | line | n_outer / k_outer |
|---|---|---|---|
| `input_read_bytes` (DRAM) | `input_read_elements * input_bytes` | :442 | 458,227,712 / 33,554,432 |
| `weight_read_bytes` (DRAM) | `weight_read_elements * weight_bytes` | :443 | 134,217,728 (both) |
| `output_read_bytes` (DRAM) | `dram_output_read_elements * output_bytes` | :444 | 0 / 424,673,280 |
| `output_write_bytes` (DRAM) | `dram_output_write_elements * output_bytes` | :445 | 33,554,432 / 458,227,712 |
| `input_sram_bytes` | `input_sram_elements * input_bytes` | :446 | 536,870,912 (both) |
| `weight_sram_bytes` | `weight_read_elements * weight_bytes` | :447 | 134,217,728 (both) |
| `osram_read_bytes` | `output_accum_elements * output_bytes` | :449 | 503,316,480 (both) |
| `osram_write_bytes` | `output_write_elements * output_bytes` | :450 | 536,870,912 (both) |
| `output_sram_bytes` | `osram_read_bytes + osram_write_bytes` | :451 | 1,040,187,392 (both) |

`weight_read_elements = weight_elements` unconditionally (:440) - weight DRAM traffic is
**independent of wsram size**. SRAM-side input traffic (:446) is order-invariant; DRAM-side
input traffic (:442) is order-dependent.

### A.5 Service rates - one channel handed to each operand (:335-341)

```python
bytes_per_cycle                 = _dram_bytes_per_cycle(arch)                    :335
input_prefetch_bytes_per_cycle  = min(bytes_per_cycle, isram_bytes_per_cycle)    :339
weight_prefetch_bytes_per_cycle = min(bytes_per_cycle, wsram_bytes_per_cycle)    :340
output_drain_bytes_per_cycle    = min(bytes_per_cycle, osram_bytes_per_cycle)    :341
```

All three = `274.877906944` in the reference config (SRAM is 3.7× faster, so the `min`
always collapses to DRAM). See §C.1.

### A.6 DRAM stalls - window is the whole compute span (:458-463)

```python
input_service_cycles  = _service_cycles(input_read_bytes,   input_prefetch_bytes_per_cycle)   :458
weight_service_cycles = _service_cycles(weight_read_bytes,  weight_prefetch_bytes_per_cycle)  :459
output_service_cycles = _service_cycles(output_write_bytes, output_drain_bytes_per_cycle)     :460
input_read_stall_cycles  = max(0, input_service_cycles  - compute_cycles)   :461
weight_read_stall_cycles = max(0, weight_service_cycles - compute_cycles)   :462
write_stall_cycles       = max(0, output_service_cycles - compute_cycles)   :463
```

Measured (n_outer): input `ceil(458,227,712/274.878) = 1,667,023 − 785,920 = 881,103`;
weight `488,282 − 785,920 → 0`; output `122,071 − 785,920 → 0`.

**`output_read_bytes` never enters any stall term** - :460 uses `output_write_bytes` only.

### A.7 SRAM phase stalls - three different windows (:465-480)

```python
weight_fill_window = batch * max(0, array_rows - 1) * fold_count           :465
steady_window      = batch*ceil(K/R)*ceil(N/C)*sum_M  if step_dim=='m'
                     else batch * M * fold_count                           :466-470
output_tail_window = batch * max(0, array_cols - 1) * fold_count           :471

weight_fill_stall_cycles  = max(0, _service_cycles(weight_sram_bytes, wsram_bpc)
                                   - weight_fill_window)                   :472
steady_state_stall_cycles = max(0, max(_service_cycles(input_sram_bytes, isram_bpc),
                                       _service_cycles(osram_read_bytes, osram_bpc))
                                   - steady_window)                        :473-479
output_tail_stall_cycles  = max(0, _service_cycles(osram_write_bytes, osram_bpc)
                                   - (steady_window + output_tail_window)) :480
```

Measured: `weight_fill_stall = 256` (exactly 1 cy/fold, structural); `steady_state_stall = 0`;
`output_tail_stall = 0`.

### A.8 Stall combination (:481, :525-527)

```python
read_stall_cycles = max(input_read_stall_cycles, weight_read_stall_cycles) \
                  + weight_fill_stall_cycles + steady_state_stall_cycles \
                  + output_tail_stall_cycles                               :481
'stall_cycles': read_stall_cycles + write_stall_cycles                     :527
```

DRAM input/weight combine by `max`; everything else by `+`.

### A.9 Bandwidth windows (:483-494)

```python
input_transfer_window_cycles  = compute_cycles if input_read_bytes  > 0 else 0    :484
weight_transfer_window_cycles = _read_active_span_cycles(compute_cycles,
                                  weight_elements, weight_read_elements,
                                  wsram_active_elements)                          :485-490
output_transfer_window_cycles = compute_cycles if output_write_bytes > 0 else 0    :491
input_sram_window_cycles      = compute_cycles if input_sram_bytes  > 0 else 0     :492
weight_sram_window_cycles     = compute_cycles if weight_sram_bytes > 0 else 0     :493
output_sram_window_cycles     = compute_cycles if output_sram_bytes > 0 else 0     :494
```

`_read_active_span_cycles` (:300-314):

```python
demand_rate  = demand_elements / compute_cycles                                   :311
demand_span  = min(compute_cycles, dram_elements / max(1e-12, demand_rate))       :312
startup_span = min(dram_elements, max(1.0, startup_buffer_elements)) / demand_rate :313
return max(1.0, demand_span + startup_span)                                       :314
```

Since `weight_read_elements == weight_elements`, `demand_span == compute_cycles` exactly, so
the weight window is always **strictly longer than compute** and grows with wsram size.
Measured: `816,620` vs `785,920` compute (+3.9 %). Nothing caps the sum.

### A.10 The six reported bandwidth factors (:528-533)

| key | numerator | denominator | line | measured (n_outer) |
|---|---|---|---|---|
| `input_sram_bandwidth` | `input_sram_bytes` (:446) | compute (:492) | :528 | 636.197 GiB/s |
| `weight_sram_bandwidth` | `weight_sram_bytes` (:447) | compute (:493) | :529 | 159.049 |
| `output_sram_bandwidth` | `osram_read + osram_write` (:451) | compute (:494) | :530 | 1232.632 |
| `input_dram_bandwidth` | `input_read_bytes` (:442) | compute (:484) | :531 | 543.004 |
| `weight_dram_bandwidth` | `weight_read_bytes` (:443) | **816,620** (:485) | :532 | 153.070 |
| `output_dram_bandwidth` | `output_write_bytes` **only** (:445) | compute (:491) | :533 | 39.762 |

All six pass through `_bandwidth_gib_per_second` (:47-51) → **GiB/s**.

Two consistency defects visible in this table alone:
- Identical byte counts (:443 == :447) report **two different numbers** - 153.070 vs 159.049  -
  purely because weight uses the startup-span window.
- `output_dram_bandwidth` excludes `output_read_bytes` from its numerator yet is attached to
  the `dram_output_read` event (:676) as well as `dram_output_write` (:681).
- `output_sram_bandwidth` is a read+write composite attached to **both** directions (:628, :636).

Full measured `_ws_schedule` output, `loop_order='auto'`, reference config:

```
loop_order = n_outer
input_count = 87.4          weight_count = 25.6
output_read_count = 96.0    output_write_count = 102.4
input_read_bytes = 458227712.0     weight_read_bytes = 134217728.0
output_read_bytes = 0.0            output_write_bytes = 33554432.0
input_sram_bytes = 536870912.0     weight_sram_bytes = 134217728.0
output_sram_bytes = 1040187392.0
input_transfer_window_cycles = 785920   weight_transfer_window_cycles = 816620.0
output_transfer_window_cycles = 785920
compute_cycles = 785920
input_read_stall_cycles = 881103   weight_read_stall_cycles = 0
weight_fill_stall_cycles = 256     steady_state_stall_cycles = 0
output_tail_stall_cycles = 0
read_stall_cycles = 881359   write_stall_cycles = 0   stall_cycles = 881359
input_sram_bandwidth = 636.1970684039088
weight_sram_bandwidth = 159.0492671009772
output_sram_bandwidth = 1232.6318200325734
input_dram_bandwidth = 543.00413846193
weight_dram_bandwidth = 153.06997134530135
output_dram_bandwidth = 39.7623167752443
mapping_efficiency = 1.0   compute_utilization = 0.6671009771986971
```

Forced `k_outer`, same config (only the order-gated fields differ):

```
input_read_bytes = 33554432.0    output_read_bytes = 424673280.0
output_write_bytes = 458227712.0
input_read_stall_cycles = 0      write_stall_cycles = 881103
read_stall_cycles = 256          stall_cycles = 881359
input_dram_bandwidth = 21.689820984363894
output_dram_bandwidth = 296.2016178177194
```

### A.11 Entry points

**`gemm()` (:539-599)** consumes only `compute_cycles` and `total_steps` (:544, :580-596) and
re-derives its own tile counts (:549-574). **It emits no bandwidth.** Because
`count × factor == compute_cycles/total_steps`, the array path contributes exactly
`compute_cycles`. Note `m_tiles` uses `array_rows` (:556) - a different decomposition from the
schedule's `k_folds×n_folds`, reconciled only through `compute_cycles`.

**`sram()` (:601-653)** - pure pass-through of :528-530. Counts are **buffer fills**
(dimensionless), `factor.cycle_count = runtime = 0`; time comes only from the stall chains
(:639-650, `aggregation: 'sequential'`, one `cycle_reference` per stall cycle).

| event | count | bandwidth factor |
|---|---|---|
| `sram_input_write_mapping` (:607-614) | `input_count` | `input_sram_bandwidth` |
| `sram_weight_write_mapping` (:615-622) | `weight_count` | `weight_sram_bandwidth` |
| `sram_output_read_mapping` (:623-630) | `output_read_count` | `output_sram_bandwidth` |
| `sram_output_write_mapping` (:631-638) | `output_write_count` | `output_sram_bandwidth` |

**`dram()` (:655-697)** - counts are **bytes** despite the `*_count` names (:660-663).
Events `dram_input_read` (:666-669), `dram_weight_read` (:670-673), `dram_output_read`
(:674-677), `dram_output_write` (:678-682); stalls at :683-694. Only `dram_output_write`
carries `'aggregation': 'sequential'` (:680); the other three default to parallel.

### A.12 The reporting half: model → CSV → figure

```
_ws_schedule            -> '*_dram_bandwidth' / '*_sram_bandwidth'      mapping.py:528-533
dram() / sram()         -> subevent factor['bandwidth']                 mapping.py:665-682, 606-638
fig_2_query.collect_mapping_samples -> re-calls the python fn, reads
                           subevent['factor']['bandwidth']              fig_2_query.py:64, 73
fig_2_query.summarize_bandwidth     -> active_avg + peak                fig_2_query.py:37-47
CSV row (6 bandwidth columns, 4 of them aliases)                        fig_2_query.py:176-185
fig_2.py groupby(...).mean() -> two-panel line plot                     fig_2.py:63-77, 89-110
```

`collect_mapping_samples` **does not read bandwidth from the event graph**. It re-invokes the
Python performance function in-process (`getattr(llama_model, event)(...)`, fig_2_query.py:64)
and reads the factor directly (:73); the graph supplies only the multiplicity `event_count`
(:56-60). The graph-level `bandwidth` metric (`metric.yaml`, `aggregation: specified`, with
every leaf set to `{'value': 1, 'unit': 'GiB/s'}` in memory.py:9,21,33,…,117) is therefore
**dead for reporting** - fortunately, since `aggregation: specified` would sum it into a
meaningless extensive quantity.

Per sample (fig_2_query.py:72-82):

```python
count           = float(subevent.get('count', 0)) * event_count
bandwidth       = float(subevent.get('factor', {}).get('bandwidth', 0))
data_moved      = count * spec.get('count_to_bytes', 1)
transfer_window = data_moved / (bandwidth * 2**30) * frequency_mhz * 1e6     # :26-29
```

Aggregation (fig_2_query.py:37-47):

```python
data_moved      = sum(s['data_moved'] for s in samples)
transfer_window = sum(s['transfer_window_cycles'] for s in samples)
active_avg      = bandwidth_from_window_cycles(data_moved, transfer_window, frequency_mhz)
peak            = max((s['bandwidth'] for s in samples if s['data_moved'] > 0), default=0)
```

Because `bandwidth_from_window_cycles` (:31-35) is the exact inverse of
`transfer_window_cycles` (:26-29), **`active_avg` collapses algebraically to a byte-weighted
harmonic mean of the per-event bandwidths**:

```
active_avg = ( Σ D_i ) / ( Σ D_i / BW_i )
```

Frequency and `2**30` cancel exactly. It is **not** a time average over the actual execution  -
it never touches `array_execution_time`, `compute_cycles`, or any stall count, and it is
invariant to rescaling every `D_i` by a common constant (which is why the SRAM unit bug in
§D.7 does not visibly corrupt the SRAM bandwidth columns even though the SRAM *window* columns
are nonsense).

**CSV columns and their aliases** (fig_2_query.py:176-185), verified against the emitted file:

| column | source | line |
|---|---|---|
| `input_dram_bandwidth` | `dram['input']['active_avg_bandwidth']` | 176 |
| `weight_dram_bandwidth` | `dram['weight']['active_avg_bandwidth']` | 177 |
| `input_sram_bandwidth` | `sram['input']['active_avg_bandwidth']` | 178 |
| `weight_sram_bandwidth` | `sram['weight']['active_avg_bandwidth']` | 179 |
| `input_active_avg_required_bandwidth` | **same as 176** | 180 |
| `weight_active_avg_required_bandwidth` | **same as 177** | 181 |
| `input_peak_required_bandwidth` | `dram['input']['peak_required_bandwidth']` | 182 |
| `weight_peak_required_bandwidth` | `dram['weight']['peak_required_bandwidth']` | 183 |
| `input_required_bandwidth` | **same as 176** | 184 |
| `weight_required_bandwidth` | **same as 177** | 185 |

So `input_dram_bandwidth ≡ input_active_avg_required_bandwidth ≡ input_required_bandwidth`  -
three columns, one number (audit hygiene bullet confirmed), and likewise for weight.
`fig_2.py` plots the `*_required_bandwidth` aliases (fig_2.py:65, 73, 91, 104). The SRAM
columns are written and **never plotted**. **No output bandwidth reaches any CSV or figure**  -
`movement_specs` requests only `input` and `weight` (fig_2_query.py:144-147, 155-159).

### A.13 `query_cycle_breakdown` (query/utils.py:53-81)

Three buckets, keyed purely on the event-name suffix over the transitive subtree:
`*_arr` → `compute_cycle_count` (:62), `*_sram` → `sram_cycle_count` (:67),
`*_dram` → `memory_stall_cycle_count` (:72); each `Σ path_count × query_cycle_count(...)`,
scaled by `total/local` (:57). A residual is emitted at :80.

Measured on `runs/llama_3_8b/arch_73/config_4`:

```
llama_array: {'cycle_count': 19069287820368.004, 'compute_cycle_count': 19069287820368.004,
              'sram_cycle_count': 0.0, 'memory_stall_cycle_count': 0.0,
              'unattributed_cycle_count': 0.0}
llama:       {'cycle_count': 19141491711871.004, 'compute_cycle_count': 19044254449824.004,
              'sram_cycle_count': 216426496.0, 'memory_stall_cycle_count': 97020835551.0,
              'unattributed_cycle_count': 0.0}
```

Both residuals are exactly `0.0` - the decomposition is exhaustive on today's graph. Three
structural caveats in §D.10-D.12.

---

## Part B - Verification of the `[FIXED]` items

Verified against the committed state (`git status --porcelain zoo/chiplet4ai/common` is empty,
so the code read is HEAD). No regressions found.

| item | audit label | **verdict** | note |
|---|---|---|---|
| 1 output tail window | FIXED | **HOLDS** | window is `M+C-1`/fold; audit text says `M+C-2` |
| 2 DRAM output writes | FIXED | **HOLDS** | gating symmetric; spill *reads* still untaxed |
| 3 loop order | FIXED | **HOLDS** | fully threaded, both operands from one order |
| 4 osram accumulate | FIXED | **HOLDS** | separate vars used everywhere; spill double-charged |
| 6 chunked windows | MOSTLY FIXED | **HOLDS** | helpers gone; weight window errs 3.9 % *low* |

### B.1 Item 1 - output tail stall window - HOLDS

```python
weight_fill_window = batch * max(0, array_rows - 1) * fold_count                    :465
steady_window      = ... else batch * M * fold_count                                :466-470
output_tail_window = batch * max(0, array_cols - 1) * fold_count                    :471
output_tail_stall_cycles = max(0, _service_cycles(osram_write_bytes, osram_bpc)
                                  - (steady_window + output_tail_window))           :480
```

The window is steady + tail: `256 * (2048 + 511) = 655,104` cy, not the old `(C-1)*folds =
130,816`. The pre-fix number reproduces exactly from the code shape:
`osram_write_bytes/osram_bpc = 536,870,912/1024 = 524,288`; `524,288 − 130,816 = 393,472`  -
the audit's figure to the cycle. Post-fix `524,288 − 655,104 < 0` → **0**, confirmed live.

Two corrections to the audit text:
- The audit says the window is `M + C − 2` per fold. **The code gives `M + C − 1`** (`steady_window`
  uses `M`, `output_tail_window` uses `C−1`) - one extra cycle per fold, 256 cy here.
- **"Stays positive for an undersized osram" is true only under a *bandwidth* reading.** A
  capacity-undersized osram does not affect this stall at all - both `osram_write_bytes` (:450)
  and the window are capacity-independent. Measured at 1 MiB osram, bank 1024:
  `output_tail_stall_cycles: 0`. It is bank/width that matters (10 MiB capacity held fixed):
  `bank512 → 393,472`; `bank256 → 1,442,048`; `bank128 → 3,539,200`. Correctly monotone in
  osram service bandwidth.

### B.2 Item 2 - DRAM output write traffic and symmetric gating - HOLDS

`:423` and `:424` apply the *same* `(1.0 − output_fit)` gate to the same
`output_accum_elements`; writes additionally carry `output_final_elements`. Consumed at
:437-445, emitted at :662-663.

Reference numbers reproduce the audit: `output_write_bytes: 33,554,432` (33.5 MB = `M*N*2`
final writes; old model `M*N*k_folds*2 = 536,870,912` = 537 MB), `write_stall_cycles: 0`
(old 1,167,205), `output_dram_bandwidth: 39.76` GiB/s (old 636; audit predicted "~40").

The spill path is live and non-degenerate - forced `k_outer` gives `output_read_bytes:
424,673,280`, `output_write_bytes: 458,227,712`, `write_stall_cycles: 881,103`.

**Remaining gap (not a regression):** the fix is one-sided on stalls - see §C.2.

### B.3 Item 3 - loop order threaded through both operands - HOLDS

Parameter present and threaded to every entry point: `_ws_schedule` :316, `gemm` :539→:542,
`sram` :601→:603, `dram` :655→:657, default `'auto'` everywhere. Repo-wide grep finds
`loop_order` only in `mapping.py` - no caller overrides the default, which is the intended path.

Both operands come from a single `_order_traffic` call per order (:429-430, :437-438), so the
"opposite loop orders" defect is genuinely gone. The `'auto'` rule minimises `dram_bytes`
(:425, :434), correctly excluding weights (:440).

Reference config reproduces the audit exactly: `loop_order: n_outer`, `input_read_bytes:
458,227,712` (458 MB), `input_read_stall_cycles: 881,103` (881k), `output_read_bytes: 0.0`.

### B.4 Item 4 - osram accumulate traffic tracked separately - HOLDS

```python
osram_read_bytes  = output_accum_elements * output_bytes     :449
osram_write_bytes = output_write_elements * output_bytes     :450
output_sram_bytes = osram_read_bytes + osram_write_bytes     :451
output_read_count  = output_accum_elements / osram_active_elements    :455
output_write_count = output_write_elements / osram_active_elements    :456
```

Used at every site the audit named: `sram_output_read_mapping` count = `output_read_count`
(:624), i.e. osram accumulate reads `M*N*(k_folds−1)`, **not** DRAM spill reads;
`sram_output_write_mapping` likewise (:632); steady-state pressure uses `osram_read_bytes`
(:476-478); the tail stall uses `osram_write_bytes` (:480); `output_sram_bandwidth` uses
`output_sram_bytes` (:530).

Confirmed order-invariant: `output_sram_bytes: 1,040,187,392` identical for `auto`, `k_outer`,
`n_outer`, and identical at 1 MiB osram - exactly `M*N*(15 reads + 16 writes)*2 B`.

**Caveat:** `osram_read/write_bytes` are gate-free, so a partial sum that spills to DRAM is
charged **both** as a full osram access and as DRAM spill traffic. Visible at 1 MiB osram:
`output_sram_bytes: 1,040,187,392` while `output_read_bytes: 495,452,160` /
`output_write_bytes: 529,006,592`. Defensible as "osram write then evict", but osram energy is
never reduced by a too-small osram.

### B.5 Item 6 - chunked-window bandwidth - HOLDS

Repo-wide grep for `_window_from_chunks` and `_chunk_count`: **zero hits**. Both helpers gone.
Every bandwidth window is the compute span (:484, :491-494), with the single documented
exception of `_read_active_span_cycles` for weight (:485-490).

Measured, that exception errs in the *opposite* direction from the original bug:
`weight_transfer_window_cycles = 816,620` vs `compute_cycles = 785,920`, so
`weight_dram_bandwidth = 153.07` **understates** the compute-span average (159.05, which
`weight_sram_bandwidth` reports on identical bytes). Near-average as claimed; inflation is not
the direction of error. `weight_fill_window` / `steady_window` / `output_tail_window`
(:465-471) are phase spans feeding stall terms only (:472-480), not bandwidth windows.

The audit's "remaining" note stands unchanged - see §C.3.

---

## Part C - The open items, as they stand today

### C.1 Item 5a - each operand gets the full DRAM channel

The architecture declares exactly one DRAM (`arch['dram']`, one `bandwidth` field, :19, :40),
yet :339/:340/:341 hand that same `bytes_per_cycle` to each of three operands. In the reference
config the model provisions **3× the declared channel (824.6 B/cy aggregate)** at the point
where stalls are computed.

**Under-count, measured, reference config (n_outer, `'auto'`):**

```
dram_bpc 274.877906944
total dram bytes 625,999,872   (458,227,712 in + 134,217,728 wt + 33,554,432 out-w + 0 out-r)
shared-channel service cycles 2,277,375
compute 785,920      true shared-channel stall 1,491,455
model read_stall 881,359   write_stall 0   sum 881,359
model total runtime 1,667,279 cy    shared-channel runtime 2,277,375 cy
```

The model's `max(881,103, 0)` discards the weight stream's 488,282 service cycles and the
output stream's 122,071 entirely, because each individually fits under `compute_cycles`. On one
shared channel they do not: true deficit **1,491,455** vs modelled **881,359** - a **41 %
under-count of DRAM stall**, and total GEMM runtime **27 % optimistic** (1.667 ms vs 2.277 ms
at 1 GHz).

`k_outer` is worse, because `output_read_bytes` carries no stall term at all:

```
k_outer total dram bytes 1,050,673,152   shared service 3,822,327 cy
        model stall 881,359   model runtime 1,667,279 cy
        (dram_output_read 424,673,280 bytes -> no stall term)
```

**3,822,327 vs 1,667,279 cycles - 2.29× optimistic.**

General shape: with per-operand demands `s_i` and compute span `T`, the model charges
`max_i max(0, s_i − T)` (:461-463, :481); one shared channel charges `max(0, Σs_i − T)`. The gap
is largest exactly in the regime of interest - several streams each near but under the compute
span, where the model reports **zero** stall and the hardware is saturated.

### C.2 Item 5b - stalls combined coarsely (`max` on DRAM, `+` on everything else)

`weight_fill_stall`, `steady_state_stall`, `output_tail_stall` and `write_stall` are all
measured against windows that are subsets of, or equal to, the same `compute_cycles` span
(:461-463, :465-471, :480) and are then summed as disjoint wall-clock (:481, :527). Two
specific double-budgets:

1. **`steady_window` is charged twice** - once as the budget for `steady_state_stall_cycles`
   (:478) and again as part of the budget for `output_tail_stall_cycles` (:480). The two
   shortfalls are then added (:481): the same osram is credited `steady_window` cycles of
   service twice.
2. Any DRAM stall overlapping any SRAM phase stall in real time is added rather than maxed.

**Measured demonstration** (reference config, osram narrowed to `bank=64, depth=81920` so
`osram_bpc = 64` B/cy at unchanged element capacity):

```
osram_bpc 64.0   compute_cycles 785,920
weight_fill_stall_cycles 256
steady_state_stall_cycles 7,340,032
output_tail_stall_cycles 7,733,504
input_read_stall_cycles 881,103   weight_read_stall_cycles 0
read_stall_cycles 15,954,895   write_stall_cycles 0   stall_cycles 15,954,895
osram total bytes 1,040,187,392  ->  osram service cycles 16,252,928
output_sram_bandwidth 1232.6318200325734
```

osram is now the sole bottleneck at **16,252,928** unavoidable cycles, so ≈16.25 M is the
correct answer. The model reports `785,920 + 15,954,895 = 16,740,815`: it adds the
881,103-cycle DRAM input stall on top of a window in which a 20×-slower osram fully hides it
(**5.4 % of wall-clock counted twice**), and splits the osram deficit into 7,340,032 + 7,733,504
across two stalls whose budgets overlap by the full 524,288-cycle `steady_window`. Where DRAM
and SRAM stalls are comparable, the `+` at :481 approaches a clean 2× over-count.

Note also that `output_sram_bandwidth` stays at **1232.6 GiB/s** in that run (:530, denominator
hardwired to `compute_cycles` at :494) although the osram can physically deliver
`64 B/cy × 1 GHz / 2**30 = 59.6 GiB/s`. **The reported bandwidths are demand-over-compute-span,
never reconciled against modelled supply, and never recomputed once a stall extends the real
span.**

### C.3 Item 6-remaining - `peak_required_bandwidth` is a max of per-event averages

```python
peak = max((sample['bandwidth'] for sample in samples if sample['data_moved'] > 0), default=0)
                                                                    # fig_2_query.py:41
bandwidth = float(subevent.get('factor', {}).get('bandwidth', 0))   # fig_2_query.py:73
```

which for the DRAM path is `schedule['input_dram_bandwidth']` = `input_read_bytes /
compute_cycles × f / 2**30` (:531, :484, :47-51).

**What it computes:** the maximum, over the ~19 GEMM-level `*_dram` events under the workload,
of each GEMM's *whole-GEMM average* DRAM read bandwidth. For stepped decode GEMMs (`qkt_dc`,
`av_dc`) the per-event value is itself already averaged over the entire prefill→max_seq_len
trajectory (:376-385). It is a max over averages of averages.

**What it does mean:** "of the GEMMs in this model, the one with the highest sustained DRAM read
demand needs this many GiB/s averaged over its own duration." A legitimate bound on the
*sustained* requirement - a kernel whose average demand exceeds the channel can never be served.

**What it does not mean:** not a burst, not an instantaneous peak, not a percentile, not a
windowed maximum. **No sub-kernel time resolution exists anywhere in the model** - every window
in `_ws_schedule` is `compute_cycles` or `compute_cycles + startup_span` (:484-494), so there is
no shorter interval to take a max over. It is also **unweighted by bytes or time**. Measured
per-event values for `runs/llama_3_8b/arch_73/config_4`:

```
input  proj_k_pf_dram     bytes=1.7130e+10  bw=922.1204
input  proj_v_pf_dram     bytes=1.7130e+10  bw=922.1204
input  a_proj_pf_dram     bytes=6.8367e+10  bw=920.0883
...
input  down_proj_dc_dram  bytes=2.7682e+13  bw=288.4647
...
input  av_dc_dram         bytes=3.5186e+13  bw=3.5028
input  qkt_dc_dram        bytes=6.2449e+12  bw=0.6217
```

The reported `input_peak_required_bandwidth` (~924 GiB/s) is set by the two **smallest** prefill
events, while >90 % of the bytes move at 0.6-290 GiB/s. Nothing in the query layer compares it
against the service rate (`_dram_bytes_per_cycle` = 274.88 B/cy = 256 GiB/s here), so the
"peak" is 3.6× the channel and is silently a *demand*, never a roofline ratio.

---

## Part D - Bandwidth-relevant behaviour not covered by the audit

### D.1 DRAM output reads are traffic without a stall
`output_read_bytes` (:444) is counted for energy, emitted as `dram_output_read` (:674), and
reported - but the only output stall term (:460, :463) uses `output_write_bytes`. Under
`k_outer` the reference config moves **424,673,280 bytes for free**. This is the one-sided half
of the item-2 fix.

### D.2 `dram_output_read` carries the *write* bandwidth
`output_dram_bandwidth`'s numerator is `output_write_bytes` alone (:533), yet it is attached to
`dram_output_read` (:676). Anyone reading `factor.bandwidth` off that event gets 39.76 GiB/s
regardless of the actual 424.7 MB read.

### D.3 `output_sram_bandwidth` is a read+write composite on both directions
`:451` sums the directions, `:530` divides by compute, and `sram()` attaches the result to both
`sram_output_read_mapping` (:628) and `sram_output_write_mapping` (:636). A consumer that adds
the two events double-reports osram bandwidth.

### D.4 Weight uses a different window than every other operand
See §A.9/§B.5: identical bytes yield 153.070 (DRAM) vs 159.049 (SRAM) GiB/s. `min(compute, …)`
at :312 caps the demand part; nothing caps `demand_span + startup_span`.

### D.5 `weight_fill_stall_cycles` is a structural artifact
Service is `R*C/C = R` cycles per fold against an `R−1` window (:465, :472), so the term is
≈1 × `fold_count` for *any* balanced wsram - measured exactly 256 for 256 folds. It is
nonetheless summed into `read_stall_cycles` (:481) and emitted as `sram_weight_fill_stall`
(:639-642). (The audit lists this under hygiene; recording it here because it is a permanent
non-zero floor on reported stall.)
[`sram_weight_fill_stall` is now named `sram_fill_stall`.]

### D.6 `_active_fraction` default 0.5 silently halves every buffer
`:55`, `:64`. No YAML under `zoo/` or `src/` sets `active_fraction` or `active_buffer_fraction`
(verified by grep). Every `_fit_fraction` test (:420-421) and every `*_count` divisor
(:453-456) therefore uses **half** the declared SRAM. Meanwhile `fig_2.py`'s x-axis is
`bits_to_mib(asram_size)` = the *full* `bank*depth*width` (fig_2_query.py:167, fig_2.py:32, 69),
so **the plotted SRAM size is 2× the capacity the model credits with reuse**. Separately,
`_sram_bytes_per_cycle` hardcodes `bank/2` (:71) rather than deriving it from
`_active_fraction`, so the two disagree the moment anyone sets a non-0.5 fraction.

### D.7 The SRAM samples in `fig_2_query` are dimensionally wrong
```python
'input':  {'event': 'sram_input_write_mapping',  'count_to_bytes': isram_width / 8},   # :155-159
```
But `subevent['count']` for `sram_*_write_mapping` is a **buffer-fill count** (:453-454,
`input_read_elements / isram_active_elements`), not a word count. Fills × bytes-per-word is
neither bytes nor words; the correct factor would be `isram_active_elements * width / 8`, i.e.
**786,432× larger** for these configs. Consequences:
- `input_sram_window_cycles` / `weight_sram_window_cycles` in the CSV (fig_2_query.py:174-175)
  are off by ~6 orders of magnitude and are physically meaningless - measured
  `input_sram_window_cycles = 1.86e8` against `input_transfer_window_cycles = 5.02e13`.
- `input_sram_bandwidth` / `weight_sram_bandwidth` survive only because `active_avg` is
  scale-invariant (§A.12), but the per-event **weights** are still wrong: they weight by
  DRAM-read-derived fill counts, not by `input_sram_bytes`.

Related: the `sram_input_write_mapping` **count** equals `input_read_elements` exactly
(node.py:50, :54 give one mapping = `depth × bank/2` = `isram_active_elements`), i.e. DRAM
reads - while the **bandwidth factor attached to that same event** is computed from
`input_sram_bytes` (:446, :612). Count and reported bandwidth describe different quantities.

### D.8 The weight panel of fig_2 has essentially no signal
`weight_read_elements = weight_elements` (:440) - weight DRAM traffic is independent of wsram
size. The only wsram dependence is `startup_buffer_elements` inside `_read_active_span_cycles`
(:313), which *enlarges* the denominator as wsram grows. From the regenerated CSV:

```
llama_3_70b  wsram  8,388,608 -> 125.011666
             wsram 83,886,080 -> 124.854858     (0.13 % over a 10x sweep)
llama_3_8b   wsram  8,388,608 -> 115.658858
             wsram 83,886,080 -> 115.505853
```

`fig_2.py:121` sets `ylim(0, max)`, so it renders dead flat - but any reader who zooms is
reading a heuristic startup-span term, not a reuse effect. The input panel by contrast carries
real signal (70B: 22.10 → 14.01; 8B: 5.98 → 2.74), driven by `_fit_fraction` at :420.

### D.9 Operands averaged over different windows, then harmonic-meaned across incomparable spans
`input_dram_bandwidth` uses `compute_cycles`, `weight_dram_bandwidth` uses
`compute_cycles + startup_span` (:484 vs :485-490). Within each operand,
`summarize_bandwidth` (fig_2_query.py:37-47) then harmonic-means across events whose windows are
prefill spans, single-decode-step spans, and full-decode-trajectory spans (:376-385) - a
well-defined number, but not a bandwidth over any single time interval.

### D.10 `array_execution_time` and the bandwidth samples come from different graph views
`fig_2_query.py:131-136` queries `array_execution_time` on `event='llama_array'`, which fans out
to `*_arr` nodes only (llama_model.py:47-52, 60-70) and is therefore **stall-free by
construction** (§A.13: its breakdown is 100 % compute, 0 stalls, always). The bandwidth samples
in the same CSV row come from `*_dram` events under the `llama` branch. **Each row mixes two
different views of the same computation.**

### D.11 `memory_stall_cycle_count` is a misnomer [WRONG AS WRITTEN - see banner]
It is the `_dram` bucket only (query/utils.py:72). SRAM stalls (`sram_weight_fill_stall`,
`sram_steady_state_stall`, `sram_output_tail_stall`, :639-650) land in `sram_cycle_count`, so a
caller reading it as "all memory stalls" understates them (2.16e8 vs 9.70e10 in the measured
run - small here, but the name is wrong).
[The three are now `sram_fill_stall`, `sram_steady_stall`, `sram_tail_stall`.]
[The metric itself is now `dram_stall_cycle_count`; the misnomer this heading names is no longer live. See banner item 2.]

### D.12 Latent subtree under-count in `_component_event_counts` (query/utils.py:83-107) [WRONG AS WRITTEN - see banner]
The dedup key is `(name, path_count)` (:90-92) and the `continue` at :91 skips expanding
children, while `component_counts[child] += child_count` (:104) already ran for the first
arrival. A node reached by two distinct paths with the **same** multiplicity is counted twice
(correct) but its entire subtree once (wrong). Not triggered today
(`unattributed_cycle_count == 0.0` on both roots), but it bites the moment the graph gains a
diamond with equal edge counts.
[WRONG AS WRITTEN. The bug was live, not latent: old versus new differ on 40/40 sampled root queries, up to 2.37x. The published numbers were safe only because the three buckets filter on `_arr`/`_sram`/`_dram` suffixes above the diamond. Also, the residual is `<= 2e-16` relative, ULP-nonzero on 11 of 40 sampled runs, not flatly `== 0.0`. See banner items 1 and 3.]

### D.13 `lm_head_dc` multiplicity also biases the bandwidth totals
The audit lists the count mismatch (`decode()` charges 1, llama_model.py:86, vs one per decode
step, llama_model.py:51) as a cycle bug. It additionally biases `input_data_moved` /
`weight_data_moved` and the harmonic-mean weights, since `collect_mapping_samples` weights by
`aggregate_event_count` under the `llama` root (fig_2_query.py:56-60). Measured magnitude:
`lm_head_dc_dram bytes=6.60e+08 bw=199.42` against `down_proj_dc_dram bytes=2.77e+13`  -
negligible here, but it scales with vocab size.

### D.14 Silent skips, silent defaults, and undocumented filtering in the query layer
- `fig_2_query.py:53` - `if not event.endswith(suffix) or not hasattr(llama_model, event):
  continue`. Any `*_dram` node without a same-named Python function is dropped **with no
  warning**. This is what keeps `dram_*_stall` events out of the samples (they live in
  `mapping`, not `llama_model`) - correct, but by accident of module membership.
- `fig_2_query.py:69-70` - a GEMM with no `dram_input_read` subevent silently contributes nothing.
- `fig_2_query.py:73` - a missing bandwidth factor silently becomes `0`, which makes
  `transfer_window_cycles` return `0` (:27-28) while `data_moved` still accumulates, **inflating
  `active_avg`**. Not triggered today; a fail-loud condition handled quietly.
- `fig_2_query.py:121-129` - `if array_dim != [512, 512]: continue` and `if batch_size != 512:
  continue`. **The entire bandwidth figure covers one array size and one batch size out of 1,240
  configurations** (200 rows survive). Documented nowhere in the script or the figure.
- `fig_2.py:63-68, 71-76` - an unweighted arithmetic mean of already-harmonic-mean bandwidths,
  taken across the 10 `wsram_size` variants at each `asram_size` (and symmetrically for weight).
  Harmless for the input panel; for the weight panel it averages over the strongly varying
  `asram_size` axis, which is why that curve looks so clean.

### D.15 Unit and counting asymmetries
- **GB vs GiB (:42).** `bandwidth_gbs * 2**30 / frequency_hz` - the YAML field is conventionally
  decimal GB/s but is scaled by `2**30`. Verified live on `runs/llama_3_8b/arch_73/config_4`:
  `{'bandwidth': 256, ...}` → `_dram_bytes_per_cycle 274.877906944` instead of `256.0`, a
  **7.37 % systematic over-provision of DRAM** propagating into every stall (:458-463). It is
  self-consistent with `_bandwidth_gib_per_second` (:51) and with `memory.py:8,20,32,44`
  (`1000/(bandwidth*2**30)`), so the whole stack is internally coherent in GiB - the error is
  only in what "256" in the YAML buys you. (Audit hygiene item confirmed still open.)
- **The 1000 MHz default is duplicated, not shared.** `mapping._frequency_mhz` (:33-37) and
  `fig_2_query.py:137` each carry their own literal. They agree today. Frequency cancels out of
  `active_avg_bandwidth` entirely (§A.12); it affects only the window columns.
- **DRAM counts are bytes, SRAM counts are words.** `:660-663` vs node.py. Both feed
  `dynamic_energy` through the same CACTI leaf mechanism, so DRAM energy is charged per byte and
  SRAM per `width`-bit word. If the CACTI DRAM number is a per-access (row/burst) energy rather
  than per-byte, **DRAM energy is inflated by the burst width**. Runtime is safely zeroed on all
  traffic events (`cycle_count: 0, runtime: 0`), so this is an energy-only concern.
- **Inconsistent `aggregation` on DRAM traffic events.** Only `dram_output_write` carries
  `'aggregation': 'sequential'` (:680); `dram_input_read` (:666), `dram_weight_read` (:670),
  `dram_output_read` (:674) default to parallel. No timing impact today (zero cycle factors),
  but unexplained and will bite if a nonzero factor is added.
- **`_sram_elements` is width-independent** (:58-60): `width*bank*depth // width == bank*depth`.
  Capacity in *elements* is unchanged by datatype width while byte counts (:327-329) scale with
  it, so halving `width` halves modelled traffic but leaves modelled capacity alone.

### D.16 Modelling shortcuts worth knowing when reading any bandwidth number
- **`_fit_fraction` is linear, not a reuse model** (:291-294, :298). A working set 2× capacity
  refetches 50 % of the excess. There is no notion of *tiling* the working set to make it fit  -
  which is exactly what a real mapper would do, and what would make the `k_outer`/`n_outer`
  dichotomy unnecessary.
- **`_sram_bytes_per_cycle` conflates the double-buffer split with port width** (:68-72).
  `bank/2` is the active half of a double buffer, used here as a bandwidth term. For the
  reference config it coincidentally gives `512 words/cy = array_cols`, exactly the array's
  steady-state demand - so `steady_state_stall_cycles` is structurally 0 for any config where
  `bank/2 ≈ array_cols`, and only fires when `bank` is deliberately mis-sized.
- **Ceil placement** (:74-77). One `ceil` on a whole-GEMM aggregate; per-transfer/per-burst
  rounding is never modelled. Conversely `max(1.0, bank/2)` is float (:71), so an odd `bank`
  yields a fractional rate.
- **`compute_utilization` divides a product of sums by `total_steps`** (:497). Correct for the
  three stepped branches as written, wrong for any future branch stepping two dimensions.

### D.17 Dead code that reads as live
None of the following are reachable from any live path (verified: zero references outside their
own definitions, repo-wide):
`_ws_folds` :22-24 and `_ws_fold_infos` :26-31 (the canonical fold enumeration - only the
*comment* at :405 still refers to it); `_ws_phase_cycles` :79-83; `_phase_stall` :85-88;
`_touch_resident` :127-137 and `_touch_chunked_resident` :139-174; `_launch_output_drain`
:176-190, `_service_output_write` :192-237, `_drain_remaining_output` :239-269;
`_cycle_reference` :98-99 (the live one is `cycle_reference` :101-102);
`_fit_2d_tile` (common/performance/utils.py:26-42).
Unused locals in `_ws_schedule`: `isram/wsram/osram_prefetch_elements` :331-333,
`max_weight_tile` :413, `input_tile`/`output_tile` :437.

**This matters for an audit:** `_service_output_write` / `_drain_remaining_output` implement a
credible *occupancy-based* write-stall model (back-pressure against a draining osram, :206-237)
that is **not** what the model uses. Reading them gives a false impression of the fidelity of
`write_stall_cycles` (:463).

Query-layer dead/stale code beyond the audit's `gemm_test.py` note:
- `best.py:4-5` reads `array_performance_metrics_8b_scientific.csv` and
  `array_performance_metrics_70b.csv`; **neither exists** in `results/csv/`. It cannot run.
- `fig_7.py` is a 6-line truncated stub reading the same two non-existent CSVs; the file ends
  mid-statement with no plotting code.
- `fig_2_query.py:18-24` defines `memory_sizes()` and never calls it (the row builder inlines
  the arithmetic at :167). Copied from `fig_1_query.py:16-22`, where it *is* used.
- `fig_2_query.py:89` - `tag='onchip'` assigned at module scope, never read.
- `fig_1_query.py:79` - `array_query_df.to_csv(...)` sits **inside** the per-config loop,
  rewriting the whole CSV on all 1,240 iterations.
- `query/utils.py:41-47`, `query_bandwidth` - a **different, simpler** bandwidth definition
  (`data_moved / execution_time / 2**30`, dividing by real aggregated runtime) with **no
  callers**. Anyone reaching for it expecting agreement with `fig_2_query` will not get it.
- `query/utils.py` annotation drift: `query_cycle_breakdown`/`query_cycle_count`/
  `query_execution_time` are annotated `-> OrderedDict` (:49, :109, :113, :120) while returning
  floats. **The audit's `np.float64`-without-numpy note (scalesim_audit.md:36) is stale** - that
  annotation is no longer present in today's file.

---

## Part E - Incidental side effect during this analysis (no code changed)

One analysis subagent imported `zoo/chiplet4ai/results/query/fig_2_query.py`, which is a
**script**, not a module - the import re-ran the full query and **overwrote**
`results/csv/bandwidth_performance_metrics.csv` and
`results/csv/bandwidth_performance_metrics_scientific.csv` in the working tree.

The regenerated content is **not** identical to the staged copies. Row 1, `input` columns:

```
staged (index):     input_data_moved 2.80688821534720e14   input_dram_bandwidth 51.35537478717252
                    input_peak_required_bandwidth 81486.31029986963
regenerated (wt):   input_data_moved 1.28821469642752e15   input_dram_bandwidth 23.88711211179669
                    input_peak_required_bandwidth 924.5879150256443
```

The staged copies were produced by an older code state; the regenerated ones reflect today's
`mapping.py`. Nothing is lost - the staged content is recoverable with
`git show :zoo/chiplet4ai/results/csv/bandwidth_performance_metrics.csv`. No git-destructive
command was run, and **no source file was modified by this task**. Flagging for whoever owns
those CSVs to decide which version should stand.

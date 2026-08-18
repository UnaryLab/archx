# SCALE-Sim v3 bandwidth & stall model - reference spec

Purpose: the reference against which `zoo/chiplet4ai/common/performance/` is audited. Scope is
weight-stationary (WS) only. All paths are relative to `SCALE-Sim/scalesim/`; every claim carries
`file:line` from the working tree as read on 2026-08-17. Companion to `zoo/chiplet4ai/scalesim_audit.md`;
§5 answers that audit's open items 5 and 6 directly.

Headline results, stated up front because they are the two the audit turns on:

- **There is no shared DRAM channel.** ifmap, filter and ofmap each get an independent, uncontended
  words/cycle budget. No code anywhere sums their concurrent demand against one bus (§1.3, §4).
- **There is no peak or required bandwidth number in SCALE-Sim.** Every bandwidth the simulator
  reports is an average. `grep -rniE "peak|required.?bandwidth|req_bw|max_bw" --include=*.py` over the
  whole tree returns zero hits; the README's mention of "maximum bandwidths"
  (`SCALE-Sim/README.md:110`) is stale relative to the header actually written at
  `simulator.py:186-191` (§3).

---

## 1. Bandwidth modes and per-operand backing bandwidth

### 1.1 Mode selection

`InterfaceBandwidth` in the config selects the mode (`scale_config.py:81-85`):

```python
bw_mode_string = config.get(section, 'InterfaceBandwidth')
if bw_mode_string == 'USER':
    self.use_user_bandwidth = True
elif bw_mode_string == 'CALC':
    self.use_user_bandwidth = False
```

The list-form config repeats this at `scale_config.py:192-198`. The numeric `Bandwidth` key is parsed
**only** in USER mode (`scale_config.py:136-138`):

```python
if self.use_user_bandwidth:
    self.bandwidths = [int(x.strip())
                       for x in config.get(section, 'Bandwidth').strip().split(',')]
```

Accessors: `use_user_dram_bandwidth()` (`scale_config.py:317-329`), `get_bandwidths_as_list()`
(`scale_config.py:481` and again at `:488` - defined twice, the second shadows the first; both return
`self.bandwidths`), `get_min_dram_bandwidth()` (`scale_config.py:500-509`, raises in CALC mode).

Caveat: `section` was reassigned to `'architecture_presets'` at `scale_config.py:108`, so `Bandwidth`
is in practice read from the architecture section rather than `run_presets`.

### 1.2 Per-operand backing bandwidth

`single_layer_sim.py:246-263`, verified verbatim:

```python
ifmap_backing_bw = 1
filter_backing_bw = 1
ofmap_backing_bw = 1
estimate_bandwidth_mode = False
if self.config.use_user_dram_bandwidth():
    bws = self.config.get_bandwidths_as_list()
    ifmap_backing_bw = self.config.ifmap_sram_bank_bandwidth
    filter_backing_bw = self.config.filter_sram_bank_bandwidth
    ofmap_backing_bw = bws[0]
else:
    arr_row, arr_col = self.config.get_array_dims()
    estimate_bandwidth_mode = True
    # The number 10 elems per cycle is arbitrary
    ifmap_backing_bw = 10
    filter_backing_bw = 10
    ofmap_backing_bw = arr_col
```

| mode | ifmap | filter | ofmap |
|---|---|---|---|
| USER | `ifmap_sram_bank_bandwidth` (`scale_config.py:128`) | `filter_sram_bank_bandwidth` (`scale_config.py:131`) | `bandwidths[0]` |
| CALC | 10 (hardcoded) | 10 (hardcoded) | `arr_col` |

Two things matter here for the audit:

1. In **USER** mode the ifmap and filter backing bandwidths do **not** come from the `Bandwidth` config
   list at all - they come from the `[layout]` keys `IfmapSRAMBankBandwidth` / `FilterSRAMBankBandwidth`.
   Only ofmap uses `bws[0]`. So a config reading `Bandwidth = 10` does not mean "10 words/cycle of DRAM
   in total"; it sets the ofmap drain width alone.
2. In **CALC** mode the read-side numbers are arbitrary constants, by the source's own comment
   (`single_layer_sim.py:260`).

These values become `self.req_gen_bandwidth` inside the buffers: `read_buffer.py:89`,
`write_buffer.py:82`; in the estimate buffer they become `self.default_bandwidth` /
`self.prefetch_bandwidth` (`read_buffer_estimate_bw.py:85-86`).

### 1.3 Buffer instantiation per mode

`double_buffered_scratchpad_mem.py:105-165`:

- CALC: ifmap/filter become `ReadBufferEstimateBw` (`:105-107`), configured with
  `backing_buf_default_bw=` (`:113`, `:121`).
- USER: plain `read_buffer` (`:125-126`), configured with `backing_buf_bw=` (`:143`, `:154`) plus the
  layout params `num_bank`, `num_port`, `enable_layout_evaluation`.
- ofmap is always a `write_buffer` in both modes (`:161-165`); only its bandwidth number differs.

Prefetch matrices are installed into the read buffers **only in USER mode**
(`single_layer_sim.py:288-292`). The CALC estimate buffer never receives a fetch matrix; it discovers
addresses from the demand stream itself.

### 1.4 What CALC mode actually does

CALC is not "infinite bandwidth" and not a constrained simulation - it is **stall-free by construction
on the read side**, and back-computes the bandwidth that would have been required
(`read_buffer_estimate_bw.py:119-121`):

```python
outcycles = incoming_cycles_arr + self.hit_latency
# In estimate mode, operation is stall free.
# Therefore its always a hit
```

Each time a prefetch buffer's worth of unique addresses has been consumed, the elapsed window is
divided into the element count (`read_buffer_estimate_bw.py:180-183`):

```python
elems_to_prefetch = self.num_sets_prefetch_buffer * self.num_items_per_set
cycles_needed = self.last_prefetch_end_cycle - self.last_prefetch_start_cycle + 1
self.prefetch_bandwidth = math.ceil(elems_to_prefetch / cycles_needed)
```

The first (active-buffer) prefetch is the only one using the configured default, and it is scheduled
*backwards in time* so it lands before the first request (`read_buffer_estimate_bw.py:162-171`), which
is why DRAM start cycles go negative and inflate `overall_cycles` (§3.4). The resulting per-prefetch
`prefetch_bandwidth` stamps the DRAM trace, whose row width therefore **varies per prefetch**, forcing
column padding when traces are concatenated (`read_buffer_estimate_bw.py:278-292`, `:306-316`).

**Consequence for the audit:** in CALC mode the `Avg IFMAP/FILTER DRAM BW` report columns stop meaning
"achieved traffic under a fixed budget" and start meaning "bandwidth that would have been required for
stall-free operation". That is the closest thing SCALE-Sim has to a required-bandwidth figure, and it
is still delivered through the *average* columns - never as a separate peak column.

---

## 2. How stall cycles arise

### 2.1 Read stalls: prefetch buffer vs active buffer

Buffer partition (`read_buffer.py:84-86`):

```python
self.total_size_elems = math.floor(self.total_size_bytes / self.word_size)
self.active_buf_size = int(math.ceil(self.total_size_elems * self.active_buf_frac))
self.prefetch_buf_size = self.total_size_elems - self.active_buf_size
```

`active_buf_frac` is asserted into `[0.5, 1)` (`read_buffer.py:79`) and hardcoded to 0.5 by the caller
(`single_layer_sim.py:239`).

`prepare_hashed_buffer()` (`read_buffer.py:173-222`) chops the fetch matrix into sets ("lines"):

```python
elems_per_set = math.ceil(self.total_size_elems / 100)      # :178  - the whole SRAM is 100 sets by default
if self.enable_layout_evaluation:
    elems_per_set = self.req_gen_bandwidth                  # :180  - one set == one DRAM fetch line
max_num_active_buf_lines   = int(math.ceil(self.active_buf_size / elems_per_set))     # :205
max_num_prefetch_buf_lines = int(math.ceil(self.prefetch_buf_size / elems_per_set))   # :206
```

`active_buffer_hit(addr)` (`read_buffer.py:225-271`) linearly scans the active window's line ids,
handling wraparound when `start_id >= end_id`.

The core stall arithmetic, plain (non-layout) path, `read_buffer.py:346-370`:

```python
offset = self.hit_latency                       # :296
...
for addr in request_line:
    if addr == -1:
        continue
    while not self.active_buffer_hit(addr):
        self.new_prefetch()
        potential_stall_cycles = self.last_prefetch_cycle - (cycle + offset)
        offset += potential_stall_cycles        # Offset increments if there were potential stalls
        if potential_stall_cycles > 0:
            offset += potential_stall_cycles
out_cycles = cycle + offset                     # :369
```

Semantics:

- A **miss does not fetch the missing address on demand.** It calls `new_prefetch()`, which *rotates*
  the active/prefetch windows forward and issues the next sequential block of the fetch matrix. The
  `while` repeats until rotation happens to bring `addr` into the active window, so one miss can cost
  several rotations.
- The cost of a miss is `last_prefetch_cycle - (cycle + offset)`: how long the array waits for that
  prefetch to land.
- **Defect worth flagging:** `potential_stall_cycles` is added **twice** when positive (`:362`
  unconditionally, then `:364` again), and when negative - prefetch already complete, i.e. no stall -
  line `:362` still applies it and *subtracts* from `offset`. The layout-evaluation path
  (`read_buffer.py:315-318`) does not have this bug: it adds once, and only if positive.

The layout path additionally models SRAM bank conflicts (`read_buffer.py:327-334`):

```python
bank_id = column_addr // self.bw_per_bank
...
offset += math.ceil(max_line_request_among_all_banks / self.num_port) - 1
```

with `self.bw_per_bank = self.req_gen_bandwidth // self.num_bank` (`read_buffer.py:94`).

Prefetch scheduling - this is the **only place DRAM read bandwidth is enforced in USER mode**
(`read_buffer.py:475-479`, `:505-509`):

```python
num_lines = math.ceil(self.prefetch_buf_size / self.req_gen_bandwidth)
requested_data_size = num_lines * self.req_gen_bandwidth
self.num_access += requested_data_size
...
cycles_arr[i][0] = self.last_prefetch_cycle + i + 1      # one fetch line per cycle, back-to-back
```

`self.last_prefetch_cycle = np.amax(response_cycles_arr)` (`read_buffer.py:517`). Because prefetch issue
cycles chain off `last_prefetch_cycle` rather than off the current demand cycle, the prefetch pipeline
drifts ahead of or behind the array, and `potential_stall_cycles` is exactly the measured gap. Window
rotation is `read_buffer.py:466-469`.

### 2.2 Write stalls: write-buffer drain

Partition (`write_buffer.py:84-87`), mirroring the read buffer: `active_buf_size` plus `drain_buf_size`,
`free_space` initialized to `total_size_elems`.

`service_writes` (`write_buffer.py:189-235`), per element:

```python
current_cycle = cycle[0] + offset                                  # :205
self.store_to_trace_mat_cache(elem)                                # :212
if current_cycle < self.drain_end_cycle:
    if not self.free_space > 0:
        offset += max(self.drain_end_cycle - current_cycle, 0)     # :216
        current_cycle = self.drain_end_cycle
elif self.free_space < (self.total_size_elems - self.drain_buf_size):
    self.append_to_trace_mat(force=True)
    self.drain_end_cycle = self.empty_drain_buf(empty_start_cycle=current_cycle)   # :221
```

Policy:

- Occupancy tracked by `free_space`, decremented per stored element (`write_buffer.py:127`).
- A drain is **triggered** when occupancy exceeds the active-buffer size - `free_space < active_buf_size`
  (`:219`) - and only when no drain is in flight.
- A **write stall occurs only if** a drain is in flight *and* the buffer is completely full
  (`free_space <= 0`). The array then waits until `drain_end_cycle` (`:216`). Writes are stall-free as
  long as the drain keeps up; the drain buffer is pure slack.
- Issuing a drain itself costs nothing - the `current_cycle = self.drain_end_cycle` after issue is
  commented out (`write_buffer.py:222-223`).

Drain rate (`write_buffer.py:238-273`):

```python
lines_to_fill_dbuf = int(math.ceil(self.drain_buf_size / self.req_gen_bandwidth))
...
data_sz_to_drain = num_lines * requests_arr_np.shape[1]
self.num_access += data_sz_to_drain
cycles_arr = [x + empty_start_cycle for x in range(num_lines)]
```

**Elements per cycle = `req_gen_bandwidth`**: each trace row is exactly that wide
(`write_buffer.py:123`) and `cycles_arr` assigns one row per consecutive cycle. The `-1` padding
compensation is applied only to the **last** row (`write_buffer.py:253-255`), so mid-trace partial rows
over-count `num_access`.

Final drain: `empty_all_buffers` (`write_buffer.py:276-287`), called from
`double_buffered_scratchpad_mem.py:288` with `ofmap_serviced_cycles[-1]`.

### 2.3 CALC mode collapses the read-side stalls entirely

Since `ReadBufferEstimateBw.service_reads` always returns `incoming + hit_latency`
(`read_buffer_estimate_bw.py:119-121`), `ifmap_stalls` and `filter_stalls` at
`double_buffered_scratchpad_mem.py:263,270` are **identically zero** in CALC mode, and the per-line
combine of §4 reduces to `ofmap_stalls` alone. Only the write buffer can stall in CALC mode.

---

## 3. Reported bandwidth definitions

### 3.1 There is no peak / required bandwidth

Verified by grep over the whole tree: zero hits for `peak`, `required bandwidth`, `req_bw`, `max_bw` in
any `.py`. The report header written at `simulator.py:186-192` contains only `Avg` columns:

```python
if self.conf.sparsity_support is True:
    header = ('LayerID, Avg IFMAP SRAM BW, Avg FILTER SRAM BW, Avg FILTER Metadata SRAM BW,'
              ' Avg OFMAP SRAM BW, ')                                   # :187-188
else:
    header = 'LayerID, Avg IFMAP SRAM BW, Avg FILTER SRAM BW, Avg OFMAP SRAM BW, '   # :190
header += 'Avg IFMAP DRAM BW, Avg FILTER DRAM BW, Avg OFMAP DRAM BW,\n'  # :191
```

Rows come from `single_layer_sim.get_bandwidth_report_items()` (`single_layer_sim.py:408-425`), in
exactly header order; the file is named at `simulator.py:184`. Units are words/cycle
(`simulator.py:136-150` prints `' words/cycle'`).

### 3.2 SRAM bandwidths - `single_layer_sim.py:342-350`

```python
self.avg_ifmap_sram_bw  = self.ifmap_sram_reads  / self.total_cycles   # :345
self.avg_filter_sram_bw = self.filter_sram_reads / self.total_cycles   # :347
self.avg_ofmap_sram_bw  = self.ofmap_sram_writes / self.total_cycles   # :350
```

| number | numerator | denominator | window |
|---|---|---|---|
| Avg IFMAP SRAM BW | `ifmap_reads`, accumulated at `systolic_compute_ws.py:292` (also `:285`, `:289-290` sparse) | `total_cycles` = `max(ofmap_serviced_cycles)` (`double_buffered_scratchpad_mem.py:307`) | cycle 0 → last ofmap service cycle, **stalls included** |
| Avg FILTER SRAM BW | `filter_reads`, `systolic_compute_ws.py:373` | same | same |
| Avg OFMAP SRAM BW | `ofmap_writes`, `systolic_compute_ws.py:457` | same | same |
| Avg FILTER Metadata SRAM BW | `metadata_reads`, `single_layer_sim.py:182` | same | same (sparsity runs only) |

Important nuance: the SRAM **numerators come from the compute model's static request counts**, not from
the traces the memory system produced. They count real (non-`-1`) operands only. Only the denominator is
simulated. So `Avg * SRAM BW` = "useful operands per simulated cycle", and by construction
`BW × Total Cycles` reproduces the counts in `DETAILED_ACCESS_REPORT.csv` exactly
(`single_layer_sim.py:435-437`).

### 3.3 DRAM bandwidths - `single_layer_sim.py:374-379`

```python
self.avg_ifmap_dram_bw = self.ifmap_dram_reads / \
                        (self.ifmap_dram_stop_cycle - self.ifmap_dram_start_cycle + 1)
self.avg_filter_dram_bw = self.filter_dram_reads / \
                        (self.filter_dram_stop_cycle - self.filter_dram_start_cycle + 1)
self.avg_ofmap_dram_bw = self.ofmap_dram_writes / \
                        (self.ofmap_dram_stop_cycle - self.ofmap_dram_start_cycle + 1)
```

| number | numerator | denominator | window |
|---|---|---|---|
| Avg IFMAP DRAM BW | `ifmap_buf.get_num_accesses()` (`double_buffered_scratchpad_mem.py:595`; `read_buffer.py:559-564`, incremented as `num_lines * req_gen_bandwidth` at `:389-390`, `:479`) | `stop - start + 1` from the DRAM trace cycle column (`read_buffer.py:572-573`: `np.amin`/`np.amax` of `trace_matrix[:,0]`) | **the ifmap DRAM active window only** - not `total_cycles` |
| Avg FILTER DRAM BW | `filter_buf.get_num_accesses()` (`:609`) | same construction on the filter trace | filter DRAM window |
| Avg OFMAP DRAM BW | `ofmap_buf.get_num_accesses()` (`:623`; `write_buffer.py:256`) | `np.amin`/`np.amax` of `cycles_vec` (`write_buffer.py:324-325`) `+ 1` | ofmap drain window |

Two asymmetries to carry into the audit:

1. **The DRAM denominators are per-operand active windows, not the compute span.** Each operand's
   average is taken over its own busy interval. This is *not* an average over the layer.
2. **The DRAM numerators include padding.** Prefetch counts whole `req_gen_bandwidth`-wide lines
   including `-1` slots (`read_buffer.py:399-404`, `:498-502`), so in USER mode `Avg * DRAM BW` comes out
   ≈ the configured backing bandwidth almost by construction. `ReadBufferEstimateBw` is the exception -
   it counts real addresses only (`read_buffer_estimate_bw.py:276`).

The `avg_ifmap_dram_bw` / `avg_filter_dram_bw` / `avg_ofmap_dram_bw` fields declared in
`double_buffered_scratchpad_mem.py:52-54` are **dead** - never written after `__init__`. The live copies
are the `single_layer_sim` ones above.

### 3.4 The two cycle counts

`COMPUTE_REPORT.csv` header (`simulator.py:175-176`) is
`LayerID, Total Cycles (incl. prefetch), Total Cycles, Stall Cycles, Overall Util %, Mapping Efficiency %, Compute Util %`,
fed by `single_layer_sim.py:399-404`:

- **Total Cycles** = `max(ofmap_serviced_cycles)` (`double_buffered_scratchpad_mem.py:307`).
- **Total Cycles (incl. prefetch)** = `overall_cycles`, the DRAM-inclusive span
  (`single_layer_sim.py:371`):
  `ofmap_dram_stop_cycle - min(ifmap_dram_start_cycle, filter_dram_start_cycle)`. In CALC mode the read
  start cycles are negative (§1.4), which is what makes this exceed Total Cycles.
- **Stall Cycles** is reported but **never subtracted** from Total Cycles anywhere; it is an independent
  statistic (`single_layer_sim.py:335-336`, where the subtraction is present but commented out).
- `overall_util = (num_compute * 100) / (total_cycles * num_mac_unit)` (`single_layer_sim.py:337`) with
  `num_compute = num_ofmap_px * window_size` (`:203-204`).

---

## 4. How per-operand stalls combine into total cycles

**Per-line max across operands, accumulated as a running sum over lines.** Verified verbatim at
`double_buffered_scratchpad_mem.py:279-280`:

```python
self.stall_cycles += int(max(ifmap_stalls[0], filter_stalls[0], ofmap_stalls[0]))
#self.stall_cycles += ifmap_stalls[0] + filter_stalls[0] + ofmap_stalls[0]
```

The summing alternative on line 280 is present but disabled.

The loop context (`double_buffered_scratchpad_mem.py:254-277`):

```python
for i in tqdm(range(ofmap_lines), disable=pbar_disable):
    cycle_arr = np.zeros((1,1)) + i + self.stall_cycles           # :256
    ...
    ifmap_stalls  = ifmap_cycle_out[0]  - cycle_arr[0] - ifmap_hit_latency   # :263
    filter_stalls = filter_cycle_out[0] - cycle_arr[0] - filter_hit_latency  # :270
    ofmap_stalls  = ofmap_cycle_out[0]  - cycle_arr[0]                       # :277
```

Reading of the model:

- The issue cycle of demand line `i` is `i + (all stalls accumulated so far)`. One demand line = one
  cycle, plus the running offset.
- Per-line stall = `serviced_cycle - request_cycle - hit_latency`, i.e. everything above ideal latency.
  ifmap/filter subtract a hit latency so a clean hit yields exactly 0; **ofmap does not subtract one**,
  because `write_buffer.service_writes` starts `offset = 0` (`write_buffer.py:197`).
- The three operands are combined with `max` - this is a **"the whole array stalls together"** model.
  Whichever operand is slowest on line `i` sets that line's stall, which is then pushed into the issue
  cycle of every subsequent line.
- All three demand matrices have identical row counts (§6.3), which is what makes the single index `i`
  valid across all three and `ofmap_lines` a safe loop bound.
- The `offset` inside each buffer is re-initialized to `hit_latency` on every `service_reads` call
  (`read_buffer.py:296`); persistent cross-call state is `last_prefetch_cycle` and the window limits.
  Since each call here processes exactly one line, the returned value is a genuine per-line stall.

Total cycles is then taken from the ofmap service trace, **not** from `lines + stall_cycles`
(`double_buffered_scratchpad_mem.py:307`, with the previous `[-1][0]` version commented out at `:305`
and a comment flagging it as a suspected fault at `:306`):

```python
self.total_cycles = int(max(ofmap_serviced_cycles))
```

---

## 5. Answers to the open items in `scalesim_audit.md`

### Open item 5 - DRAM channel sharing across operands

**SCALE-Sim gives each operand a completely independent, uncontended bandwidth budget. It does not
model a shared DRAM channel at all.** Evidence:

1. Three separate buffer objects with three separate `req_gen_bandwidth` values and three separate port
   objects (`double_buffered_scratchpad_mem.py:31-37`, `:139-165`). Nothing is passed between them.
2. `service_memory_requests` calls the three buffers sequentially on the *same* `cycle_arr`
   (`double_buffered_scratchpad_mem.py:256-275`); they are never told about each other's cycle
   consumption. They interact solely through the `max()` at `:279`, which couples **stalls, not
   bandwidth**.
3. The default (non-Ramulator) port is a pure constant-latency shifter with no occupancy model at all
   (`read_port.py:81-83`):
   ```python
   if self.ramulator_trace is False:
       out_cycles_arr = incoming_cycles_arr + self.latency
       return out_cycles_arr
   ```
   Same for `write_port.service_writes` (`write_port.py:69-72`), with `latency = 0`
   (`write_port.py:19,44`).
4. `read_port.bw` / `write_port.bw` are set from `get_bandwidths_as_list()[0]` (`read_port.py:39`,
   `write_port.py:41`) but are **never read anywhere** - dead fields. They are only assigned inside
   `def_params`, which `double_buffered_scratchpad_mem.py:135-137` calls **only when
   `use_ramulator_trace` is True**.
5. Even the Ramulator-trace path keeps operands independent: separate per-operand `.npy` latency files
   (`double_buffered_scratchpad_mem.py:132-134`), and each port's queue-pressure stall
   (`read_port.py:87-108`) is reset to 0 at the end of every call (`read_port.py:109`,
   `write_port.py:100`), so it does not even persist across prefetches.

Bearing on the audit: chiplet4ai's "each operand services at `min(dram_bpc, sram_bpc)`"
(mapping.py:349-351) **does faithfully match SCALE-Sim**. If chiplet4ai wants concurrent streams to
share one declared DRAM channel, that is a deliberate *improvement over* SCALE-Sim, not a bug fix
toward it, and the spec cannot be cited as authority for either choice. What chiplet4ai does **not**
match is the combine rule: SCALE-Sim takes a per-line `max` across operands within a single timeline
(§4), whereas chiplet4ai takes `max` over whole-GEMM aggregates and then sums phase stalls - which can
double-count the same wall-clock deficit. The per-line-max structure is the part worth importing.

### Open item 6 - peak vs average bandwidth definition

**SCALE-Sim offers no peak/burst definition to borrow.** Every reported bandwidth is an average (§3.1),
and the two families of average are not even taken over the same window:

- SRAM averages: static compute-model operand counts ÷ `total_cycles` (the full simulated span,
  including stalls) - §3.2.
- DRAM averages: padded access counts ÷ **that operand's own DRAM active window** - §3.3.

The only bandwidth-as-a-requirement notion anywhere is CALC mode's back-computed `prefetch_bandwidth`
(`read_buffer_estimate_bw.py:180-183`), which is `elems_to_prefetch ÷ (window between consecutive
prefetch deadlines)` - still an average, just over a prefetch interval instead of a layer, and surfaced
only through the ordinary `Avg * DRAM BW` columns. If chiplet4ai wants a burst/peak requirement
distinct from the average, it must define one itself; the nearest SCALE-Sim analogue to copy is the
CALC per-prefetch-window average, which is a defensible definition precisely because its window is the
interval a buffer must be refilled within, not the whole layer.

---

## 6. Supporting detail: the WS demand matrices

Included because §4's combine rule and §3's denominators only make sense against the matrix shapes.

### 6.1 Shape parameters - `systolic_compute_ws.py:102-110`

```python
self.Sr = self.ifmap_op_mat.shape[1]      # :102  window size / GEMM K
self.Sc = self.filter_op_mat.shape[1]     # :103  num filters / GEMM N
self.T  = self.ifmap_op_mat.shape[0]      # :104  num ofmap px / GEMM M
self.arr_row, self.arr_col = self.config.get_array_dims()                             # :106
self.row_fold = math.ceil(self.Sr / self.arr_row)                                     # :108 (prefetch only)
self.row_fold_demand_matrices = math.ceil(self.filter_op_mat.shape[0] / self.arr_row) # :109
self.col_fold = math.ceil(self.Sc / self.arr_col)                                     # :110
```

`row_fold` (`:108`) drives only the prefetch matrices; the demand matrices iterate
`row_fold_demand_matrices` (`:263`, `:361`, `:451`). They differ only under sparsity-optimized mapping.

### 6.2 One line = one cycle

A row of a demand matrix is exactly one cycle of array operation; the columns are the spatial ports of
that array edge; `-1` is a bubble. Enforced at `systolic_compute_ws.py:229-231`:

```python
assert self.ifmap_demand_matrix.shape[1] == self.arr_row, 'IFMAP demands exceed the rows'
assert self.filter_demand_matrix.shape[1] == self.arr_col,'Filter demands exceed the cols'
assert self.ofmap_demand_matrix.shape[1] == self.arr_col, 'OFMAP demands exceed the cols'
```

Operands per line: ifmap `arr_row` (west edge), filter `arr_col` (north edge, weight load), ofmap
`arr_col` (south edge, psum drain). The consumer confirms one-line-per-cycle
(`double_buffered_scratchpad_mem.py:254-256`).

### 6.3 Per-fold block heights are equal across operands

Fold loops are identical in all three generators - column-fold outer, row-fold inner (`:261/:263`,
`:359/:361`, `:449/:451`), blocks concatenated at `:329`, `:418`, `:481`.

- **ifmap** (`:236-334`): prefix `arr_row` bubbles (`:242-243`), suffix `arr_col - 1` (`:245-247`), then
  `skew_matrix` (`:325`), which maps `R x C` → `(R + C - 1) x C` (`:614-633`). Height
  `T + 2*arr_row + arr_col - 2`, width `arr_row`.
- **filter** (`:339-425`): slice flipped vertically (`:386`) so the top weight enters last, suffix
  `arr_row + arr_col + T - 2` (`:345`), **no skew** (comment `:425`). Height
  `T + 2*arr_row + arr_col - 2`, width `arr_col`.
- **ofmap** (`:428-488`): prefix `2*arr_row - 1` (`:434-435`), then skew (`:472`). Height
  `T + 2*arr_row + arr_col - 2`, width `arr_col`.

All three therefore have
`col_fold * row_fold_demand_matrices * (T + 2*arr_row + arr_col - 2)` rows. This equality is what makes
the shared index `i` and the per-line `max` in §4 well-defined.

Note this per-fold height `T + 2*arr_row + arr_col - 2` is the padded demand-matrix height, larger than
the `M + R + C - 2` per-fold figure that `scalesim_audit.md` item 2 matches against - the extra
`arr_row` is the weight-load prefix, which in WS overlaps the previous fold's drain in wall-clock but is
materialized as distinct bubble rows here.

### 6.4 Static request counters

Accumulated on pre-padding slices, so they count real operands only: `ifmap_reads`
(`systolic_compute_ws.py:292`, sparse variants `:285`, `:289-290`), `filter_reads` (`:373`),
`ofmap_writes` (`:457`). Exposed via `get_ifmap_requests` (`:589`), `get_filter_requests` (`:597`),
`get_ofmap_requests` (`:605`). These are the numerators of §3.2.

Per-fold efficiency metrics are also computed here (`systolic_compute_ws.py:400-406`), with `sum_sparse`
counted *before* the suffix is appended (`:388` precedes `:391`); `get_avg_mapping_efficiency`
(`:561-572`) and `get_avg_compute_utilization` (`:575-586`) are plain unweighted means over folds.

---

## 7. Defects observed in SCALE-Sim while writing this spec

Recorded because a model calibrated against SCALE-Sim inherits these unless it deliberately does not.

1. `read_buffer.py:362-364` - read stall double-counted when positive; and a negative
   `potential_stall_cycles` (prefetch already landed, no stall) still applies at `:362`, *reducing*
   `offset`. The layout path (`:315-318`) is correct. USER-mode read stalls are therefore inflated on
   misses and can be pulled backwards on hits.
2. DRAM read `num_access` counts padded prefetch lines including `-1` slots (`read_buffer.py:389-390`,
   `:479`, `:399-404`, `:498-502`), inflating `Avg * DRAM BW` toward the configured bandwidth.
3. `write_buffer.py:253-255` - the `-1` compensation in `empty_drain_buf` inspects only the last row, so
   mid-trace partial rows over-count `num_access`.
4. `double_buffered_scratchpad_mem.py:307` - `int(max(ofmap_serviced_cycles))` takes a max over a list of
   length-1 arrays; the comment at `:306` flags it as a suspected fault.
5. `scale_config.py:481` / `:488` - `get_bandwidths_as_list` defined twice.
6. `scale_config.py:108,136` - `Bandwidth` read from `architecture_presets` rather than `run_presets`
   because `section` was reassigned.
7. `write_buffer.py:98` - `reset()` sets `self.backing_buffer = write_buffer()`, should be `write_port()`;
   `:101` uses `total_size_elems` before recomputing it.
8. `read_buffer.py:102-142` - `reset()` does not reset `num_bank`, `num_port`, `bw_per_bank`,
   `enable_layout_evaluation`, `num_lines`, inconsistent with `set_params`.
9. `write_port.py:75,91` - leftover `print` statements.
10. `read_port.py:68-69`, `write_port.py:57-58` - `find_latency` silently clamps any latency > 10000 to
    1 (read) / 0 (write).
11. `SCALE-Sim/README.md:110` claims "average and maximum bandwidths"; no maximum is computed or
    reported anywhere (§3.1).

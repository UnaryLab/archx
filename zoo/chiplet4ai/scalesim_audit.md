# chiplet4ai vs SCALE-Sim: memory/bandwidth model audit

Scope: `zoo/chiplet4ai/common/performance/` and `zoo/chiplet4ai/results/query/` checked against SCALE-Sim v3 semantics (`SCALE-Sim/scalesim/`). Goal is a relatively accurate event-based model, not trace accuracy. Everything is closed-form: **no per-cycle and no per-fold trace loops**, folds enter only as multipliers. Fixes land item by item; anything under "Open" is still live.

Supporting documents:
- `log/bw_audit_scalesim_spec.md` - SCALE-Sim reference semantics, `file:line` on every claim (2026-08-17).
- `log/bw_audit_chiplet4ai_state.md` - the chiplet4ai formula chain end to end, with measured numbers (2026-08-17; carries an erratum banner, its per-phase numbers and event names predate T9).

Reference config used throughout: 512x512 array, 10 MiB SRAMs (bank 1024, width 16, depth 5120), DRAM 256 GB/s, frequency 1000, GEMM batch=1, M=2048, K=N=8192, `loop_order='auto'`. Compute span 785,920 cy. `'auto'` selects `n_outer`.

**All numbers below are final and independently re-derived**, not taken from the implementing task's own reports. Where a figure could not be independently reproduced it is labelled as such.

---

## Correct (keep)

1. **Event decomposition.** Compute lives in `*_arr` (parallel factors, aggregating to compute_cycles once); stalls are `sequential` chains of `cycle_reference` under `*_sram`/`*_dram`. `query_cycle_breakdown` (query/utils.py:53-80) decomposes along it exactly, bucketing on `endswith('_arr'/'_sram'/'_dram')` over the reachable component events and adding each matched node's whole subtree count.
2. **WS fold structure and per-fold cycles.** `ceil(K/R) * ceil(N/C)` folds at `M + R + C - 2` cycles each matches SCALE-Sim's demand-matrix pipeline. SCALE-Sim's materialized per-fold matrix height is `T + 2*arr_row + arr_col - 2` (spec §6.3); the extra `arr_row` is the weight-load prefix, which overlaps the previous fold's drain in wall-clock and is only padding rows in the trace. Our `M + R + C - 2` is the right wall-clock figure. **This overlap is load-bearing** and is reused twice below (weight proration, osram drain eligibility).
3. **Decode-step (KV growth) machinery.** Closed-form trajectory sums (`_range_sum`, `_ceil_sum`) divided by `total_steps` and multiplied back by per-step edge counts. Exact and cheap; verified uniform across all eleven `_dc` GEMM branches.
4. **Weight DRAM traffic.** Every weight fetched exactly once: correct for weight-stationary, matches SCALE-Sim filter traffic.
5. **Input DRAM refetch model (all-or-nothing, replaced 2026-08-17 -- see Resolved 13).** `reads = unique (M*K)` when the whole input matrix fits the active half of the double buffer, else the full streamed ceiling `M*K * ceil(N/C)`; loop-order independent, because SCALE-Sim's WS traversal is fixed and its rotating sequential prefetch window gives NO partial credit. The earlier linear `_fit_fraction` interpolation this replaces was a stand-in with no SCALE-Sim analogue.
6. **Double buffering.** Active/prefetch split via `active_fraction`, prefetch bandwidth `min(DRAM, SRAM)`, mirroring SCALE-Sim's double-buffered scratchpad (`read_buffer.py:84-86`; `single_layer_sim.py:239` also hardcodes 0.5). **The prefetch half is what makes DRAM demand elastic across phase boundaries** - see Open 1.
7. **SRAM access bookkeeping.** SRAM writes = DRAM reads (fills x active-buffer words); SRAM reads = streamed elements via the array path; energy plumbed once (array-path osram accesses carry `dynamic_energy: 0`). Self-consistent.
8. **Byte conservation across phases.** Every stream's bytes are assigned to phases exactly once, verified to ULP over 576+ direct `_ws_schedule` evaluations spanning degenerate (1x1, M=K=N=1), asymmetric (512x1, 1x512, 3x7), prime (8191^3) and stepped shapes in both loop orders: max relative residual 1.3e-16 (DRAM), 0 (wsram), 1.1e-16 (osram).

---

## Resolved

All re-verified against the working tree on the stated date. Verification for items 1-6 is `log/bw_audit_chiplet4ai_state.md` Part B plus the T4/T7/T9 review passes; for items 7-12, the T6/T8/T9 review passes.

1. **[verified 2026-08-17] Output tail stall window.** Window is steady + tail, not the drain tail alone. Reference stall 393,472 -> 0, correctly monotone in osram service bandwidth. Two corrections to this document's earlier text: the window is **`M + C - 1`** per fold, not `M + C - 2`; and "stays positive for an undersized osram" is true only under a **bandwidth** reading (a capacity-undersized osram does not move this stall at all - at 1 MiB it is still 0; at fixed 10 MiB, bank512 -> 393,472, bank256 -> 1,442,048, bank128 -> 3,539,200).
2. **[verified 2026-08-17] DRAM output write traffic and symmetric gating.** Same `(1 - output_fit)` gate both directions. Reference: writes 537 MB -> 33.5 MB, write stall 1,167,205 -> 0, output DRAM bandwidth 636 -> 39.76 GiB/s. Spill path live and non-degenerate under forced `k_outer`. **Superseded 2026-08-17** by the output write-through ruling (Resolved 14): the fit gate is gone in both directions and no DRAM spill reads exist any more.
3. **[verified 2026-08-17] Loop order.** `loop_order` in {'auto','k_outer','n_outer'} threaded to every entry point; both operands derive from one `_order_traffic` call per order; 'auto' = argmin over `dram_bytes`, weights correctly excluded.
4. **[verified 2026-08-17] osram accumulate traffic tracked separately** from DRAM spill traffic, and used at every consuming site. Order-invariant at 1,040,187,392 B, exactly `M*N*(15 reads + 16 writes)*2`.
5. **[verified 2026-08-17] Chunked-window bandwidth.** `_window_from_chunks`/`_chunk_count` gone. `_read_active_span_cycles` since deleted too; the weight window is now the compute span like every other operand, removing the 3.9%-low bias that made identical byte counts report 153.07 vs 159.05 GiB/s.
6. **[verified 2026-08-17] Untaxed DRAM output spill reads.** Spill reads now enter the service terms symmetrically with spill writes. Previously 424,673,280 B moved for free under `k_outer`; this was the single largest contributor to that order's optimism. **Superseded 2026-08-17** by Resolved 14: DRAM output spill reads no longer exist at all (write-through), so there is nothing to tax.
7. **[verified 2026-08-17] Per-operand DRAM bandwidth and the combine rule.** Concurrent streams now share the single declared channel instead of each receiving `min(dram_bpc, sram_bpc)`. Note for the record: the old per-operand behaviour **faithfully matched SCALE-Sim**, which models no shared channel at all (three independent `req_gen_bandwidth` values, `read_port.bw`/`write_port.bw` dead fields never read, default port a constant-latency shifter with no occupancy model). Sharing the channel is a deliberate improvement over the reference, not a correction toward it. The combine rule moved from max-of-whole-GEMM-aggregates-plus-summed-phases to a per-phase form; that form is itself now under revision, see Open 1.
8. **[verified 2026-08-17] Boundary guard at zero-length boundary windows.** At `pe=[1,1]` the model previously dropped the **entire weight stream, 134,217,728 B**, from every DRAM and wsram phase. Corrected constants: the bogus zero-window stall was **131,072 cy** (`output_final_bytes/256`), not 131,233. **Attribution correction:** that stall is fixed by the final-write proration (item 9), not by the guard - a guard-only build still reports 131,072. The guard is independently necessary where the weight stream actually binds: witness `pe=[1,1]`, `M=1`, `K=N=256`, DRAM 1 GB/s, wsram bank 2, where the old model reported 512 cycles against 66,560, a **130x under-report**, while silently losing 131,072 B.
9. **[verified 2026-08-17] Final-write proration across steady+tail**, applied on both the DRAM side and the osram drain. Tail-only osram would have reintroduced Resolved 1 on the SRAM side, so the symmetry is required. Byte conservation exact. The assignment *rule* (by window length) is superseded by Open 1.
10. **[verified 2026-08-17] Phase-truthful stall event names.** `dram_{fill,steady,tail}_stall` / `sram_{fill,steady,tail}_stall`. The old names were genuinely misattributed - `dram_input_read_stall` was fed by the steady phase, whose bytes under `k_outer` are 96.2% output spill, and `dram_output_write_stall` by the tail phase, ~67% of which is weight bytes. Exact six/six bijection with `description.py`; `event.yaml` machine-regenerated (839 -> 839 lines, keys still lexicographic with renamed keys moved to their new sorted positions); 124 architecture yamls byte-identical; all 1240 per-run `event.yaml` share one md5 with the master.
11. **[verified 2026-08-17] `lm_head_dc` multiplicity and root double-count.** `decode()` now charges `lm_head_dc` `max_seq_len - prefill_seq_len` times, character-for-character the expression used by every other `_dc` branch; `lm_head_dc` is non-stepped (`step_dim=None`, `total_steps=1`), so there is no internal per-step sum to double. The root's two edges are both `parallel`, so the engine's `sequential_acc + parallel_max` yields `max(llama, llama_array)`; a `dynamic_energy: 0.0` factor on the `llama_array` edge dedups energy, which aggregates by summation and ignores edge aggregation. Dominance `llama >= llama_array` is structural: identical `*_arr` multisets with identical multiplicities, all llama-side edges sequential, added `*_sram`/`*_dram` terms non-negative by construction. **Known limitation, deferred:** `aggregate_event_count` reads neither aggregation nor factor, so raw event counts of shared `*_arr` nodes are still doubled under the root. The `description.py` separate-root restructure that would fix it is deferred.
12. **[verified 2026-08-17] Query-layer collapse and hygiene.** CSV is 12 columns x 200 rows, no NaN, no inf, no near-duplicate pair (smallest all-pairs max-relative-difference 7.735526e-01). The four window columns collapsed into one `demand_window_cycles` legitimately: they were **structurally** identical, all set to the same literal `compute_cycles if <bytes> > 0 else 0`. Collapse validated physically - `data_moved / (demand_window/1e9) / 2^30` reproduces both bandwidth columns to 3.8e-16. The `_scientific` CSV is now sorted numerically before stringification (it previously re-sorted lexicographically on `%.3e` strings, so its row 1 was a different configuration than the plain file's); it now reproduces the plain CSV string-exact across all 200x12. SRAM sample dimensions fixed (`count_to_bytes` was `width/8` applied to a buffer-fill count, wrong by a factor of `active_elements`). `execution_time` now read from the stall-carrying `llama` branch. `memory_stall_cycle_count` renamed `dram_stall_cycle_count`.

13. **[applied 2026-08-17] Input traffic: all-or-nothing step law** (user ruling "Apply it then", grounded in an independent reconciliation against ~90 real SCALE-Sim USER-mode points). `input_fits = max_input_matrix <= isram_active_elements`; `reads = unique` if it fits, else `unique * ceil(N/C)`. The fit keys on the WHOLE input matrix for both loop orders -- SCALE-Sim has one fixed traversal and no strip-residency mode -- so DRAM traffic became loop-order independent and `'auto'` now resolves to `'n_outer'`, the order whose semantics the law matches. No set-quantization margin is applied (SCALE-Sim's set-granular window can under-hold ~2% at the boundary; a 0.98 factor would be a magic constant for a sub-2% edge, and the exact active half keeps one capacity definition across the model). This closes the "linear stand-in" residual recorded under the aggregator section; the remaining granularity note is that stepped decode GEMMs test the fit once per event at the largest step. `_fit_fraction` and `_resident_refetch_elements` are deleted.

14. **[applied 2026-08-17] Output traffic: pure write-through** (same ruling; reconciliation finding: `write_buffer.py:256` is a drain, ofmap DRAM writes are independent of osram size, and partial sums are never read back). `dram_output_write = output_write_elements` (~`M*N*k_folds`), `dram_output_read = 0` always. The `dram_output_read` event and its model functions are kept with zero counts. osram-side accumulate traffic is unchanged (on-chip). Note for readers of the historical entries: every pre-ruling measured constant in this document that depends on the old traffic laws (e.g. the 458 MB reference `input_read_bytes`, the 2,445,312 / 4,104,192 exact-identity stalls, the 33.5 MB gated writes) predates 13/14 and is a record of the superseded model, not of the current one.

---

## The stall aggregator: accepted design, rulings, and verification

**Resolved, verified 2026-08-17.** This was the model's dominant error source and is now its core specification. Spec and code agree; the section is written so that either can be reconstructed from the other.

### The defect this replaced

`stall = SUM_p max(0, max(dram_p, sram_p) - window_p)` over the three disjoint windows discards slack: a phase with an idle resource cannot lend capacity to a starved phase, even though double buffering and cross-fold pipelining are exactly the mechanisms that let real hardware do so. Two measured consequences:

- **Stranded bandwidth under re-splitting.** In the `bank=64` osram probe (10 MiB capacity held, `osram_bpc = 64`), osram must move 1,040,187,392 B at 64 B/cy = **16,252,928 cy** against a 785,920 cy compute span, so the split-free bound on total stall is **15,467,008**. The old model reported 15,912,777, an excess of **445,769 (2.9%)**, decomposing exactly as +26,123 stranded osram tail bandwidth (tail service 104,694 against a 130,816 window) plus +157,502 tail DRAM stall that the previously oversized osram tail had been masking under `max()`. Over the design's own grid (25 pe shapes x 5 bank values) 55/125 configs showed a saturated steady phase beside an idle-tail SRAM, **systematically upward, concentrated exactly in the SRAM-bound regime fig_2 is about**. (The pre/post up/down/unchanged split reported by the implementing task could not be independently reproduced - the immediate predecessor aggregator is absent from git history and from every stash - so it is recorded as author-reported only. What *is* independently verified is the post-fix grid state, below.)
- **Discarded DRAM slack.** On the reference PE array at 1 TB/s, `k_outer`: windows `[130816, 524288, 130816]`, DRAM services `[65536, 888433, 72080]`, stalls `[0, 364145, 0]`, runtime 1,150,065 against a roofline of 1,026,048 - a **12.1%** overstatement decomposing exactly as unused fill slack 65,280 + unused tail slack 58,736 + 1 cycle of rounding. Worst observed relative error **74.1%** (`pe=[1,2]`, bw=1, M=5, K=1024, N=999, `step_dim='m'`, `k_outer`); worst absolute 308,398,718,976 cy (`pe=[1024,1]`, bw=1024); error exceeds 2 cycles in **18.5%** of a 57,820-point sweep.

#### Accepted design (implemented; closed-form, no loops)

The physics that picks the design: **DRAM is decoupled from instantaneous demand by the prefetch half of the double buffer** (Correct 6), so charging it per phase models a barrier the hardware does not have. **The array's SRAM demand is genuinely real-time** - the active half must supply its words in the cycle they are consumed - so per-phase resolution is meaningful there and must be retained. Hence a hybrid.

**(a) Split-free floor.** For each resource `r` in {dram_channel, isram, wsram, osram}, with a **single** ceil over whole-GEMM bytes:

```
service_r = ceil(total_bytes_r / rate_r)
floor     = max(0, max_r(service_r) - compute_cycles)
```

**(b) Real-time phase pressure, SRAM only.** For each phase `p` in {fill, steady, tail} and each SRAM buffer `b`:

```
rt = SUM_p max(0, max_b(ceil(bytes_{b,p} / rate_b)) - window_p)
```

DRAM contributes to `rt` not at all; it is represented entirely by `floor`.

**(c) Combine.**

```
stall_cycles = max(floor, rt)
```

**(d) Slack-aware assignment of prorated streams**, replacing proration by window length. Slack is **per resource**, measured against the **realized** phase length:

```
realized_p       = max(window_p, max_b exclusive_service_{b,p})   # exclusive = non-prorated bytes
slack_{b,p}      = max(0, realized_p - exclusive_service_{b,p})
total_slack      = SUM_{p in eligible(S)} slack_{b,p}
if s <= total_slack:  assign s proportional to slack_{b,p}        # fits in slack, adds no stall
else:                 fill each eligible p to slack_{b,p}, then distribute the remainder by window length
```

With three fixed phases this is a closed-form three-way expression, not an iteration.

**This is the corrected form of the rule, and the correction matters.** An earlier draft of this spec wrote a single phase-wide slack against the nominal window, `slack_p = max(0, window_p - exclusive_service_p)`. That was **wrong, and self-contradictory with (b)**: a phase-wide scalar necessarily aggregates across buffers, so a saturated isram would consume the room an idle osram has in the same phase - functionally a sum across physically separate ports, which is precisely the stranding this whole section exists to remove. Two properties make the per-resource realized form correct and safe:

- **Non-circular.** `realized_p` depends only on the fixed windows and the exclusive byte totals, both computed before any assignment. It never reads the prorated phase bytes, so the chain runs one way. It also omits the prorated streams' own contribution, making it a *lower* bound on the true realized length and therefore conservative.
- **It cannot understate.** In the fits-in-slack branch each phase receives service `<= slack_{b,p}`, so a non-binding buffer filled to the realized length can never raise `max_b` above what the binding buffer already forces. Verified: the binding phase's `rt` term is bit-identical under both readings.

Measured difference on a case where isram binds and osram idles (`pe=[512,512]`, isram bank 2, osram bank 8, DRAM 1024): the steady phase genuinely lasts 268,435,456 cycles because isram cannot feed the array faster, leaving the osram port idle for 142,606,336 of them, and the final drain fits there for free. The phase-wide reading refuses to see that room and charges **1,309,184 cycles** of stranded bandwidth. Across a 2,592-config comparison the two differ on 96 points, 92 of them phase-wide higher.

**(e) Eligible phase sets**, each justified by the WS overlap of Correct 2:

| stream | eligible phases | justification |
|---|---|---|
| weight load (wsram, DRAM) | fill + tail | weight-load prefix overlaps the previous fold's drain |
| input stream (isram) | steady | array consumes inputs in real time; not deferrable |
| osram accumulate (read + write) | steady | partial sums are produced and consumed in the steady region |
| final output drain (osram, DRAM) | fill + steady + tail | outputs retire column-by-column throughout, and drain of fold *i* overlaps fill of fold *i+1* - the same overlap already accepted for weights |
| DRAM, all streams | whole span | prefetch buffer decouples DRAM from instantaneous demand |

#### What it bought, verified

- **The `bank=64` probe collapses to the bound up to rounding.** Excess over the split-free bound: **445,769 -> 1**. The residual 1 cycle is pure per-phase ceil rounding (`rt = 43,719 + 15,379,571 + 43,719 = floor + 1`); the real-time component is **exactly 0**, and 1 sits inside the `{0..P_live-1}` bound of identity 2 below. **Ruling, recorded:** an earlier draft of this section promised "collapses to the bound **exactly**" and an excess of 0. That was an over-precise promise extrapolated from a single-resource toy calculation that modelled osram in isolation against nominal-window slack; the implementation assigns under realized-window slack with every resource present, producing a different and equally valid three-way split whose per-phase ceils sum to `floor + 1`. (For the record, the discrepancy is **not** un-ceiled arithmetic - the toy gives 0 with and without ceils.) The regression target is therefore `<= P_live - 1`, never 0.
- **The exact byte identity is restored.** A single whole-span ceil for the DRAM channel removed the double-ceil that cost +1 cycle in both loop orders. Reference config: `n_outer` **2,445,312** and `k_outer` **4,104,192**, each exactly equal to one-channel service (625,999,872 and 1,050,673,152 bytes at 256 B/cy). In both orders `total_bytes / 256` is an exact integer, so the identity is tested with zero rounding slack. The six stall events sum to exactly 1,659,392.
- **The floor is guaranteed by construction**, since `stall = max(floor, rt) >= floor`.
- **A cleaner decomposition than the protocol asked for.** Because `compute + floor == bound` identically, `runtime - bound == max(0, rt - floor)` exactly. So **all** excess over the split-free bound is real-time phase pressure; rounding never contributes on its own and enters only inside `rt`.

#### Residual pessimism, stated honestly

`rt` can still exceed `floor` when a buffer's real-time demand within one phase exceeds that phase's window even though its whole-span service fits. That is the physically correct behaviour - the array genuinely stalls - but the phase boundaries are themselves a modelling abstraction of the WS pipeline, so the residual is an artifact of a three-phase discretization, not a hardware property. It must be reported per config rather than hidden. Two further limits remain and are **not** addressed by this fix: DRAM elasticity is treated as unbounded, whereas real prefetch is capped by the prefetch buffer capacity (a config whose per-phase DRAM burst exceeds the prefetch half would stall in hardware and will not here). The second limit this paragraph used to carry -- `_fit_fraction` as a linear stand-in for reuse -- is resolved by the all-or-nothing step law (Resolved 12a, 2026-08-17); what remains of it is only the granularity note that stepped decode GEMMs test the fit once per event at the largest step's working set.

#### Verification results (independently re-derived)

Coverage: a 32,144-point structured grid plus a 60,000-point randomized stress sweep (PE rows/cols including 1, 2, 3, 7; banks including 1, 3, 127; DRAM down to 1 B/cy; primes, degenerates, all step dims, both loop orders), plus targeted probes.

1. **Byte conservation.** Assert is now genuinely absolute - `max(0.5, 4*ulp(total))` - replacing the old 1e-6 relative tolerance that permitted ~232 KB of slack at reference scale. Worst measured absolute residual **2.44e-4 B**, about 2000x inside the bound.
2. **Floor property.** `runtime >= max(compute_cycles, max_r service_r)`: **zero violations in 92,144 configs**, zero negative stalls, zero assertion failures. Also structural, and self-asserted in code.
3. **Exact identity.** Residual **0** in both loop orders (numbers above).
4. **Excess decomposition.** Design grid (25 pe x 5 bank): **0/125 below the bound, 44/125 exactly on it, 81/125 above**, median 2.10% of the bound, worst 15.58% (`pe=[512,128]`, bank 512). Stress sweep worst relative excess **0.78%**.
5. **Regression.** `bank=64` probe excess **445,769 -> 1** (rounding only). The pre/post grid split is author-reported and not independently reproducible; see the note under "The defect this replaced".
6. **Degenerate shapes.** All clean: no bytes above 0.5 B ever placed in a zero-length window across all 92,144 configs, checked directly rather than via the in-code assert. The `pe=[1,1]`, `M=1`, `K=N=256`, bw=1, wsram bank 2 witness reproduces **66,560** - and note it is set by the **DRAM floor**, not by wsram, consistent with the attribution correction in Resolved 8.
7. **Rounding bound.** `E = SUM_p ceil(b_p/r) - ceil(B/r)` computed per resource per config: **max E = 2 with P_live = 3**, zero violations, and independent of the number of prorated streams.

---

## Open, ranked

### 1. Reported bandwidths are demand-over-compute-span, never reconciled against supply

Every reported bandwidth divides bytes by `compute_cycles`, and none is recomputed once a stall extends the real span. In the narrowed-osram run the model reports 1232.6 GiB/s of osram bandwidth while the osram can physically deliver 59.6. The numbers are honest *demands*, but nothing in the model or the query layer compares a demand against the corresponding service rate, so a demand above the roofline is reported without comment. Either divide by `compute_cycles + stall_cycles`, or emit the demand/supply ratio alongside each bandwidth.

### 2. `peak_required_bandwidth` was dropped, and the definition question stays open

SCALE-Sim has no peak or required bandwidth metric to borrow: a grep for `peak|required.?bandwidth|req_bw|max_bw` over its whole tree returns zero hits, the report header is all `Avg` columns, and the README's "maximum bandwidths" claim is stale. Its two average families do not even share a window (SRAM avg = static compute-model counts / `total_cycles`; DRAM avg = padded access counts / that operand's own DRAM active window). Our old column was a max over per-GEMM averages - for stepped decode GEMMs, a max over averages of averages - and measured ~924 GiB/s set by the two *smallest* prefill events while >90% of bytes moved at 0.6-290 GiB/s, at 3.6x the channel with no roofline comparison. It has been dropped, which is the right default. **If a requirement number is ever wanted**, the only definition with real sub-kernel resolution is the per-phase demand rate `max_p (bytes_p / window_p)`, reported as a ratio against `dram_bytes_per_cycle`. That is also the nearest analogue to the one requirement-shaped number SCALE-Sim has, CALC mode's back-computed `prefetch_bandwidth = elems / (window between consecutive prefetch deadlines)` (`read_buffer_estimate_bw.py:180-183`), likewise an average over a refill interval rather than a burst.

### 3. `_active_banks` semantics at bank=1 and odd banks (resolved as documented)

Unifying the capacity view and the port view onto `_active_banks = max(1, min(bank, floor(bank*fraction)))` closed the previously divergent cases (bank=768 at fractions 0.333 and 0.001) and is right. But "fraction 0.5 unchanged" is **false in general**: at bank=1 the clamp yields active = total and prefetch = 1, i.e. **no double buffer at all, for every fraction including 0.0**, which doubles effective capacity and therefore feeds `_fit_fraction`, DRAM traffic, and the auto loop-order choice. At bank=3, fraction 0.5 gives one bank of three (33%, not 50%). No effect on the shipped sweep - `description.py` builds `sram_banks = [64,128,256,512,1024]`, all even powers of two, and no YAML sets `active_fraction`, verified bit-identical - but it is a live trap for any future single-bank or odd-bank config.

**Resolved as documented rather than asserted, 2026-08-17**, which is the right call: single-bank architectures are legal to model, and an assert would reject them for a modelling nicety. The degeneracy bullets in the code are accurate (verified: `bank=1` gives active = total and prefetch = 1 at every fraction including 0.0; `bank=3` at fraction 0.5 gives 1 of 3). An earlier draft of the code justified rejecting the assert by claiming `bank` "is optional in the schema (it defaults to 1 here)". That was false on both clauses - there is no schema anywhere in the repo, and `bank` is hard-required by `utils.py:28` and `cacti7.py:43,247` - and it has been **replaced with the true reason, verified 2026-08-17**: a single-bank SRAM is a legal architecture to model, so the degeneracy is documented rather than refused. The dead `.get('bank', 1)` fallbacks were removed at the same time; the caller set was traced closed (`_active_banks` reachable only from `_buffer_elements` and `_sram_bytes_per_cycle`, both of which sit behind `_sram_bits`, which indexes `query['bank']` and raises first), so a bank-less query raised `KeyError` before the change and still does. The failure stays loud even under a hostile reordering of `_ws_schedule`: it would raise from a different line rather than silently model a single-bank SRAM.

### 4. The weight-load window justification is config-specific

The argument that a fold's `R*C` weights served at `C` words/cycle need exactly `R` cycles holds only because `_sram_bytes_per_cycle` gives `bank * active_fraction * width/8`, and `1024 * 0.5 = 512 = C` in the reference arch. `wsram_bytes_per_cycle` is independent of `array_cols`. The reasoning survives but should be scoped to `bank/2 ~= C` configs; the same coincidence makes `steady_state_stall` structurally 0 for any config where `bank/2 ~= array_cols`, firing only when `bank` is deliberately mis-sized.

### 5. Final-write tail share is geometrically too heavy (superseded)

Length-proration puts `(C-1)/(M+C-1)` = 19.97% of finals in the tail, where the WS drain geometry implies `(C-1)/(2M)` = 12.48% (of the `M*C` finals per fold, only `C(C-1)/2` emerge in the last `C-1` cycles). Direction is right, magnitude ~1.6x too tail-heavy. **Superseded**: length-proration is no longer used for this stream - the accepted design assigns it by per-resource slack, which sidesteps the geometric question entirely. Recorded only in case a geometric rule is ever preferred to a slack rule.

---

## Model identities and error characterization

State all three clauses. **Never quote the rounding bound alone** - it covers a genuinely bounded 2-cycle artifact while the model's real error is unbounded, and quoting it alone licenses a 3x10^11-cycle error under a 2-cycle banner.

1. **Byte conservation: exact.** Every stream's bytes are assigned across phases exactly once, max relative residual 1.3e-16 over the full shape sweep. This is a hard invariant and should be asserted with an absolute tolerance below one byte.
2. **Rounding error: bounded, and stream-count independent.** With `E = SUM_p ceil(b_p/r) - ceil(B/r) = SUM_p delta_p - delta_B`, `E` is an integer in `(-1, P)` and therefore in `{0..P-1}`; the supremum `P-1` is attained (`r = P`, `b_p = 1`), witnessed in the real code at `pe=[2,2]`, bw=2, M=2, K=3, N=5, `step_dim='k'`, `k_outer`. **The number of prorated streams is irrelevant**: the ceil is applied to the per-phase *sum* of all streams' fragments, so residues from different streams add before rounding and never each buy a cycle. A tenth prorated stream still cannot exceed `P-1`. `P` here is the number of phases with a **nonzero** window, so degenerate arrays lower the bound. Against the un-ceil'd `B/r` the bound is `< P`, not `<= P-1`.
3. **Slack error: unbounded, and the one that matters.** When any phase has slack, `runtime - max(compute, ceil(B/r)) = SUM_{slack phases}(window_p - service_p) + rounding`, bounded above only by the roofline itself, i.e. up to a 2x overstatement of runtime. Measured 12.1% on the reference PE at 1 TB/s, 74.1% worst relative, >2 cycles in 18.5% of a 57,820-point sweep. **This was the aggregator defect, fixed 2026-08-17** (see the aggregator section). It was a first-class modelling assumption - no cross-phase bandwidth borrowing - never a footnote about rounding. Post-fix, all excess over the split-free bound is real-time phase pressure, bounded on the design grid at 15.58% worst and 2.10% median, with 44/125 points sitting exactly on the bound and none below it.

**Direction: always pessimistic.** `SUM_p ceil(x_p) >= ceil(SUM_p x_p)` by superadditivity, and `max(0, .)` only inflates, so `runtime >= max(compute_cycles, ceil(B/r))`. The model never understates the roofline. This guarantee is contingent on byte conservation being exact, which is why clause 1's assert tolerance matters.

**Two tolerances that must not be confused.** The `2e-16` relative tolerance belongs to `query_cycle_breakdown`'s **unattributed residual**, a sum decomposition where ULP drift is real: the residual is exactly 0.0 on the audited configuration but ULP-nonzero on 11 of 40 sampled runs, always negative, `|residual|/total <= 2.0e-16`. The safe phrasing there is "residual <= 2e-16 relative", never "== 0.0". It does **not** belong to the `root == llama` check: the root aggregates by `max` (`aggregate.rs:322-389`, `sequential_acc + parallel_max`), and a max returns one operand **unchanged**, so the root is either bit-identical to `llama` or is `llama_array`'s structurally different value. There is no summation-order path to a one-ULP difference, and `2e-16` is below one ULP of a double anyway. That check uses exact `!=` (T13).

---

## SCALE-Sim behaviours we deliberately do NOT replicate

Recorded so calibration targets are chosen knowingly. Every item is a real defect in SCALE-Sim (`log/bw_audit_scalesim_spec.md` §7); a model tuned to reproduce its numbers inherits them.

1. **Read-stall double-count** (`read_buffer.py:362-364`). `potential_stall_cycles` is added twice when positive, and when negative - prefetch already landed, no stall - it is still applied, **subtracting** from `offset`. The layout-evaluation path (`:315-318`) is correct. **Do not calibrate our DRAM read stall against SCALE-Sim USER-mode stall cycles.**
2. **DRAM read traffic counts padding.** `num_access` accrues whole `req_gen_bandwidth`-wide prefetch lines including `-1` slots, so USER-mode `Avg * DRAM BW` comes out at roughly the configured bandwidth almost by construction. Our byte counts are exact.
3. **Write drain `-1` compensation inspects only the last row** (`write_buffer.py:253-255`), over-counting `num_access` on mid-trace partial rows.
4. **Per-operand uncontended DRAM.** SCALE-Sim gives each operand an independent budget and models no shared channel. We share the declared channel; no SCALE-Sim number can validate that stall.
5. **No peak/required bandwidth exists to match.** See Open 3.
6. **`Total Cycles` excludes `Stall Cycles` by construction.** SCALE-Sim reports `total_cycles = max(ofmap_serviced_cycles)` and reports `stall_cycles` as an independent statistic never added or subtracted. Our `stall_cycles` is additive on top of `compute_cycles`. Not the same quantity; do not compare directly.
7. **Read misses do not fetch the missing address.** SCALE-Sim rotates the prefetch window and retries, so one miss can cost several rotations and the stall depends on window phase, not on the address.
8. **CALC mode is stall-free on reads by construction** (`read_buffer_estimate_bw.py:119-121`). Only USER-mode runs are meaningful stall references.
9. **In USER mode, ifmap/filter backing bandwidth does not come from the `Bandwidth` key**; it comes from the `[layout]` SRAM bank bandwidth keys, and only ofmap uses `bandwidths[0]`. A SCALE-Sim config that appears to set DRAM bandwidth is setting the ofmap drain width.
10. Minor SCALE-Sim defects with no bearing here: `get_bandwidths_as_list` defined twice; `Bandwidth` read from the wrong section because `section` was reassigned; `write_buffer.reset()` assigns `write_buffer()` where `write_port()` is meant; `read_buffer.reset()` misses several fields; leftover prints; `find_latency` silently clamps latency >10000 to 1/0.

---

## Minor / hygiene

- **`sram_*_stall` is a residual, not a measurement.** `_phase_stalls` computes `sram_stall = total - dram_stall`, so the emitted per-phase SRAM stalls are identically 0 whenever DRAM dominates - which is every phase of the reference config. `sram_cycle_count == 0.0` on balanced-SRAM llama configs is therefore **legitimate and structural**, not a bug; downstream readers must not treat it as one.
- **`count * factor == 1` structurally in node.py**, since count and factor are exact reciprocals. Consequently `cycle_count`/`runtime` are **insensitive to SRAM access counts** and only `dynamic_energy` responds. Relevant to anyone sweeping `active_fraction`, and it means energy is the only witness for a regression in the active-element derivation. Reciprocity is exact only when `floor(total*fraction)` is dyadic: at fraction 0.333333, `count*factor == 0.9999999999999999` and the companion identity in mapping drifts 1 ulp. Nil impact today, since no YAML sets `active_fraction`.
- **The `_dram`/`_sram` prefixes on the stall event names are not load-bearing**, contrary to the comments at `mapping.py:105` and `description.py:71-72`. `query_cycle_breakdown` buckets on `endswith` over the parent composite events (`query/utils.py:62,67,72`); prefixes never participate. The stall events are counted because they sit inside the `*_dram`/`*_sram` wrappers' subtrees, which is why `unattributed_cycle_count` is 0. The comment overstates the constraint; correcting it is queued with T12.
- **Conservation assert tolerance.** `mapping.py` uses 1e-6 **relative** (~232 KB at reference scale) while observed residuals are 1e-16. Tighten to an absolute tolerance below one byte so it can catch loss of a small stream; queued with T12.
- **`_component_event_counts` diamond bug: the fix was real, and the "latent" claim was wrong.** The old dedup counted a node reached by two equal-multiplicity paths twice but its subtree once. Earlier text in this document and in the state doc called it "not triggered today". **That is false**: old versus new differ on 40/40 sampled root queries, by up to 2.37x (`pe` 1.84e19 vs 7.76e18). It never reached published numbers only because the three buckets filter on `_arr`/`_sram`/`_dram` suffixes that sit above the diamond, so the frozen CSVs were always safe. The rewrite is a genuine Kahn topological traversal, verified against a synthetic diamond fixture (3 -> 6), raising `ValueError` on a cycle, O(V+E) with 2^30-path fixtures resolving exactly in 0.1 ms, and deterministic.
- **`weight_dram_bandwidth` is constant per model** (125.220955 for 70B, 115.941055 for 8B) because `weight_read_elements = weight_elements` unconditionally: weight DRAM traffic is independent of wsram size. This is correct WS physics, not a bug. It does mean the weight panel of fig_2 carries no reuse signal; whether to keep the panel is a presentation decision.
- **`demand_window_cycles` is written but read by nothing.** Defensible as provenance for the bandwidth columns, but it should be a deliberate choice rather than an accident.
- **`fig_2.py` NaN contract.** Empty groups now correctly yield NaN plus a warning rather than a fabricated 0.0 point; `set_ylim(0, max)` is the one place that contract is unhandled and will raise on an all-NaN panel (queued with T13).
- **Figure scope is undocumented in the figure.** The sweep filter keeps one array size and one batch size, 200 rows of 1240. The `root == llama` assertion deliberately runs *before* the filter, so its coverage is the full 1240.
- **Unit convention.** DRAM bandwidth is now decimal GB/s via 1e9 (`dram_bytes_per_cycle` 274.877906944 -> 256.0 at the reference), consistent in `memory.py`. DRAM traffic counts are bytes while SRAM counts are words; both feed `dynamic_energy` through the same CACTI leaf, so if the CACTI DRAM figure is per-access rather than per-byte, DRAM energy is inflated by the burst width. Energy-only: runtime is zeroed on all traffic events.
- **Dead and stale code.** The audit's earlier dead-code list is deleted (zero repo references). Remaining query-layer staleness: `gemm_test.py` cannot run (calls `query_performance_metrics` without its required `workload`/`metrics` arguments, reads a `checkpoint.gt` path that does not exist), and `best.py`/`fig_7.py` read `array_performance_metrics_{8b_scientific,70b}.csv`, neither of which is written any more; `fig_7.py` is a truncated stub. `query/utils.py`'s `query_bandwidth` is an unused, differently-defined bandwidth (data_moved / real aggregated runtime) that will not agree with `fig_2_query`.
- **Verification trap worth knowing.** `.gitignore` ignores `**/description/`, and the working snapshots are stashes without an untracked-files parent. `git diff <snapshot> -- <any description artifact>` therefore returns empty for **all** of them regardless of whether they changed, and `configurations.csv`/`runs.txt` are untracked so byte-identity to git cannot be asserted either way. Verify these by content and mtime against an extracted baseline, never by an empty `git diff`.

---

## Ranked fix order

**Closed, verified 2026-08-17.** The aggregator revision and its full verification protocol (see the aggregator section); conservation assert moved from 1e-6 relative to absolute sub-byte; the prefix-vs-`endswith` comment correction; exact `!=` on the `root == llama` assertion with the misleading summation-order comment removed; `_buffer_elements` routed through `_architecture`; the collapsed-window self-check dropped as structurally unreachable; `set_ylim` guarded against an all-NaN panel; `figure_generation.py` reordered to `[fig_1_query, fig_1, fig_2_query, fig_2]`, with stale outputs deleted before the loop so a halt leaves absence rather than staleness.

**Text-accuracy items closed, verified 2026-08-17** (no behavioural change to the model): the false `bank`-is-optional rationale replaced with the true one and the dead defaults removed (Open 3); the `2e-16` comment in `fig_2_query.py` rephrased to describe the *relative* residual it actually measures, with both scoping points retained; the `aggregate.rs` citation symbolized to `sequential_acc`/`parallel_max` so no Python-side refactor can rot it; and the stale claim that fraction 0.5 reproduces the old hardcoded `bank/2`, which holds on dyadic banks but not in general. Reference numbers re-derived unchanged after each: `n_outer` 2,445,312 and `k_outer` 4,104,192.

**Then, in order:** Open 1 (reconcile reported bandwidth against supply); Open 4 (scope the weight-window justification); Open 2 only if a requirement metric is actually wanted; the `description.py` separate-root restructure for `aggregate_event_count` doubling (deferred, documented limitation under Resolved 11). All **event-based-feasible**.

**Inherent limitations, document rather than attempt.** Sub-kernel time resolution finer than a phase: any true burst, percentile, or windowed-maximum bandwidth, and any stall depending on the instantaneous interleaving of operand requests, requires a per-cycle or per-line timeline; the per-phase decomposition is the finest resolution this model will have. Address-level reuse and prefetch-window-phase miss costs likewise have no closed form. Bounded prefetch elasticity (Open 1's residual) sits at the boundary: a capacity-aware cap is expressible in closed form, but its accuracy would not be verifiable without a trace.

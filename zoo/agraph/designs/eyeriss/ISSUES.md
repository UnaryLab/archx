# Eyeriss model issues

Known gaps between the fg model and the Eyeriss paper (Chen et al., JSSC 2017,
[paper/eyeriss.pdf](paper/eyeriss.pdf)). Model numbers come from
[results/fg](results/fg) and from the traffic code in
[common/performance_shared.py](common/performance_shared.py).

Out of scope by decision: parts and results that the paper can't validate or that come
from the PnR flow, such as the clock network, I/O pads, absolute power, leakage and area.

Last checked: 2026-10-28.

## What matches

- The 12x14 PE array, the GLB (25 ifmap/psum banks + 2 filter banks of 512x64b), the
  filter, ifmap and psum spads (224, 12 and 24 entries of 16b), and the 12-b zero
  buffer.
- The GIN and GON data widths (16b ifmap, 64b filter and psum), the 200 MHz core clock and
  the 60 MHz link clock.
- The Table III mapping parameters (m, n, e, p, q, r, t), the GLB ifmap and psum
  allocations, and the Table V MACs, active PEs and zeros in ifmaps.
- GLB accesses are within 2.5% of Table V for CONV1, CONV2, CONV4 and CONV5 (see issue 5
  for CONV3).
- Total AlexNet DRAM traffic matches Table V (15.4 MB) now that issues 1 and 2 are fixed.
- fg's total AlexNet latency is within 5% of Table V (120.1 vs 115.3 ms), and its
  processing latency within 3% (106.0 vs 103.5 ms), now that issue 17 is fixed. Per layer,
  CONV4 and CONV5 are high (see issue 17). cg doesn't have the issue 17 cycles and stays
  at 109.5 and 95.4 ms.

## Open issues

Per-layer values are listed CONV1 / CONV2 / CONV3 / CONV4 / CONV5.

### Traffic

**5. CONV3 GLB accesses are 40% too high.**
The model gives 70.5 MB against Table V's 50.2 MB. With q * r = 4 channels per pass, CONV3
takes 64 channel passes, so each psum makes 63 round trips through the GLB
([performance_shared.py:224](common/performance_shared.py#L224)). The other layers are
within 2.5%, so either the CONV3 psum schedule differs from the one the Table III values
imply, or the paper counts its accesses differently.

### Structure ([fg/description.py](fg/description.py))

**13. GIN multicast fan-out is not modeled. To model.**
Each NoC transfer charges one Y-bus and one X-bus. The paper multicasts one value to every
X-bus whose row ID matches, in the same cycle (Sec. V-B1, Fig. 10, Fig. 11). Unmatched
X-buses are gated, so a transfer costs one Y-bus plus one X-bus for each PE row that
receives it.
- Fix: charge `bus_gin_*_x` the average number of X-buses each transfer reaches,
  per layer, from the PE-set placement (Fig. 5) and the delivery patterns (Fig. 4). The
  Y-bus stays at one per transfer.
  - Filter GIN: a filter row is reused horizontally across one set row (Fig. 4a). That's
    one X-bus per segment, so 2 for CONV2 and 1 for the other layers.
  - Ifmap GIN: an ifmap row is reused diagonally (Fig. 4b), reaching up to R rows of each
    set that uses it (fewer at the set edges). Sets that share the ifmap and sit in
    different rows multiply that: the four stacked CONV3 sets, and every other set in
    CONV4/5. Sets side by side in the same rows, like CONV1's two 11x7 sets, share X-buses.
  - Psum GIN: a psum goes into one set row, so one X-bus.
- Not covered: each X-bus's energy stays at the 14-destination PnR value. How many of its
  PEs match (the rest are gated by their multicast controllers) isn't modeled.

# increase noc widths

## Resolved

- **1. DRAM traffic missed the ifmap and filter re-fetches** (fixed 2026-10-28). The
  model read each ifmap from DRAM once and each filter once per image group, giving
  2.65 / 3.83 / 2.35 / 2.02 / 1.46 MB. It now re-fetches both, giving
  4.78 / 4.20 / 3.50 / 2.71 / 1.81 MB ([performance_shared.py:228](common/performance_shared.py#L228)).
  - Ifmaps: the GLB holds psums for only m ofmap channels, so each group's ifmaps are
    re-read ceil(M_group / m) times (1 / 2 / 6 / 3 / 2). The rows where neighboring
    strips overlap are re-read too (CONV1 reads 276 rows instead of 227).
  - Filters: the 8 KB filter GLB holds only one pass ahead, so every filter read out of
    it is first fetched from DRAM. That's once per strip, 8x for CONV1.
  - What's left of the gap is issues 2 (now fixed) and 3.
- **2. RLC compression used one fixed ratio and compressed the CONV1 input** (fixed
  2026-10-28). `RLC_RATIO = 1.5` was applied to every fmap, giving 4.78 / 4.20 / 3.50 /
  2.71 / 1.81 MB of DRAM traffic (17.0 MB in total). Now DRAM traffic is 5.63 / 3.92 /
  2.67 / 1.91 / 1.30 MB against Table V's 5.0 / 4.0 / 3.0 / 2.1 / 1.3 MB, and 15.4 MB in
  total against Table V's 15.4 MB.
  - CONV1's input image is no longer compressed (Sec. IV-B).
  - Each ratio now comes from the Fig. 8 format and the fmap's zero ratio
    ([performance_shared.py:23](common/performance_shared.py#L23)). Three 21-b
    (run, level) pairs fit in one 64-b word, and a run stops at 31 zeros. With zeros
    placed independently, that gives 0.75 · (1 − z³²) / (1 − z): 1.22x at CONV2's 38.7%
    zeros and 3.62x at CONV4's 79.3%. The CONV3 ifmap ratio, 2.73x, matches the roughly
    2.7x that Fig. 9 implies.
  - Output zero ratios are a new workload parameter, `ofmap_zero_ratio`
    ([description_shared.py:200](common/description_shared.py#L200)). The paper doesn't
    give them, so each layer uses the next layer's ifmap zero ratio, and CONV5 reuses its
    own. Pooling follows CONV1 and CONV2, so their ratios are lower bounds. That's why
    CONV1 is 13% high: matching Table V would need about 59% zeros in CONV1's output,
    against the 38.7% assumed.
  - CONV3 and CONV4 are 11% and 9% low. Padding (issue 3) didn't change this: the
    border is all zeros, and the coded size depends mostly on the non-zero values.
- **3. Ifmaps were not padded** (fixed 2026-10-28). The workload now uses Table III's
  padded sizes, 31x31 for CONV2 and 15x15 for CONV3-5, with a new `conv_pad` parameter
  for the border ([description_shared.py:148](common/description_shared.py#L148)).
  - Table III's GLB ifmap allocation is now checked alongside the psum one
    ([performance_shared.py:173](common/performance_shared.py#L173)), and every layer
    passes.
  - Table V's zero ratios are read as activation statistics without the border (Sec.
    IV-B), so the border adds zeros on top
    ([performance_shared.py:66](common/performance_shared.py#L66)). RLC codes the padded
    pixel share: 53.5 / 79.3 / 84.5 / 83.2% zeros for CONV2-5. The zero buffer gates the
    share of MAC taps that land on zeros: 44.0 / 75.2 / 81.4 / 79.8%. That share is
    lower, because border pixels sit in fewer windows.
  - The other reading, where Table V already includes the border, gives 16.2 MB of DRAM
    traffic instead of 15.4. Its per-layer error is about the same.
  - DRAM traffic and latency are unchanged. Ifmap NoC transfers rose 24%, and about 6%
    more MACs are gated. GLB accesses rose 1-2%: 18.4 / 78.5 / 70.5 / 38.2 / 25.5 MB
    against Table V's 18.5 / 77.6 / 50.2 / 37.4 / 24.9 MB.
- **4. The RLC decoder was charged on filter reads** (fixed 2026-10-28). `dram_read` no
  longer includes the decoder. A new `rlc_decode` event charges it once per decoded 16-b
  value ([performance_shared.py:257](common/performance_shared.py#L257)), because the RTL
  emits one value per cycle (`rlc_decoder.v`, `o_ifmap_data`). Only coded ifmaps count,
  so filters and CONV1's raw image are excluded (Fig. 2). The count went from 1.57M
  64-b transfers to 3.85M values.
- **6. DRAM stall cycles counted the filter loads** (fixed 2026-10-28). Every DRAM
  transfer stalled the PE array, giving 10.0 / 8.7 / 7.3 / 5.6 / 3.8 ms of stall. Now
  only the ifmap share of the DRAM reads stalls, through a `factor` on `dram_read`
  ([performance_shared.py:324](common/performance_shared.py#L324)). Sec. V-A says the GLB
  preloads the next pass's filters while the PE array works, and Sec. VI-A says only
  ifmap loads and ofmap dumps stop processing. Energy and DRAM access counts don't change.
  - After this fix the stall was 5.3 / 3.6 / 3.6 / 2.9 / 1.9 ms. With issue 2 also fixed
    it's 7.1 / 3.0 / 1.9 / 1.2 / 0.9 ms, against the paper's 4.4 / 2.7 / 1.8 / 2.4 / 0.5 ms
    (total latency minus processing latency). CONV1 is high for the same reason its DRAM
    traffic is: the ofmap zero ratio assumed for it is too low.
  - An assertion checks that the hidden filter fetch fits inside the layer's MAC time.
    It takes 1.8-5.1 ms against 9.6-33.2 ms of MACs.
- **12. The local network had no module** (closed 2026-10-28). There's no reliable way to
  get a wire as a module, so the local network is represented by the 2:1 psum input mux
  that selects between the PE below and the psum GIN. That mux already existed:
  `psum_in_mux` is pe.v:132 (`w_psum_ififo_data = i_cfg_psum_ln ? i_psum_ln_packed :
  i_psum_packed`), so no second mux was added, which would have counted it twice. Its
  comment now says it stands in for the local network
  ([fg/description.py:53](fg/description.py#L53)).
  - It's charged on every psum that comes in from either source, once per 64-b pack, in
    `pe_psum_acc`. The count is all psums out minus the final ofmaps, which is the local
    network words plus the GIN words.
  - The muxes scale linearly from the 16-b `mux_2_to_1` row: the 64-b `psum_in_mux` has
    exactly 4x the area and leakage of the 16-b muxes. A missing `width` in a mux query
    would scale it by 1/16.
- **14. The ifmap spad sizing didn't match the paper** (fixed 2026-10-28).
  - The ifmap spad check is now `q * S <= 12`, as Sec. IV-A sizes the spad
    ([performance_shared.py:144](common/performance_shared.py#L144)). Every Table III layer
    passes (11 / 10 / 12 / 9 / 9).
  - `mult_gate_mux` (the RTL's zero-ifmap product gate, pe.v:299) stays, even though Fig. 12
    doesn't draw it. The RTL holds the operand registers on a zero ifmap, so the multiplier
    still gives the old product, and the mux zeroes it. Fig. 12 doesn't draw the local
    network/GIN input mux (pe.v:132) either. It was removed briefly and then restored.
- **16. The RLC encoder and ReLU were charged once per DRAM write transfer** (fixed
  2026-10-28). `dram_write` now charges only the DRAM. Two new zero-cycle events follow
  the RTL ([performance_shared.py:260](common/performance_shared.py#L260)): `ofmap_relu`
  charges ReLU once per 64-b pack of four ofmap values (`relu.v`, `i_data`), and
  `rlc_encode` charges the encoder once per 16-b ofmap value (`rlc_encoder.v`,
  `i_ofmap_data`). The counts went from 0.36M each to 0.65M (ReLU) and 2.60M (encoder).
  Codec dynamic power rose from 0.08 to 0.13 mW. Latency, GLB and DRAM are unchanged.
- **17. Ifmap loads and psum accumulations took no PE cycles** (fixed 2026-10-28). In
  `pe_ctrl.v`, S_LD_IFM and S_ACC are states of their own: each does one op per cycle, and
  the PE issues no MAC during either. Fig. 12 also shares the one adder between the MACs
  and the input psums. `pe_ifmap_load` and `pe_psum_acc` now take one cycle of their PE.
  A conv-level `factor` of 1 / used_pe spreads those cycles over the active PEs
  ([fg/performance.py](fg/performance.py)). That adds (p + q·U) / (p·q·S) of the MAC
  cycles: 11 / 11 / 10 / 13 / 13%.
  - Processing latency went from 15.1 / 34.7 / 20.3 / 15.2 / 10.1 ms (95.4 ms in total)
    to 16.6 / 38.5 / 22.3 / 17.1 / 11.4 ms (106.0 ms). Table V gives 16.5 / 39.2 / 21.8 /
    16.0 / 10.0 ms (103.5 ms). Total latency is now 120.1 ms against 115.3 ms.
  - CONV4 and CONV5 are now 7% and 14% high. In Table V, CONV5 takes only 4.4% longer than
    its MACs alone (9.6 ms on 156 PEs). So at least in CONV5, the chip must overlap part
    of these cycles with the MACs. The RTL doesn't.
  - Not modeled: a one-cycle bubble after each ifmap load, while the last spad write is
    still pending (`r_ifm_wr_v`). That would add about 0.6% more.
  - cg is unchanged: its PE events have no ifmap load or psum accumulation to carry the
    cycles.
  - Energy and access counts are unchanged. Power falls with the longer runtime.
- **18. The GIN Y-buses left out the col tag** (fixed 2026-10-28). A GIN Y-bus carries
  the col tag to the X-buses along with the data (Fig. 10). The Y-bus PnR rows were built
  at data width only. Their queries now add the col ID width from Sec. V-B1: 16 + 5 b for
  ifmap, and 64 + 4 b for filter and psum
  ([description_shared.py](common/description_shared.py)). The interface scales linearly
  from the data-width row, so those Y-buses cost x1.31 and x1.06. CONV1 NoC dynamic power
  rose 9%. The GON Y-bus carries only psum data, so it's unchanged. Transfer counts are
  unchanged too: the model takes each network's data width from its X-bus.
- **ifmap spad charged on every MAC.** fg now charges the ifmap spad and zero buffer once
  per map_p MACs through `pe_ifmap_read`.
- **15. The FIFO classes depended on stale CSV tables** (fixed 2026-10-28). The
  descriptions now match the new interface in [interface/](interface), and both
  variants run again.
  - Class renames: `fifo_pack*` became `pe_*_fifo`, and `pe_ifmap_spad`,
    `pe_psum_spad` and `pe_zero_buffer` became `ifmap_spad`, `psum_spad` and
    `zero_buffer`.
  - Query parameters now match the synthesized ones. The PE FIFOs and RLC codec have 4
    entries, and `pe_filt_fifo` has a 16-b read port. The interface scales a module by
    query/row for each parameter column, so a missing or wrong key would quietly rescale
    it rather than fail.
  - The register-file spads have separate read and write power reports. The fg events
    charge each access with its action. A MAC or an accumulated psum reads and writes the
    psum spad, which takes two edges, so that's two conv-level events,
    `pe_psum_spad_read` and `pe_psum_spad_write`.
- **PnR power charged at 2x.** The reports were run at 2.5 ns but labeled 200 MHz, which
  doubled the energy. The spot-checked reports (multiplier, psum spad, pe_control, psum
  output FIFO, filter X-bus) now use a 5 ns clock.

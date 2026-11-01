# sc_cnn_dse model issues

Known gaps between this model and the U-SFQ paper (Gonzalez-Guerrero et al., ACM JETC
2024, [sc_cnn.pdf](sc_cnn.pdf)). Model numbers come from [results/dse.csv](results/dse.csv)
and [fig9.py](fig9.py), with layer tables from [scale_sim_conv/](scale_sim_conv/). Paper
Fig. 9 numbers are exact bar heights, read from the vector shapes in the PDF.

Out of scope by decision: memory, data loading and bandwidth. Only array utilization
pulls throughput below peak.

Last checked: 2026-10-31.

## What matches

- 64x64x32 peak throughput: 409.6 TMAC/s, against Table 3's 409, both at 5 bits.
- 64x64x32 area: 15.3M JJ, against Table 3's 15.4M.
- Fig. 9: 13 of the 15 bars are within 2% of the paper, now that issues 1 and 2 are
  fixed. The other two are issue 3.

## Differences from the paper

Fig. 9 bar totals, in millions of epochs. Each cell is 8x8x512 / 16x16x512 / 64x64x32.

| | Paper | Ours |
|---|---|---|
| VGG16 | 1.505 / 0.399 / 0.499 | 1.504 / 0.398 / 0.499 |
| ResNet18 | 0.264 / 0.087 / 0.345 | **0.335** / 0.087 / 0.345 |
| ResNet50 | 0.310 / 0.123 / 0.648 | **0.449** / 0.125 / 0.647 |
| GoogLeNet | 0.266 / 0.078 / 0.186 | 0.266 / 0.077 / 0.185 |
| AlexNet | 0.099 / 0.025 / 0.118 | 0.098 / 0.025 / 0.117 |

**1. Fixed: the layer tables now come from SCALE-Sim, as the paper's do.**
[scale_sim_conv/convert.py](scale_sim_conv/convert.py) reads SCALE-Sim's topology tables
at commit 9f98c43. It drops FC rows and the ResNet 1x1 shortcut convs, as the paper does.
SCALE-Sim has no VGG16, so VGG16 comes from torchvision's graph in [hf/](hf/). That graph
matches all 39 of the paper's VGG16 segments, once issue 2's padding is applied.
Compared with the old torchvision tables for the other four:
- AlexNet has the original 96-256-384-384-256 channels.
- GoogLeNet has real 5x5 convs.
- ResNet50 is v1, with the stride on the first 1x1 conv of each downsampling block.

**2. Fixed: we now tile the padded size, as the paper does.**
[performance.py:9](performance.py#L9) tiles R = (H_in + 2·pad) / stride, with
pad = K // 2, instead of the output size. For VGG16 conv1 that's 226, not 224. MAC counts
still use the true output size. This overcounts tiles relative to real hardware, since
padding adds no outputs, but it's the paper's rule. It only matters when the padding
crosses a tile boundary, which happens mostly on 8x8 and 16x16 arrays.

**3. The paper's ResNet bars at 8x8x512 look wrong.**
They are 21% (ResNet18) and 31% (ResNet50) below ours. The same layers match on 16x16x512
and 64x64x32, and no ceil rule fits all three. This is likely a data error in the paper.

**4. Fixed: `bitwidth` now means the paper's b.**
[performance.py:35](performance.py#L35) uses 2^(b−1) time slots per epoch, the paper's
t_max formula (Sec. 3.1.1, p. 9), and `bitwidth=5`. The sign bit rides on a separate pulse
stream, so it costs no time. The paper's prose isn't consistent: the Fig. 3 and Fig. 6
examples and Sec. 3.2.2 count 2^b slots. But every number in it uses 2^(b−1): Fig. 5(a),
Fig. 8(a), the PE counts in Sec. 3.2.1, and Tables 3 and 4. `dse.csv` is unchanged, since
we used to run 2^b with b = 4, which is also 16 slots.

**5. Power at 64x64x32: 7.0 W against Table 4's 3.9 W.**
Table 4 matches PE leakage alone: 131K PEs × 0.03 mW. Our total adds splitter and PTL
leakage (3.1 W), whose area the paper does count in Table 3's 15.4M JJ.

## Open issues (model)

**6. Dynamic energy is about 63× too high.**
- csv_sc computes mW/GHz, which is pJ, but labels it nJ: 1000× too high
  ([csv_sc.py:68](../../../../src/archx/interface/csv_sc/csv_sc.py#L68)).
- One mac event is counted per MAC instead of per cycle: 16× too low
  ([performance.py:43](performance.py#L43)).

The interface is shared with sc_cnn and sc_fir, so it isn't fixed here. Power is mostly
leakage (median dynamic share 1%), so the power totals barely move.

**7. No accumulator reset, bias or read phases.**
Fig. 6 shows these around each run of accumulation epochs. Neither the paper's epoch
counts nor ours include them.

**8. The paper's array has two modes; ours is one fixed shape.**
Sec. 3.3.2 builds 131K PEs from 16 subarrays that run as 64x64x32 (mode A) or 16x16x512
(mode B). Our DSE and [figures/pareto_area_throughput.png](figures/pareto_area_throughput.png) only model fixed shapes.
Assuming each layer runs in its better mode, the paper's array gets 91–314 TMAC/s,
1.1–1.6× our best fixed shape at the same area. Fixed 64x64x32 gets 16–96.

**9. No bandwidth roof.**
Sec. 3.2.3 says TM beyond 32 hits a 300 GB/s HBM roof. We model no memory, so the frontier's
TM = 128–256 shapes may be optimistic.

## Open decisions

- `figures/<model>/*.pdf`: plot.py saved 15 PDF copies that weren't requested. Keep or delete them?

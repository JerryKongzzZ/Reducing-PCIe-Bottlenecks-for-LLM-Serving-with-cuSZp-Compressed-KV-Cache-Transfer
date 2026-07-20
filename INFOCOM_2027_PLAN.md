# INFOCOM 2027 冲刺计划

## 2026-07-18 历史状态（最新结论见文末）

- Gate A：基本完成。数据抽样、计时口径、round-trip、error-bound 与结果
  schema 已修正；测试 21/21 通过。Qwen2.5-0.5B 已用保守范围和显式
  attention mask 重跑，并完成 14/24 层的联合验证；仍需扩充校准语料。
- Gate B：已达到单模型原型门槛。Qwen2.5-0.5B 在 vLLM 0.23 的真实
  GPU→CPU 和 CPU→GPU offload 路径已跑通，raw/cuSZp/INT8 使用同一协议。
- Gate C：静态全路径前置门槛已通过，自适应策略门槛仍未通过。最新 BF16
  single-grid fixed path 在 Qwen2.5-1.5B、4K、并发 8 的五次公平异步配对中，
  GPU→CPU 相对 `async_raw` 为 -26.322±8.631 ms，initial E2E 为
  -188.285±22.608 ms，且质量一致。deadline-pressure controller 机制已接入，
  但仍需在同一 trace 上同时优于 `async_raw` 和这个新 static baseline。
  本节后续负结果保留为演进记录；文末 2026-07-19 milestone 覆盖其当前结论。
- 五次预热独立进程结果进一步确认：static cuSZp 约 2.054x，但 initial E2E
  为 115.43±25.37 ms，raw 为 96.84±14.14 ms；当前只能主张减少字节，不能
  主张端到端加速。优化 store 路径和扩大 transfer batch 是当前首要问题。
- 真实可变长度并发 probe 已产生 GREEN→YELLOW→RED，压力从 0.351 增至
  1.405，bound 从 raw 切到 1e-4 和 1e-3，且 6/6 输出一致。Gate C 的
  “机制有效”已满足，“相对 static 有性能收益”仍未满足。
- 生产 baseline 已纠正为 stock vLLM 异步 CPU offloader。同批五次实验中，
  stock、同步 cuSZp、异步 cuSZp 的 initial E2E 分别为 62.27±3.71、
  100.44±5.31、82.96±1.67 ms。异步 store 相对同步实现降低 17.4%，但仍比
  stock 高 33.2%，所以尚不能主张端到端优势。下一硬门槛是消除 cuSZp
  wrapper 内的 host/device 同步并证明压缩能与调度或 PCIe 传输重叠。
- cuSZp wrapper 已合并 min/max reduction 并删除外层重复同步；GPU→CPU handler
  从 29.40 降至约 27.85 ms，但端到端置信区间重叠，不能写成显著加速。
- zstd/LZ4 已作为真实 connector 的异步无损 baseline 接入。同批五次结果为：
  stock 61.24±2.78 ms；LZ4 74.33±1.84 ms、1.000x；zstd 75.10±1.39 ms、
  1.239x；cuSZp 82.60±4.06 ms、2.054x。cuSZp 当前优势是字节削减，不是
  512-token 小负载延迟。
- 约 1.7K-token 的 2K 配置也已完成五次独立进程对比。质量匹配的 cuSZp
  `1e-5` 为 1.441x、168.31±4.99 ms；zstd 为 1.239x、136.84±3.71 ms；
  LZ4 为 1.000x、129.54±2.86 ms；stock 为 82.88±2.18 ms，四种方法均
  保持 6/6 输出一致。长上下文本身尚未抵消压缩路径开销，不能把 2K 结果
  写成加速；它将 wrapper/分段融合确认为当前系统瓶颈。
- Gate D：未达到。常见无损算法原型对比已完成，但主实验表、正式质量任务、
  4K 与多模型、消融和论文正文仍是主要剩余工作。

当前最高优先级不是继续调大 error bound，而是依次完成：改造 cuSZp 库内部
的每调用分配、同步长度读取和释放；合并 adaptive 的 layer gather/scatter 与
raw/cuSZp 分段调用；扩大上下文、并发和 transfer batch；扩充
校准语料并重新标定 layer sensitivity；在长上下文上重复比较 stock、INT8、
zstd/LZ4、static cuSZp 和 adaptive cuSZp；最后才生成论文主图表。

## 目标定位

把原毕设升级为一个端到端系统：在 GPU 上对即将 offload 的 KV-cache block 进行 cuSZp 压缩，只传输压缩后的 payload；根据 PCIe 待传输量和离线测得的层敏感度，在线选择 error bound；在恢复 block 时先传输压缩 payload，再在 GPU 上解压并恢复原始 KV dtype。

暂定贡献：

1. 面向单机 GPU–CPU KV offload 的 error-bounded GPU compression data path。
2. 联合 PCIe pressure 与 layer sensitivity 的动态 error-bound controller。
3. 在真实 vLLM 请求上验证 latency/throughput/quality trade-off，而不是只报告 tensor microbenchmark。

## 投稿硬门槛

- 所有论文图必须能追溯到 raw JSON/CSV 和运行命令，禁止模拟或手填数据。
- 无压缩 round-trip 输出必须与原 vLLM 一致；有损路径必须报告质量变化。
- Static 与 adaptive 必须使用相同 block 粒度、相同计时边界和相同负载。
- 至少提供 baseline、INT8、通用无损压缩、static cuSZp、adaptive cuSZp 五组结果。
- 若 7 月 23 日前没有真实端到端结果，不把 microbenchmark 描述成完整系统结果。

## 时间表

### 7 月 18–19 日：正确性与实验基础

- 禁用模拟 Pareto、queue 和 ablation 图。
- 修复 DynamicCache 被 baseline forward 原地修改的问题。
- 修复 INT8 与 cuSZp 的不公平计时。
- 输出实验环境、版本、随机种子、trial-level 原始结果。
- 使用完整 K/V、多层数据替代“第 0 层 key 重复填充”。
- 增加 round-trip、error-bound、scheduler 和结果 schema 测试。

### 7 月 19–22 日：vLLM 0.23 端到端路径

- 新建自定义 `CompressedCPUOffloadingSpec`，通过 vLLM 的 `spec_module_path` 加载，不修改 site-packages。
- GPU→CPU：读取真实 KV block，转换为 cuSZp 支持的 FP32，GPU 压缩，仅拷贝有效 compressed bytes 到 pinned CPU memory。
- CPU→GPU：拷贝 compressed bytes，GPU 解压，转换回原 KV dtype，并用 CUDA event 保证消费前完成。
- 记录每个 job 的原始字节、压缩字节、压缩/传输/解压时间和 queue depth。
- 验证 block ID、layer/tensor mapping、并发请求和失败回退路径。

### 7 月 22–24 日：adaptive controller 与摘要

- 用修复后的 sensitivity sweep 为每层生成 high/medium/low sensitivity。
- GREEN/YELLOW/RED 阈值从实测 PCIe service rate 和 pending bytes 推导，不使用任意常数。
- controller 输出每次选择的 error bound、原因和状态变化 trace。
- 对 static 和 adaptive 使用完全相同的 block-wise pipeline。
- 7 月 24 日前注册摘要；摘要只写已经实现或可以在全文截止前验证的贡献。

### 7 月 24–28 日：主实验

- 模型：至少 GPT-2/OPT 用于调试，Qwen2.5-0.5B、Qwen2.5-1.5B、TinyLlama-1.1B 用于主结果；显存允许时增加更大模型。
- 上下文：512、2K、4K；至少三种并发或到达率。
- 负载：稳定、突发、内存压力三类。
- 指标：compression ratio、max/mean error、P50/P95/P99 TTFT、TPOT、tokens/s、PCIe bytes、GPU/CPU memory、SLO attainment。
- 质量：perplexity 加至少一个 LongBench/RULER/GSM8K 类任务。
- 每个配置至少 5 次 trial，保存均值、标准差和 95% CI。

### 7 月 27–29 日：对比和消融

- Uncompressed vLLM。
- FP16/BF16 原始传输。
- INT8 对称量化；若硬件/框架可用，再加入 FP8。
- zlib 作为通用 CPU 无损基线；优先补充 zstd/lz4 中至少一个常见快速无损基线。
- static cuSZp：多组固定 error bound。
- adaptive cuSZp：完整 controller。
- 消融：去掉 queue signal、去掉 layer sensitivity、去掉异步/overlap、不同阈值。

### 7 月 28–30 日：论文

- Introduction：PCIe-local KV offload 问题、现有方法不足、三项贡献。
- Related Work：CacheGen、KVComp、KVServe、KIVI/ZipCache 等。
- Design：data path、metadata、controller、正确性与失败回退。
- Evaluation：环境、问题列表、主结果、质量、消融、开销。
- Limitations：FP32 转换成本、硬件依赖、模型规模和 cuSZp dtype 限制。
- 所有结论只能引用自动生成的表格和图。

### 7 月 30–31 日：投稿检查

- 匿名化、页数、字体、引用、图中文字和 PDF 可读性。
- 从干净环境执行最小复现流程。
- 核对每个摘要/结论数字与 raw result。
- 预留至少 12 小时上传和 PDF 检查。

## 决策门槛

- Gate A（7 月 20 日）：压缩 round-trip、sensitivity 和公平 microbenchmark 全部正确。
- Gate B（7 月 23 日）：真实 vLLM offload 至少在一个模型上跑通并产生端到端结果。
- Gate C（7 月 27 日）：adaptive 在至少一种真实压力负载下优于 static，同时质量损失可控。
- Gate D（7 月 29 日）：主表、质量表、消融和相关工作完整。

如果 Gate C 未通过，论文应把贡献收缩为 static PCIe compression system，不虚构 adaptive 优势；如果 Gate B 未通过，则当前工作只能作为 microbenchmark/prototype，不能声称是端到端 vLLM 系统。

## 2026-07-18 RTX 5080 / Qwen2.5-1.5B latest update

- A real 4K, concurrency-eight workload now restores about 64.8 MB per event.
  Stage profiling confirms that per-page synchronization limited H2D efficiency.
- Batched pinned-memory restore reduces raw CPU-to-GPU handler time from
  16.964 to 10.848 ms (-36.1%) and cuSZp from 43.916 to 31.907 ms (-27.3%).
- Weighted H2D throughput is 157.17 Gbit/s for raw and 154.12 Gbit/s for
  cuSZp, while cuSZp decompression is only 24.65 Gbit/s. The current bottleneck
  is the decoder, not PCIe.
- cuSZp 1e-5 reduces restore bytes from 64.82 to 41.48 MB (1.563x), but replay
  E2E increases from 355.16 to 434.55 ms. This is not a latency-win result.
- Stock/raw itself reproduces only 7/8 Qwen2.5-1.5B replay sequences. This
  workload is valid for performance diagnosis, not codec-quality attribution.
- A real two-trial Qwen2.5-0.5B cost-aware probe reaches GREEN/YELLOW/RED, but
  the RTX 5080-calibrated gate falls back to raw for every job and preserves
  6/6 exact replay. This verifies no-regret fallback, not adaptive speedup.
- The complete test suite now passes 25/25 tests. H2D and decompression
  throughput use total bytes divided by total stage time.

The updated paper direction is **cost-aware, no-regret adaptive KV transfer**,
not always-on compression. Next priorities are a 1.5B-specific sensitivity
profile, controlled real PCIe contention, at least five trials per main-table
configuration, formal quality tasks, P95/P99/SLO metrics, and a substantially
faster or replaced cuSZp decoder.

### Five-trial common-codec milestone

The 1.5B/4K/concurrency-eight run now has five isolated trials for stock, raw,
cuSZp 1e-5, INT8, zstd, and LZ4. Stock remains the production baseline.
cuSZp reduces restore bytes by 36.0%, but its 37.929 +/- 0.698 ms CPU-to-GPU
handler is slower than raw at 9.673 +/- 0.422 ms. INT8 restores in
15.275 +/- 0.477 ms but reproduces only 3/8 sequences. zstd and LZ4 are slower
because CPU decode dominates. Gate D still requires formal quality, controlled
PCIe contention, adaptive/static comparison on the same trace, and paper
figures generated from the canonical aggregate.

### Gate C joint-mode milestone

- Gate C now jointly selects raw/cuSZp, a model-capped per-layer error bound,
  and cuSZp's existing plain/fixed/outlier mode from measured restore costs.
- In Qwen2.5-1.5B 4K/concurrency-eight calibration at 1e-5, fixed improves
  compression ratio from plain's 1.563x to 1.615x and reduces profiled restore
  from 36.27 to 29.51 ms; outlier restores in 41.14 ms.
- At the measured 154 Gbit/s H2D rate, two actual-link Gate C trials correctly
  choose raw for every RED-pressure event: 1.000x bytes, 8.04 ms profiled
  restore, zero decode time, and the same 7/8 replay behavior as raw/reference.
- A Qwen2.5-1.5B individual profile and joint 1e-3 debug profile now exist.
  The joint profile was calibrated on one short text and conflicts with the poor
  uniform-1e-3 replay result, so it is not yet paper-quality robustness evidence.
- Gate C therefore remains not passed under the original positive-performance
  definition. The defensible result is a no-regret fallback plus a better cuSZp
  mode, not an adaptive latency win over raw.

See `data/GATE_C_REPORT.md` for paths, numbers, and remaining evidence.

## 2026-07-19 batched cuSZp restore milestone

This milestone keeps cuSZp's fixed encoding and error-bound semantics
unchanged. It is a systems optimization, not a new compression algorithm:
persistent metadata workspaces remove per-page allocation, restore pages are
launched as one batch, the fixed decoder writes BF16 directly, and BF16 pairs
are stored vectorially. The RTX 5080 build embeds compute-90 PTX because CUDA
12.0 cannot name the GPU's native Blackwell architecture.

On Qwen2.5-1.5B, 4K, concurrency eight, fixed cuSZp at 1e-5:

- compression ratio is 1.61445x and output agreement remains the same as raw
  (7/8 exact and token match);
- in the five-trial profiled run, mean CPU-to-GPU handler time falls from
  9.125 +/- 0.287 ms for raw to 8.882 +/- 0.485 ms for cuSZp;
- profiled restore falls from 8.481 +/- 0.289 to 8.016 +/- 0.487 ms;
- compressed H2D is 1.991 ms versus raw's 2.958 ms, while the new decoder
  costs 0.554 ms and reaches about 947 Gbit/s;
- without profiling, handler time is 4.550 +/- 0.223 ms for raw and
  4.282 +/- 0.603 ms for cuSZp. Replay E2E is 321.44 +/- 10.46 ms for raw
  and 320.13 +/- 16.20 ms for cuSZp.

This is the first mean restore result below raw, so the engineering break-even
gate is passed for this batch shape. It is not yet a paper-level speedup:
confidence intervals overlap, replay E2E is effectively tied, and initial E2E
still averages 1254.46 ms for raw versus 1368.47 ms for cuSZp because
compression/store preprocessing remains on the initial request path.

Concurrency sixteen is also a negative result: handler time is 4.131 ms for
raw and 4.206 ms for cuSZp, while replay E2E is 613.71 versus 617.57 ms.
Nominal concurrency alone does not create a win; restore page count and
fragmentation determine whether the fused decoder amortizes its launch cost.

The old five-trial data ran methods in groups. The repeated runner now defaults
to interleaved method order within every trial to reduce thermal and temporal
drift. All final tables must be rerun with that order. The next hard gate is:
(1) interleaved five-trial reproduction, (2) real-KV batch-size break-even
curves, (3) remove or overlap initial store preprocessing, and only then
proceed to the cross-model Gate D table.

## 2026-07-19 fair-direct Gate C update

The restore engineering gate now passes on Qwen2.5-1.5B, 4K, concurrency
eight. Raw and cuSZp both write directly into final KV pages; compressed pages
share one pinned host slab and one GPU payload allocation. Fixed cuSZp decodes
the batch in one kernel directly to BF16 destinations. The redundant
Python-side current-stream synchronization has been removed after verifying
that H2D, metadata copies, and decoder launches use the current CUDA stream.
The full suite passes 29/29 tests.

Five interleaved trials establish three operating regions:

1. No synthetic contention: completed profiled restore is
   1.610 ms raw versus 1.478 ms cuSZp (paired -0.133 +/- 0.057 ms). In normal
   execution the handler is 1.561 versus 1.106 ms
   (paired -0.455 +/- 0.346 ms, cuSZp wins 5/5).
2. Medium contention: profiled restore is 2.131 versus 1.671 ms
   (paired -0.461 +/- 0.149 ms), but the profiled handler is statistically tied
   at paired +0.038 +/- 0.157 ms.
3. High contention: profiled restore is 3.177 versus 2.226 ms
   (paired -0.950 +/- 0.293 ms), handler is 3.384 versus 2.918 ms
   (paired -0.466 +/- 0.290 ms), and replay E2E is 418.252 versus 348.168 ms
   (paired -70.084 +/- 24.777 ms).

All configurations preserve the same 7/8 output agreement as raw. Without
contention replay E2E is tied, while initial E2E remains worse for cuSZp
(paired +184.114 +/- 86.026 ms). Static CPU-to-GPU restore break-even is
solved, but the full INFOCOM gate is not. Next optimize GPU-to-CPU
compression/store, reproduce across all planned models, compare INT8/zstd/LZ4,
replace synthetic contention with real concurrent pressure for the main table,
and demonstrate that cost-aware adaptive raw/cuSZp plus error-bound selection
beats both always-raw and always-static-cuSZp on the same trace.

## 2026-07-19 GPU-to-CPU store update

The per-segment GPU-to-pageable-CPU, pin-memory, and packed-slab copies have
been replaced by one direct D2H batch into the final pinned slab. Fixed cuSZp
compression now reuses persistent offset/flag workspace rather than allocating
and freeing three metadata buffers per page. The rebuilt native extension and
29/29 tests pass, including pinned CPU payload and round-trip checks.

In two-trial probes, direct pinned storage measures 98.658 ms raw versus
138.665 ms cuSZp GPU-to-CPU handler time. Persistent fixed workspace further
reduces cuSZp to 127.914 ms while raw is 100.174 ms. Initial E2E remains slower
for cuSZp (1112.142 versus 744.753 ms), so the full pipeline gate is still
open. The earlier five-trial initial-E2E table used the old store
implementation and must not be mixed with these results.

Next implementation task: batch same-shaped fixed-mode page compression so
per-page relative-range reductions and compressed-size D2H reads are submitted
once per job, then rerun five interleaved trials before expanding models.

## 2026-07-19 BF16 single-grid GPU-to-CPU milestone

The compression-side hard gate is now passed on the Qwen2.5-1.5B, 4K,
concurrency-eight workload. The production path no longer converts every BF16
page to a temporary FP32 tensor or launches per-page min/max and compression
kernels. It performs one batched BF16 range reduction, one cuSZp-compatible
fixed compression grid, one packed D2H publication, and reuses a grow-only GPU
output slab. Per-page actual absolute bounds and compressed sizes are returned
with one stream synchronization. A regression test asserts that this BF16 API
is really used; the full suite passes 38/38 tests.

The experiment protocol was also corrected: asynchronous cuSZp must be paired
with the new `async_raw` baseline. Five interleaved trials report:

- GPU-to-CPU handler: 72.892 ms raw versus 46.570 ms cuSZp,
  paired -26.322 +/- 8.631 ms, cuSZp wins 5/5;
- CPU-to-GPU handler: 1.565 versus 0.970 ms,
  paired -0.596 +/- 0.141 ms, cuSZp wins 5/5;
- initial E2E: 932.269 versus 743.984 ms,
  paired -188.285 +/- 22.608 ms, cuSZp wins 5/5;
- replay E2E: paired -21.728 +/- 30.908 ms, favorable but not significant;
- compression ratio 1.61445x and identical 0.875 token/exact agreement.

This supersedes the negative GPU-to-CPU conclusion in the preceding historical
sections. The static full-datapath prerequisite is passed; the original
adaptive Gate C is still open. The next task is to rerun adaptive policy traces
against both `async_raw` and this improved static BF16 cuSZp path, then expand
the same fair protocol to Gate D models and common codecs.

Canonical aggregate:
`data/vllm_qwen1.5b_4k_concurrency8_async_fair_bf16_single_grid_reduction_5trial/aggregate.json`.

## 2026-07-20 adaptive indexed-layer fast-path milestone

Adaptive fixed-mode segments now use cuSZp-compatible indexed BF16 kernels.
Compression reads selected layers directly from the flattened KV page, while
restore writes decoded values back through the same layer map. This removes
the temporary GPU gather and decoded temporary/scatter paths without changing
the fixed bitstream or defining a new codec. Unsupported layouts fall back to
the previous safe paths. The maintained suite passes 38/38 tests.

Two RED-state engineering trials reduced adaptive GPU-to-CPU from 83.231 ms to
54.219 ms and CPU-to-GPU from 4.151 ms to 2.842 ms, at 2.81565x compression.
This does not pass Gate C: the debug 27/28-layer 1e-3 profile produced only
0.28125 token match and 0.25 exact match. The 2 ms deadline is a controller
probe, not evidence of real PCIe congestion. Before any five-trial adaptive
claim, rebuild sensitivity from multiple prompts/tasks and reject a profile
unless it passes the complete 4K concurrency-eight workload quality gate.

Engineering aggregate:
`data/vllm_qwen1.5b_4k_concurrency8_adaptive_indexed_bidir_red_probe_2trial/aggregate.json`.

NaN

NaN
NaN
NaN
NaN
NaN

NaN
NaN
NaN
NaN
NaN
NaN

NaN
NaN
NaN
NaN
NaN

NaN
NaN

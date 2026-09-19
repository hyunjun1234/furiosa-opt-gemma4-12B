//! `ops::sliding_attention_output`: output19's projection and hop, with a RE-SHAPED TAIL.
//!
//! output19's tail after the hop reload is four serial Main/Sub commands whose total the device
//! trace puts at 4,445 cycles with the DMA idle:
//!   sum of squares (1,063) -> ring-16 reduce + eps + sqrt (661) -> `to_vrf` of the rms (439)
//!   -> final pass `z * s / rms * w + r` (2,276).
//! The last one is by far the biggest, and the only thing it does that the (same-length) sum of
//! squares pass does not is the **`DivF` by the rms**. So this module removes the division from the
//! final pass:
//!   * the rms pass computes 1 / rms instead of rms. `t = ms + eps` is stashed before the square
//!     root, and the FpDiv stage then divides the root by the stash: `sqrt(t) / t = 1 / sqrt(t)`.
//!     The stage order is Fp -> IntraSliceReduce -> FpDiv, so the root (Fp, FpFpu) and the division
//!     (a separate stage) live in ONE pass. That pass writes eight values per slice, so whatever the
//!     divider costs per element it costs it 8 times instead of 240 times.
//!   * the final pass then needs THREE multiplies (s, 1/rms, w) and the vector engine has THREE
//!     multiply ALUs -- `FpMul0`, `FpMul1` and `FpFma` -- so all three ride one pass. The module
//!     used to precompute `sw = s * w` because its author believed there were only two; that pass
//!     and the `to_vrf` that read it back are gone.
//! The sum of squares pass still needs `s` on its own, so the channel-scale VRF stays.

use furiosa_opt_std::prelude::*;

use crate::axes::{Ds, Gs, H, Ns, Qs};
use crate::device::layout::{OutputClusters, SlidingOutputColumns, SlidingOutputRows};
use crate::{Chip, EPS};

axes![Lq = 2, Vr = 16, Zp = 3];
/// The tail runs on cluster 0 only (the cluster that stores): the final sync is then the cheap kind.
type Vc = m![1 # 2];

const H_F32: f32 = H::SIZE as f32;
const X_SCALE: f32 = 16.0;
/// z stays 16 times too large to the end; the RMSNorm divides that out, given a matching epsilon.
const EPS_SCALED: f32 = EPS * X_SCALE * X_SCALE;

type Levels = DmTensor<f8e4m3, Chip, OutputClusters, SlidingOutputColumns, m![Lq, Qs % 256]>;

pub(crate) type Rows = SlidingOutputRows;
/// The tail's home: slices 0..16 of both clusters (replicated), slice g holding rows 240g..240g+240.
pub(crate) type Tail = m![1 # 16, H / 240];
pub(crate) type TailDm<D> = DmTensor<D, Chip, Vc, Tail, m![H % 240]>;
pub(crate) type TailVrf = VrfTensor<f32, Chip, Vc, Tail, m![H % 240]>;

/// The three per-row operands of the tail (residual, channel scale, RMSNorm weight) share ONE DM
/// tensor. The point is the allocator: the pool is born with the FIRST load, which happens while the
/// projection `z` is still alive, so the pool cannot be placed on z's address (that is what used to
/// draw a second Core wait, `DramReuse`).
pub(crate) type Pool = DmTensor<bf16, Chip, Vc, Tail, m![Zp, H % 240]>;
pub(crate) type PoolVrf = TailVrf;

/// Loads `v` into slot `slot` of the pool. The pool's slots are written in program order (one DM
/// tensor = one version chain), so the caller lists them in the order it wants them issued.
pub(crate) fn pool_load(ctx: &mut Context, pool: &mut Pool, v: &HbmTensor<bf16, Chip, m![H]>, slot: usize) {
    v.view()
        .to_dm_view(&mut ctx.tdma, pool.view_mut().tile::<m![Zp], 1, m![Zp = 1 #{!} 3, H % 240]>(slot));
}

/// Slot `slot` of the pool in the VRF, ready as a per-row operand of the tail's passes. The
/// length-one slot axis is dropped in the collect, so the passes see exactly the operand shape they
/// saw when every operand had its own tensor.
pub(crate) fn pool_vrf(ctx: &mut Context, pool: &Pool, slot: usize) -> PoolVrf {
    ctx.sub
        .begin(pool.view().tile::<m![Zp], 1, m![Zp = 1 # 3, H % 240]>(slot))
        .fetch::<m![Zp = 1], m![H % 240]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .to_vrf()
}

pub(crate) type XTrf = TrfTensor<f8e4m3, Chip, OutputClusters, SlidingOutputColumns, m![Lq], m![Qs % 256]>;
pub(crate) type Z = DmTensor<bf16, Chip, OutputClusters, Rows, m![H % 120]>;

pub(crate) type WeightTileA = DmTensor<f8e4m3, Chip, OutputClusters, SlidingOutputColumns, m![H % 120 = 104, Qs % 256]>;
pub(crate) type WeightTileB = DmTensor<f8e4m3, Chip, OutputClusters, SlidingOutputColumns, m![H % 120 = 16, Qs % 256]>;

/// Rows `offset..offset + 104` of every row slice: the two levels summed inside the pass.
pub(crate) fn contract_tile_a(ctx: &mut Context, weight: &WeightTileA, x_trf: &XTrf, z: &mut Z, offset: usize) {
    ctx.main
        .begin(weight.view())
        .fetch::<m![H % 120 = 104, Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![H % 120 = 104, Qs / 32 % 8], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 104, Qs / 64 % 4], m![Qs % 64], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 104]>()
        .contract_lane::<m![H % 120 = 104, Lq], m![1 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<Lq, m![H % 120 = 104], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<Rows, m![H % 120 = 104]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 104 / 4], m![H % 120 = 104 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 104 % 4]>()
        .commit_view(z.view_mut().tile::<m![H % 120], 104, m![H % 120 = 104 #{!} 120]>(offset));
}

/// Rows `offset..offset + 16` of every row slice.
pub(crate) fn contract_tile_b(ctx: &mut Context, weight: &WeightTileB, x_trf: &XTrf, z: &mut Z, offset: usize) {
    ctx.main
        .begin(weight.view())
        .fetch::<m![H % 120 = 16, Qs / 32 % 8], m![Qs % 32]>()
        .collect::<m![H % 120 = 16, Qs / 32 % 8], m![Qs % 32]>()
        .contract_outer::<m![H % 120 = 16, Qs / 64 % 4], m![Qs % 64], _, _, _>(x_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 120 = 16]>()
        .contract_lane::<m![H % 120 = 16, Lq], m![1 # 8]>(LaneMode::Sequential)
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<Lq, m![H % 120 = 16], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_inter_slice_reduce::<Rows, m![H % 120 = 16]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![1 # 16]>()
        .transpose::<m![H % 120 = 16 / 4], m![H % 120 = 16 % 4 # 16]>()
        .commit_trim::<m![H % 120 = 16 % 4]>()
        .commit_view(z.view_mut().tile::<m![H % 120], 16, m![H % 120 = 16 #{!} 120]>(offset));
}

/// x as two exact f8 levels in the TRF: q0 = f8(16x), q1 = f8(16x - q0).
pub(crate) fn quantise_x(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, OutputClusters, SlidingOutputColumns, m![Qs % 256]>,
) -> XTrf {
    let mut levels: Levels = DmTensor::new();
    ctx.main
        .begin(x.view())
        .fetch::<m![1], m![Qs % 256]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), X_SCALE)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit_view(levels.view_mut().tile::<m![Lq], 1, m![Lq = 1 #{!} 2, Qs % 256]>(0));
    let q0_vrf: VrfTensor<f32, Chip, OutputClusters, SlidingOutputColumns, m![Lq = 1, Qs % 256]> = ctx
        .sub
        .begin(levels.view().tile::<m![Lq], 1, m![Lq = 1 # 2, Qs % 256]>(0))
        .fetch::<m![Lq = 1], m![Qs % 256]>()
        .fetch_cast::<f32>()
        .collect::<m![Lq = 1, Qs / 8 % 32], m![Qs % 8]>()
        .to_vrf();
    ctx.main
        .begin(x.view())
        .fetch::<m![1], m![Qs % 256]>()
        .fetch_cast::<f32>()
        .collect::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![Qs / 4 % 64], m![Qs % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), X_SCALE)
        .vector_fp_binary(FpBinaryOp::SubF, &q0_vrf)
        .vector_widen_concat::<m![Qs / 8 % 32], m![Qs % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Qs % 8 # 32]>()
        .commit_trim::<m![Qs % 8]>()
        .commit_view(levels.view_mut().tile::<m![Lq], 1, m![Lq = 1 #{!} 2, Qs % 256]>(1));
    ctx.sub
        .begin(levels.view())
        .fetch::<m![Lq], m![Qs % 256]>()
        .collect::<m![Lq, Qs / 32 % 8], m![Qs % 32]>()
        .to_trf()
}

pub(crate) fn project_normalize_add(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![Ns, Gs, Ds]>,
    weight: &HbmTensor<f8e4m3, Chip, m![H, Qs]>,
    weight_scale: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    residual_hbm: &mut HbmTensor<bf16, Chip, m![H]>,
) {
    // The weight in two tiles, 104 and 16 rows per slice: the big tile is contracted while the
    // small one is still on its way.
    let weight_a: WeightTileA = weight
        .view()
        .tile::<m![H % 120], 104, m![H / 120, H % 120 = 104 # 120, Qs]>(0)
        .to_dm(&mut ctx.tdma);
    let weight_b: WeightTileB = weight
        .view()
        .tile::<m![H % 120], 16, m![H / 120, H % 120 = 16 # 120, Qs]>(104)
        .to_dm(&mut ctx.tdma);

    // One DMA load hands every slice the 256 columns it contracts, once per row group.
    let x: DmTensor<bf16, Chip, OutputClusters, m![H / 120 % 16, Ns, Gs], m![Ds]> = x.to_dm(&mut ctx.tdma);
    let x: DmTensor<bf16, Chip, OutputClusters, SlidingOutputColumns, m![Qs % 256]> = unsafe { x.reshape() };
    let x_trf = quantise_x(ctx, &x);

    // 16 * (weight @ x), bf16, 120 rows per row slice.
    let mut z: Z = DmTensor::new();
    contract_tile_a(ctx, &weight_a, &x_trf, &mut z, 0);
    contract_tile_b(ctx, &weight_b, &x_trf, &mut z, 104);

    // The residual, loaded here so that the DMA has work while the small tile is contracted; it
    // opens the pool, which is what keeps the two loads behind the store off z's address.
    let mut pool: Pool = DmTensor::new();
    pool_load(ctx, &mut pool, residual_hbm, 0);
    // Each slot is read into the VRF right after its own load: a read of the pool must finish
    // before the next slot is written (one version chain), so the three `to_vrf` passes end up
    // spread over the DMA commands instead of queueing up behind the last load, where they would
    // all land in the tail after the reload.
    let residual = pool_vrf(ctx, &pool, 0);

    // The hop: all 3,840 projected values go to HBM in row order and come back, 240 consecutive
    // rows per slice, into slices 0..16.
    let mut hop: HbmTensor<bf16, Chip, m![H]> = HbmTensor::new();
    z.view().to_hbm_view(&mut ctx.tdma, hop.view_mut());

    // These two are what the model puts between the store and the reload: they cover the sync.
    pool_load(ctx, &mut pool, weight_scale, 1);
    let scale = pool_vrf(ctx, &pool, 1);
    pool_load(ctx, &mut pool, rms_weight, 2);
    let rms_weight = pool_vrf(ctx, &pool, 2);

    let z_tail: TailDm<bf16> = hop.to_dm(&mut ctx.tdma);

    // Sum of squares of the 240 scaled values of each slice, over H.
    let partial: DmTensor<f32, Chip, Vc, Tail, m![1 # 8]> = ctx
        .main
        .begin(z_tail.view())
        .fetch::<m![1], m![H % 240]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();

    // Ring of sixteen: the mean square in every slice, epsilon, and the RECIPROCAL of the root --
    // `t` is stashed before the root and the FpDiv stage divides by it: sqrt(t) / t = 1 / sqrt(t).
    // Eight values per slice, so the divider runs 8 times instead of 240 times.
    let inv_rms: DmTensor<f32, Chip, Vc, m![1 # 16, Vr], m![1 # 8]> = ctx
        .main
        .begin(partial.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 16, Vr], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, EPS_SCALED)
        .vector_stash()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_fp_div(Stash)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let inv_rms: DmTensor<f32, Chip, Vc, Tail, m![1 # 8]> = unsafe { inv_rms.reshape() };
    let inv_rms_vrf: VrfTensor<f32, Chip, Vc, Tail, m![1 # 8]> = ctx
        .sub
        .begin(inv_rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    let out: TailDm<bf16> = ctx
        .main
        .begin(z_tail.view())
        .fetch::<m![1], m![H % 240]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 30], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 60], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &scale)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Fma), &rms_weight)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &inv_rms_vrf)
        .vector_widen_concat::<m![H / 8 % 30], m![H % 8]>()
        .vector_clip(ClipBinaryOpF32::Add, &residual)
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();
    // `Vc` IS `m![1 # 2]`, so the reshape that used to stand here was a no-op that still cost a
    // cross-resource wait point in the schedule.
    out.view().to_hbm_view(&mut ctx.tdma, residual_hbm.view_mut());
}

//! Feed-forward, fourth layout (= `ffn6` with the block scales applied by the CONTRACTION ENGINE):
//! the block sums leave every block pass as bf16, the block scales are converted f8 -> bf16 by the
//! Cast Engine (no Vector Engine), and one short contraction pass per matrix / tile multiplies the
//! sums by the scales held in the TRF and accumulates over the blocks in the Time Reducer. No scale
//! VRF exists any more: nothing gates the next tile's decode table behind a running pass, the eight
//! Vector-Engine scale tiles are gone, and the down projection needs three tiles instead of four.

use furiosa_opt_std::prelude::*;

use super::mlp::UpGateClusters;
use super::rmsnorm::ReducingSlices;
use crate::axes::{Dummy8, H, L};
use crate::device::layout::Cluster;
use crate::{Chip, EPS};

// Columns inside the contraction: Ut / Dt count the 64-column packets of an up/gate row (3840
// columns) or of a down half-row (7680), Xp is the packet. Sb / Sd are the 16-column blocks of a
// row (240) or of a half-row (480): the block sums and the block scales are indexed by them.
axes![Rep8 = 8, Q4 = 4, Rep32 = 32, Ring8 = 8, T2 = 2, Xh = 3840, Rep4 = 4, Ring64 = 64, Xg = 120, Xd = 7680, Ut = 60, Dt = 120, Xp = 64, C2 = 2, Lead4 = 4, Hg = 128, Pw = 64, Sb = 240, Sd = 480, Slot = 2];

const H_F32: f32 = H::SIZE as f32;
const INVSQRT2: f32 = 0.70710678118f32;

/// Gain applied to the activation before it is split into f8 terms (a power of two, undone
/// exactly after the contraction).
const UP_GAIN: f32 = 128.0;
/// The GeGLU output is quantized BEFORE the up projection's global scale is applied (that scale
/// moved to the last pass), so it is ~1e4 times larger than the scaled value.
const DOWN_GAIN: f32 = 0.25;
/// The up and gate projections come out of their scale passes UP_GAIN times too large: the
/// gate's factor rides its global scale (with the erf's 1/sqrt(2)), the up's the GeGLU's divisor.
const GATE_ERF_FACTOR: f32 = INVSQRT2 / UP_GAIN;
const GEGLU_DIVISOR: f32 = 2.0 * UP_GAIN * INVSQRT2;

/// Eight real copies of eight 480-column pieces: only the first ring of every four holds data after
/// the load (`Loaded`); the passes run on all of them (`Pieces`: the other three compute on whatever
/// their memory holds and are overwritten by the Switch before anything reads them).
type Loaded = m![Rep8, 1 # 4, H / 480];
type Pieces = m![Rep8, Q4, H / 480];
type PiecesX = m![Rep8, Q4, Xh / 480];
/// The same 256 slices once each ring of eight has gathered the whole vector.
type Gathered = m![Rep8, Q4, Ring8];
/// One cluster's 7680 rows, 30 whole rows a slice.
pub(crate) type RowSlices = m![L / 30 % 256];

/// RMSNorm on the pieces (rounded to bf16 as the reference does), split into two exact f8 terms,
/// gathered into every slice and stored in the TRF as two lanes of 3840 columns.
fn normalize_quantize(
    ctx: &mut Context,
    x: &HbmTensor<bf16, Chip, m![H]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> (TrfTensor<f8e4m3, Chip, UpGateClusters, Gathered, m![1], m![T2, Ut, Xp]>, DmTensor<bf16, Chip, UpGateClusters, Pieces, m![Slot, H % 480]>) {
    // WARM-UP. The first Sub command of this kernel costs ~10x its model cycles (2,696 device for a
    // 267-cycle StoVrf) and the ISSUER stalls on it, which is why the DMA idles 2,464 cycles between
    // the norm-weight load and the decode-table load and the up weight load starts at 9,209 instead
    // of ~6,3K. This command depends on nothing, so the issuer hands it over at once and the warm-up
    // is paid off the critical chain; its value is never read.
    let warm: DmTensor<f32, Chip, UpGateClusters, Pieces, m![1 # 8]> = DmTensor::new();
    let _warm_vrf: VrfTensor<f32, Chip, UpGateClusters, Pieces, m![1 # 8]> = ctx
        .sub
        .begin(warm.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    // POOL TILE CHAIN (BRIEF2 UPDATE 14 / 47, the qkv lever): the rms weight and the hidden state
    // are two TILES of ONE DM tensor and the WEIGHT is written first, so the x load carries a real
    // DMA -> DMA dependency and the scheduler lists the RMSNorm chain one DMA slot later.
    let mut xpool: DmTensor<bf16, Chip, UpGateClusters, Loaded, m![Slot, H % 480]> = DmTensor::new();
    {
        let w = unsafe { rms_weight.view().reshape::<Chip, m![Slot = 1, H]>() };
        w.to_dm_view(&mut ctx.tdma, xpool.view_mut().tile::<m![Slot], 1, m![Slot = 1 #{!} 2, H % 480]>(1));
        let xv = unsafe { x.view().reshape::<Chip, m![Slot = 1, H]>() };
        xv.to_dm_view(&mut ctx.tdma, xpool.view_mut().tile::<m![Slot], 1, m![Slot = 1 #{!} 2, H % 480]>(0));
    }
    let xpool: DmTensor<bf16, Chip, UpGateClusters, Pieces, m![Slot, H % 480]> = unsafe { xpool.reshape() };

    let mean_square: DmTensor<f32, Chip, UpGateClusters, Pieces, m![1 # 8]> = ctx
        .main
        .begin(xpool.view().tile::<m![Slot], 1, m![Slot = 1 # 2, H % 480]>(0))
        .fetch::<m![Slot = 1, H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, UpGateClusters, Gathered, m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<Gathered, m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, UpGateClusters, Pieces, m![1 # 8]> = unsafe { rms.reshape() };

    let weight_vrf: VrfTensor<f32, Chip, UpGateClusters, Pieces, m![H % 480]> = ctx
        .sub
        .begin(xpool.view().tile::<m![Slot], 1, m![Slot = 1 # 2, H % 480]>(1))
        .fetch::<m![Slot = 1, H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();
    let rms_vrf: VrfTensor<f32, Chip, UpGateClusters, Pieces, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let normalized: DmTensor<bf16, Chip, UpGateClusters, Pieces, m![H % 480]> = ctx
        .main
        .begin(xpool.view().tile::<m![Slot], 1, m![Slot = 1 # 2, H % 480]>(0))
        .fetch::<m![Slot = 1, H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit();

    // Rename the columns Xh; the two f8 terms sit side by side under T2.
    let normalized: DmTensor<bf16, Chip, UpGateClusters, PiecesX, m![T2 = 1, Xh % 480]> =
        unsafe { normalized.reshape() };
    let mut q: DmTensor<f8e4m3, Chip, UpGateClusters, PiecesX, m![T2, Xh % 480]> = DmTensor::new();
    ctx.main
        .begin(normalized.view())
        .fetch::<m![T2 = 1, Xh / 16 % 30], m![Xh % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![T2 = 1, Xh / 8 % 60], m![Xh % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![T2 = 1, Xh / 4 % 120], m![Xh % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), UP_GAIN)
        .vector_widen_concat::<m![T2 = 1, Xh / 8 % 60], m![Xh % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Xh % 8 # 32]>()
        .commit_trim::<m![Xh % 8]>()
        .commit_view(q.view_mut().tile::<m![T2], 1, m![T2 = 1 #{!} 2, Xh % 480]>(0));
    let q1_vrf: VrfTensor<f32, Chip, UpGateClusters, PiecesX, m![T2 = 1, Xh % 480]> = ctx
        .sub
        .begin(q.view().tile::<m![T2], 1, m![T2 = 1 # 2, Xh % 480]>(0))
        .fetch::<m![T2 = 1, Xh / 32 % 15], m![Xh % 32]>()
        .fetch_cast::<f32>()
        .collect::<m![T2 = 1, Xh / 8 % 60], m![Xh % 8]>()
        .to_vrf();
    ctx.main
        .begin(normalized.view())
        .fetch::<m![T2 = 1, Xh / 16 % 30], m![Xh % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![T2 = 1, Xh / 8 % 60], m![Xh % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![T2 = 1, Xh / 4 % 120], m![Xh % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), UP_GAIN)
        .vector_fp_binary(FpBinaryOp::SubF, &q1_vrf)
        .vector_widen_concat::<m![T2 = 1, Xh / 8 % 60], m![Xh % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Xh % 8 # 32]>()
        .commit_trim::<m![Xh % 8]>()
        .commit_view(q.view_mut().tile::<m![T2], 1, m![T2 = 1 #{!} 2, Xh % 480]>(1));

    // Only the first ring of every four holds real terms: an all-gather over the four (stride 8) gives
    // every slice the four rings' pieces in ring order; member 0 is the real one.
    let spread: DmTensor<f8e4m3, Chip, UpGateClusters, PiecesX, m![Q4, T2, Xh % 480]> = ctx
        .main
        .begin(q.view())
        .fetch::<m![1], m![T2, Xh % 480]>()
        .switch::<PiecesX, m![1, Q4]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 8 })
        .collect::<m![1, Q4, T2, Xh / 32 % 15], m![Xh % 32]>()
        .commit_trim::<m![Xh % 32]>()
        .commit();
    // All-gather along each ring of eight: every slice ends with both terms of the whole vector.
    let gathered: DmTensor<f8e4m3, Chip, UpGateClusters, Gathered, m![Q4 = 1, T2, Xh]> = ctx
        .main
        .begin(spread.view().tile::<m![Q4], 1, m![Q4 = 1 # 4, T2, Xh % 480]>(0))
        .fetch::<m![Q4 = 1, T2], m![Xh % 480]>()
        .switch::<Gathered, m![Q4 = 1, T2, Xh / 480]>(SwitchConfig::Broadcast1 { slice1: 8, slice0: 1 })
        .collect::<m![Q4 = 1, T2, Xh / 32], m![Xh % 32]>()
        .commit_trim::<m![Xh % 32]>()
        .commit();
    let gathered: DmTensor<f8e4m3, Chip, UpGateClusters, Gathered, m![T2, Ut, Xp]> =
        unsafe { gathered.reshape() };
    let x_trf = ctx
        .sub
        .begin(gathered.view())
        .fetch::<m![T2, Ut, Xp / 32], m![Xp % 32]>()
        .collect::<m![T2, Ut, Xp / 32], m![Xp % 32]>()
        .to_trf();
    (x_trf, xpool)
}

/// A whole up or gate matrix (30 whole rows a slice) in ONE pass, one decode table: per-block sums
/// (the two activation terms already added) rounded to bf16, 240 a row.
macro_rules! block_sums_all {
    ($ctx:ident, $w:ident, $x_trf:ident) => {{
        let packed: DmTensor<f4e2m1, Chip, UpGateClusters, RowSlices, m![L % 30, H]> = $w.to_dm(&mut $ctx.tdma);
        let packed: DmTensor<f4e2m1, Chip, UpGateClusters, Gathered, m![L % 30, Ut, Xp]> =
            unsafe { packed.reshape() };
        let z: DmTensor<bf16, Chip, UpGateClusters, Gathered, m![L % 30, Ut, Xp / 16]> = $ctx
            .main
            .begin(packed.view())
            .fetch::<m![L % 30, Ut], m![Xp]>()
            .fetch_table_lookup::<f8e4m3>()
            .collect::<m![L % 30, Ut, Xp / 32], m![Xp % 32]>()
            .contract_outer::<m![L % 30, Ut, T2], m![Xp], _, _, _>(&$x_trf)
            .contract_packet::<m![Xp / 16]>()
            .contract_time::<m![L % 30, Ut]>()
            .contract_lane::<m![L % 30, Ut], m![Xp / 16 # 8]>(LaneMode::Sequential)
            .cast::<bf16, m![Xp / 16 # 16]>()
            .commit_trim::<m![Xp / 16]>()
            .commit();
        let z: DmTensor<bf16, Chip, UpGateClusters, Gathered, m![L % 30, Sb]> = unsafe { z.reshape() };
        z
    }};
}

/// The block scales of an up / gate matrix, loaded as today (30 rows a slice) and converted
/// f8 -> bf16 by the Fetch Adapter and the Cast Engine (no Vector Engine).
fn scales_bf16(
    ctx: &mut Context,
    scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
) -> DmTensor<bf16, Chip, UpGateClusters, Gathered, m![L % 30, Sb]> {
    let s: DmTensor<f8e4m3, Chip, UpGateClusters, RowSlices, m![L % 30, H / 16]> = scale.to_dm(&mut ctx.tdma);
    let s: DmTensor<f8e4m3, Chip, UpGateClusters, Gathered, m![L % 30, Sb]> = unsafe { s.reshape() };
    ctx.sub
        .begin(s.view())
        .fetch::<m![L % 30, Sb / 8], m![Sb % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L % 30, Sb / 8], m![Sb % 8]>()
        .cast::<bf16, m![Sb % 8 # 16]>()
        .commit_trim::<m![Sb % 8]>()
        .commit()
}

/// sum over the blocks of (block sum x block scale) for all 30 rows of a slice: the scales sit in
/// the TRF, the contraction engine multiplies and its Time Reducer adds the blocks; two rows a
/// commit packet (Transpose Engine) so the result is already in the `RowSlices` layout.
fn apply_scales(
    ctx: &mut Context,
    z: &DmTensor<bf16, Chip, UpGateClusters, Gathered, m![L % 30, Sb]>,
    s: &DmTensor<bf16, Chip, UpGateClusters, Gathered, m![L % 30, Sb]>,
) -> DmTensor<f32, Chip, UpGateClusters, RowSlices, m![L % 30]> {
    let s_trf: TrfTensor<bf16, Chip, UpGateClusters, Gathered, m![1], m![L % 30, Sb]> = ctx
        .sub
        .begin(s.view())
        .fetch::<m![L % 30, Sb / 16], m![Sb % 16]>()
        .collect::<m![L % 30, Sb / 16], m![Sb % 16]>()
        .to_trf();
    let y: DmTensor<f32, Chip, UpGateClusters, Gathered, m![L % 30]> = ctx
        .main
        .begin(z.view())
        .fetch::<m![L % 30, Sb / 16], m![Sb % 16]>()
        .collect::<m![L % 30, Sb / 16], m![Sb % 16]>()
        .contract_outer::<m![L % 30, Sb / 16], m![Sb % 16], _, _, _>(&s_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![L % 30]>()
        .contract_lane::<m![L % 30], m![1 # 8]>(LaneMode::Sequential)
        .transpose::<m![L % 30 / 2], m![L % 30 % 2 # 8]>()
        .commit_trim::<m![L % 30 % 2]>()
        .commit();
    unsafe { y.reshape() }
}

/// A cluster's 7680 GeGLU rows in groups of 120, four real copies of each group.
type Groups = m![L / 120 % 64, Rep4];
/// The same slices named by the down projection's columns (a cluster owns 7680 of them).
type GroupsX = m![Xd / 120, Rep4];
/// Every slice of a cluster once the 64 groups have been gathered.
type DownSlices = m![Ring64, Rep4];
/// Down-projection rows: 15 a slice, every row in both clusters (each cluster contracts its own
/// half of the columns).
type DownRowSlices = m![H / 30, H % 2];

/// Four neighbouring slices (30 rows each) meet: every one of the four ends with the 120-value
/// row group the GeGLU works on.
fn regroup(
    ctx: &mut Context,
    y: &DmTensor<f32, Chip, UpGateClusters, RowSlices, m![L % 30]>,
) -> DmTensor<f32, Chip, UpGateClusters, Groups, m![L % 120]> {
    ctx.main
        .begin(y.view())
        .fetch::<m![L / 2 % 15], m![L % 2 # 8]>()
        .switch::<Groups, m![L / 2 % 15, L / 30 % 4]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 1 })
        .collect::<m![L / 2 % 15, L / 30 % 4], m![L % 2 # 8]>()
        .commit_trim::<m![L % 2]>()
        .commit()
}

/// The gate's global scale (times the erf's 1/sqrt(2) and the undo of UP_GAIN) as a VRF in every
/// slice of the GeGLU layout. The scalar is loaded into one slice of every group of four (a cheap
/// strided load) and handed to the other three by the Switch; a plain load into all 256 slices
/// cost 8-9K device cycles of DMA time.
fn gate_factor_vrf(
    ctx: &mut Context,
    global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> VrfTensor<f32, Chip, UpGateClusters, Groups, m![1 # 8]> {
    let global_scale: DmTensor<f32, Chip, UpGateClusters, m![L / 120 % 64, 1 # 4], m![1 # 8]> =
        global_scale.to_dm(&mut ctx.tdma);
    // Only the first slice of a group holds the scalar. An all-gather over the group hands every
    // slice the four slices' words in slice order; word 0 is the scalar, the other three are never read.
    let global_scale: DmTensor<f32, Chip, UpGateClusters, m![L / 120 % 64, Lead4], m![1 # 8]> =
        unsafe { global_scale.reshape() };
    let global_scale: DmTensor<f32, Chip, UpGateClusters, Groups, m![Lead4, 1 # 8]> = ctx
        .main
        .begin(global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .switch::<Groups, m![1, Lead4]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 1 })
        .collect::<m![1, Lead4], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), GATE_ERF_FACTOR)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    ctx.sub
        .begin(global_scale.view().tile::<m![Lead4], 1, m![Lead4 = 1 # 4, 1 # 8]>(0))
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf()
}

/// GeGLU: up x gate x gs x (1 + erf(gate x gs / sqrt 2)) / 2, with `up` and `gate` UP_GAIN times too
/// large: `gate_factor` = gs / (sqrt 2 x UP_GAIN) makes the erf argument right, the stashed product
/// carries that factor, and the final divisor 2 x UP_GAIN x (1 / sqrt 2) undoes the rest.
fn geglu(
    ctx: &mut Context,
    up: &DmTensor<f32, Chip, UpGateClusters, Groups, m![L % 120]>,
    gate: &DmTensor<f32, Chip, UpGateClusters, Groups, m![L % 120]>,
    gate_factor: &VrfTensor<f32, Chip, UpGateClusters, Groups, m![1 # 8]>,
) -> DmTensor<bf16, Chip, UpGateClusters, Groups, m![L % 120]> {
    let gelu: DmTensor<f32, Chip, UpGateClusters, Groups, m![L % 120]> = ctx
        .sub
        .begin(gate.view())
        .fetch::<m![L / 8 % 15], m![L % 8]>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), gate_factor)
        .vector_stash()
        .vector_fp_unary(FpUnaryOp::Erf)
        .vector_fp_binary(FpBinaryOp::AddF, 1f32)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), Stash)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .commit_trim::<m![L % 8]>()
        .commit();
    let gelu_vrf: VrfTensor<f32, Chip, UpGateClusters, Groups, m![L % 120]> = ctx
        .sub
        .begin(gelu.view())
        .fetch::<m![L / 8 % 15], m![L % 8]>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .to_vrf();
    ctx.main
        .begin(up.view())
        .fetch::<m![L / 8 % 15], m![L % 8]>()
        .collect::<m![L / 8 % 15], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 30], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &gelu_vrf)
        .vector_fp_div(GEGLU_DIVISOR)
        .vector_widen_concat::<m![L / 8 % 15], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit()
}

/// The GeGLU output, group by group, split into two exact f8 terms; then all 64 groups of the
/// cluster gathered into every slice and stored in the TRF (two lanes of 7680 columns, named as
/// four quarters of 1920).
fn quantize_gather(
    ctx: &mut Context,
    x: DmTensor<bf16, Chip, UpGateClusters, Groups, m![L % 120]>,
) -> TrfTensor<f8e4m3, Chip, UpGateClusters, DownSlices, m![1], m![T2, Dt, Xp]> {
    let x: DmTensor<bf16, Chip, UpGateClusters, Groups, m![T2 = 1, Xg]> = unsafe { x.reshape() };
    let mut q: DmTensor<f8e4m3, Chip, UpGateClusters, Groups, m![T2, Xg]> = DmTensor::new();
    ctx.main
        .begin(x.view())
        .fetch::<m![T2 = 1, Xg / 8], m![Xg % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![T2 = 1, Xg / 8], m![Xg % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![T2 = 1, Xg / 4], m![Xg % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), DOWN_GAIN)
        .vector_widen_concat::<m![T2 = 1, Xg / 8], m![Xg % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Xg % 8 # 32]>()
        .commit_trim::<m![Xg % 8]>()
        .commit_view(q.view_mut().tile::<m![T2], 1, m![T2 = 1 #{!} 2, Xg]>(0));
    let q1_vrf: VrfTensor<f32, Chip, UpGateClusters, Groups, m![T2 = 1, Xg]> = ctx
        .sub
        .begin(q.view().tile::<m![T2], 1, m![T2 = 1 # 2, Xg]>(0))
        .fetch::<m![T2 = 1, Xg / 8], m![Xg % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![T2 = 1, Xg / 8], m![Xg % 8]>()
        .to_vrf();
    ctx.main
        .begin(x.view())
        .fetch::<m![T2 = 1, Xg / 8], m![Xg % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![T2 = 1, Xg / 8], m![Xg % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![T2 = 1, Xg / 4], m![Xg % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), DOWN_GAIN)
        .vector_fp_binary(FpBinaryOp::SubF, &q1_vrf)
        .vector_widen_concat::<m![T2 = 1, Xg / 8], m![Xg % 8]>()
        .vector_final()
        .cast::<f8e4m3, m![Xg % 8 # 32]>()
        .commit_trim::<m![Xg % 8]>()
        .commit_view(q.view_mut().tile::<m![T2], 1, m![T2 = 1 #{!} 2, Xg]>(1));

    // The 64 groups are the columns the cluster owns: name them Xd / 120 and gather along the
    // ring of 64 (stride 4: slices with the same copy index form a ring).
    let q: DmTensor<f8e4m3, Chip, UpGateClusters, GroupsX, m![T2, Xd % 120]> = unsafe { q.reshape() };
    let gathered: DmTensor<f8e4m3, Chip, UpGateClusters, DownSlices, m![T2, Xd]> = ctx
        .main
        .begin(q.view())
        .fetch::<m![T2, Xd / 24 % 5], m![Xd % 24]>()
        .switch::<DownSlices, m![T2, Xd / 24 % 5, Xd / 120]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 4 })
        .collect::<m![T2, Xd / 24 % 5, Xd / 120], m![Xd % 24 # 32]>()
        .commit_trim::<m![Xd % 24]>()
        .commit();
    let gathered: DmTensor<f8e4m3, Chip, UpGateClusters, DownSlices, m![T2, Dt, Xp]> =
        unsafe { gathered.reshape() };
    ctx.sub
        .begin(gathered.view())
        .fetch::<m![T2, Dt, Xp / 32], m![Xp % 32]>()
        .collect::<m![T2, Dt, Xp / 32], m![Xp % 32]>()
        .to_trf()
}

/// The down projection's block scales (15 half-rows a slice, 480 blocks each), f8 -> bf16.
fn down_scales_bf16(
    ctx: &mut Context,
    scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
) -> DmTensor<bf16, Chip, UpGateClusters, DownSlices, m![H / 2 % 15, Sd]> {
    let s: DmTensor<f8e4m3, Chip, UpGateClusters, DownRowSlices, m![H / 2 % 15, L / 16 % 480]> = scale.to_dm(&mut ctx.tdma);
    let s: DmTensor<f8e4m3, Chip, UpGateClusters, DownSlices, m![H / 2 % 15, Sd]> = unsafe { s.reshape() };
    ctx.sub
        .begin(s.view())
        .fetch::<m![H / 2 % 15, Sd / 8], m![Sd % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 2 % 15, Sd / 8], m![Sd % 8]>()
        .cast::<bf16, m![Sd % 8 # 16]>()
        .commit_trim::<m![Sd % 8]>()
        .commit()
}

/// A tile of the down projection (a cluster's half of the columns, `$len` rows): the block pass
/// (one decode table, bf16 block sums out) and the scale contraction (the tile's scales in the TRF,
/// blocks added in the Time Reducer), one f32 value per row into `$out`.
macro_rules! down_tile {
    ($ctx:ident, $w:ident, $x_trf:ident, $scale:ident, $out:ident, $start:literal, $len:literal) => {{
        let packed: DmTensor<f4e2m1, Chip, UpGateClusters, DownRowSlices, m![H / 2 % 15 = $len, L % 7680]> = $w
            .view()
            .tile::<m![H / 2 % 15], $len, m![H / 30, H / 2 % 15 = $len # 15, H % 2, L]>($start)
            .to_dm(&mut $ctx.tdma);
        let packed: DmTensor<f4e2m1, Chip, UpGateClusters, DownSlices, m![H / 2 % 15 = $len, Dt, Xp]> =
            unsafe { packed.reshape() };
        let z: DmTensor<bf16, Chip, UpGateClusters, DownSlices, m![H / 2 % 15 = $len, Dt, Xp / 16]> = $ctx
            .main
            .begin(packed.view())
            .fetch::<m![H / 2 % 15 = $len, Dt], m![Xp]>()
            .fetch_table_lookup::<f8e4m3>()
            .collect::<m![H / 2 % 15 = $len, Dt, Xp / 32], m![Xp % 32]>()
            .contract_outer::<m![H / 2 % 15 = $len, Dt, T2], m![Xp], _, _, _>(&$x_trf)
            .contract_packet::<m![Xp / 16]>()
            .contract_time::<m![H / 2 % 15 = $len, Dt]>()
            .contract_lane::<m![H / 2 % 15 = $len, Dt], m![Xp / 16 # 8]>(LaneMode::Sequential)
            .cast::<bf16, m![Xp / 16 # 16]>()
            .commit_trim::<m![Xp / 16]>()
            .commit();
        let z: DmTensor<bf16, Chip, UpGateClusters, DownSlices, m![H / 2 % 15 = $len, Sd]> = unsafe { z.reshape() };
        let s_trf: TrfTensor<bf16, Chip, UpGateClusters, DownSlices, m![1], m![H / 2 % 15 = $len, Sd]> = $ctx
            .sub
            .begin($scale.view().tile::<m![H / 2 % 15], $len, m![H / 2 % 15 = $len # 15, Sd]>($start))
            .fetch::<m![H / 2 % 15 = $len, Sd / 16], m![Sd % 16]>()
            .collect::<m![H / 2 % 15 = $len, Sd / 16], m![Sd % 16]>()
            .to_trf();
        $ctx.main
            .begin(z.view())
            .fetch::<m![H / 2 % 15 = $len, Sd / 16], m![Sd % 16]>()
            .collect::<m![H / 2 % 15 = $len, Sd / 16], m![Sd % 16]>()
            .contract_outer::<m![H / 2 % 15 = $len, Sd / 16], m![Sd % 16], _, _, _>(&s_trf)
            .contract_packet::<m![1]>()
            .contract_time::<m![H / 2 % 15 = $len]>()
            .contract_lane::<m![H / 2 % 15 = $len], m![1 # 8]>(LaneMode::Sequential)
            .commit_trim::<m![1 # 8]>()
            .commit_view($out.view_mut().tile::<m![H / 2 % 15], $len, m![H / 2 % 15 = $len #{!} 15, 1 # 8]>($start));
    }};
}

pub(crate) fn feedforward(
    ctx: &mut Context,
    residual: &HbmTensor<bf16, Chip, m![H]>,
    pre_ff_rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    up_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    gate_weight_packed: &HbmTensor<f4e2m1, Chip, m![L, H]>,
    down_weight_packed: &HbmTensor<f4e2m1, Chip, m![H, L]>,
    up_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    gate_weight_scale: &HbmTensor<f8e4m3, Chip, m![L, H / 16]>,
    down_weight_scale: &HbmTensor<f8e4m3, Chip, m![H, L / 16]>,
    up_global_scale: &HbmTensor<f32, Chip, m![1]>,
    gate_global_scale: &HbmTensor<f32, Chip, m![1]>,
    down_global_scale: &HbmTensor<f32, Chip, m![1]>,
) -> (DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]>, DmTensor<bf16, Chip, Cluster, ReducingSlices, m![Slot, H % 480]>) {
    let (x_trf, residual_pieces) = normalize_quantize(ctx, residual, pre_ff_rms_weight);
    // The input was loaded as 32 real copies of eight 480-column pieces: copy 0 of cluster 0 IS the
    // residual spread over the eight reducing slices, byte for byte (real data viewed as padded).
    let residual_spread: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![Slot, H % 480]> =
        unsafe { residual_pieces.reshape() };

    let up_z = block_sums_all!(ctx, up_weight_packed, x_trf);
    let up_s = scales_bf16(ctx, up_weight_scale);
    let up = apply_scales(ctx, &up_z, &up_s);
    let gate_z = block_sums_all!(ctx, gate_weight_packed, x_trf);
    let gate_s = scales_bf16(ctx, gate_weight_scale);
    let gate = apply_scales(ctx, &gate_z, &gate_s);

    let up = regroup(ctx, &up);
    let gate = regroup(ctx, &gate);
    let gate_factor = gate_factor_vrf(ctx, gate_global_scale);
    let x = geglu(ctx, &up, &gate, &gate_factor);
    let x_trf = quantize_gather(ctx, x);

    let down_s = down_scales_bf16(ctx, down_weight_scale);
    let mut partial: DmTensor<f32, Chip, UpGateClusters, DownSlices, m![H / 2 % 15, 1 # 8]> = DmTensor::new();
    // TWO tiles instead of three. The bytes are identical, but each tile costs one weight command
    // (~1,700 by the corrected cost law) AND, for every tile after the first, one compiler-generated
    // f4 decode-table load (~2,000): the tables are the DmaLoads with an empty description, and
    // three tiles need two of them. The LAST tile keeps its shipped 3-unit shape, because its block
    // pass is the one that cannot hide under a following load; only the first two are merged.
    down_tile!(ctx, down_weight_packed, x_trf, down_s, partial, 0, 10);
    down_tile!(ctx, down_weight_packed, x_trf, down_s, partial, 10, 5);

    // Two neighbouring slices meet (rows interleaved two ways, so the pair is 30 consecutive rows).
    let partial: DmTensor<f32, Chip, UpGateClusters, DownRowSlices, m![H / 2 % 15, 1 # 8]> = unsafe { partial.reshape() };
    let partial: DmTensor<f32, Chip, UpGateClusters, m![H / 30, 1 # 2], m![H % 30 # 64]> = ctx
        .main
        .begin(partial.view())
        .fetch::<m![H / 2 % 15], m![1 # 8]>()
        .switch::<m![H / 30, 1 # 2], m![H / 2 % 15, H % 2]>(SwitchConfig::Broadcast1 { slice1: 2, slice0: 1 })
        .collect::<m![H / 2 % 15, H % 2], m![1 # 8]>()
        .transpose::<m![H / 2 % 15], m![H % 2 # 8]>()
        .commit_trim::<m![H % 2]>()
        .commit();

    // The two clusters' partial results meet in HBM and come back divided over the eight slices the
    // post-norm reduces on; the add and both remaining global scales ride one pass there.
    // A scattered store of 120-byte pieces is slow (unaligned tails: 4.5K device cycles), so every
    // slice's 30 values sit at the head of a 256-byte element and the store is block-aligned; only
    // the heads come back. (An HBM tensor cannot end in padding: the pad words are a real axis.)
    let partial: DmTensor<f32, Chip, m![C2], m![Hg, 1 # 2], m![Pw]> = unsafe { partial.reshape() };
    let partial: HbmTensor<f32, Chip, m![C2, Hg, Pw]> = partial.to_hbm(&mut ctx.tdma);
    let partial: DmTensor<f32, Chip, Cluster, m![1 # 32, Hg / 16], m![C2, Hg % 16, Pw = 30]> = partial
        .view()
        .tile::<m![Pw], 30, m![C2, Hg, Pw = 30 # 64]>(0)
        .to_dm(&mut ctx.tdma);
    let partial: DmTensor<f32, Chip, Cluster, ReducingSlices, m![C2, H % 480]> = unsafe { partial.reshape() };

    let down_global_scale: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> =
        down_global_scale.to_dm(&mut ctx.tdma);
    let down_global_scale: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> =
        unsafe { down_global_scale.reshape() };
    let down_global_scale_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(down_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let up_global_scale: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> =
        up_global_scale.to_dm(&mut ctx.tdma);
    let up_global_scale: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> =
        unsafe { up_global_scale.reshape() };
    let up_global_scale_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(up_global_scale.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    // The DOWN_GAIN of the quantized GeGLU output is undone here (FpDiv stage, after the reduce).
    let out: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .main
        .begin(partial.view())
        .fetch::<m![H / 4 % 120, C2], m![H % 4]>()
        .collect::<m![H / 4 % 120, C2], m![H % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &down_global_scale_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &up_global_scale_vrf)
        .vector_intra_slice_reduce::<C2, m![H / 4 % 120], m![H % 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(DOWN_GAIN)
        .vector_widen_pad::<m![H % 4 # 8]>()
        .vector_final()
        .cast::<bf16, m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>()
        .commit();
    (out, residual_spread)
}

/// Post-norm, residual add and layer gate: (x / rms * w + r) * g in ONE pass, rounded to bf16 once at
/// the end (the three-pass form rounded after each step).
pub(crate) fn normalize_add_gate_in_place(
    ctx: &mut Context,
    x: &DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]>,
    rms_weight: &HbmTensor<bf16, Chip, m![H]>,
    residual_dm: &DmTensor<bf16, Chip, Cluster, ReducingSlices, m![Slot, H % 480]>,
    layer_scalar: &HbmTensor<bf16, Chip, m![1 # 8]>,
) -> DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> {
    let mean_square: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![1], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(H_F32)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, m![1 # 32, Dummy8], m![1 # 8]> = ctx
        .main
        .begin(mean_square.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![1 # 32, Dummy8], m![1]>(InterSliceReduceOpF32::Add)
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_fp_binary(FpBinaryOp::AddF, EPS)
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_widen_pad::<m![1 # 8]>()
        .vector_final()
        .commit_trim::<m![1 # 8]>()
        .commit();
    let rms: DmTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = unsafe { rms.reshape() };

    let weight_dm: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![H % 480]> = rms_weight.to_dm(&mut ctx.tdma);
    let weight_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();
    let rms_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(rms.view())
        .fetch::<m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();
    let residual_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![H % 480]> = ctx
        .sub
        .begin(residual_dm.view().tile::<m![Slot], 1, m![Slot = 1 # 2, H % 480]>(0))
        .fetch::<m![Slot = 1, H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .to_vrf();
    let scalar: DmTensor<bf16, Chip, Cluster, ReducingSlices, m![1 # 8]> = layer_scalar.to_dm(&mut ctx.tdma);
    let scalar_vrf: VrfTensor<f32, Chip, Cluster, ReducingSlices, m![1 # 8]> = ctx
        .sub
        .begin(scalar.view())
        .fetch::<m![1], m![1 # 8]>()
        .fetch_cast::<f32>()
        .collect::<m![1], m![1 # 8]>()
        .to_vrf();

    ctx.main
        .begin(x.view())
        .fetch::<m![H / 16 % 30], m![H % 16]>()
        .fetch_cast::<f32>()
        .collect::<m![H / 8 % 60], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![H / 4 % 120], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::DivF, &rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &weight_vrf)
        .vector_fp_binary(FpBinaryOp::AddF, &residual_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &scalar_vrf)
        .vector_widen_concat::<m![H / 8 % 60], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}

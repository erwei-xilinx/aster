// RUN: mlir-air-opt %s \
// RUN:   --transform-interpreter --canonicalize --cse \
// RUN:   --air-par-to-launch="has-air-segment=true" --canonicalize --cse \
// RUN:   --air-copy-to-dma \
// RUN:   --air-dma-to-channel \
// RUN:   --air-to-amdgcn --canonicalize \
// RUN:   --convert-memspace-to-amdgcn \
// RUN:   --convert-linalg-to-amdgcn \
// RUN:   --amdgcn-preload-library="library-paths=%p/../../../mlir_kernels/library/common/register-init.mlir,%p/../../../mlir_kernels/library/common/indexing.mlir,%p/../../../mlir_kernels/library/common/indexing_ptr.mlir,%p/../../../mlir_kernels/library/common/futures.mlir,%p/../../../contrib/kittens/library/compute_16x16_f16.mlir,%p/../../../contrib/kittens/library/global_16x64_b.mlir,%p/../../../contrib/kittens/library/lds_16x64_b.mlir,%p/../../../contrib/kittens/library/lds_mfma_16x64_b.mlir" \
// RUN:   --inline --symbol-dce --canonicalize \
// RUN:   --mlir-air-to-asm \
// RUN: | aster-translate --mlir-to-asm \
// RUN: | FileCheck %s

// CHECK-LABEL: matmul_f16_64x64:
// CHECK:   global_load_dwordx4
// CHECK:   ds_write_b64
// CHECK:   ds_read_b64
// CHECK:   v_mfma_f32_16x16x16_f16
// CHECK:   global_store_dword
// CHECK:   s_endpgm

// Real AIR pipeline (adapted from xrt/12, tile-using-pad path):
//
//   1. linalg.generic on tensors (64x64 matmul, no AIR ops)
//   2. transform: tile_using_forall (2x1 herd) → tile_using_for (compute)
//      → pad → bufferize_to_allocation (L1) → one_shot_bufferize
//      → forall_to_parallel → par_to_herd
//   3. air-copy-to-dma → air-dma-to-channel
//   4. air-to-amdgcn (herd → wavefront index)
//   5. convert-memspace → convert-linalg → preload → asm

!sx2 = !amdgcn.sgpr<[? + 2]>
!vx2 = !amdgcn.vgpr<[? + 2]>
!ax4 = !amdgcn.agpr<[? + 4]>
!lds_write_token = !amdgcn.write_token<shared>
!future_lds_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>
!future_global_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>

module attributes {transform.with_named_sequence} {
  amdgcn.library @linalg_lib isa = [#amdgcn.isa<cdna3>] {
    func.func private @zero_C() -> !ax4
    func.func private @mfma_f32_16x16x16_f16(!vx2, !vx2, !ax4) -> !ax4
    func.func private @store_global_C_mfma_f32_16x16x16_f16(
        !ax4, !aster_utils.any, index, index, index)
    func.func private @prepare_ptr(!sx2) -> !aster_utils.any
    func.func private @load_global_tile_16x64_b(
        !aster_utils.any, index, index, index) -> !future_global_read
    func.func private @store_global_tile_to_lds_16x64_b(
        index, !future_global_read) -> (!lds_write_token, !lds_write_token)
    func.func private @load_lds_A_swizzled(
        index, index, index) -> !future_lds_read
    func.func private @load_lds_B_swizzled(
        index, index, index) -> !future_lds_read
    func.func private @get_lds_read_value_vx2(!future_lds_read) -> !vx2

    func.func private @copy_f16_16x32(
        %src_ptr: !sx2, %src_stride: index,
        %row_offset: index, %col_offset: index,
        %lds_dst: index) {
      %ptr = func.call @prepare_ptr(%src_ptr) : (!sx2) -> !aster_utils.any
      %gfut = func.call @load_global_tile_16x64_b(
          %ptr, %row_offset, %col_offset, %src_stride)
          : (!aster_utils.any, index, index, index) -> !future_global_read
      %t0, %t1 = func.call @store_global_tile_to_lds_16x64_b(%lds_dst, %gfut)
          : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)
      amdgcn.wait deps %t0 : !lds_write_token
      amdgcn.wait deps %t1 : !lds_write_token
      return
    }

    func.func private @mfma_matmul_f16_16x32(
        %lds_A: index, %lds_B: index,
        %C_ptr: !sx2, %C_stride: index,
        %C_row_offset: index, %C_col_offset: index) {
      %C_prepared = func.call @prepare_ptr(%C_ptr) : (!sx2) -> !aster_utils.any
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c32 = arith.constant 32 : index
      %acc = func.call @zero_C() : () -> !ax4
      %A0f = func.call @load_lds_A_swizzled(%lds_A, %c0, %c2)
          : (index, index, index) -> !future_lds_read
      %A0 = func.call @get_lds_read_value_vx2(%A0f) : (!future_lds_read) -> !vx2
      %B0f = func.call @load_lds_B_swizzled(%lds_B, %c0, %c2)
          : (index, index, index) -> !future_lds_read
      %B0 = func.call @get_lds_read_value_vx2(%B0f) : (!future_lds_read) -> !vx2
      %acc0 = func.call @mfma_f32_16x16x16_f16(%A0, %B0, %acc)
          : (!vx2, !vx2, !ax4) -> !ax4
      %A1f = func.call @load_lds_A_swizzled(%lds_A, %c32, %c2)
          : (index, index, index) -> !future_lds_read
      %A1 = func.call @get_lds_read_value_vx2(%A1f) : (!future_lds_read) -> !vx2
      %B1f = func.call @load_lds_B_swizzled(%lds_B, %c32, %c2)
          : (index, index, index) -> !future_lds_read
      %B1 = func.call @get_lds_read_value_vx2(%B1f) : (!future_lds_read) -> !vx2
      %acc1 = func.call @mfma_f32_16x16x16_f16(%A1, %B1, %acc0)
          : (!vx2, !vx2, !ax4) -> !ax4
      func.call @store_global_C_mfma_f32_16x16x16_f16(
          %acc1, %C_prepared, %C_row_offset, %C_col_offset, %C_stride)
          : (!ax4, !aster_utils.any, index, index, index) -> ()
      return
    }

    func.func private @fill_f16_16x32(%val: f16, %lds_dst: index) { return }
  }

  amdgcn.module @matmul_mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
    // 64x64 tensor-based matmul. No AIR ops — transform generates hierarchy.
    func.func @matmul_f16_64x64(
        %A: memref<64x64xf16>, %B: memref<64x64xf16>, %C: memref<64x64xf32>)
        attributes {gpu.kernel} {
      %cst = arith.constant 0.000000e+00 : f32
      %a = bufferization.to_tensor %A restrict writable : memref<64x64xf16> to tensor<64x64xf16>
      %b = bufferization.to_tensor %B restrict writable : memref<64x64xf16> to tensor<64x64xf16>
      %empty = tensor.empty() : tensor<64x64xf32>
      %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
      // matmul_transpose_b: C[m,n] += A[m,k] * B[n,k]
      %result = linalg.generic {
        indexing_maps = [
          affine_map<(m, n, k) -> (m, k)>,
          affine_map<(m, n, k) -> (n, k)>,
          affine_map<(m, n, k) -> (m, n)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
      } ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>)
        outs(%fill : tensor<64x64xf32>) {
      ^bb0(%av: f16, %bv: f16, %cv: f32):
        %a_ext = arith.extf %av : f16 to f32
        %b_ext = arith.extf %bv : f16 to f32
        %prod = arith.mulf %a_ext, %b_ext : f32
        %sum = arith.addf %cv, %prod : f32
        linalg.yield %sum : f32
      } -> tensor<64x64xf32>
      bufferization.materialize_in_destination %result in writable %C
          : (tensor<64x64xf32>, memref<64x64xf32>) -> ()
      return
    }
  }

  // Transform adapted from xrt/12 (tile-using-pad, no packing).
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.generic"]} in %arg0
        : (!transform.any_op) -> !transform.any_op

    // Outer tiling: 2x1 forall on M (becomes 2-wavefront air.herd).
    // 64/32 = 2 iterations — non-trivial, survives canonicalization.
    %outer_tiled, %outer_forall =
      transform.structured.tile_using_forall %matmul
        tile_sizes [32, 0, 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Compute tiling inside forall: 16x16 tiles, K=32.
    %tiled, %lm, %ln, %lk = transform.structured.tile_using_for %outer_tiled
        tile_sizes [16, 16, 32]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
                                   !transform.any_op, !transform.any_op)

    // Pad A and B only (not C — the matmul stores C directly to global).
    %padded, %pad, %copy_back = transform.structured.pad %tiled {
      padding_values = [0.0 : f16, 0.0 : f16, 0.0 : f32],
      padding_dimensions = [0, 1, 2],
      pack_paddings = [1, 1, 0],
      nofold_flags = [1, 1, 0],
      copy_back_op = "linalg.copy"
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
                                 !transform.any_op)
    %pad_dps = transform.structured.rewrite_in_destination_passing_style %pad
        : (!transform.any_op) -> !transform.any_op

    // Promote A,B,C pads to L1 (memory_space=2) via bufferize_to_allocation.
    %padded_lhs = transform.get_producer_of_operand %padded[0]
        : (!transform.any_op) -> (!transform.any_op)
    %buf_a, %new_a = transform.structured.bufferize_to_allocation %padded_lhs
        {memory_space = 2, bufferize_destination_only, emit_dealloc}
        : !transform.any_op

    %padded_rhs = transform.get_producer_of_operand %padded[1]
        : (!transform.any_op) -> (!transform.any_op)
    %buf_b, %new_b = transform.structured.bufferize_to_allocation %padded_rhs
        {memory_space = 2, bufferize_destination_only, emit_dealloc}
        : !transform.any_op

    // Canonicalize.
    %func_0 = transform.structured.match ops{["func.func"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_0 {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_0 : !transform.any_op

    // One-shot bufferize.
    %func_1 = transform.structured.match ops{["func.func"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %func_buf = transform.bufferization.one_shot_bufferize %func_1 {
      allow_return_allocs_from_loops = true
    } : (!transform.any_op) -> !transform.any_op

    // Cleanup.
    %func_2 = transform.structured.match ops{["func.func"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_2 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_2 : !transform.any_op
    %func_3 = transform.air.remove_uninitialized_copy %func_2
        : (!transform.any_op) -> !transform.any_op

    // Convert outer forall → parallel → air.herd (now on memrefs).
    %forall_2 = transform.structured.match ops{["scf.forall"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %parallel = transform.loop.forall_to_parallel %forall_2
        : (!transform.any_op) -> !transform.any_op
    %herd = transform.air.par_to_herd %parallel
        : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}

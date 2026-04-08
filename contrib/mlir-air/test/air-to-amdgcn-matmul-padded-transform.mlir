// Transform sequence for padded matmul: M=40, N=40, K=64.
// tile_using_forall [16,0,0] → 3 wavefronts (ceil(40/16)=3).
// tile_using_for [16,16,0] → 16x16 compute tiles, untiled K.
// Boundary tiles (8 rows or 8 cols) are padded to 16.
// bufferize_to_allocation on pad ops BEFORE DPS rewrite uses the
// dedicated PadOp allocation path that handles the full pad result.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.generic"]} in %arg0
        : (!transform.any_op) -> !transform.any_op

    // Outer tiling: forall on M, one wavefront per 16 M rows.
    %outer_tiled, %outer_forall =
      transform.structured.tile_using_forall %matmul
        tile_sizes [16, 0, 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Compute tiling: 16x16 output tiles, K untiled.
    %tiled, %lm, %ln = transform.structured.tile_using_for %outer_tiled
        tile_sizes [16, 16, 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
                                   !transform.any_op)

    // Pad all operands. Boundary tiles get zero-padded to 16.
    %padded, %pad, %copy_back = transform.structured.pad %tiled {
      padding_values = [0.0 : f16, 0.0 : f16, 0.0 : f32],
      padding_dimensions = [0, 1, 2],
      pack_paddings = [1, 1, 1],
      nofold_flags = [1, 1, 1]
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
                                 !transform.any_op)

    // Pre-allocate pad buffers with explicit memory spaces BEFORE DPS
    // rewrite. The PadOp-specific path in bufferize_to_allocation handles
    // the full pad (fill + copy) in one allocation.
    // A, B → LDS (memory_space=2). C → global (memory_space=0 is default,
    // but we don't pre-allocate C — it writes directly to the global subview).
    %padded_lhs = transform.get_producer_of_operand %padded[0]
        : (!transform.any_op) -> (!transform.any_op)
    %buf_a, %new_a = transform.structured.bufferize_to_allocation %padded_lhs
        {memory_space = 2} : !transform.any_op

    %padded_rhs = transform.get_producer_of_operand %padded[1]
        : (!transform.any_op) -> (!transform.any_op)
    %buf_b, %new_b = transform.structured.bufferize_to_allocation %padded_rhs
        {memory_space = 2} : !transform.any_op

    // C pad: allocate in LDS (memory_space=2). The matmul computes the full
    // 16x16 tile in LDS, then copies back only the valid region to global C.
    %padded_out = transform.get_producer_of_operand %padded[2]
        : (!transform.any_op) -> (!transform.any_op)
    %buf_c, %new_c = transform.structured.bufferize_to_allocation %padded_out
        {memory_space = 2} : !transform.any_op

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

    // Convert outer forall → parallel (par_to_herd runs as pipeline pass).
    %forall_2 = transform.structured.match ops{["scf.forall"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %parallel = transform.loop.forall_to_parallel %forall_2
        : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}

// Transform sequence for padded matmul: actual M=40, N=40, K=64.
// Uses pad_tiling_interface to pad 40→48 (next multiple of 16) BEFORE tiling.
// After padding, all dimensions are tile-aligned → no affine.min, uniform loops.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(
      %arg0: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.generic"]} in %arg0
        : (!transform.any_op) -> !transform.any_op

    // Step 1: Pad the iteration domain to tile-aligned sizes.
    // M: 40→48, N: 40→48, K: unchanged (already 64, divisible by any tile).
    %padded_matmul, %pad_op = transform.structured.pad_tiling_interface %matmul
        to padding_sizes [16, 16, 0] pad_to_multiple_of
        { padding_values = [0.0 : f16, 0.0 : f16, 0.0 : f32] }
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Step 2: Outer tiling — 3 wavefronts (48/16=3, exact).
    %outer_tiled, %outer_forall =
      transform.structured.tile_using_forall %padded_matmul
        tile_sizes [16, 0, 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Step 3: Compute tiling — 16x16 output tiles, K untiled.
    // 48 % 16 == 0 → no affine.min, all tiles are full.
    %tiled, %lm, %ln = transform.structured.tile_using_for %outer_tiled
        tile_sizes [16, 16, 0]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op,
                                   !transform.any_op)

    // Step 4: Pad A and B operands to promote to LDS.
    // All tiles are full (no boundary tiles) so pad is a no-op on shapes
    // but nofold forces allocation for the copy to LDS.
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

    // Step 5: Promote padded A,B to LDS (memory_space=2).
    %padded_lhs = transform.get_producer_of_operand %padded[0]
        : (!transform.any_op) -> (!transform.any_op)
    %buf_a, %new_a = transform.structured.bufferize_to_allocation %padded_lhs
        {memory_space = 2, bufferize_destination_only}
        : !transform.any_op

    %padded_rhs = transform.get_producer_of_operand %padded[1]
        : (!transform.any_op) -> (!transform.any_op)
    %buf_b, %new_b = transform.structured.bufferize_to_allocation %padded_rhs
        {memory_space = 2, bufferize_destination_only}
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

    // Convert outer forall → parallel → air.herd.
    %forall_2 = transform.structured.match ops{["scf.forall"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %parallel = transform.loop.forall_to_parallel %forall_2
        : (!transform.any_op) -> !transform.any_op
    %herd = transform.air.par_to_herd %parallel
        : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}

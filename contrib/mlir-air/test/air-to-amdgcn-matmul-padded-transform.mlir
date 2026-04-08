// Transform sequence for 48x48x64 matmul (host-padded from 40x40x64).
// tile_using_forall [16,0,0] → 3 wavefronts (48/16=3, exact).
// tile_using_for [16,16,0] → 16x16 compute tiles, untiled K.
// All tiles are full (48 is a multiple of 16). No boundary padding needed.

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

    // Pad A and B to tile size (16). Boundary tiles (8 rows/cols) get
    // zero-padded to 16.
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

    // Promote padded A,B to LDS (memory_space=2).
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

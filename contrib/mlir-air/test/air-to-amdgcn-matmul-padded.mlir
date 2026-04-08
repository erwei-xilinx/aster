// Padded matmul: actual M=40, N=40, K=64.
// Kernel operates on actual (non-tile-aligned) dimensions.
// The transform pads boundary tiles to full 16-element tiles.
// Host over-allocates C (48*48 elements) to accommodate OOB boundary stores.

!sx2 = !amdgcn.sgpr<[? + 2]>
!vx2 = !amdgcn.vgpr<[? + 2]>
!ax4 = !amdgcn.agpr<[? + 4]>
!lds_write_token = !amdgcn.write_token<shared>
!future_lds_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<shared>>
!future_global_read = !aster_utils.struct<value: !aster_utils.any, token: !amdgcn.read_token<flat>>

module {
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
    func.func private @fill_lds_16x64_b(index)
    func.func private @store_lds_C_mfma_f32_16x16x16_f16(!ax4, index)
    func.func private @mfma_index_C_16x16_f32() -> !aster_utils.struct<i: index, j: index>
    func.func private @mfma_c_16x16_f32_byte_offset(index, index, index, index, index, index, index) -> index
    func.func private @global_addr_from_offset(!sx2, index) -> !vx2
    func.func private @alloc_vgpr() -> !amdgcn.vgpr

    func.func private @copy_f16_16x64(
        %src_ptr: !sx2, %src_stride: index,
        %row_offset: index, %col_offset: index,
        %lds_dst: index) {
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      %col1 = arith.addi %col_offset, %c32 : index
      %lds_dst1 = arith.addi %lds_dst, %c1024 : index
      %ptr = func.call @prepare_ptr(%src_ptr) : (!sx2) -> !aster_utils.any
      %gfut0 = func.call @load_global_tile_16x64_b(
          %ptr, %row_offset, %col_offset, %src_stride)
          : (!aster_utils.any, index, index, index) -> !future_global_read
      %t0, %t1 = func.call @store_global_tile_to_lds_16x64_b(%lds_dst, %gfut0)
          : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)
      %gfut1 = func.call @load_global_tile_16x64_b(
          %ptr, %row_offset, %col1, %src_stride)
          : (!aster_utils.any, index, index, index) -> !future_global_read
      %t2, %t3 = func.call @store_global_tile_to_lds_16x64_b(%lds_dst1, %gfut1)
          : (index, !future_global_read) -> (!lds_write_token, !lds_write_token)
      amdgcn.wait deps %t0 : !lds_write_token
      amdgcn.wait deps %t1 : !lds_write_token
      amdgcn.wait deps %t2 : !lds_write_token
      amdgcn.wait deps %t3 : !lds_write_token
      return
    }

    func.func private @mfma_matmul_f16_16x64(
        %lds_A: index, %lds_B: index,
        %C_ptr: !sx2, %C_stride: index,
        %C_row_offset: index, %C_col_offset: index) {
      %C_prepared = func.call @prepare_ptr(%C_ptr) : (!sx2) -> !aster_utils.any
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      %lds_A2 = arith.addi %lds_A, %c1024 : index
      %lds_B2 = arith.addi %lds_B, %c1024 : index
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
      %A2f = func.call @load_lds_A_swizzled(%lds_A2, %c0, %c2)
          : (index, index, index) -> !future_lds_read
      %A2 = func.call @get_lds_read_value_vx2(%A2f) : (!future_lds_read) -> !vx2
      %B2f = func.call @load_lds_B_swizzled(%lds_B2, %c0, %c2)
          : (index, index, index) -> !future_lds_read
      %B2 = func.call @get_lds_read_value_vx2(%B2f) : (!future_lds_read) -> !vx2
      %acc2 = func.call @mfma_f32_16x16x16_f16(%A2, %B2, %acc1)
          : (!vx2, !vx2, !ax4) -> !ax4
      %A3f = func.call @load_lds_A_swizzled(%lds_A2, %c32, %c2)
          : (index, index, index) -> !future_lds_read
      %A3 = func.call @get_lds_read_value_vx2(%A3f) : (!future_lds_read) -> !vx2
      %B3f = func.call @load_lds_B_swizzled(%lds_B2, %c32, %c2)
          : (index, index, index) -> !future_lds_read
      %B3 = func.call @get_lds_read_value_vx2(%B3f) : (!future_lds_read) -> !vx2
      %acc3 = func.call @mfma_f32_16x16x16_f16(%A3, %B3, %acc2)
          : (!vx2, !vx2, !ax4) -> !ax4
      func.call @store_global_C_mfma_f32_16x16x16_f16(
          %acc3, %C_prepared, %C_row_offset, %C_col_offset, %C_stride)
          : (!ax4, !aster_utils.any, index, index, index) -> ()
      return
    }

    func.func private @fill_f16_16x64(%val: f16, %lds_dst: index) { return }
    func.func private @fill_f16_16x32(%val: f16, %lds_dst: index) { return }

    // Zero-fill 16x16 f32 LDS tile (1024 bytes).
    // Uses MFMA C fragment layout: each thread zeros its 4 positions.
    func.func private @fill_f32_16x16(%val: f32, %lds_dst: index) {
      func.call @fill_lds_16x64_b(%lds_dst) : (index) -> ()
      return
    }

    // Copy 16x16 f32 tile from global to LDS.
    // Reuses the 16x64-byte tile load (16x16 f32 = 16 rows × 64 bytes).
    func.func private @copy_f32_16x16(
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

    // MFMA matmul with all operands in LDS (including C accumulator).
    // Stores result to LDS C via store_lds_C (not global_store).
    func.func private @mfma_matmul_lds_c_f16_16x64(
        %lds_A: index, %lds_B: index, %lds_C: index) {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      %lds_A2 = arith.addi %lds_A, %c1024 : index
      %lds_B2 = arith.addi %lds_B, %c1024 : index
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
      %A2f = func.call @load_lds_A_swizzled(%lds_A2, %c0, %c2)
          : (index, index, index) -> !future_lds_read
      %A2 = func.call @get_lds_read_value_vx2(%A2f) : (!future_lds_read) -> !vx2
      %B2f = func.call @load_lds_B_swizzled(%lds_B2, %c0, %c2)
          : (index, index, index) -> !future_lds_read
      %B2 = func.call @get_lds_read_value_vx2(%B2f) : (!future_lds_read) -> !vx2
      %acc2 = func.call @mfma_f32_16x16x16_f16(%A2, %B2, %acc1)
          : (!vx2, !vx2, !ax4) -> !ax4
      %A3f = func.call @load_lds_A_swizzled(%lds_A2, %c32, %c2)
          : (index, index, index) -> !future_lds_read
      %A3 = func.call @get_lds_read_value_vx2(%A3f) : (!future_lds_read) -> !vx2
      %B3f = func.call @load_lds_B_swizzled(%lds_B2, %c32, %c2)
          : (index, index, index) -> !future_lds_read
      %B3 = func.call @get_lds_read_value_vx2(%B3f) : (!future_lds_read) -> !vx2
      %acc3 = func.call @mfma_f32_16x16x16_f16(%A3, %B3, %acc2)
          : (!vx2, !vx2, !ax4) -> !ax4
      // Store to LDS C (not global).
      func.call @store_lds_C_mfma_f32_16x16x16_f16(%acc3, %lds_C)
          : (!ax4, index) -> ()
      return
    }

    // Copy 16x16 f32 tile from LDS to global (C writeback).
    // Uses MFMA C fragment layout: each thread reads its 4 values from LDS
    // and writes to global memory.
    func.func private @store_global_f32_16x16(
        %lds_src: index,
        %dst_ptr: !sx2, %dst_stride: index,
        %row_offset: index, %col_offset: index) {
      // Read from LDS and write to global using the MFMA C fragment layout.
      %dst_prepared = func.call @prepare_ptr(%dst_ptr) : (!sx2) -> !aster_utils.any
      %mfma_idx = func.call @mfma_index_C_16x16_f32() : () -> !aster_utils.struct<i: index, j: index>
      %col, %row_base = aster_utils.struct_extract %mfma_idx ["i", "j"]
          : !aster_utils.struct<i: index, j: index> -> index, index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c64 = arith.constant 64 : index
      %elt_size = arith.constant 4 : index
      %col_byte = arith.muli %col, %elt_size : index
      // Manually unrolled: 4 iterations for MFMA C fragment (4 consecutive rows).
      // Avoids {aster.constexpr} loop which causes --inline canonicalize hang.
      %off0 = func.call @mfma_c_16x16_f32_byte_offset(%row_offset, %col_offset, %row_base, %col, %dst_stride, %elt_size, %c0) : (index, index, index, index, index, index, index) -> index
      %off1 = func.call @mfma_c_16x16_f32_byte_offset(%row_offset, %col_offset, %row_base, %col, %dst_stride, %elt_size, %c1) : (index, index, index, index, index, index, index) -> index
      %off2 = func.call @mfma_c_16x16_f32_byte_offset(%row_offset, %col_offset, %row_base, %col, %dst_stride, %elt_size, %c2) : (index, index, index, index, index, index, index) -> index
      %off3 = func.call @mfma_c_16x16_f32_byte_offset(%row_offset, %col_offset, %row_base, %col, %dst_stride, %elt_size, %c3) : (index, index, index, index, index, index, index) -> index
      %addr0 = func.call @global_addr_from_offset(%dst_ptr, %off0) : (!sx2, index) -> !vx2
      %addr1 = func.call @global_addr_from_offset(%dst_ptr, %off1) : (!sx2, index) -> !vx2
      %addr2 = func.call @global_addr_from_offset(%dst_ptr, %off2) : (!sx2, index) -> !vx2
      %addr3 = func.call @global_addr_from_offset(%dst_ptr, %off3) : (!sx2, index) -> !vx2
      // Read 4 f32 values from LDS at MFMA C fragment positions.
      %row0_off = arith.muli %row_base, %c64 : index
      %lds0 = arith.addi %lds_src, %row0_off : index
      %lds0b = arith.addi %lds0, %col_byte : index
      %lds1b = arith.addi %lds0b, %c64 : index
      %lds2b = arith.addi %lds1b, %c64 : index
      %lds3b = arith.addi %lds2b, %c64 : index
      %la0 = arith.index_cast %lds0b : index to i32
      %la1 = arith.index_cast %lds1b : index to i32
      %la2 = arith.index_cast %lds2b : index to i32
      %la3 = arith.index_cast %lds3b : index to i32
      %lv0 = lsir.to_reg %la0 : i32 -> !amdgcn.vgpr
      %lv1 = lsir.to_reg %la1 : i32 -> !amdgcn.vgpr
      %lv2 = lsir.to_reg %la2 : i32 -> !amdgcn.vgpr
      %lv3 = lsir.to_reg %la3 : i32 -> !amdgcn.vgpr
      %c0_i32 = arith.constant 0 : i32
      %d0 = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
      %d1 = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
      %d2 = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
      %d3 = func.call @alloc_vgpr() : () -> !amdgcn.vgpr
      %v0, %t0 = amdgcn.load ds_read_b32 dest %d0 addr %lv0 offset c(%c0_i32) : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>
      %v1, %t1 = amdgcn.load ds_read_b32 dest %d1 addr %lv1 offset c(%c0_i32) : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>
      %v2, %t2 = amdgcn.load ds_read_b32 dest %d2 addr %lv2 offset c(%c0_i32) : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>
      %v3, %t3 = amdgcn.load ds_read_b32 dest %d3 addr %lv3 offset c(%c0_i32) : dps(!amdgcn.vgpr) ins(!amdgcn.vgpr, i32) -> !amdgcn.read_token<shared>
      amdgcn.wait deps %t0 : !amdgcn.read_token<shared>
      amdgcn.wait deps %t1 : !amdgcn.read_token<shared>
      amdgcn.wait deps %t2 : !amdgcn.read_token<shared>
      amdgcn.wait deps %t3 : !amdgcn.read_token<shared>
      // Fire-and-forget global stores.
      amdgcn.store global_store_dword data %v0 addr %addr0 : ins(!amdgcn.vgpr, !vx2) -> !amdgcn.write_token<flat>
      amdgcn.store global_store_dword data %v1 addr %addr1 : ins(!amdgcn.vgpr, !vx2) -> !amdgcn.write_token<flat>
      amdgcn.store global_store_dword data %v2 addr %addr2 : ins(!amdgcn.vgpr, !vx2) -> !amdgcn.write_token<flat>
      amdgcn.store global_store_dword data %v3 addr %addr3 : ins(!amdgcn.vgpr, !vx2) -> !amdgcn.write_token<flat>
      return
    }
  }

  amdgcn.module @matmul_mod target = #amdgcn.target<gfx942> isa = #amdgcn.isa<cdna3> {
    func.func @matmul_f16_40x40(
        %A: memref<40x64xf16>, %B: memref<40x64xf16>, %C: memref<40x40xf32>)
        attributes {gpu.kernel} {
      %cst = arith.constant 0.000000e+00 : f32
      %a = bufferization.to_tensor %A restrict writable : memref<40x64xf16> to tensor<40x64xf16>
      %b = bufferization.to_tensor %B restrict writable : memref<40x64xf16> to tensor<40x64xf16>
      %c = bufferization.to_tensor %C restrict writable : memref<40x40xf32> to tensor<40x40xf32>
      %fill = linalg.fill ins(%cst : f32) outs(%c : tensor<40x40xf32>) -> tensor<40x40xf32>
      %result = linalg.generic {
        indexing_maps = [
          affine_map<(m, n, k) -> (m, k)>,
          affine_map<(m, n, k) -> (n, k)>,
          affine_map<(m, n, k) -> (m, n)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
      } ins(%a, %b : tensor<40x64xf16>, tensor<40x64xf16>)
        outs(%fill : tensor<40x40xf32>) {
      ^bb0(%av: f16, %bv: f16, %cv: f32):
        %a_ext = arith.extf %av : f16 to f32
        %b_ext = arith.extf %bv : f16 to f32
        %prod = arith.mulf %a_ext, %b_ext : f32
        %sum = arith.addf %cv, %prod : f32
        linalg.yield %sum : f32
      } -> tensor<40x40xf32>
      bufferization.materialize_in_destination %result in writable %C
          : (tensor<40x40xf32>, memref<40x40xf32>) -> ()
      return
    }
  }
}

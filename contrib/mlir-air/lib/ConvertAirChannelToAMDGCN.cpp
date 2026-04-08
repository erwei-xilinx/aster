// Copyright 2026 The ASTER Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- ConvertAirChannelToAMDGCN.cpp - air.channel -> library calls -------===//
//
// Lowers air.channel.put and air.channel.get to AMDGCN library function calls.
//
// For each air.channel.put/get:
//   - The memref operand is decomposed the same way as in ConvertLinalgToAMDGCN:
//     * Global memref -> (!sgpr<[?+2]>, byte_stride: index)
//     * Promoted buffer (memref.view of memref.alloca with memory space)
//       -> LDS byte offset (index)
//   - A func.call to a named library function is emitted:
//       copy_<dtype>_<MxN>(src_args..., dst_args...)
//     where the shape comes from the channel's memref type.
//
// The channel.put sends data INTO the channel (producer side).
// The channel.get receives data FROM the channel (consumer side).
// Together they represent a point-to-point copy: the put's src is the copy
// source, the get's dst is the copy destination.
//
// This pass matches put/get pairs by channel name and emits a single copy call
// at the get site (the consumer), erasing both ops and the channel declaration.
//===----------------------------------------------------------------------===//

#include "air/Dialect/AIR/AIRDialect.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNDialect.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNOps.h"
#include "aster/Dialect/AMDGCN/IR/AMDGCNTypes.h"
#include "aster/Dialect/LSIR/IR/LSIRDialect.h"
#include "aster/Dialect/LSIR/IR/LSIROps.h"
#include "aster/Interfaces/ModuleOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "mlir/Dialect/Ptr/IR/PtrOps.h"
#include "mlir/Dialect/Ptr/IR/PtrTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::aster;
using namespace mlir::aster::amdgcn;

namespace {

// ---------------------------------------------------------------------------
// Utilities (shared with ConvertLinalgToAMDGCN.cpp)
// ---------------------------------------------------------------------------

static std::string buildCopyFuncName(MemRefType ty) {
  std::string name;
  llvm::raw_string_ostream os(name);
  os << "copy";
  Type elt = ty.getElementType();
  if (elt.isF16())
    os << "_f16";
  else if (elt.isF32())
    os << "_f32";
  else if (elt.isBF16())
    os << "_bf16";
  else
    os << "_unknown";
  auto shape = ty.getShape();
  for (size_t i = 0; i < shape.size(); ++i)
    os << (i == 0 ? "_" : "x") << shape[i];
  return name;
}

static void ensureDecl(OpBuilder &builder, Block &block, Location loc,
                       StringRef name, FunctionType funcTy) {
  for (auto &op : block)
    if (auto fn = dyn_cast<func::FuncOp>(&op))
      if (fn.getName() == name)
        return;
  auto savedIP = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(&block);
  auto decl = func::FuncOp::create(builder, loc, name, funcTy);
  decl.setPrivate();
  builder.restoreInsertionPoint(savedIP);
}

static bool isPromotedBuffer(Value v) {
  if (auto viewOp = v.getDefiningOp<memref::ViewOp>()) {
    if (auto allocaOp = viewOp.getSource().getDefiningOp<memref::AllocaOp>())
      return allocaOp.getMemref().getType().getMemorySpace() != nullptr;
  }
  if (auto allocOp = v.getDefiningOp<memref::AllocOp>())
    return allocOp.getMemref().getType().getMemorySpace() != nullptr;
  return false;
}

static Value emitLDSOffset(OpBuilder &builder, Location loc, Value memrefVal,
                           DenseMap<Value, Value> &ldsCache) {
  auto it = ldsCache.find(memrefVal);
  if (it != ldsCache.end())
    return it->second;

  int64_t sizeBytes = 0;
  Value byteShift;
  if (auto viewOp = memrefVal.getDefiningOp<memref::ViewOp>()) {
    auto allocaOp = viewOp.getSource().getDefiningOp<memref::AllocaOp>();
    sizeBytes = allocaOp.getMemref().getType().getNumElements();
    byteShift = viewOp.getByteShift();
  } else if (auto allocOp = memrefVal.getDefiningOp<memref::AllocOp>()) {
    auto mrTy = allocOp.getMemref().getType();
    unsigned eltBits = mrTy.getElementType().getIntOrFloatBitWidth();
    sizeBytes = mrTy.getNumElements() * eltBits / 8;
  }
  auto ldsAlloc = AllocLDSOp::create(builder, loc, /*dynamic_size=*/Value(),
                                     sizeBytes, /*alignment=*/16,
                                     /*offset=*/IntegerAttr{});
  auto ldsOffset =
      GetLDSOffsetOp::create(builder, loc, builder.getIndexType(), ldsAlloc);
  Value result = ldsOffset.getResult();
  if (byteShift)
    result = builder.create<arith::AddIOp>(loc, result, byteShift);
  ldsCache[memrefVal] = result;
  return result;
}

static std::pair<Value, Value>
decomposeGlobalMemref(OpBuilder &builder, Location loc, Value memref) {
  auto mrTy = cast<MemRefType>(memref.getType());
  unsigned eltBytes = mrTy.getElementType().getIntOrFloatBitWidth() / 8;
  auto metadata =
      memref::ExtractStridedMetadataOp::create(builder, loc, memref);
  Value baseBuffer = metadata.getBaseBuffer();
  Value offset = metadata.getOffset();
  Value leadingStride = metadata.getStrides()[0];
  Value eltSize = arith::ConstantIndexOp::create(builder, loc, eltBytes);
  Value byteStride =
      arith::MulIOp::create(builder, loc, leadingStride, eltSize);
  Value byteOffset = arith::MulIOp::create(builder, loc, offset, eltSize);
  auto addrSpace = cast<ptr::MemorySpaceAttrInterface>(mrTy.getMemorySpace());
  auto ptrTy = ptr::PtrType::get(builder.getContext(), addrSpace);
  Value ptrVal = ptr::ToPtrOp::create(builder, loc, ptrTy, baseBuffer);
  auto sx2Ty = amdgcn::SGPRType::get(builder.getContext(), Register(),
                                     /*size=*/2, /*alignment=*/2);
  Value rawPtr = lsir::ToRegOp::create(builder, loc, sx2Ty, ptrVal);
  Value ptrFromReg = lsir::FromRegOp::create(builder, loc, ptrTy, rawPtr);
  Value adjusted =
      ptr::PtrAddOp::create(builder, loc, ptrTy, ptrFromReg, byteOffset);
  Value result = lsir::ToRegOp::create(builder, loc, sx2Ty, adjusted);
  return {result, byteStride};
}

/// Emit decomposed args for a memref operand (either LDS offset or global ptr).
static void emitDecomposedArgs(OpBuilder &builder, Location loc, Value memref,
                               SmallVectorImpl<Value> &callArgs,
                               SmallVectorImpl<Type> &argTypes,
                               DenseMap<Value, Value> &ldsCache) {
  auto indexTy = builder.getIndexType();
  auto sx2Ty = amdgcn::SGPRType::get(builder.getContext(), Register(),
                                     /*size=*/2, /*alignment=*/2);
  if (isPromotedBuffer(memref)) {
    callArgs.push_back(emitLDSOffset(builder, loc, memref, ldsCache));
    argTypes.push_back(indexTy);
  } else {
    auto [ptrVal, byteStride] = decomposeGlobalMemref(builder, loc, memref);
    callArgs.push_back(ptrVal);
    argTypes.push_back(sx2Ty);
    callArgs.push_back(byteStride);
    argTypes.push_back(indexTy);
  }
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct ConvertAirChannelToAMDGCN
    : public PassWrapper<ConvertAirChannelToAMDGCN,
                         InterfacePass<aster::ModuleOpInterface>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertAirChannelToAMDGCN)
  StringRef getArgument() const override {
    return "convert-air-channel-to-amdgcn";
  }
  StringRef getDescription() const override {
    return "Convert air.channel.put/get pairs to AMDGCN library calls";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<ptr::PtrDialect>();
    registry.insert<lsir::LSIRDialect>();
    registry.insert<amdgcn::AMDGCNDialect>();
  }

  void runOnOperation() override {
    Operation *moduleOp = getOperation();
    MLIRContext *ctx = &getContext();

    Operation *declParent = moduleOp;
    if (isa<mlir::ModuleOp>(moduleOp))
      moduleOp->walk([&](amdgcn::ModuleOp m) { declParent = m; });
    auto &declBlock = declParent->getRegion(0).front();

    OpBuilder builder(ctx);
    SmallVector<Operation *> toErase;
    DenseMap<Value, Value> ldsCache;

    // ---------------------------------------------------------------
    // Path 1: Convert air.dma_memcpy_nd directly (no channels).
    // ---------------------------------------------------------------
    moduleOp->walk([&](xilinx::air::DmaMemcpyNdOp dma) {
      Value dst = dma.getDstMemref();
      Value src = dma.getSrcMemref();
      auto dstTy = dyn_cast<MemRefType>(dst.getType());
      if (!dstTy)
        return;

      builder.setInsertionPoint(dma);
      Location loc = dma.getLoc();

      std::string name = buildCopyFuncName(dstTy);

      SmallVector<Value> callArgs;
      SmallVector<Type> argTypes;

      // src: if the DMA has src offsets/sizes/strides, create a subview.
      Value srcForDecompose = src;
      auto srcOffsets = dma.getSrcOffsets();
      auto srcSizes = dma.getSrcSizes();
      auto srcStrides = dma.getSrcStrides();
      if (!srcOffsets.empty()) {
        SmallVector<OpFoldResult> offsets, sizes, strides;
        for (auto v : srcOffsets)
          offsets.push_back(v);
        for (auto v : srcSizes)
          sizes.push_back(v);
        for (auto v : srcStrides)
          strides.push_back(v);
        srcForDecompose = memref::SubViewOp::create(
            builder, loc, src, offsets, sizes, strides);
      }
      emitDecomposedArgs(builder, loc, srcForDecompose, callArgs, argTypes,
                         ldsCache);
      // dst args.
      emitDecomposedArgs(builder, loc, dst, callArgs, argTypes, ldsCache);

      auto funcTy = builder.getFunctionType(argTypes, {});
      ensureDecl(builder, declBlock, loc, name, funcTy);
      func::CallOp::create(builder, loc, name, TypeRange{}, callArgs);

      toErase.push_back(dma);
    });

    // ---------------------------------------------------------------
    // Path 2: Convert air.channel.put/get pairs (if channels present).
    // ---------------------------------------------------------------
    DenseMap<StringRef, SmallVector<xilinx::air::ChannelPutOp>> putsByChannel;
    moduleOp->walk([&](xilinx::air::ChannelPutOp put) {
      putsByChannel[put.getChanName()].push_back(put);
    });

    moduleOp->walk([&](xilinx::air::ChannelGetOp get) {
      StringRef chanName = get.getChanName();
      auto it = putsByChannel.find(chanName);
      if (it == putsByChannel.end() || it->second.empty())
        return;

      xilinx::air::ChannelPutOp put = it->second.front();

      Value src = put.getSrc();
      Value dst = get.getDst();
      auto dstTy = dyn_cast<MemRefType>(dst.getType());
      if (!dstTy)
        return;

      builder.setInsertionPoint(get);
      Location loc = get.getLoc();

      std::string name = buildCopyFuncName(dstTy);

      SmallVector<Value> callArgs;
      SmallVector<Type> argTypes;

      // src args.
      emitDecomposedArgs(builder, loc, src, callArgs, argTypes, ldsCache);
      // dst args.
      emitDecomposedArgs(builder, loc, dst, callArgs, argTypes, ldsCache);

      auto funcTy = builder.getFunctionType(argTypes, {});
      ensureDecl(builder, declBlock, loc, name, funcTy);
      func::CallOp::create(builder, loc, name, TypeRange{}, callArgs);

      toErase.push_back(get);
      toErase.push_back(put);
    });

    for (auto *op : toErase)
      op->erase();

    // Clean up channel declarations that are now unused.
    SmallVector<Operation *> deadChannels;
    moduleOp->walk([&](xilinx::air::ChannelOp chan) {
      deadChannels.push_back(chan);
    });
    for (auto *op : deadChannels)
      op->erase();
  }
};

} // namespace

namespace mlir::aster::mlir_air {
std::unique_ptr<Pass> createConvertAirChannelToAMDGCN() {
  return std::make_unique<ConvertAirChannelToAMDGCN>();
}
} // namespace mlir::aster::mlir_air

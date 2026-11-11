//===-- GCNPreRAOptimizations.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass combines split register tuple initialization into a single pseudo:
///
///   undef %0.sub1:sreg_64 = S_MOV_B32 1
///   %0.sub0:sreg_64 = S_MOV_B32 2
/// =>
///   %0:sreg_64 = S_MOV_B64_IMM_PSEUDO 0x200000001
///
/// This is to allow rematerialization of a value instead of spilling. It is
/// supposed to be done after register coalescer to allow it to do its job and
/// before actual register allocation to allow rematerialization.
///
/// Right now the pass only handles 64 bit SGPRs with immediate initializers,
/// although the same shall be possible with other register classes and
/// instructions if necessary.
///
/// This pass also adds register allocation hints to COPY.
/// The hints will be post-processed by SIRegisterInfo::getRegAllocationHints.
/// When using True16, we often see COPY moving a 16-bit value between a VGPR_32
/// and a VGPR_16. If we use the VGPR_16 that corresponds to the lo16 bits of
/// the VGPR_32, the COPY can be completely eliminated.
///
/// This pass also unties relaxable tied operands whose input is a copy of a
/// register that stays live after the instruction. Such an instruction can
/// either overwrite its accumulator in place or write a disjoint result
/// register. When the source stays live anyway, the disjoint form avoids the
/// copy that the tied form needs to preserve it:
///
///   %t = COPY %src
///   %t = MFMA ..., %t(tied-def 0)
///   ... use of %src
/// =>
///   early-clobber %t = MFMA ..., %src
///   ... use of %src
///
//===----------------------------------------------------------------------===//

#include "GCNPreRAOptimizations.h"
#include "AMDGPU.h"
#include "GCNPreRAAntiHints.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-pre-ra-optimizations"

static cl::opt<bool>
    EnableAntiHints("amdgpu-anti-hints", cl::Hidden,
                    cl::desc("Enable register allocation anti-hints."),
                    cl::init(true));

namespace {

class GCNPreRAOptimizationsImpl {
private:
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  MachineRegisterInfo *MRI;
  LiveIntervals *LIS;
  TargetSchedModel SchedModel;

  bool processReg(Register Reg);
  void hintTrue16Copy(const MachineInstr &MI);
  bool optimizeBVHStack(MachineInstr &MI);
  bool isLiveAfter(Register Reg, SlotIndex Idx) const;
  bool allLanesLiveAfter(Register Reg, SlotIndex CopyIdx, SlotIndex Idx) const;
  MachineInstr *shouldUntieMAI(const MachineInstr &MAI) const;
  void untieMAI(MachineInstr &MAI, MachineInstr &Copy);
  bool relaxMAITies(MachineFunction &MF);

public:
  GCNPreRAOptimizationsImpl(LiveIntervals *LS) : LIS(LS) {}
  bool run(MachineFunction &MF);
};

class GCNPreRAOptimizationsLegacy : public MachineFunctionPass {
public:
  static char ID;

  GCNPreRAOptimizationsLegacy() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return "AMDGPU Pre-RA optimizations";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveIntervalsWrapperPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};
} // End anonymous namespace.

INITIALIZE_PASS_BEGIN(GCNPreRAOptimizationsLegacy, DEBUG_TYPE,
                      "AMDGPU Pre-RA optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervalsWrapperPass)
INITIALIZE_PASS_END(GCNPreRAOptimizationsLegacy, DEBUG_TYPE,
                    "Pre-RA optimizations", false, false)

char GCNPreRAOptimizationsLegacy::ID = 0;

char &llvm::GCNPreRAOptimizationsID = GCNPreRAOptimizationsLegacy::ID;

bool GCNPreRAOptimizationsImpl::processReg(Register Reg) {
  MachineInstr *Def0 = nullptr;
  MachineInstr *Def1 = nullptr;
  uint64_t Init = 0;
  bool Changed = false;
  SmallSet<Register, 32> ModifiedRegs;
  bool IsAGPRDst = TRI->isAGPRClass(MRI->getRegClass(Reg));

  for (MachineInstr &I : MRI->def_instructions(Reg)) {
    switch (I.getOpcode()) {
    default:
      return false;
    case AMDGPU::V_ACCVGPR_WRITE_B32_e64:
      break;
    case AMDGPU::COPY: {
      // Some subtargets cannot do an AGPR to AGPR copy directly, and need an
      // intermdiate temporary VGPR register. Try to find the defining
      // accvgpr_write to avoid temporary registers.

      if (!IsAGPRDst)
        return false;

      Register SrcReg = I.getOperand(1).getReg();

      if (!SrcReg.isVirtual())
        break;

      // Check if source of copy is from another AGPR.
      bool IsAGPRSrc = TRI->isAGPRClass(MRI->getRegClass(SrcReg));
      if (!IsAGPRSrc)
        break;

      // def_instructions() does not look at subregs so it may give us a
      // different instruction that defines the same vreg but different subreg
      // so we have to manually check subreg.
      Register SrcSubReg = I.getOperand(1).getSubReg();
      for (auto &Def : MRI->def_instructions(SrcReg)) {
        if (SrcSubReg != Def.getOperand(0).getSubReg())
          continue;

        if (Def.getOpcode() == AMDGPU::V_ACCVGPR_WRITE_B32_e64) {
          const MachineOperand &DefSrcMO = Def.getOperand(1);

          // Immediates are not an issue and can be propagated in
          // postrapseudos pass. Only handle cases where defining
          // accvgpr_write source is a vreg.
          if (DefSrcMO.isReg() && DefSrcMO.getReg().isVirtual()) {
            // Propagate source reg of accvgpr write to this copy instruction
            I.getOperand(1).setReg(DefSrcMO.getReg());
            I.getOperand(1).setSubReg(DefSrcMO.getSubReg());

            // Reg uses were changed, collect unique set of registers to update
            // live intervals at the end.
            ModifiedRegs.insert(DefSrcMO.getReg());
            ModifiedRegs.insert(SrcReg);

            Changed = true;
          }

          // Found the defining accvgpr_write, stop looking any further.
          break;
        }
      }
      break;
    }
    case AMDGPU::S_MOV_B32:
      if (I.getOperand(0).getReg() != Reg || !I.getOperand(1).isImm() ||
          I.getNumOperands() != 2)
        return false;

      switch (I.getOperand(0).getSubReg()) {
      default:
        return false;
      case AMDGPU::sub0:
        if (Def0)
          return false;
        Def0 = &I;
        Init |= Lo_32(I.getOperand(1).getImm());
        break;
      case AMDGPU::sub1:
        if (Def1)
          return false;
        Def1 = &I;
        Init |= static_cast<uint64_t>(I.getOperand(1).getImm()) << 32;
        break;
      }
      break;
    }
  }

  // For AGPR reg, check if live intervals need to be updated.
  if (IsAGPRDst) {
    if (Changed) {
      for (Register RegToUpdate : ModifiedRegs) {
        LIS->removeInterval(RegToUpdate);
        LIS->createAndComputeVirtRegInterval(RegToUpdate);
      }
    }

    return Changed;
  }

  // For SGPR reg, check if we can combine instructions.
  if (!Def0 || !Def1 || Def0->getParent() != Def1->getParent())
    return Changed;

  LLVM_DEBUG(dbgs() << "Combining:\n  " << *Def0 << "  " << *Def1
                    << "    =>\n");

  if (SlotIndex::isEarlierInstr(LIS->getInstructionIndex(*Def1),
                                LIS->getInstructionIndex(*Def0)))
    std::swap(Def0, Def1);

  LIS->RemoveMachineInstrFromMaps(*Def0);
  LIS->RemoveMachineInstrFromMaps(*Def1);
  auto NewI = BuildMI(*Def0->getParent(), *Def0, Def0->getDebugLoc(),
                      TII->get(AMDGPU::S_MOV_B64_IMM_PSEUDO), Reg)
                  .addImm(Init);

  Def0->eraseFromParent();
  Def1->eraseFromParent();
  LIS->InsertMachineInstrInMaps(*NewI);
  LIS->removeInterval(Reg);
  LIS->createAndComputeVirtRegInterval(Reg);

  LLVM_DEBUG(dbgs() << "  " << *NewI);

  return true;
}

bool GCNPreRAOptimizationsLegacy::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;
  LiveIntervals *LIS = &getAnalysis<LiveIntervalsWrapperPass>().getLIS();
  return GCNPreRAOptimizationsImpl(LIS).run(MF);
}

PreservedAnalyses
GCNPreRAOptimizationsPass::run(MachineFunction &MF,
                               MachineFunctionAnalysisManager &MFAM) {
  LiveIntervals *LIS = &MFAM.getResult<LiveIntervalsAnalysis>(MF);
  GCNPreRAOptimizationsImpl(LIS).run(MF);
  return PreservedAnalyses::all();
}

void GCNPreRAOptimizationsImpl::hintTrue16Copy(const MachineInstr &MI) {
  Register Dst = MI.getOperand(0).getReg();
  Register Src = MI.getOperand(1).getReg();
  const TargetRegisterClass *DstRC = TRI->getRegClassForReg(*MRI, Dst);
  bool IsDst16Bit = AMDGPU::VGPR_16RegClass.hasSubClassEq(DstRC);
  if (Dst.isVirtual() && IsDst16Bit && Src.isPhysical() &&
      TRI->getRegClassForReg(*MRI, Src) == &AMDGPU::VGPR_32RegClass)
    MRI->setRegAllocationHint(Dst, 0, TRI->getSubReg(Src, AMDGPU::lo16));
  if (Src.isVirtual() && MRI->getRegClass(Src) == &AMDGPU::VGPR_16RegClass &&
      Dst.isPhysical() && DstRC == &AMDGPU::VGPR_32RegClass)
    MRI->setRegAllocationHint(Src, 0, TRI->getSubReg(Dst, AMDGPU::lo16));
  if (!Dst.isVirtual() || !Src.isVirtual())
    return;
  if (MRI->getRegClass(Dst) == &AMDGPU::VGPR_32RegClass &&
      MRI->getRegClass(Src) == &AMDGPU::VGPR_16RegClass) {
    MRI->setRegAllocationHint(Dst, AMDGPURI::Size32, Src);
    MRI->setRegAllocationHint(Src, AMDGPURI::Size16, Dst);
  }
  if (IsDst16Bit && MRI->getRegClass(Src) == &AMDGPU::VGPR_32RegClass)
    MRI->setRegAllocationHint(Dst, AMDGPURI::Size16, Src);
}

bool GCNPreRAOptimizationsImpl::optimizeBVHStack(MachineInstr &MI) {
  SmallVector<Register, 2> UseRegs;

  // Find BVH sources for this DS_BVH_STACK instruction.
  auto CheckUse = [&](MachineOperand &Use) {
    Register Reg = Use.getReg();
    for (const MachineInstr &Src : MRI->def_instructions(Reg)) {
      if (!SIInstrInfo::isImage(Src))
        continue;
      const AMDGPU::MIMGInfo *Info = AMDGPU::getMIMGInfo(Src.getOpcode());
      const AMDGPU::MIMGBaseOpcodeInfo *BaseInfo =
          AMDGPU::getMIMGBaseOpcodeInfo(Info->BaseOpcode);
      if (!BaseInfo->BVH)
        continue;
      UseRegs.push_back(Reg);
      break;
    }
  };
  CheckUse(*TII->getNamedOperand(MI, AMDGPU::OpName::data0));
  CheckUse(*TII->getNamedOperand(MI, AMDGPU::OpName::data1));

  if (UseRegs.empty())
    return false;

  // Add implicit uses for entire BVH source registers.
  // This avoids partial reallocation of register which could
  // introduce a premature s_wait_bvhcnt.
  for (Register Reg : UseRegs) {
    MI.addOperand(MachineOperand::CreateReg(Reg, false, true));
    LIS->removeInterval(Reg);
    LIS->createAndComputeVirtRegInterval(Reg);
  }
  LLVM_DEBUG(dbgs() << "Added implicit uses to: " << MI);

  return true;
}

/// Return true if \p Reg is live after the instruction at \p Idx.
bool GCNPreRAOptimizationsImpl::isLiveAfter(Register Reg, SlotIndex Idx) const {
  return LIS->getInterval(Reg).Query(Idx).valueOut() != nullptr;
}

/// Return true if every lane of \p Reg live at \p CopyIdx is still live
/// after the instruction at \p Idx.
bool GCNPreRAOptimizationsImpl::allLanesLiveAfter(Register Reg,
                                                  SlotIndex CopyIdx,
                                                  SlotIndex Idx) const {
  for (const LiveInterval::SubRange &SR : LIS->getInterval(Reg).subranges())
    if (SR.Query(CopyIdx).valueIn() && !SR.Query(Idx).valueOut())
      return false;
  return true;
}

/// Decide whether it is beneficial for register allocation to convert the MFMA
/// to its untied form. Returns the copy instruction that supplies the tied
/// operand if the MFMA should be untied or nullptr when the tied form should be
/// kept.
MachineInstr *
GCNPreRAOptimizationsImpl::shouldUntieMAI(const MachineInstr &MAI) const {
  const MachineOperand *Src2 = TII->getNamedOperand(MAI, AMDGPU::OpName::src2);
  if (!Src2 || !Src2->isReg() || !Src2->isTied())
    return nullptr;
  unsigned UseIdx = MAI.getOperandNo(Src2);
  if (!MAI.isTiedMayBeDisjoint(UseIdx))
    return nullptr;
  const MachineOperand &DefMO = MAI.getOperand(MAI.findTiedOperandIdx(UseIdx));
  Register TiedReg = Src2->getReg();
  if (!TiedReg.isVirtual() || Src2->getSubReg() || DefMO.getSubReg())
    return nullptr;
  // The tied register must not appear in any other operand.
  for (const MachineOperand &MO : MAI.operands())
    if (&MO != Src2 && &MO != &DefMO && MO.isReg() && MO.getReg() == TiedReg)
      return nullptr;

  // Identify the COPY instruction supplying the tied operand.
  SlotIndex MIIdx = LIS->getInstructionIndex(MAI);
  const VNInfo *VNI = LIS->getInterval(TiedReg).Query(MIIdx).valueIn();
  if (!VNI || VNI->isPHIDef())
    return nullptr;
  MachineInstr *Copy = LIS->getInstructionFromIndex(VNI->def);
  if (!Copy || !Copy->isFullCopy())
    return nullptr;
  if (Copy->getParent() != MAI.getParent())
    return nullptr;
  Register SrcReg = Copy->getOperand(1).getReg();
  if (!SrcReg.isVirtual() || SrcReg == TiedReg)
    return nullptr;

  // If the copy source register isn't used after the MAI, keep the tied form as
  // the copy source register and the accumulator & result can occupy the same
  // register.
  if (!isLiveAfter(SrcReg, MIIdx))
    return nullptr;

  // If not all lanes of the source register are needed after the MAI, it is
  // better to keep the tied form. Consider this example:
  // ```
  // %1 = COPY %0
  // %1 = MAI ..., %1
  // ... = ..., %0.sub1
  // ```
  // If %0 is a tuple of N registers, we need N+1 registers with the tied form.
  //
  // With the untied form, we get:
  // ```
  // %1 = MAI ..., %0
  // ... = ..., %0.sub1
  // ```
  // In this case, because %0 and %1 need to be disjoint register tuples for
  // MAI, we need 2 * N registers.
  SlotIndex CopyIdx = LIS->getInstructionIndex(*Copy);
  if (!allLanesLiveAfter(SrcReg, CopyIdx, MIIdx))
    return nullptr;

  // If the copy source register is redefined between copy and MAI, keep the
  // tied form.
  const LiveInterval &SrcLI = LIS->getInterval(SrcReg);
  if (SrcLI.Query(CopyIdx).valueIn() != SrcLI.Query(MIIdx).valueIn())
    return nullptr;

  // If the copy destination register is read between copy and MAI, we can't
  // remove the copy, keep the tied form.
  for (auto I = std::next(Copy->getIterator()); &*I != &MAI; ++I)
    if (!I->isDebugInstr() && I->readsRegister(TiedReg, TRI))
      return nullptr;

  // If copy source and destination register have different classes, keep the
  // copy and tied form.
  const TargetRegisterClass *OpRC = MAI.getRegClassConstraint(UseIdx, TII, TRI);
  if (OpRC && !TRI->getCommonSubClass(OpRC, MRI->getRegClass(SrcReg)))
    return nullptr;

  return Copy;
}

/// Convert the MFMA to its untied form and drop the COPY instruction.
void GCNPreRAOptimizationsImpl::untieMAI(MachineInstr &MAI,
                                         MachineInstr &Copy) {
  MachineOperand &UseMO = *TII->getNamedOperand(MAI, AMDGPU::OpName::src2);
  unsigned UseIdx = MAI.getOperandNo(&UseMO);
  MachineOperand &DefMO = MAI.getOperand(MAI.findTiedOperandIdx(UseIdx));
  Register TiedReg = UseMO.getReg();
  Register SrcReg = Copy.getOperand(1).getReg();
  if (const TargetRegisterClass *OpRC =
          MAI.getRegClassConstraint(UseIdx, TII, TRI)) {
    [[maybe_unused]] bool Constrained = MRI->constrainRegClass(SrcReg, OpRC);
    assert(Constrained && "copy source has no common register class");
  }

  // Update debug instructions between copy and MAI. There can be no non-debug
  // uses or definitions, otherwise we would have kept the tied form.
  for (auto I = std::next(Copy.getIterator()); &*I != &MAI; ++I)
    if (I->isDebugInstr())
      for (MachineOperand &MO : I->operands())
        if (MO.isReg() && MO.getReg() == TiedReg)
          MO.setReg(SrcReg);

  MAI.untieRegOperand(UseIdx);
  UseMO.setReg(SrcReg);
  // There is always another use of SrcReg after the MAI, otherwise we would
  // have kept the tied form, so it cannot be killed here.
  UseMO.setIsKill(false);
  DefMO.setIsEarlyClobber(true);
  LIS->RemoveMachineInstrFromMaps(Copy);
  Copy.eraseFromParent();

  SmallVector<LiveInterval *, 4> SplitLIs;
  for (Register Reg : {TiedReg, SrcReg}) {
    LIS->removeInterval(Reg);
    LiveInterval &NewLI = LIS->createAndComputeVirtRegInterval(Reg);
    LIS->splitSeparateComponents(NewLI, SplitLIs);
  }
  assert(SplitLIs.empty() && "rewrite split a live interval");
}

/// Convert MFMAs in the function to their untied form if it benefits register
/// allocation.
bool GCNPreRAOptimizationsImpl::relaxMAITies(MachineFunction &MF) {
  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!TII->isMAI(MI))
        continue;
      if (MachineInstr *Copy = shouldUntieMAI(MI)) {
        untieMAI(MI, *Copy);
        Changed = true;
      }
    }
  }
  return Changed;
}

bool GCNPreRAOptimizationsImpl::run(MachineFunction &MF) {
  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  TII = ST.getInstrInfo();
  MRI = &MF.getRegInfo();
  TRI = ST.getRegisterInfo();

  bool Changed = false;
  if (ST.hasMAIInsts())
    Changed |= relaxMAITies(MF);

  // Anti-hints only steer register allocation, so they do not count as a
  // modification of the function.
  if (EnableAntiHints) {
    SchedModel.init(&ST);
    AMDGPU::HazardContext HCtx{TII, TRI, MRI, LIS, &ST, &SchedModel};
    AMDGPU::applyAntiHintRules(MF, HCtx);
  }

  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    if (!LIS->hasInterval(Reg))
      continue;
    const TargetRegisterClass *RC = MRI->getRegClass(Reg);
    if ((RC->getSizeInBits() != 64 || !TRI->isSGPRClass(RC)) &&
        (ST.hasGFX90AInsts() || !TRI->isAGPRClass(RC)))
      continue;

    Changed |= processReg(Reg);
  }

  const bool HasBVHStack = ST.hasBVHDualAndBVH8Insts();
  const bool HasRealTrue16 = ST.useRealTrue16Insts();

  if (!HasRealTrue16 && !HasBVHStack)
    return Changed;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      // Add RA hints to improve True16 COPY elimination.
      if (HasRealTrue16 && MI.getOpcode() == AMDGPU::COPY) {
        hintTrue16Copy(MI);
        continue;
      }
      // Add implicit uses to avoid early wait on intersect ray instructions.
      if (HasBVHStack &&
          (MI.getOpcode() == AMDGPU::DS_BVH_STACK_RTN_B32 ||
           MI.getOpcode() == AMDGPU::DS_BVH_STACK_PUSH8_POP1_RTN_B32 ||
           MI.getOpcode() == AMDGPU::DS_BVH_STACK_PUSH8_POP2_RTN_B64)) {
        Changed |= optimizeBVHStack(MI);
        continue;
      }
    }
  }

  return Changed;
}

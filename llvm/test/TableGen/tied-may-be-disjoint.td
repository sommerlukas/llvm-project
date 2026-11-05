// RUN: llvm-tblgen -gen-instr-info -I %p/../../include %s | FileCheck %s
// RUN: not llvm-tblgen -gen-instr-info -I %p/../../include -DERR_UNTIED %s 2>&1 | FileCheck %s --check-prefix=ERR-UNTIED
// RUN: not llvm-tblgen -gen-instr-info -I %p/../../include -DERR_NOT_CODEGEN_ONLY %s 2>&1 | FileCheck %s --check-prefix=ERR-NOT-CODEGEN-ONLY
// RUN: not llvm-tblgen -gen-instr-info -I %p/../../include -DERR_MULTIPLE %s 2>&1 | FileCheck %s --check-prefix=ERR-MULTIPLE
// RUN: not llvm-tblgen -gen-instr-info -I %p/../../include -DERR_MISSING_OPERAND %s 2>&1 | FileCheck %s --check-prefix=ERR-MISSING-OPERAND

include "llvm/Target/Target.td"

def archInstrInfo : InstrInfo {}
def arch : Target {
  let InstructionSet = archInstrInfo;
}

def R0 : Register<"r0">;
def R32 : RegisterClass<"MyNS", [i32], 0, (add R0)>;

// CHECK: { MyNS::R32RegClassID, 0, MCOI::OPERAND_REGISTER, 0 }, { MyNS::R32RegClassID, 0|(1<<MCOI::TiedMayBeDisjoint), MCOI::OPERAND_REGISTER, MCOI_TIED_TO(0) },
class TestInst : Instruction {
  let Namespace = "MyNS";
  let OutOperandList = (outs R32:$dst);
  let InOperandList = (ins R32:$src);
  let Constraints = "$dst = $src, @tied_may_be_disjoint $src";
  let isCodeGenOnly = 1;
  let hasSideEffects = false;
  field bits<8> Inst;
  field bits<8> SoftFail = 0;
}

def Inst : TestInst;

#ifdef ERR_UNTIED
// ERR-UNTIED: '@tied_may_be_disjoint' requires a preceding tied constraint on operand '$src' in 'Untied'
def Untied : TestInst {
  let Constraints = "@tied_may_be_disjoint $src";
}
#endif

#ifdef ERR_NOT_CODEGEN_ONLY
// ERR-NOT-CODEGEN-ONLY: '@tied_may_be_disjoint' is only supported on codegen-only instructions
def NotCodeGenOnly : TestInst {
  let isCodeGenOnly = 0;
}
#endif

#ifdef ERR_MULTIPLE
// ERR-MULTIPLE: Operand '$src' of 'Multiple' has multiple '@tied_may_be_disjoint' constraints!
def Multiple : TestInst {
  let Constraints = "$dst = $src, @tied_may_be_disjoint $src, @tied_may_be_disjoint $src";
}
#endif

#ifdef ERR_MISSING_OPERAND
// ERR-MISSING-OPERAND: Illegal format for @tied_may_be_disjoint constraint in 'MissingOperand': '@tied_may_be_disjoint'
def MissingOperand : TestInst {
  let Constraints = "@tied_may_be_disjoint";
}
#endif

defm : RemapAllTargetPseudoPointerOperands<R32>;

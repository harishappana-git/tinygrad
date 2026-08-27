from __future__ import annotations
import ctypes, os, struct
from dataclasses import dataclass, replace
from typing import Callable, Sequence
from tinygrad.runtime.autogen import libc, mesa
from test.mockgpu.qcom.pm4 import PM4Packet, PM4Type4Packet, PM4Type7Packet

# Payload fields and units follow Mesa 25.2.7 at 461196a1c827769168304ff3f5b36360f16618ca:
# adreno_pm4.xml, a6xx.xml, a6xx_descriptors.xml, tu_shader.cc, tu_cmd_buffer.cc, ir3_shader.h,
# ir3.xml, ir3-common.xml, ir3-cat[0-7].xml, ir3.h, ir3.c, ir3_a6xx.c, ir3_compiler.c, ir3_compiler_nir.c,
# ir3_delay.c, ir3_legalize.c,
# ir3_nir_analyze_ubo_ranges.c, ir3_nir_imul.py, ir3_nir_lower_64b.c, ir3_rpt.c, nir_lower_int64.c,
# nir_lower_system_values.c, nir_opcodes.py, isaspec.h, and isaspec_decode_impl.c.

@dataclass(frozen=True)
class A630MemoryRange:
  address:int
  size:int
  read:bool
  write:bool
  purpose:str

@dataclass(frozen=True)
class A630LoadState:
  kind:str
  address:int
  size:int
  units:int

@dataclass(frozen=True)
class A630Wait:
  word_offset:int
  address:int
  reference:int
  mask:int

@dataclass(frozen=True)
class A630Write:
  word_offset:int
  address:int
  size:int
  value:int|None
  purpose:str

@dataclass(frozen=True)
class A630IR3Operand:
  kind:str
  value:int

@dataclass(frozen=True)
class A630IR3Instruction:
  index:int
  category:int
  raw:int
  name:str|None
  fields:tuple[tuple[str, int|str], ...]
  opcode:str|None = None
  dst:A630IR3Operand|None = None
  srcs:tuple[A630IR3Operand, ...] = ()

@dataclass(frozen=True)
class A630ExecutionWrite:
  address:int
  data:bytes

@dataclass
class A630ExecutionBudget:
  lane_instruction_steps:int = 0
  memory_events:int = 0

@dataclass
class _A630LaneState:
  full:dict[int, int]
  half:dict[int, int]
  predicates:dict[int, bool]

@dataclass
class _A630WorkgroupState:
  pc:int
  lanes:list[_A630LaneState]
  shared:dict[int, int]
  active:list[bool]
  predicated:bool
  local:dict[int, int]
  steps:int

@dataclass(frozen=True)
class A630Dispatch:
  word_offset:int
  registers:tuple[tuple[int, int], ...]
  loads:tuple[A630LoadState, ...]
  shader_address:int
  shader_size:int
  shader_image:bytes
  constants_address:int
  constants_size:int
  constants_image:bytes
  stack_base:int
  stack_offset:int
  local_size:tuple[int, int, int]
  global_size:tuple[int, int, int]
  groups:tuple[int, int, int]
  instructions:tuple[A630IR3Instruction, ...] = ()

@dataclass(frozen=True)
class A630Submission:
  dispatches:tuple[A630Dispatch, ...]
  memory_ranges:tuple[A630MemoryRange, ...]
  waits:tuple[A630Wait, ...]
  writes:tuple[A630Write, ...]

@dataclass(frozen=True)
class _A630MemoryScheduleState:
  force_ss:bool = False
  force_sy:bool = False
  needs_ss:frozenset[int] = frozenset()
  needs_sy:frozenset[int] = frozenset()
  needs_ss_war_full:frozenset[int] = frozenset()
  needs_ss_war_half:frozenset[int] = frozenset()
  needs_ss_or_sy_war_full:frozenset[int] = frozenset()

@dataclass(frozen=True)
class _A630DelayState:
  full_alu:tuple[tuple[int, int], ...] = ()
  full_non_alu:tuple[tuple[int, int], ...] = ()
  half_alu:tuple[tuple[int, int], ...] = ()
  half_non_alu:tuple[tuple[int, int], ...] = ()
  predicate:tuple[tuple[int, int], ...] = ()

Resolver = Callable[[int, int], memoryview]
ReadObserver = Callable[[int, int, str], None]
_MAX_INVOCATIONS = 0x10000
_MAX_CONTROL_FLOW_ITERATIONS = 0x1000
_MAX_SHADER_INSTRUCTIONS = 0x1000
# Emulator resource policy, not an A630 hardware limit. A largest-shape dispatch may execute up to 64 decoded
# instruction steps per invocation; smaller dispatches retain proportionally deeper straight-line/control programs.
_MAX_LANE_INSTRUCTION_STEPS = 64 * _MAX_INVOCATIONS
# Emulator resource policy, not an A630 hardware limit. It bounds work and the retirement journal while retaining
# four memory operations for every invocation at the largest admitted dispatch.
_MAX_MEMORY_EVENTS = 4 * _MAX_INVOCATIONS
# Emulator policy bound, not an A630 hardware limit. It caps immutable state copied while staging one command.
_MAX_A630_SNAPSHOT_BYTES = 1 << 20
# Emulator resource policy, not an A630 hardware limit. It bounds fail-closed CFG dataflow validation even for a
# maximum-size image whose branches repeatedly grow the abstract scoreboard or fixed-delay state.
_MAX_SCHEDULE_VALIDATION_STEPS = 8 * _MAX_SHADER_INSTRUCTIONS
_OPERAND_CONTRACTS:dict[str, tuple[str|None, tuple[tuple[str, ...], ...]]] = {
  "nop":(None, ((),)), "end":(None, ((),)), "bar.g":(None, ((),)), "prede":(None, ((),)),
  "br.p0":(None, (("pred", "iim"),)), "jump":(None, (("iim",),)), "predt.p0":(None, (("pred",),)),
  "mov.u32":("gpr", (("gpr",), ("shared",), ("const",), ("uim",))),
  "ashr.b":("gpr", (("gpr", "iim"), ("shared", "iim"))),
  "shl.b":("gpr", (("gpr", "iim"), ("shared", "iim"))),
  "shrg":("gpr", (("iim", "gpr", "gpr"),)),
  "add.u":("gpr", (("gpr", "gpr"), ("const", "gpr"), ("gpr", "const"), ("gpr", "iim"))),
  "sub.u":("gpr", (("gpr", "gpr"),)), "shr.b":("gpr", (("gpr", "gpr"),)),
  "max.s":("gpr", (("gpr", "gpr"),)), "max.u":("gpr", (("gpr", "gpr"),)),
  "xor.b":("gpr", (("gpr", "gpr"),)), "and.b":("gpr", (("gpr", "gpr"),)),
  "or.b":("gpr", (("gpr", "gpr"),)),
  "mull.u":("gpr", (("gpr", "gpr"), ("shared", "iim"), ("gpr", "iim"))),
  "madsh.m16":("gpr", (("gpr", "gpr", "gpr"), ("const", "gpr", "gpr"))),
  "cmps.s.lt":("half", (("gpr", "gpr"),)), "cmps.s.eq":("half", (("gpr", "gpr"),)),
  "cmps.u.lt":("half", (("gpr", "gpr"), ("gpr", "const"))),
  "cov.u16s32":("gpr", (("half",),)),
  "add.f":("gpr", (("gpr", "gpr"), ("gpr", "flut"), ("flut", "gpr"))),
  "add.f.rpt4":("gpr", (("gpr", "gpr"),)), "add.u.rpt2":("gpr", (("const", "gpr"),)),
  "cmps.s.ge.p0":("pred", (("gpr", "const"),)), "cmps.s.eq.p0":("pred", (("gpr", "iim"),)),
  "ldg.u32":("gpr", (("gpr",),)), "ldg.u32x4":("gpr", (("gpr",),)),
  "stg.u32":(None, (("gpr", "gpr"),)), "stg.u32x4":(None, (("gpr", "gpr"),)),
  "stg.u8":(None, (("gpr", "half"),)), "ldl.u32x4":("gpr", (("gpr",),)),
  "stl.u32":(None, (("gpr", "gpr"),)),
}

def _require(condition:bool, message:str):
  if not condition: raise ValueError(message)

def _field_values(fields:tuple[tuple[str, int|str], ...], name:str) -> tuple[int|str, ...]:
  return tuple(value for field,value in fields if field.partition(":align=")[0] == name)

def _same_int_field(fields:tuple[tuple[str, int|str], ...], name:str) -> int:
  values = _field_values(fields, name)
  _require(bool(values) and all(isinstance(value, int) and value == values[0] for value in values), f"inconsistent IR3 {name} field")
  assert isinstance(values[0], int)
  return values[0]

def _int_field_is(fields:tuple[tuple[str, int|str], ...], name:str, *allowed:int) -> bool:
  values = _field_values(fields, name)
  return bool(values) and all(isinstance(value, int) and value in allowed for value in values)

def _has_no_repeat(fields:tuple[tuple[str, int|str], ...]) -> bool:
  repeats = _field_values(fields, "REPEAT")
  return (bool(repeats) and all(value == 0 for value in repeats)) or \
         (not repeats and _int_field_is(fields, "NOP", 1, 2, 3))

def _register_operand(value:int, full:bool) -> A630IR3Operand|None:
  if 0 <= value < 0xc0: return A630IR3Operand("gpr" if full else "half", value)
  if full and 0xc0 <= value < 0xe0: return A630IR3Operand("shared", value)
  return None

def _multisrc_operand(encoded:int, full:bool) -> A630IR3Operand|None:
  selector = encoded >> 11 & 0x7
  if selector == 0 and encoded == encoded & 0xff: return _register_operand(encoded, full)
  if selector == 2 and encoded == 0x1000 | (encoded & 0x7ff): return A630IR3Operand("const", encoded & 0x7ff)
  if selector == 4 and encoded == 0x2000 | (encoded & 0x7ff):
    value = encoded & 0x7ff
    return A630IR3Operand("iim", value - 0x800 if value & 0x400 else value)
  if selector == 5 and encoded == 0x2800 | (encoded & 0x3ff): return A630IR3Operand("flut", encoded & 0x3ff)
  return None

def _normalize_ir3(raw:int, category:int, name:str|None,
                   fields:tuple[tuple[str, int|str], ...]) -> tuple[str|None, A630IR3Operand|None, tuple[A630IR3Operand, ...]]:
  if category == 0 and name == "nop" and raw & ~((0x7 << 40) | (1 << 44) | (1 << 60)) == 0 and \
     _int_field_is(fields, "REPEAT", *range(8)) and all(_int_field_is(fields, field, 0) for field in ("EQ", "JP")):
    return "nop", None, ()
  if category == 0 and name == "end" and raw == 6 << 55: return "end", None, ()
  if category == 0 and name == "br" and raw & ~0xffffffff == 0x0080000000000000 and \
     all(_int_field_is(fields, field, 0) for field in ("SY", "SS", "EQ", "JP", "INV1", "COMP1")):
    immediate = _same_int_field(fields, "IMMED")
    _require(0 <= immediate < 1 << 32, "invalid Cat0 branch immediate")
    return "br.p0", None, (A630IR3Operand("pred", 0),
                            A630IR3Operand("iim", immediate - (1 << 32) if immediate & 0x80000000 else immediate))
  if category == 0 and name == "jump" and raw & ~0xffffffff == 0x0100000000000000 and \
     all(_int_field_is(fields, field, 0) for field in ("SY", "SS", "JP")):
    immediate = _same_int_field(fields, "IMMED")
    _require(0 <= immediate < 1 << 32, "invalid Cat0 jump immediate")
    return "jump", None, (A630IR3Operand("iim", immediate - (1 << 32) if immediate & 0x80000000 else immediate),)
  # ir3-cat0.xml defines these exact predicate-region controls. PREDT captures p0.x for all fibers; PREDE
  # returns to unpredicated execution. Keep the implicit predicate operand explicit in the normalized contract.
  if category == 0 and name == "predt" and raw == 0x0682000000000000:
    return "predt.p0", None, (A630IR3Operand("pred", 0),)
  if category == 0 and name == "prede" and raw == 0x0782000000000000: return "prede", None, ()
  # These cat1 leaves have no NAME callback, so fixed leaf bits and typed callback fields identify them without parsing text.
  cat1_schedule = (1 << 44) | (1 << 60)
  mov_gpr_variable = (0xff << 32) | 0xff | cat1_schedule
  if category == 1 and raw & ~mov_gpr_variable == 0x200cc00000000000 and \
     (_same_int_field(fields, "SRC_TYPE"), _same_int_field(fields, "DST_TYPE"), _same_int_field(fields, "DST_HALF"),
      _same_int_field(fields, "HALF")) == (3, 3, 0, 0) and _has_no_repeat(fields) and \
     all(_int_field_is(fields, field, 0) for field in ("JP", "UL", "ROUND", "SRC_R", "LAST")):
    dst,src = _register_operand(_same_int_field(fields, "DST"), True), _register_operand(_same_int_field(fields, "SRC"), True)
    if dst is not None and dst.kind == "gpr" and src is not None: return "mov.u32", dst, (src,)
  mov_const_variable = (0xff << 32) | 0x7ff | cat1_schedule
  if category == 1 and raw & ~mov_const_variable == 0x202cc00000000000 and \
     (_same_int_field(fields, "SRC_TYPE"), _same_int_field(fields, "DST_TYPE"), _same_int_field(fields, "DST_HALF"),
      _same_int_field(fields, "HALF")) == (3, 3, 0, 0) and _has_no_repeat(fields) and \
     all(_int_field_is(fields, field, 0) for field in ("JP", "UL", "ROUND", "SRC_R")):
    dst = _register_operand(_same_int_field(fields, "DST"), True)
    if dst is not None and dst.kind == "gpr": return "mov.u32", dst, (A630IR3Operand("const", _same_int_field(fields, "SRC")),)
  mov_immediate_variable = (0xff << 32) | 0xffffffff | cat1_schedule
  if category == 1 and raw & ~mov_immediate_variable == 0x204cc00000000000 and \
     (_same_int_field(fields, "SRC_TYPE"), _same_int_field(fields, "DST_TYPE"), _same_int_field(fields, "DST_HALF")) == (3, 3, 0) and \
     _has_no_repeat(fields) and all(_int_field_is(fields, field, 0) for field in ("JP", "UL", "ROUND")):
    dst = _register_operand(_same_int_field(fields, "DST"), True)
    if dst is not None and dst.kind == "gpr": return "mov.u32", dst, (A630IR3Operand("uim", _same_int_field(fields, "SRC")),)
  cov_variable = (0xff << 32) | 0xff | (1 << 44) | (1 << 60)
  if category == 1 and raw & ~cov_variable == 0x2009400000000000:
    if (_same_int_field(fields, "SRC_TYPE"), _same_int_field(fields, "DST_TYPE"), _same_int_field(fields, "DST_HALF"),
        _same_int_field(fields, "HALF")) == (2, 5, 0, 1) and _has_no_repeat(fields) and \
       all(_int_field_is(fields, field, 0) for field in ("JP", "UL", "ROUND", "SRC_R")):
      return "cov.u16s32", A630IR3Operand("gpr", _same_int_field(fields, "DST")), \
             (A630IR3Operand("half", _same_int_field(fields, "SRC")),)
  # ir3_rpt.c merges four scalar ALU leaves only when the destination registers are consecutive; each source marked (r)
  # advances with the destination. Keep this vector form separate from scalar ADD.F and require the exact repeat contract.
  repeated_add_variable = (0xff << 32) | (0xff << 16) | 0xff | (1 << 44) | (1 << 60)
  if category == 2 and name == "add.f" and raw & ~repeated_add_variable == 0x40180b0000000000 and \
     (_same_int_field(fields, "REPEAT"), _same_int_field(fields, "DST_HALF")) == (3, 0) and \
     all(_int_field_is(fields, field, 0) for field in ("JP", "SAT", "UL", "EI", "LAST", "ABSNEG")) and \
     _int_field_is(fields, "SRC_R", 1) and (raw >> 52 & 1, raw >> 46 & 1) == (1, 0):
    dst = _register_operand(_same_int_field(fields, "DST"), True)
    srcs = (_multisrc_operand(_same_int_field(fields, "SRC1"), True),
            _multisrc_operand(_same_int_field(fields, "SRC2"), True))
    if dst is not None and dst.kind == "gpr" and dst.value + 3 < 0xc0 and \
       all(src is not None and src.kind == "gpr" and src.value + 3 < 0xc0 for src in srcs):
      assert srcs[0] is not None and srcs[1] is not None
      return "add.f.rpt4", dst, (srcs[0], srcs[1])
  # ir3_rpt.c advances a consecutive destination and only sources carrying IR3_REG_R. This exact two-component
  # addressing form therefore keeps the constant fixed while advancing the GPR source and destination once.
  if category == 2 and name == "add.u" and _field_values(fields, "REPEAT") == (1,) and \
     _same_int_field(fields, "DST_HALF") == 0 and \
     tuple(_field_values(fields, "SRC_R")) == (0, 1) and \
     all(_int_field_is(fields, field, 0) for field in ("JP", "SAT", "UL", "EI", "LAST", "ABSNEG")) and \
     (raw >> 52 & 1, raw >> 46 & 1) == (1, 0):
    dst = _register_operand(_same_int_field(fields, "DST"), True)
    srcs = (_multisrc_operand(_same_int_field(fields, "SRC1"), True),
            _multisrc_operand(_same_int_field(fields, "SRC2"), True))
    if dst is not None and dst.kind == "gpr" and dst.value + 1 < 0xc0 and \
       srcs[0] is not None and srcs[0].kind == "const" and \
       srcs[1] is not None and srcs[1].kind == "gpr" and srcs[1].value + 1 < 0xc0:
      return "add.u.rpt2", dst, (srcs[0], srcs[1])
  if category == 2 and name == "cmps.s" and _same_int_field(fields, "COND") in (3, 4) and \
     (_same_int_field(fields, "DST_HALF"), _same_int_field(fields, "DST")) == (0, 0xf8) and _has_no_repeat(fields) and \
     all(_int_field_is(fields, field, 0) for field in ("JP", "SAT", "UL", "EI", "LAST", "ABSNEG", "SRC_R", "SY", "SS")) and \
     (raw >> 52 & 1, raw >> 46 & 1) == (1, 0) and \
     ((_same_int_field(fields, "COND") == 3 and raw >> 51 & 1 == 1) or
      (_same_int_field(fields, "COND") == 4 and (raw >> 43 & 1, raw >> 51 & 1) == (0, 0))) and \
     all(value == 0 for value in _field_values(fields, "HALF")):
    srcs = (_multisrc_operand(_same_int_field(fields, "SRC1"), True),
            _multisrc_operand(_same_int_field(fields, "SRC2"), True))
    if srcs[0] is not None and srcs[1] is not None:
      return ("cmps.s.ge.p0" if _same_int_field(fields, "COND") == 3 else "cmps.s.eq.p0"), \
             A630IR3Operand("pred", 0), (srcs[0], srcs[1])
  cat2_compare = name in {"cmps.u", "cmps.s"}
  if category == 2 and name in {"ashr.b", "shl.b", "shr.b", "add.u", "sub.u", "max.s", "max.u", "xor.b", "and.b", "or.b",
                                "mull.u", "cmps.u", "cmps.s", "add.f"} and \
     _has_no_repeat(fields) and all(_int_field_is(fields, field, 0) for field in ("JP", "SAT", "UL", "EI", "LAST", "ABSNEG", "SRC_R")) and \
     (raw >> 52 & 1, raw >> 46 & 1) == (1, int(cat2_compare)):
    dst_half = bool(_same_int_field(fields, "DST_HALF"))
    dst = _register_operand(_same_int_field(fields, "DST"), not dst_half) if cat2_compare else \
      A630IR3Operand("half" if dst_half else "gpr", _same_int_field(fields, "DST"))
    full = bool(raw >> 52 & 1)
    srcs = (_multisrc_operand(_same_int_field(fields, "SRC1"), full),
            _multisrc_operand(_same_int_field(fields, "SRC2"), full))
    if dst is not None and all(src is not None for src in srcs):
      if name in {"max.s", "max.u", "shr.b", "xor.b", "and.b", "or.b"} and (dst.kind != "gpr" or not 0 <= dst.value < 0xc0 or
                                         any(src.kind != "gpr" for src in srcs if src is not None)):
        return None, None, ()
      opcode = {("cmps.s", 0):"cmps.s.lt", ("cmps.u", 0):"cmps.u.lt", ("cmps.s", 4):"cmps.s.eq"}.get(
        (name, _same_int_field(fields, "COND"))) if cat2_compare else name
      if opcode is not None:
        assert srcs[0] is not None and srcs[1] is not None
        return opcode, dst, (srcs[0], srcs[1])
  if category == 3 and name == "madsh.m16" and _has_no_repeat(fields) and \
     all(_int_field_is(fields, field, 0) for field in ("JP", "SAT", "UL", "SRC1_NEG", "SRC2_NEG", "SRC3_NEG", "LAST")) and \
     (raw >> 13 & 1, raw >> 29 & 1, raw >> 42 & 1, raw >> 46 & 1) == (0, 0, 0, 0) and \
     _same_int_field(fields, "DST_HALF") == 0 and all(value == 0 for value in _field_values(fields, "HALF")):
    dst = _register_operand(_same_int_field(fields, "DST"), True)
    src1,src2,src3 = (_same_int_field(fields, field) for field in ("SRC1", "SRC2", "SRC3"))
    src1_op = _multisrc_operand(src1, True)
    src2_op = _register_operand(src2, True) if src2 == src2 & 0xff else None
    src3_op = _multisrc_operand(src3, True)
    if dst is not None and dst.kind == "gpr" and src1_op is not None and src1_op.kind in {"gpr", "const"} and \
       src2_op is not None and src2_op.kind == "gpr" and src3_op is not None and src3_op.kind == "gpr":
      assert src1_op is not None and src2_op is not None and src3_op is not None
      return "madsh.m16", dst, (src1_op, src2_op, src3_op)
  if category == 3 and name == "shrg" and _has_no_repeat(fields) and \
     all(_int_field_is(fields, field, 0) for field in ("JP", "SAT", "UL", "SRC1_NEG", "SRC2_NEG", "SRC3_NEG",
                                                               "SRC1_R", "SRC2_R", "SRC3_R")) and \
     (raw >> 13 & 1, raw >> 42 & 1, raw >> 46 & 1) == (1, 1, 0):
    src1,src2,src3 = (_same_int_field(fields, field) for field in ("SRC1", "SRC2", "SRC3"))
    if src1 == 0x1000 | (src1 & 0xfff) and src2 == src2 & 0xff and src3 == src3 & 0xff and \
       _same_int_field(fields, "DST_HALF") == 0 and all(value == 0 for value in _field_values(fields, "HALF")):
      value = src1 & 0xfff
      immediate = value - 0x1000 if value & 0x800 else value
      return "shrg", A630IR3Operand("gpr", _same_int_field(fields, "DST")), \
             (A630IR3Operand("iim", immediate), A630IR3Operand("gpr", src2), A630IR3Operand("gpr", src3))
  cat6_schedule = 1 << 60
  ldg_variable = (0xff << 14) | (0xff << 32) | (0x7 << 24) | cat6_schedule
  ldg_size = _same_int_field(fields, "SIZE") if category == 6 and name == "ldg" else 0
  if category == 6 and name == "ldg" and (_same_int_field(fields, "TYPE"), _same_int_field(fields, "TYPE_HALF"),
                                           _same_int_field(fields, "OFF")) == (3, 0, 0) and ldg_size in (1, 4) and \
     _int_field_is(fields, "JP", 0) and raw & ~ldg_variable == 0xc006000000800001:
    dst,address = _register_operand(_same_int_field(fields, "DST"), True), _register_operand(_same_int_field(fields, "SRC1"), True)
    if dst is not None and dst.kind == "gpr" and dst.value + ldg_size - 1 < 0xc0 and \
       address is not None and address.kind == "gpr" and address.value + 1 < 0xc0:
      return ("ldg.u32" if ldg_size == 1 else "ldg.u32x4"), dst, (address,)
  stg_variable = (0xff << 1) | (0xff << 41) | (0x7 << 24) | cat6_schedule
  stg_size = _same_int_field(fields, "SIZE") if category == 6 and name == "stg" else 0
  if category == 6 and name == "stg" and (_same_int_field(fields, "TYPE"), _same_int_field(fields, "TYPE_HALF"),
                                           _same_int_field(fields, "OFF")) == (3, 0, 0) and stg_size in (1, 4) and \
     _int_field_is(fields, "JP", 0) and raw & ~stg_variable == 0xc0c6010000800000:
    address,data = _register_operand(_same_int_field(fields, "SRC1"), True), _register_operand(_same_int_field(fields, "SRC3"), True)
    if address is not None and address.kind == "gpr" and address.value + 1 < 0xc0 and \
       data is not None and data.kind == "gpr" and data.value + stg_size - 1 < 0xc0:
      return ("stg.u32" if stg_size == 1 else "stg.u32x4"), None, (address, data)
  stg_u8_variable = (0xff << 1) | (0xff << 41) | cat6_schedule
  if category == 6 and name == "stg" and (_same_int_field(fields, "TYPE"), _same_int_field(fields, "TYPE_HALF"),
                                           _same_int_field(fields, "OFF"), _same_int_field(fields, "SIZE")) == (6, 1, 0, 1) and \
     _int_field_is(fields, "JP", 0) and raw & ~stg_u8_variable == 0xc0cc010001800000:
    address,data = _register_operand(_same_int_field(fields, "SRC1"), True), _register_operand(_same_int_field(fields, "SRC3"), False)
    if address is not None and address.kind == "gpr" and address.value + 1 < 0xc0 and data is not None and data.kind == "half":
      return "stg.u8", None, (address, data)
  # Local-memory offsets are byte-addressed scalar GPRs, unlike the adjacent GPR pair used by global memory.
  # ir3-cat6.xml gives LDL's address in SRC and STL's address in DST; SIZE is the U32 component count.
  ldl_variable = (0xff << 32) | (0xff << 14) | (0x7 << 24) | cat6_schedule
  if category == 6 and name == "ldl" and (_same_int_field(fields, "TYPE"), _same_int_field(fields, "OFF"),
                                           _same_int_field(fields, "SIZE")) == (3, 0, 4) and \
     _int_field_is(fields, "JP", 0) and raw & ~ldl_variable == 0xc046000000800001:
    dst,address = _register_operand(_same_int_field(fields, "DST"), True), _register_operand(_same_int_field(fields, "SRC"), True)
    if dst is not None and dst.kind == "gpr" and dst.value + 3 < 0xc0 and address is not None and address.kind == "gpr":
      return "ldl.u32x4", dst, (address,)
  stl_variable = (0xff << 41) | (0xff << 1) | (0x7 << 24) | cat6_schedule
  if category == 6 and name == "stl" and (_same_int_field(fields, "TYPE"), _same_int_field(fields, "OFF"),
                                           _same_int_field(fields, "SIZE")) == (3, 0, 1) and \
     _int_field_is(fields, "JP", 0) and raw & ~stl_variable == 0xc106010000800000:
    address,data = _register_operand(_same_int_field(fields, "DST"), True), _register_operand(_same_int_field(fields, "SRC"), True)
    if address is not None and address.kind == "gpr" and data is not None and data.kind == "gpr":
      return "stl.u32", None, (address, data)
  # ir3-cat7.xml distinguishes this exact workgroup execution barrier from FENCE and from other BAR scopes.
  if category == 7 and name == "bar" and raw == 0xe042000000000000: return "bar.g", None, ()
  return None, None, ()

def decode_a630_ir3(image:bytes) -> tuple[A630IR3Instruction, ...]:
  _require(bool(image), "empty IR3 image")
  _require(len(image) % 8 == 0, "IR3 image size must be a multiple of 8")
  _require(len(image) // 8 <= _MAX_SHADER_INSTRUCTIONS, "IR3 image exceeds the emulator instruction limit")
  instructions:list[A630IR3Instruction] = []
  current = [-1]
  callback_errors:list[str] = []
  pre_indices:list[int] = []
  post_indices:list[int] = []
  pre_images:list[bytes] = []
  callback_fields:list[list[tuple[str, int|str]]] = [[], [], []]

  @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(mesa.struct_isa_decode_value))
  def field_cb(_data, name, value):
    try:
      if not name or not value: raise ValueError("null field callback value")
      field_name = ctypes.string_at(name).decode("ascii")
      if field_name.partition(":align=")[0] == "NAME":
        if not value.contents.str: raise ValueError("null NAME callback value")
        field_value:int|str = ctypes.string_at(value.contents.str).decode("ascii")
      else: field_value = int(value.contents.num)
      if not 0 <= current[0] < len(callback_fields): raise ValueError("field outside instruction callbacks")
      callback_fields[current[0]].append((field_name, field_value))
    except Exception as error: callback_errors.append(str(error))

  @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p)
  def pre_cb(_data, index, instruction):
    try:
      current[0] = int(index)
      if not instruction or not 0 <= current[0] < len(callback_fields): raise ValueError("invalid pre-instruction callback")
      pre_indices.append(current[0])
      pre_images.append(ctypes.string_at(instruction, 8))
    except Exception as error: callback_errors.append(str(error))

  @ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p)
  def post_cb(_data, index, _instruction):
    try:
      if current[0] != int(index): raise ValueError("mismatched post-instruction callback")
      post_indices.append(int(index))
      current[0] = -1
    except Exception as error: callback_errors.append(str(error))

  sink = libc.fopen(os.devnull.encode(), b"w")
  _require(bool(sink), "failed to open IR3 decoder output sink")
  mesa_sink = ctypes.cast(sink, ctypes.POINTER(mesa.struct__IO_FILE))
  try:
    for index in range(len(image) // 8):
      instruction = image[index*8:(index+1)*8]
      # IR3 has no generated <decode> map in this Mesa revision, so this call supplies only the leaf-match result.
      matched = mesa.ir3_isa_decode(None, instruction, mesa.struct_isa_decode_options(gpu_id=630, show_errors=False))
      _require(matched, f"unmatched IR3 encoding at instruction {index}")

      current[0] = -1
      callback_errors.clear()
      pre_indices.clear()
      post_indices.clear()
      pre_images.clear()
      callback_fields[:] = [[], [], []]
      # Mesa exposes reserved/assert failures only through its disassembler. In the pin, errors > max_errors is tested before
      # the next word; two copies with max_errors=1 therefore expose an error by stopping before this zero callback sentinel.
      probe = instruction * 2 + bytes(8)
      mesa.ir3_isa_disasm(probe, len(probe), mesa_sink, mesa.struct_isa_decode_options(
        gpu_id=630, show_errors=True, max_errors=1, branch_labels=False,
        field_cb=field_cb, pre_instr_cb=pre_cb, post_instr_cb=post_cb))
      _require(not callback_errors, f"IR3 callback failure at instruction {index}: {callback_errors[0] if callback_errors else ''}")
      _require(pre_indices == [0, 1, 2] and post_indices == [0, 1, 2] and
               pre_images == [instruction, instruction, bytes(8)], f"invalid or reserved IR3 encoding at instruction {index}")
      _require(callback_fields[0] == callback_fields[1], f"inconsistent IR3 fields at instruction {index}")

      fields = tuple(callback_fields[0])
      names = [value for field,value in fields if field.partition(":align=")[0] == "NAME"]
      name = names[0] if names else None
      _require(len(names) <= 1 and (name is None or isinstance(name, str)), f"ambiguous IR3 name at instruction {index}")
      assert name is None or isinstance(name, str)
      _require(name not in {"ldp", "stp", "call", "ret"}, f"unsupported IR3 instruction {name}")
      raw = int.from_bytes(instruction, "little")
      category = raw >> 61
      opcode,dst,srcs = _normalize_ir3(raw, category, name, fields)
      instructions.append(A630IR3Instruction(index, category, raw, name, fields, opcode, dst, srcs))
  finally:
    libc.fclose(sink)

  # Accept only the observed pinned-compiler image boundary: one canonical end followed by zero-filled allocation padding.
  ends = [instruction.index for instruction in instructions if instruction.name == "end"]
  _require(len(ends) == 1, "missing end instruction" if not ends else "multiple end instructions")
  end = ends[0]
  _require(instructions[end].raw == 6 << 55, "unsupported end instruction encoding")
  _require(all(instruction.raw == 0 for instruction in instructions[end+1:]), "nonzero instruction after end")
  return tuple(instructions)

def _address(lo:int, hi:int, alignment:int, purpose:str) -> int:
  address = lo | hi << 32
  _require(address != 0 and address % alignment == 0, f"unaligned or null {purpose} address {address:#x}")
  return address

def _registers(regs:dict[int, int], start:int, count:int, purpose:str) -> tuple[int, ...]:
  _require(all(start+i in regs for i in range(count)), f"missing {purpose} registers")
  return tuple(regs[start+i] for i in range(count))

def _validate_type4(packet:PM4Type4Packet, regs:dict[int, int]) -> None:
  values, register = packet.values, packet.register
  exact = {
    mesa.REG_A6XX_SP_CS_TSIZE: 0x80,
    mesa.REG_A6XX_SP_CS_USIZE: 0x40,
    mesa.REG_A6XX_SP_MODE_CNTL: 0x5,
    mesa.REG_A6XX_SP_PERFCTR_SHADER_MASK: 0x20,
    mesa.REG_A6XX_TPL1_MODE_CNTL: 0x2,
    mesa.REG_A6XX_TPL1_DBG_ECO_CNTL: 0x0,
  }
  if register in exact: _require(values[0] == exact[register], f"invalid fixed register {register:#x} value {values[0]:#x}")
  elif register == mesa.REG_A6XX_SP_CS_NDRANGE_0:
    ctrl = values[0]
    _require(ctrl & 0x3 == 3, "compute NDRANGE must be three-dimensional")
    local = ((ctrl >> 2 & 0x3ff) + 1, (ctrl >> 12 & 0x3ff) + 1, (ctrl >> 22 & 0x3ff) + 1)
    _require(all(local), "zero NDRANGE local size")
    _require(values[2] == values[4] == values[6] == 0, "nonzero NDRANGE global offset")
    _require(values[7] == 0xccc0cf and values[8] == 0xfc, "invalid fixed NDRANGE compute state")
    _require(all(values[i] > 0 for i in (1, 3, 5, 9, 10, 11)), "zero NDRANGE dimension")
  elif register == mesa.REG_A6XX_SP_CS_CNTL_0:
    cntl0_mask = 0x1 | 0x7e | 0x1f80 | 0xfc000 | 0x100000
    _require(values[0] & ~cntl0_mask == 0 and values[0] & 0x100001 == 0, "unsupported CS control flags")
    _require(values[0] & 0xfc000 == 0, "branch stack is not supported")
    _require(values[1] & ~0x7f == 0 and (values[1] >> 5 & 0x3) == mesa.CONSTLEN_256, "invalid CS constant RAM mode")
    _require(values[2] == 0 and values[3] == 0, "unsupported CS boolean mask or program offset")
    _address(values[4], values[5], 128, "shader")
    _require(values[6] == 0, "private memory is not supported")
    _address(values[7], values[8], 32, "private-memory base")
    _require(values[9] == 0, "private memory is not supported")
  elif register == mesa.REG_A6XX_SP_REG_PROG_ID_0:
    _require(values == (0xfcfcfcfc, 0xfcfcfcfc, 0xfcfcfcfc, 0xfc, 0x100), "invalid compute program identifiers")
  elif register == mesa.REG_A6XX_SP_CS_PVT_MEM_STACK_OFFSET:
    _require(values[0] & ~0x7ffff == 0, "reserved private-stack offset bits")
  elif register == mesa.REG_A6XX_SP_CS_INSTR_SIZE:
    _require(0 < values[0] <= 0xfffffff, "invalid instruction-group count")
  elif register in (mesa.REG_A6XX_SP_CS_SAMPLER_BASE, mesa.REG_A6XX_TPL1_CS_BORDER_COLOR_BASE,
                    mesa.REG_A6XX_SP_CS_TEXMEMOBJ_BASE, mesa.REG_A6XX_SP_CS_UAV_BASE):
    alignment = {mesa.REG_A6XX_SP_CS_SAMPLER_BASE:16, mesa.REG_A6XX_TPL1_CS_BORDER_COLOR_BASE:128,
                 mesa.REG_A6XX_SP_CS_TEXMEMOBJ_BASE:64, mesa.REG_A6XX_SP_CS_UAV_BASE:16}[register]
    _address(values[0], values[1], alignment, f"register {register:#x}")
  elif register == mesa.REG_A6XX_SP_CS_CONFIG:
    allowed = 0x100 | 0x1fe00 | 0x3e0000 | 0x1fc00000
    _require(values[0] & ~allowed == 0 and values[0] & 0x100 != 0, "invalid CS resource configuration")
  elif register == mesa.REG_A6XX_SP_CS_CONST_CONFIG_0:
    _require(values[1] & ~0xff == 0, "unsupported workgroup control flags")
  else:
    _require(register == mesa.REG_A6XX_SP_UPDATE_CNTL, f"missing semantic handler for register {register:#x}")
  for i,value in enumerate(values): regs[register+i] = value

def _load_state(values:tuple[int, ...]) -> A630LoadState:
  control = values[0]
  dst, state_type, source, block, units = control & 0x3fff, control >> 14 & 0x3, control >> 16 & 0x3, control >> 18 & 0xf, control >> 22
  _require(dst == 0 and source == mesa.SS6_INDIRECT and units > 0, "unsupported indirect state load")
  # A6xx constants use vec4 units (Mesa tu_shader.cc:1558-1569 and tu_cmd_buffer.cc:5980-5990); shader units are 128-byte groups.
  forms = {
    (mesa.ST_CONSTANTS, mesa.SB6_CS_SHADER): ("constants", 16, 16),
    (mesa.ST_SHADER, mesa.SB6_CS_SHADER): ("shader", 128, 128),
    (mesa.ST_SHADER, mesa.SB6_CS_TEX): ("samplers", 16, 16),
    (mesa.ST_CONSTANTS, mesa.SB6_CS_TEX): ("textures", 64, 64),
    (mesa.ST6_UAV, mesa.SB6_CS_SHADER): ("uavs", 64, 64),
  }
  _require((state_type, block) in forms, f"unsupported state load type/block {state_type}/{block}")
  kind,unit_size,alignment = forms[(state_type, block)]
  if kind == "constants": _require(units == 256, f"unsupported constant load size {units}")
  if kind == "shader":
    _require(units * unit_size <= _MAX_SHADER_INSTRUCTIONS * 8, "IR3 image exceeds the emulator instruction limit")
  address = _address(values[1], values[2], alignment, kind)
  return A630LoadState(kind, address, units * unit_size, units)

def _dispatch(regs:dict[int, int], loads:dict[str, A630LoadState], packet:PM4Type7Packet,
              ranges:list[A630MemoryRange]) -> A630Dispatch:
  values = packet.values
  _require(values[0] == 0 and all(value > 0 for value in values[1:]), "invalid compute dispatch dimensions")
  groups = values[1], values[2], values[3]
  ndrange = _registers(regs, mesa.REG_A6XX_SP_CS_NDRANGE_0, 12, "NDRANGE")
  cntl = _registers(regs, mesa.REG_A6XX_SP_CS_CNTL_0, 10, "CS control")
  _registers(regs, mesa.REG_A6XX_SP_REG_PROG_ID_0, 5, "program identifier")
  stack_offset = _registers(regs, mesa.REG_A6XX_SP_CS_PVT_MEM_STACK_OFFSET, 1, "private-stack offset")[0]
  instr_size = _registers(regs, mesa.REG_A6XX_SP_CS_INSTR_SIZE, 1, "instruction size")[0]
  config = _registers(regs, mesa.REG_A6XX_SP_CS_CONFIG, 1, "CS configuration")[0]
  _registers(regs, mesa.REG_A6XX_SP_CS_CONST_CONFIG_0, 2, "workgroup configuration")
  for register in (mesa.REG_A6XX_SP_UPDATE_CNTL, mesa.REG_A6XX_SP_CS_TSIZE, mesa.REG_A6XX_SP_CS_USIZE, mesa.REG_A6XX_SP_MODE_CNTL,
                   mesa.REG_A6XX_SP_PERFCTR_SHADER_MASK, mesa.REG_A6XX_TPL1_MODE_CNTL, mesa.REG_A6XX_TPL1_DBG_ECO_CNTL):
    _require(register in regs, f"missing fixed compute register {register:#x}")
  _require(regs[mesa.REG_A6XX_SP_UPDATE_CNTL] == 0, "compute state update was not cleared")

  ctrl = ndrange[0]
  local = ((ctrl >> 2 & 0x3ff) + 1, (ctrl >> 12 & 0x3ff) + 1, (ctrl >> 22 & 0x3ff) + 1)
  global_size, registered_groups = (ndrange[1], ndrange[3], ndrange[5]), (ndrange[9], ndrange[10], ndrange[11])
  _require(groups == registered_groups, "EXEC_CS and NDRANGE group counts differ")
  _require(global_size == tuple(group*size for group,size in zip(groups, local)), "global, local, and group dimensions differ")

  _require("constants" in loads and "shader" in loads, "missing constant or shader state load")
  constants, shader = loads["constants"], loads["shader"]
  shader_base = _address(cntl[4], cntl[5], 128, "shader")
  stack_base = _address(cntl[7], cntl[8], 32, "private-memory base")
  _require(shader.address == shader_base and shader.units == instr_size, "shader base or instruction size differs from state load")
  # The unchanged runtime emits this inert raw value even with zero private memory; no stack range is implied or accepted.
  _require(stack_offset == 0x1000, "unsupported private-stack offset")

  nsamp, ntex, nuav = config >> 17 & 0x1f, config >> 9 & 0xff, config >> 22 & 0x7f
  _require((nsamp, ntex, nuav) == (0, 0, 0) and not any(kind in loads for kind in ("samplers", "textures", "uavs")),
           "A630 image execution is not implemented")

  active_loads = tuple(loads[kind] for kind in ("constants", "shader"))
  return A630Dispatch(packet.word_offset, tuple(sorted(regs.items())), active_loads, shader.address, shader.size, b"", constants.address,
                      constants.size, b"", stack_base, stack_offset, local, global_size, groups)

def stage_a630(packets:Sequence[PM4Packet], resolver:Resolver) -> A630Submission:
  regs:dict[int, int] = {}
  loads:dict[str, A630LoadState] = {}
  ranges:list[A630MemoryRange] = []
  dispatches:list[A630Dispatch] = []
  waits:list[A630Wait] = []
  writes:list[A630Write] = []
  update_pending = marker_pending = False

  for packet in packets:
    if update_pending:
      _require(isinstance(packet, PM4Type4Packet) and packet.register == mesa.REG_A6XX_SP_UPDATE_CNTL and packet.values == (0,),
               "SP_UPDATE_CNTL enable was not immediately cleared")
    if isinstance(packet, PM4Type4Packet):
      if packet.register == mesa.REG_A6XX_SP_UPDATE_CNTL:
        if packet.values == (0x60,):
          _require(not update_pending, "duplicate SP_UPDATE_CNTL enable")
          update_pending = True
        else:
          _require(packet.values == (0,) and update_pending, "invalid SP_UPDATE_CNTL sequence")
          update_pending = False
      _validate_type4(packet, regs)
      continue

    values = packet.values
    if packet.opcode == mesa.CP_WAIT_REG_MEM:
      _require(values[0] == (mesa.WRITE_GE | mesa.POLL_MEMORY << 4) and values[4] == 0xffffffff and values[5] == 32,
               "invalid memory-wait control")
      address = _address(values[1], values[2], 4, "wait")
      ranges.append(A630MemoryRange(address, 4, read=True, write=False, purpose="wait value"))
      waits.append(A630Wait(packet.word_offset, address, values[3], values[4]))
    elif packet.opcode == mesa.CP_EVENT_WRITE:
      if len(values) == 1: _require(values[0] == mesa.CACHE_INVALIDATE, "invalid cache-invalidate event")
      else:
        _require(values[0] == mesa.CACHE_FLUSH_TS, "invalid cache-flush event")
        address = _address(values[1], values[2], 4, "event-write")
        ranges.append(A630MemoryRange(address, 4, read=False, write=True, purpose="event value"))
        writes.append(A630Write(packet.word_offset, address, 4, values[3], "event value"))
    elif packet.opcode == mesa.CP_REG_TO_MEM:
      expected = mesa.REG_A6XX_CP_ALWAYS_ON_COUNTER | 2 << 18 | 1 << 30
      _require(values[0] == expected, "invalid counter-to-memory control")
      address = _address(values[1], values[2], 8, "counter")
      ranges.append(A630MemoryRange(address, 8, read=False, write=True, purpose="counter value"))
      writes.append(A630Write(packet.word_offset, address, 8, None, "counter value"))
    elif packet.opcode in (mesa.CP_WAIT_MEM_WRITES, mesa.CP_WAIT_FOR_IDLE): pass
    elif packet.opcode == mesa.CP_SET_MARKER:
      _require(values == (mesa.RM6_COMPUTE,), "invalid compute marker")
      marker_pending = True
    elif packet.opcode == mesa.CP_LOAD_STATE6_FRAG:
      load = _load_state(values)
      loads[load.kind] = load
      ranges.append(A630MemoryRange(load.address, load.size, read=True, write=False, purpose=load.kind))
    elif packet.opcode == mesa.CP_EXEC_CS:
      _require(marker_pending, "compute dispatch without marker")
      dispatches.append(_dispatch(regs, loads, packet, ranges))
      marker_pending = False
    else: raise ValueError(f"missing semantic handler for opcode {packet.opcode:#x}")
  _require(not update_pending and not marker_pending, "unterminated compute-state sequence")

  resolved:list[A630MemoryRange] = []
  snapshots:dict[tuple[int, int], bytes] = {}
  read_images:dict[tuple[int, int, str], bytes] = {}
  snapshot_bytes = 0
  for memory_range in dict.fromkeys(ranges):
    if memory_range.read:
      snapshot_key = (memory_range.address, memory_range.size)
      if snapshot_key not in snapshots:
        snapshot_bytes += memory_range.size
        _require(snapshot_bytes <= _MAX_A630_SNAPSHOT_BYTES, "A630 state snapshots exceed the emulator byte limit")
        view = resolver(memory_range.address, memory_range.size)
        _require(len(view) == memory_range.size, f"short resolved {memory_range.purpose} range")
        snapshots[snapshot_key] = bytes(view)
      read_images[(memory_range.address, memory_range.size, memory_range.purpose)] = snapshots[snapshot_key]
    else:
      view = resolver(memory_range.address, memory_range.size)
      _require(len(view) == memory_range.size, f"short resolved {memory_range.purpose} range")
    resolved.append(memory_range)
  decoded_shaders:dict[tuple[int, int], tuple[A630IR3Instruction, ...]] = {}
  dispatch_instructions:list[tuple[A630IR3Instruction, ...]] = []
  for dispatch in dispatches:
    shader_key = (dispatch.shader_address, dispatch.shader_size)
    if shader_key not in decoded_shaders: decoded_shaders[shader_key] = decode_a630_ir3(snapshots[shader_key])
    dispatch_instructions.append(decoded_shaders[shader_key])
  frozen_dispatches = tuple(replace(dispatch,
    shader_image=read_images[(dispatch.shader_address, dispatch.shader_size, "shader")],
    constants_image=read_images[(dispatch.constants_address, dispatch.constants_size, "constants")],
    instructions=instructions)
    for dispatch,instructions in zip(dispatches, dispatch_instructions))
  return A630Submission(frozen_dispatches, tuple(resolved), tuple(waits), tuple(writes))

def _read_ir3_operand(operand:A630IR3Operand, full:dict[int, int], half:dict[int, int], shared:dict[int, int],
                      constants:tuple[int, ...]) -> int:
  if operand.kind == "gpr":
    _require(operand.value in full, f"read of uninitialized full register {operand.value}")
    return full[operand.value]
  if operand.kind == "half":
    _require(operand.value in half, f"read of uninitialized half register {operand.value}")
    return half[operand.value]
  if operand.kind == "shared":
    _require(operand.value in shared, f"read of unmapped shared register {operand.value}")
    return shared[operand.value]
  if operand.kind == "const":
    _require(operand.value < len(constants), f"constant register {operand.value} is out of range")
    return constants[operand.value]
  if operand.kind == "iim": return operand.value & 0xffffffff
  if operand.kind == "uim": return operand.value
  if operand.kind == "flut":
    _require(operand.value in (2, 3), f"unsupported float lookup immediate {operand.value}")
    return (0x3f800000, 0x40000000)[operand.value - 2]
  raise ValueError(f"unsupported IR3 operand kind {operand.kind}")

def _write_ir3_operand(operand:A630IR3Operand, value:int, full:dict[int, int], half:dict[int, int]) -> None:
  if operand.kind == "gpr": full[operand.value] = value & 0xffffffff
  elif operand.kind == "half": half[operand.value] = value & 0xffff
  else: raise ValueError(f"unsupported IR3 destination kind {operand.kind}")

def _system_registers(dispatch:A630Dispatch) -> tuple[int, int]:
  registers = dict(dispatch.registers)
  config = registers.get(mesa.REG_A6XX_SP_CS_CONST_CONFIG_0)
  _require(config is not None and config >> 32 == 0, "missing or overflowing A630 system-value register mapping")
  assert config is not None
  wgid,wgsz,wgoffset,lid = ((config >> shift) & 0xff for shift in (0, 8, 16, 24))
  invalid = 0xfc
  _require((wgsz, wgoffset, registers.get(mesa.REG_A6XX_SP_CS_WGE_CNTL)) == (invalid, invalid, invalid),
           "unsupported A630 system-value register mapping")
  _require(wgid == invalid or wgid % 4 == 0 and 0xc0 <= wgid <= 0xdc, "invalid A630 workgroup-id register mapping")
  _require(lid == invalid or lid % 4 == 0 and 0 <= lid <= 0xbc, "invalid A630 local-id register mapping")
  _require(dispatch.groups[0] == 1 or wgid != invalid, "multi-workgroup A630 dispatch lacks a workgroup-id mapping")
  _require(dispatch.local_size[0] == 1 or lid != invalid, "multi-lane A630 dispatch lacks a local-id mapping")
  return wgid,lid

def _gpr_address(full:dict[int, int], operand:A630IR3Operand) -> int:
  return full[operand.value] | full[operand.value + 1] << 32

def _finish_writes(writes:Sequence[A630ExecutionWrite], reads:Sequence[tuple[int, int, str]]) -> tuple[A630ExecutionWrite, ...]:
  ordered = sorted(writes, key=lambda write: write.address)
  _require(all(left.address + len(left.data) <= right.address for left,right in zip(ordered, ordered[1:])),
           "overlapping A630 global stores")
  ordered_reads = sorted(reads)
  write_index = read_index = 0
  while write_index < len(ordered) and read_index < len(ordered_reads):
    write,read = ordered[write_index],ordered_reads[read_index]
    if write.address + len(write.data) <= read[0]: write_index += 1
    elif read[0] + read[1] <= write.address: read_index += 1
    else: raise ValueError(f"A630 global store aliases snapshotted {read[2]}")
  return tuple(writes)

def _f32_add_bits(left:int, right:int) -> int:
  # The emitted shader requests RTE but neither preserves nor flushes FP32 denorms, so only zero/normal finite words are grounded.
  _require(all((bits >> 23 & 0xff) != 0xff and ((bits >> 23 & 0xff) != 0 or bits & 0x7fffff == 0) for bits in (left, right)),
           "unsupported special or subnormal float input")
  result = ctypes.c_float(struct.unpack("<f", struct.pack("<I", left))[0] +
                          struct.unpack("<f", struct.pack("<I", right))[0]).value
  value = struct.unpack("<I", struct.pack("<f", result))[0]
  exponent = value >> 23 & 0xff
  _require(exponent != 0xff and (exponent != 0 or value & 0x7fffff == 0), "unsupported special or subnormal float result")
  return value

def _validate_operand_contract(instruction:A630IR3Instruction) -> None:
  opcode = instruction.opcode
  _require(opcode is not None and opcode in _OPERAND_CONTRACTS, f"unsupported A630 semantic at instruction {instruction.index}")
  assert opcode is not None
  expected_dst,allowed_srcs = _OPERAND_CONTRACTS[opcode]
  dst_kind = instruction.dst.kind if instruction.dst is not None else None
  src_kinds = tuple(operand.kind for operand in instruction.srcs)
  _require(dst_kind == expected_dst and src_kinds in allowed_srcs,
           f"unsupported A630 operand contract at instruction {instruction.index}")
  if opcode in {"br.p0", "predt.p0"}:
    _require(instruction.srcs[0] == A630IR3Operand("pred", 0),
             f"unsupported A630 predicate at instruction {instruction.index}")
  if opcode in {"cmps.s.ge.p0", "cmps.s.eq.p0"}:
    _require(instruction.dst == A630IR3Operand("pred", 0),
             f"unsupported A630 predicate destination at instruction {instruction.index}")
  if opcode == "ashr.b":
    _require(instruction.srcs[1].value == 31, f"unsupported A630 ASHR.B shift at instruction {instruction.index}")
  if opcode == "shl.b":
    _require(instruction.srcs[1].value in (1, 2, 4, 6, 7),
             f"unsupported A630 SHL.B shift at instruction {instruction.index}")
  if opcode == "shrg":
    _require(instruction.srcs[0].value == 30, f"unsupported A630 SHRG shift at instruction {instruction.index}")
  if opcode == "cmps.s.eq.p0":
    _require(instruction.srcs[1].value == 0, f"unsupported A630 predicate immediate at instruction {instruction.index}")

def _schedule_flag(instruction:A630IR3Instruction, name:str) -> int:
  values = _field_values(instruction.fields, name)
  if not values: return 0
  _require(all(isinstance(value, int) and value in (0, 1) and value == values[0] for value in values),
           f"inconsistent IR3 {name} field at instruction {instruction.index}")
  assert isinstance(values[0], int)
  return values[0]

def _full_gpr_writes(instruction:A630IR3Instruction) -> set[int]:
  if instruction.dst is None or instruction.dst.kind != "gpr": return set()
  width = 4 if instruction.opcode in {"ldg.u32x4", "ldl.u32x4", "add.f.rpt4"} else 2 if instruction.opcode == "add.u.rpt2" else 1
  return set(range(instruction.dst.value, instruction.dst.value + width))

def _full_gpr_accesses(instruction:A630IR3Instruction) -> set[int]:
  # The admitted CS_CNTL_0 mode has MERGEDREGS clear, so Mesa's full and half register files do not alias.
  accesses = _full_gpr_writes(instruction)
  def add(operand:A630IR3Operand|None, width:int=1) -> None:
    if operand is not None and operand.kind == "gpr": accesses.update(range(operand.value, operand.value + width))

  for operand in instruction.srcs: add(operand)
  if instruction.opcode in {"ldg.u32", "ldg.u32x4", "stg.u32", "stg.u32x4", "stg.u8"}: add(instruction.srcs[0], 2)
  if instruction.opcode == "stg.u32x4": add(instruction.srcs[1], 4)
  if instruction.opcode == "add.f.rpt4":
    for operand in instruction.srcs: add(operand, 4)
  if instruction.opcode == "add.u.rpt2":
    add(instruction.srcs[1], 2)
  return accesses

def _store_source_gprs(instruction:A630IR3Instruction) -> tuple[set[int], set[int]]:
  full:set[int] = set()
  half:set[int] = set()
  def add(operand:A630IR3Operand, width:int=1) -> None:
    target = full if operand.kind == "gpr" else half if operand.kind == "half" else None
    if target is not None: target.update(range(operand.value, operand.value + width))

  if instruction.opcode in {"stg.u32", "stg.u32x4", "stg.u8"}:
    add(instruction.srcs[0], 2)
    add(instruction.srcs[1], 4 if instruction.opcode == "stg.u32x4" else 1)
  elif instruction.opcode == "stl.u32":
    add(instruction.srcs[0])
    add(instruction.srcs[1])
  return full,half

def _instruction_successors(active:Sequence[A630IR3Instruction], index:int) -> tuple[int, ...]:
  instruction = active[index]
  if instruction.opcode == "end": return ()
  if instruction.opcode == "jump": return (index + instruction.srcs[0].value,)
  if instruction.opcode == "br.p0": return tuple(dict.fromkeys((index + 1, index + instruction.srcs[1].value)))
  return (index + 1,)

def _merge_memory_schedule(left:_A630MemoryScheduleState, right:_A630MemoryScheduleState) -> _A630MemoryScheduleState:
  return _A630MemoryScheduleState(left.force_ss or right.force_ss, left.force_sy or right.force_sy,
    left.needs_ss | right.needs_ss, left.needs_sy | right.needs_sy,
    left.needs_ss_war_full | right.needs_ss_war_full, left.needs_ss_war_half | right.needs_ss_war_half,
    left.needs_ss_or_sy_war_full | right.needs_ss_or_sy_war_full)

def _schedule_value(instruction:A630IR3Instruction, name:str) -> int:
  values = _field_values(instruction.fields, name)
  if not values: return 0
  _require(all(isinstance(value, int) and value >= 0 and value == values[0] for value in values),
           f"inconsistent IR3 {name} field at instruction {instruction.index}")
  assert isinstance(values[0], int)
  return values[0]

def _is_alu_instruction(instruction:A630IR3Instruction) -> bool:
  return instruction.category in (1, 2, 3)

def _instruction_source_reads(instruction:A630IR3Instruction, component:int|None=None) -> tuple[tuple[str, int, int], ...]:
  reads:list[tuple[str, int, int]] = []
  for source_index,source in enumerate(instruction.srcs):
    if component is not None:
      advances = instruction.opcode == "add.f.rpt4" or instruction.opcode == "add.u.rpt2" and source_index == 1
      base,width = source.value + (component if advances else 0),1
    else:
      base = source.value
      if instruction.opcode == "stg.u32x4" and source_index == 1: width = 4
      elif instruction.opcode in {"ldg.u32", "ldg.u32x4", "stg.u32", "stg.u32x4", "stg.u8"} and source_index == 0: width = 2
      else: width = 1
    late = 2 if instruction.opcode == "madsh.m16" and source_index == 2 else 0
    if source.kind in {"gpr", "half", "pred"}:
      reads.extend((source.kind, base + offset, late) for offset in range(width))
  return tuple(reads)

def _instruction_destination_writes(instruction:A630IR3Instruction, component:int|None=None) -> tuple[tuple[str, int], ...]:
  if instruction.dst is None or instruction.dst.kind not in {"gpr", "half", "pred"}: return ()
  if component is not None:
    advances = instruction.opcode in {"add.f.rpt4", "add.u.rpt2"}
    return ((instruction.dst.kind, instruction.dst.value + (component if advances else 0)),)
  width = len(_full_gpr_writes(instruction)) if instruction.dst.kind == "gpr" else 1
  return tuple((instruction.dst.kind, instruction.dst.value + offset) for offset in range(width))

def _merge_delay_maps(left:tuple[tuple[int, int], ...], right:tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
  merged = dict(left)
  for register,delay in right: merged[register] = max(delay, merged.get(register, 0))
  return tuple(sorted(merged.items()))

def _merge_delay_state(left:_A630DelayState, right:_A630DelayState) -> _A630DelayState:
  return _A630DelayState(*(_merge_delay_maps(getattr(left, field), getattr(right, field))
                           for field in ("full_alu", "full_non_alu", "half_alu", "half_non_alu", "predicate")))

def _advance_delay_map(delays:dict[int, int], cycles:int) -> dict[int, int]:
  return {register:delay-cycles for register,delay in delays.items() if delay > cycles}

def _validate_fixed_alu_delays(active:Sequence[A630IR3Instruction]) -> None:
  # Pinned ir3_delay.c and the A630 compiler configuration require three cycles from Cat1-3 ALU writes to another
  # ALU and six to memory/control consumers. REPEAT advances component reads/writes; Cat3 source 2 is read two cycles
  # late. Use maximum remaining delay at CFG joins, matching ir3_legalize.c's predecessor/loop convergence.
  incoming:dict[int, _A630DelayState] = {0:_A630DelayState()}
  pending = [0]
  queued = {0}
  steps = 0
  while pending:
    index = pending.pop()
    queued.remove(index)
    steps += 1
    _require(steps <= _MAX_SCHEDULE_VALIDATION_STEPS, "A630 schedule validation exceeds the emulator work limit")
    instruction,state = active[index],incoming[index]
    full_alu,full_non_alu = dict(state.full_alu),dict(state.full_non_alu)
    half_alu,half_non_alu = dict(state.half_alu),dict(state.half_non_alu)
    predicate = dict(state.predicate)
    is_alu = _is_alu_instruction(instruction)

    def check_reads(reads:Sequence[tuple[str, int, int]], consumer_alu:bool) -> None:
      for kind,register,read_offset in reads:
        if kind == "pred": delays = predicate
        elif kind == "gpr": delays = full_alu if consumer_alu else full_non_alu
        else: delays = half_alu if consumer_alu else half_non_alu
        _require(delays.get(register, 0) <= read_offset,
                 f"A630 fixed ALU dependency lacks delay slots at instruction {instruction.index}")

    def record_writes(writes:Sequence[tuple[str, int]], producer_alu:bool) -> None:
      for kind,register in writes:
        if not producer_alu:
          if kind == "gpr":
            full_alu.pop(register, None)
            full_non_alu.pop(register, None)
          elif kind == "half":
            half_alu.pop(register, None)
            half_non_alu.pop(register, None)
          else: predicate.pop(register, None)
        elif kind == "pred": predicate[register] = max(predicate.get(register, 0), 7)
        else:
          alu_delays,non_alu_delays = (full_alu,full_non_alu) if kind == "gpr" else (half_alu,half_non_alu)
          alu_delays[register] = max(alu_delays.get(register, 0), 4)
          non_alu_delays[register] = max(non_alu_delays.get(register, 0), 7)

    def advance(cycles:int) -> None:
      nonlocal full_alu,full_non_alu,half_alu,half_non_alu,predicate
      full_alu,full_non_alu = _advance_delay_map(full_alu, cycles),_advance_delay_map(full_non_alu, cycles)
      half_alu,half_non_alu = _advance_delay_map(half_alu, cycles),_advance_delay_map(half_non_alu, cycles)
      predicate = _advance_delay_map(predicate, cycles)

    repeat = _schedule_value(instruction, "REPEAT")
    if is_alu:
      for component in range(repeat + 1):
        check_reads(_instruction_source_reads(instruction, component), True)
        record_writes(_instruction_destination_writes(instruction, component), True)
        advance(1)
      advance(_schedule_value(instruction, "NOP"))
    else:
      check_reads(_instruction_source_reads(instruction), False)
      record_writes(_instruction_destination_writes(instruction), False)
      if instruction.opcode in {"nop", "predt.p0", "prede"}: advance(1 + repeat)
    outgoing = _A630DelayState(tuple(sorted(full_alu.items())), tuple(sorted(full_non_alu.items())),
      tuple(sorted(half_alu.items())), tuple(sorted(half_non_alu.items())), tuple(sorted(predicate.items())))
    for successor in _instruction_successors(active, index):
      _require(0 <= successor < len(active), f"A630 delay successor {successor} is out of range")
      joined = outgoing if successor not in incoming else _merge_delay_state(incoming[successor], outgoing)
      if joined != incoming.get(successor):
        incoming[successor] = joined
        if successor not in queued:
          pending.append(successor)
          queued.add(successor)

def _validate_memory_schedule(active:Sequence[A630IR3Instruction]) -> None:
  # Pinned ir3_legalize.c makes BAR force both scoreboards, records local/global load destinations in needs_ss/needs_sy,
  # and inserts an SS-carrying NOP when Cat6 cannot encode SS. Model dependencies rather than compiler NOP positions.
  incoming:dict[int, _A630MemoryScheduleState] = {0:_A630MemoryScheduleState()}
  pending = [0]
  queued = {0}
  steps = 0
  while pending:
    index = pending.pop()
    queued.remove(index)
    steps += 1
    _require(steps <= _MAX_SCHEDULE_VALIDATION_STEPS, "A630 schedule validation exceeds the emulator work limit")
    instruction,state = active[index],incoming[index]
    force_ss,force_sy = state.force_ss,state.force_sy
    needs_ss,needs_sy = set(state.needs_ss),set(state.needs_sy)
    needs_ss_war_full,needs_ss_war_half = set(state.needs_ss_war_full),set(state.needs_ss_war_half)
    needs_ss_or_sy_war_full = set(state.needs_ss_or_sy_war_full)
    if _schedule_flag(instruction, "SS"):
      force_ss = False
      needs_ss.clear()
      needs_ss_war_full.clear()
      needs_ss_war_half.clear()
      needs_ss_or_sy_war_full.clear()
    if _schedule_flag(instruction, "SY"):
      force_sy = False
      needs_sy.clear()
      needs_ss_or_sy_war_full.clear()
    if instruction.opcode != "nop":
      _require(not force_ss, f"A630 barrier synchronization lacks SS before instruction {instruction.index}")
      _require(not force_sy, f"A630 barrier synchronization lacks SY before instruction {instruction.index}")
    _require(not needs_ss.intersection(_full_gpr_accesses(instruction)),
             f"A630 local-load dependency lacks SS synchronization at instruction {instruction.index}")
    _require(not needs_sy.intersection(_full_gpr_accesses(instruction)),
             f"A630 global-load dependency lacks SY synchronization at instruction {instruction.index}")
    _require(not needs_ss_war_full.intersection(_full_gpr_writes(instruction)) and
             not needs_ss_war_half.intersection({instruction.dst.value} if instruction.dst is not None and
                                                instruction.dst.kind == "half" else set()),
             f"A630 memory-source overwrite lacks SS synchronization at instruction {instruction.index}")
    _require(not needs_ss_or_sy_war_full.intersection(_full_gpr_writes(instruction)),
             f"A630 global-load source overwrite lacks SS or SY synchronization at instruction {instruction.index}")
    if instruction.opcode == "bar.g": force_ss = force_sy = True
    elif instruction.opcode == "ldl.u32x4":
      assert instruction.dst is not None
      needs_ss.update(range(instruction.dst.value, instruction.dst.value + 4))
    elif instruction.opcode in {"ldg.u32", "ldg.u32x4"}:
      assert instruction.dst is not None
      needs_sy.update(range(instruction.dst.value, instruction.dst.value + (4 if instruction.opcode == "ldg.u32x4" else 1)))
      needs_ss_or_sy_war_full.update(range(instruction.srcs[0].value, instruction.srcs[0].value + 2))
    if instruction.opcode == "ldl.u32x4": needs_ss_war_full.add(instruction.srcs[0].value)
    store_full,store_half = _store_source_gprs(instruction)
    needs_ss_war_full.update(store_full)
    needs_ss_war_half.update(store_half)
    outgoing = _A630MemoryScheduleState(force_ss, force_sy, frozenset(needs_ss), frozenset(needs_sy),
      frozenset(needs_ss_war_full), frozenset(needs_ss_war_half), frozenset(needs_ss_or_sy_war_full))
    for successor in _instruction_successors(active, index):
      _require(0 <= successor < len(active), f"A630 schedule successor {successor} is out of range")
      joined = outgoing if successor not in incoming else _merge_memory_schedule(incoming[successor], outgoing)
      if joined != incoming.get(successor):
        incoming[successor] = joined
        if successor not in queued:
          pending.append(successor)
          queued.add(successor)

def _validate_control_flow(active:Sequence[A630IR3Instruction], dispatch:A630Dispatch) -> None:
  predicated = False
  predicate_depth:list[bool] = []
  for instruction in active:
    predicate_depth.append(predicated)
    if instruction.opcode == "predt.p0":
      _require(not predicated, f"nested A630 predicate region at instruction {instruction.index}")
      predicated = True
    elif instruction.opcode == "prede":
      _require(predicated, f"A630 PREDE lacks a predicate region at instruction {instruction.index}")
      predicated = False
    elif instruction.opcode in {"br.p0", "jump", "bar.g", "end"}:
      _require(not predicated, f"unsupported A630 control inside a predicate region at instruction {instruction.index}")
  _require(not predicated, "unterminated A630 predicate region")

  controls = tuple(instruction for instruction in active if instruction.opcode in {"br.p0", "jump"})
  _require(not controls or dispatch.global_size == (1, 1, 1),
           "A630 control flow requires exactly one invocation")
  for instruction in controls:
    offset = instruction.srcs[-1].value
    target = instruction.index + offset
    _require(0 <= target < len(active), f"A630 control-flow target {target} is out of range")
    _require(predicate_depth[target] == predicate_depth[instruction.index],
             f"A630 control-flow target {target} crosses a predicate region")

  # The lockstep interpreter may treat the workgroup barrier as a host no-op only after admitting an explicit
  # store/barrier/load phase. Otherwise its instruction-major execution would silently add cross-lane visibility.
  local_stores = tuple(instruction.index for instruction in active if instruction.opcode == "stl.u32")
  local_loads = tuple(instruction.index for instruction in active if instruction.opcode == "ldl.u32x4")
  barriers = tuple(instruction.index for instruction in active if instruction.opcode == "bar.g")
  _require(not controls or not (local_stores or local_loads or barriers),
           "A630 local memory with control flow is unsupported")
  if dispatch.local_size[0] > 1 and (local_stores or local_loads):
    _require(len(barriers) == 1 and bool(local_stores) and bool(local_loads) and
             max(local_stores) < barriers[0] < min(local_loads),
             "A630 multi-lane local memory requires one store/barrier/load phase")
  _validate_memory_schedule(active)
  _validate_fixed_alu_delays(active)

def _validate_register_footprint(dispatch:A630Dispatch, active:Sequence[A630IR3Instruction], wgid:int, lid:int) -> None:
  registers = dict(dispatch.registers)
  full:set[int] = set()
  half:set[int] = set()

  def add_full(base:int, width:int=1) -> None:
    _require(0 <= base and base + width <= 0xc0, "unsupported shared or special IR3 register")
    full.update(range(base, base + width))

  if lid != 0xfc: add_full(lid, 3)
  for instruction in active:
    for register in _full_gpr_accesses(instruction): add_full(register)
    operands = ((instruction.dst,) if instruction.dst is not None else ()) + instruction.srcs
    for operand in operands:
      if operand.kind == "half":
        _require(0 <= operand.value < 0xc0, "unsupported shared or special IR3 register")
        half.add(operand.value)
      elif operand.kind == "shared":
        _require(wgid != 0xfc and wgid <= operand.value <= wgid + 2,
                 f"unmapped A630 shared register {operand.value}")
      elif operand.kind == "const":
        _require(0 <= operand.value < 1024, f"constant register {operand.value} is out of range")
      elif operand.kind == "flut":
        _require(operand.value in (2, 3), f"unsupported float lookup immediate {operand.value}")

  control = registers[mesa.REG_A6XX_SP_CS_CNTL_0]
  half_footprint = (control & mesa.A6XX_SP_CS_CNTL_0_HALFREGFOOTPRINT__MASK) >> mesa.A6XX_SP_CS_CNTL_0_HALFREGFOOTPRINT__SHIFT
  full_footprint = (control & mesa.A6XX_SP_CS_CNTL_0_FULLREGFOOTPRINT__MASK) >> mesa.A6XX_SP_CS_CNTL_0_FULLREGFOOTPRINT__SHIFT
  _require(control == half_footprint << mesa.A6XX_SP_CS_CNTL_0_HALFREGFOOTPRINT__SHIFT |
           full_footprint << mesa.A6XX_SP_CS_CNTL_0_FULLREGFOOTPRINT__SHIFT,
           "unsupported A630 thread or register control flags")
  expected = (max(half) // 4 + 1 if half else 0, max(full) // 4 + 1 if full else 0)
  _require((half_footprint, full_footprint) == expected,
           "A630 register footprints do not match decoded operands")

def _execution_dispatch(submission:A630Submission) -> A630Dispatch:
  _require(len(submission.dispatches) == 1, "A630 execution requires exactly one dispatch")
  dispatch = submission.dispatches[0]
  _require(dispatch.groups[1:] == (1, 1) and 1 <= dispatch.groups[0] and
           dispatch.local_size[1:] == (1, 1) and 1 <= dispatch.local_size[0] <= 64 and
           dispatch.global_size == (dispatch.groups[0] * dispatch.local_size[0], 1, 1) and
           dispatch.global_size[0] <= _MAX_INVOCATIONS,
           "A630 execution currently requires a bounded one-dimensional Thread64 dispatch")
  _require(len(dispatch.constants_image) == 4096, "unsupported A630 constant image size")
  end = next(instruction.index for instruction in dispatch.instructions if instruction.opcode == "end")
  active = dispatch.instructions[:end+1]
  _require(len(active) <= _MAX_SHADER_INSTRUCTIONS, "A630 program exceeds the emulator instruction limit")
  for instruction in active: _validate_operand_contract(instruction)
  wgid,lid = _system_registers(dispatch)
  _validate_control_flow(active, dispatch)
  _validate_register_footprint(dispatch, active, wgid, lid)
  registers = dict(dispatch.registers)
  _require(registers.get(mesa.REG_A6XX_SP_CS_BOOLEAN_CF_MASK) == 0 and
           registers.get(mesa.REG_A6XX_SP_CS_PROGRAM_COUNTER_OFFSET) == 0,
           "unsupported A630 predicate mask or program entry")
  _local_capacity(registers)
  return dispatch

def _local_capacity(registers:dict[int, int]) -> int:
  control = registers[mesa.REG_A6XX_SP_CS_CNTL_1]
  shared_units = (control & mesa.A6XX_SP_CS_CNTL_1_SHARED_SIZE__MASK) >> mesa.A6XX_SP_CS_CNTL_1_SHARED_SIZE__SHIFT
  constant_mode = (control & mesa.A6XX_SP_CS_CNTL_1_CONSTANTRAMMODE__MASK) >> mesa.A6XX_SP_CS_CNTL_1_CONSTANTRAMMODE__SHIFT
  _require(control == shared_units | constant_mode << mesa.A6XX_SP_CS_CNTL_1_CONSTANTRAMMODE__SHIFT and
           constant_mode == mesa.CONSTLEN_256, "unsupported A630 shared-memory or constant-RAM control")
  # a6xx.xml defines zero as the full 32 KiB allocation; nonzero values encode the upper KiB index.
  return 32 * 1024 if shared_units == 0 else (shared_units + 1) * 1024

def _resolved_global_view(resolver:Resolver, address:int, size:int, purpose:str) -> memoryview:
  alignment = 1 if size == 1 else 4
  _require(address != 0 and address % alignment == 0 and address <= (1 << 64) - size,
           f"invalid A630 {purpose} address")
  view = resolver(address, size)
  _require(len(view) == size, f"short A630 {purpose} range")
  return view

def _signed_u32(value:int) -> int:
  return value - (1 << 32) if value & 0x80000000 else value

def _execute_alu(opcode:str|None, src:tuple[int, ...]) -> int:
  if opcode == "mov.u32": return src[0]
  if opcode == "add.u": return src[0] + src[1]
  if opcode == "sub.u": return src[0] - src[1]
  if opcode == "shl.b": return src[0] << (src[1] & 31)
  if opcode == "ashr.b": return _signed_u32(src[0]) >> (src[1] & 31)
  if opcode == "shr.b":
    # Preserve the existing supported-domain restriction until the tinygrad/PYTHON wider-shift mismatch is resolved.
    _require(src[1] < 32, "u32 logical right shift count is outside the supported 0..31 range")
    return src[0] >> src[1]
  if opcode == "shrg": return (src[1] >> (src[0] & 31)) | src[2]
  if opcode == "max.u": return max(src)
  if opcode == "max.s": return src[0] if _signed_u32(src[0]) >= _signed_u32(src[1]) else src[1]
  if opcode == "xor.b": return src[0] ^ src[1]
  if opcode == "and.b": return src[0] & src[1]
  if opcode == "or.b": return src[0] | src[1]
  if opcode == "mull.u": return (src[0] & 0xffff) * (src[1] & 0xffff)
  if opcode == "madsh.m16": return ((src[0] & 0xffff) * (src[1] >> 16) << 16) + src[2]
  if opcode == "cmps.s.lt": return int(_signed_u32(src[0]) < _signed_u32(src[1]))
  if opcode == "cmps.u.lt": return int(src[0] < src[1])
  if opcode == "cmps.s.eq": return int(src[0] == src[1])
  if opcode == "cov.u16s32": return src[0] & 0xffff
  if opcode == "add.f": return _f32_add_bits(src[0], src[1])
  raise ValueError(f"unsupported A630 opcode {opcode}")

def _execute_a630_dispatch(dispatch:A630Dispatch, resolver:Resolver, active:Sequence[A630IR3Instruction], *,
                           read_observer:ReadObserver|None, budget:A630ExecutionBudget) -> tuple[A630ExecutionWrite, ...]:
  constants = struct.unpack("<1024I", dispatch.constants_image)
  registers = dict(dispatch.registers)
  wgid,lid = _system_registers(dispatch)
  capacity = _local_capacity(registers)
  has_control = any(instruction.opcode in {"br.p0", "jump"} for instruction in active)
  max_steps = (len(active) + 1) * _MAX_CONTROL_FLOW_ITERATIONS if has_control else len(active) + 1
  writes:list[A630ExecutionWrite] = []
  reads:list[tuple[int, int, str]] = []
  def record_memory_event() -> None:
    budget.memory_events += 1
    _require(budget.memory_events <= _MAX_MEMORY_EVENTS, "A630 execution exceeded its bounded memory-event limit")

  def record_instruction_steps(count:int) -> None:
    budget.lane_instruction_steps += count
    _require(budget.lane_instruction_steps <= _MAX_LANE_INSTRUCTION_STEPS,
             "A630 execution exceeded its bounded lane-instruction limit")

  for group in range(dispatch.groups[0]):
    lanes = [_A630LaneState({lid:lane, lid+1:0, lid+2:0} if lid != 0xfc else {}, {}, {})
             for lane in range(dispatch.local_size[0])]
    shared = {} if wgid == 0xfc else {wgid:group, wgid+1:0, wgid+2:0}
    state = _A630WorkgroupState(0, lanes, shared, [True] * len(lanes), False, {}, 0)
    while True:
      _require(0 <= state.pc < len(active), f"A630 program counter {state.pc} is out of range")
      state.steps += 1
      _require(state.steps <= max_steps, "A630 execution exceeded its deterministic instruction limit")
      record_instruction_steps(len(state.lanes))
      instruction = active[state.pc]
      opcode = instruction.opcode
      try:
        if opcode == "end":
          _require(not state.predicated, "A630 end occurs inside a predicate region")
          break
        if opcode == "nop":
          state.pc += 1
          continue
        if opcode == "predt.p0":
          _require(not state.predicated and all(0 in lane.predicates for lane in state.lanes),
                   "A630 predicate is unavailable")
          state.active = [lane.predicates[0] for lane in state.lanes]
          state.predicated = True
          state.pc += 1
          continue
        if opcode == "prede":
          _require(state.predicated, "A630 PREDE lacks a predicate region")
          state.active = [True] * len(state.lanes)
          state.predicated = False
          state.pc += 1
          continue
        if opcode == "br.p0":
          _require(len(state.lanes) == 1 and not state.predicated, "unsupported divergent A630 branch")
          predicate = instruction.srcs[0]
          _require(predicate.value in state.lanes[0].predicates, "A630 branch reads an uninitialized predicate")
          target = instruction.index + instruction.srcs[1].value
          state.pc = target if state.lanes[0].predicates[predicate.value] else state.pc + 1
          continue
        if opcode == "jump":
          _require(len(state.lanes) == 1 and not state.predicated, "unsupported divergent A630 jump")
          state.pc = instruction.index + instruction.srcs[0].value
          continue
        if opcode == "bar.g":
          _require(not state.predicated and all(state.active), "nonuniform or predicated A630 barrier")
          state.pc += 1
          continue

        if opcode == "stl.u32":
          pending:list[tuple[int, int]] = []
          for lane_index,lane in enumerate(state.lanes):
            if not state.active[lane_index]: continue
            record_memory_event()
            address = _read_ir3_operand(instruction.srcs[0], lane.full, lane.half, state.shared, constants)
            value = _read_ir3_operand(instruction.srcs[1], lane.full, lane.half, state.shared, constants)
            _require(address % 4 == 0 and address + 4 <= capacity, "A630 local store is unaligned or out of range")
            _require(all(address != other for other,_ in pending), "overlapping A630 local stores")
            pending.append((address, value))
          state.local.update(pending)
          state.pc += 1
          continue

        for lane_index,lane in enumerate(state.lanes):
          if not state.active[lane_index]: continue
          try:
            if opcode in {"cmps.s.ge.p0", "cmps.s.eq.p0"}:
              left,right = (_read_ir3_operand(operand, lane.full, lane.half, state.shared, constants)
                            for operand in instruction.srcs)
              assert instruction.dst is not None
              lane.predicates[instruction.dst.value] = _signed_u32(left) >= _signed_u32(right) \
                if opcode == "cmps.s.ge.p0" else left == right
              continue
            if opcode in {"ldg.u32", "ldg.u32x4"}:
              assert instruction.dst is not None
              record_memory_event()
              components = 1 if opcode == "ldg.u32" else 4
              address = _gpr_address(lane.full, instruction.srcs[0])
              image = bytes(_resolved_global_view(resolver, address, components * 4, "global-load"))
              purpose = f"A630 global load I{instruction.index:03d}"
              reads.append((address, components * 4, purpose))
              if read_observer is not None: read_observer(address, components * 4, purpose)
              for component,value in enumerate(struct.unpack(f"<{components}I", image)):
                lane.full[instruction.dst.value + component] = value
              continue
            if opcode in {"stg.u32", "stg.u32x4", "stg.u8"}:
              record_memory_event()
              address = _gpr_address(lane.full, instruction.srcs[0])
              if opcode == "stg.u8":
                value = _read_ir3_operand(instruction.srcs[1], lane.full, lane.half, state.shared, constants)
                data = bytes((value & 0xff,))
              else:
                components = 1 if opcode == "stg.u32" else 4
                data = struct.pack(f"<{components}I", *(lane.full[instruction.srcs[1].value + component]
                                                        for component in range(components)))
              _resolved_global_view(resolver, address, len(data), "global-store")
              writes.append(A630ExecutionWrite(address, data))
              continue
            if opcode == "ldl.u32x4":
              assert instruction.dst is not None
              record_memory_event()
              address = _read_ir3_operand(instruction.srcs[0], lane.full, lane.half, state.shared, constants)
              _require(address % 4 == 0 and address + 16 <= capacity,
                       "A630 local load is unaligned or out of range")
              addresses = tuple(address + component * 4 for component in range(4))
              _require(all(component_address in state.local for component_address in addresses),
                       "A630 local load reads an uninitialized dword")
              for component,component_address in enumerate(addresses):
                lane.full[instruction.dst.value + component] = state.local[component_address]
              continue
            if opcode == "add.f.rpt4":
              assert instruction.dst is not None
              # ir3_legalize executes each repeat cycle in order; later components may observe an earlier destination write.
              for component in range(4):
                left = lane.full[instruction.srcs[0].value + component]
                right = lane.full[instruction.srcs[1].value + component]
                lane.full[instruction.dst.value + component] = _f32_add_bits(left, right)
              continue
            if opcode == "add.u.rpt2":
              assert instruction.dst is not None
              left = _read_ir3_operand(instruction.srcs[0], lane.full, lane.half, state.shared, constants)
              # Only SRC2 carries (r), so the constant remains fixed while the GPR source advances.
              for component in range(2):
                right = lane.full[instruction.srcs[1].value + component]
                lane.full[instruction.dst.value + component] = (left + right) & 0xffffffff
              continue

            src = tuple(_read_ir3_operand(operand, lane.full, lane.half, state.shared, constants)
                        for operand in instruction.srcs)
            _require(instruction.dst is not None, f"unsupported A630 semantic {opcode}")
            assert instruction.dst is not None
            _write_ir3_operand(instruction.dst, _execute_alu(opcode, src), lane.full, lane.half)
          except (KeyError, ValueError, RuntimeError) as error:
            raise ValueError(f"A630 instruction {instruction.index} lane {lane_index} group {group}: {error}") from error
        state.pc += 1
      except (KeyError, ValueError, RuntimeError) as error:
        if isinstance(error, ValueError) and str(error).startswith("A630 instruction "): raise
        raise ValueError(f"A630 instruction {instruction.index} group {group}: {error}") from error

  return _finish_writes(writes, reads)


def execute_a630(submission:A630Submission, resolver:Resolver, *, read_observer:ReadObserver|None=None,
                 budget:A630ExecutionBudget|None=None) -> tuple[A630ExecutionWrite, ...]:
  """Execute admitted A630 images into an immutable write journal; this does not retire the KGSL submission."""
  dispatch = _execution_dispatch(submission)
  end = next(instruction.index for instruction in dispatch.instructions if instruction.opcode == "end")
  writes = _execute_a630_dispatch(dispatch, resolver, dispatch.instructions[:end+1], read_observer=read_observer,
                                  budget=budget if budget is not None else A630ExecutionBudget())
  immutable_reads = tuple((memory_range.address, memory_range.size, memory_range.purpose)
                          for memory_range in submission.memory_ranges
                          if memory_range.read and memory_range.purpose != "wait value")
  return _finish_writes(writes, immutable_reads)

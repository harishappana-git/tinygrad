from __future__ import annotations
import ctypes, os, struct
from dataclasses import dataclass, replace
from typing import Callable, Sequence
from tinygrad.runtime.autogen import libc, mesa
from test.mockgpu.qcom.pm4 import PM4Packet, PM4Type4Packet, PM4Type7Packet

# Payload fields and units follow Mesa 25.2.7 at 461196a1c827769168304ff3f5b36360f16618ca:
# adreno_pm4.xml, a6xx.xml, a6xx_descriptors.xml, tu_shader.cc, tu_cmd_buffer.cc, ir3_shader.h,
# ir3.xml, ir3-common.xml, ir3-cat[0-7].xml, ir3.h, ir3_a6xx.c, ir3_compiler_nir.c, ir3_nir_imul.py,
# nir_opcodes.py, isaspec.h, and isaspec_decode_impl.c.

@dataclass(frozen=True)
class A630MemoryRange:
  address:int
  size:int
  read:bool
  write:bool
  purpose:str
  image:bytes|None = None

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
class A630Resource:
  kind:str
  index:int
  descriptor_address:int
  address:int
  size:int
  read:bool
  write:bool
  width:int
  height:int
  pitch:int
  itemsize:int
  image:bytes|None = None

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
  resources:tuple[A630Resource, ...] = ()
  instructions:tuple[A630IR3Instruction, ...] = ()

@dataclass(frozen=True)
class A630Submission:
  dispatches:tuple[A630Dispatch, ...]
  memory_ranges:tuple[A630MemoryRange, ...]
  waits:tuple[A630Wait, ...]
  writes:tuple[A630Write, ...]

Resolver = Callable[[int, int], memoryview]
_MAX_INVOCATIONS = 0x10000

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
  cat2_compare = name in {"cmps.u", "cmps.s"}
  if category == 2 and name in {"ashr.b", "shl.b", "add.u", "sub.u", "xor.b", "and.b", "mull.u", "cmps.u", "cmps.s", "add.f"} and \
     _has_no_repeat(fields) and all(_int_field_is(fields, field, 0) for field in ("JP", "SAT", "UL", "EI", "LAST", "ABSNEG", "SRC_R")) and \
     (raw >> 52 & 1, raw >> 46 & 1) == (1, int(cat2_compare)):
    dst_half = bool(_same_int_field(fields, "DST_HALF"))
    dst = _register_operand(_same_int_field(fields, "DST"), not dst_half) if cat2_compare else \
      A630IR3Operand("half" if dst_half else "gpr", _same_int_field(fields, "DST"))
    full = bool(raw >> 52 & 1)
    srcs = (_multisrc_operand(_same_int_field(fields, "SRC1"), full),
            _multisrc_operand(_same_int_field(fields, "SRC2"), full))
    if dst is not None and all(src is not None for src in srcs):
      if name in {"xor.b", "and.b"} and (dst.kind != "gpr" or not 0 <= dst.value < 0xc0 or
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
    src1_op = _register_operand(src1, True) if src1 == src1 & 0xff else None
    src2_op = _register_operand(src2, True) if src2 == src2 & 0xff else None
    src3_op = _register_operand(src3, True) if src3 == src3 & 0xff else None
    if dst is not None and dst.kind == "gpr" and all(src is not None and src.kind == "gpr" for src in (src1_op, src2_op, src3_op)):
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
  ldg_variable = (0xff << 14) | (0xff << 32) | cat6_schedule
  if category == 6 and name == "ldg" and (_same_int_field(fields, "TYPE"), _same_int_field(fields, "TYPE_HALF"),
                                           _same_int_field(fields, "OFF"), _same_int_field(fields, "SIZE")) == (3, 0, 0, 1) and \
     _int_field_is(fields, "JP", 0) and raw & ~ldg_variable == 0xc006000001800001:
    return "ldg.u32", A630IR3Operand("gpr", _same_int_field(fields, "DST")), \
           (A630IR3Operand("gpr", _same_int_field(fields, "SRC1")),)
  stg_variable = (0xff << 1) | (0xff << 41) | cat6_schedule
  if category == 6 and name == "stg" and (_same_int_field(fields, "TYPE"), _same_int_field(fields, "TYPE_HALF"),
                                           _same_int_field(fields, "OFF"), _same_int_field(fields, "SIZE")) == (3, 0, 0, 1) and \
     _int_field_is(fields, "JP", 0) and raw & ~stg_variable == 0xc0c6010001800000:
    return "stg.u32", None, (A630IR3Operand("gpr", _same_int_field(fields, "SRC1")),
                              A630IR3Operand("gpr", _same_int_field(fields, "SRC3")))
  if category == 6 and name == "stg" and (_same_int_field(fields, "TYPE"), _same_int_field(fields, "TYPE_HALF"),
                                           _same_int_field(fields, "OFF"), _same_int_field(fields, "SIZE")) == (6, 1, 0, 1) and \
     _int_field_is(fields, "JP", 0) and raw & ~stg_variable == 0xc0cc010001800000:
    address,data = _register_operand(_same_int_field(fields, "SRC1"), True), _register_operand(_same_int_field(fields, "SRC3"), False)
    if address is not None and address.kind == "gpr" and address.value + 1 < 0xc0 and data is not None and data.kind == "half":
      return "stg.u8", None, (address, data)
  return None, None, ()

def decode_a630_ir3(image:bytes) -> tuple[A630IR3Instruction, ...]:
  _require(bool(image), "empty IR3 image")
  _require(len(image) % 8 == 0, "IR3 image size must be a multiple of 8")
  _require(len(image) <= 0x7fffffff, "IR3 image exceeds Mesa decoder size")
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
  address = _address(values[1], values[2], alignment, kind)
  return A630LoadState(kind, address, units * unit_size, units)

def _resource_descriptor(kind:str, index:int, descriptor_address:int, image:bytes) -> A630Resource:
  words = struct.unpack("<16I", image)
  fmt = words[0] >> 22 & 0xff
  formats = {mesa.FMT6_16_16_16_16_FLOAT:2, mesa.FMT6_32_32_32_32_FLOAT:4}
  _require(fmt in formats, f"unsupported {kind} descriptor format {fmt}")
  # Bit 3 and words 6/7 are opaque unchanged-runtime literals: the pinned descriptor XML does not assign them these meanings.
  expected_word0 = fmt << 22 | (0x6888 if kind == "texture" else 0)
  _require(words[0] == expected_word0, f"unsupported {kind} descriptor word 0")
  _require(words[1] & 0xc0000000 == 0, f"unsupported {kind} descriptor word 1")
  width, height = words[1] & 0x7fff, words[1] >> 15 & 0x7fff
  _require(0 < width <= 16384 and 0 < height <= 16384, f"unsupported {kind} descriptor dimensions")
  _require(words[2] & 0x70 == 0 and words[2] >> 29 == mesa.A6XX_TEX_2D, f"unsupported {kind} descriptor word 2")
  pitch, pitch_alignment = words[2] >> 7 & 0x3fffff, words[2] & 0xf
  itemsize = formats[fmt]
  _require(pitch >= 64 and pitch % 64 == 0 and pitch == width * 4 * itemsize, f"unsupported {kind} descriptor pitch")
  _require(pitch_alignment == (pitch & -pitch).bit_length() - 7, f"invalid {kind} descriptor pitch alignment")
  _require(words[3] == 0, f"unsupported {kind} descriptor word 3")
  _require(words[5] & ~0x1ffff == 0, f"unsupported {kind} descriptor depth or address")
  address = _address(words[4], words[5], 32, f"{kind} target")
  _require(words[6:] == (0x40000000, 13) + (0,) * 8, f"unsupported {kind} descriptor tail")
  size = pitch * height
  _require(address + size <= 1 << 49, f"overflowing {kind} target range")
  return A630Resource(kind, index, descriptor_address, address, size, True, kind == "uav", width, height, pitch, itemsize)

def _resources(dispatch:A630Dispatch, read_images:dict[tuple[int, int, str], bytes]) -> tuple[A630Resource, ...]:
  registers = dict(dispatch.registers)
  config = registers[mesa.REG_A6XX_SP_CS_CONFIG]
  counts = {"samplers":config >> 17 & 0x1f, "textures":config >> 9 & 0xff, "uavs":config >> 22 & 0x7f}
  loads = {load.kind:load for load in dispatch.loads}
  if (count:=counts["samplers"]):
    table = read_images[(loads["samplers"].address, count * 16, "samplers descriptors")]
    for index in range(count):
      _require(struct.unpack_from("<4I", table, index * 16) == (0x1b60, 0x30, 0, 0), f"unsupported sampler descriptor {index}")
    border_words = _registers(registers, mesa.REG_A6XX_TPL1_CS_BORDER_COLOR_BASE, 2, "border-color base")
    border_address = _address(border_words[0], border_words[1], 128, "border-color")
    _require(read_images[(border_address, 128, "border color")] == bytes(128), "unsupported border color")

  resources:list[A630Resource] = []
  for plural,kind in (("textures", "texture"), ("uavs", "uav")):
    if not (count:=counts[plural]): continue
    load = loads[plural]
    table = read_images[(load.address, count * 64, f"{plural} descriptors")]
    resources.extend(_resource_descriptor(kind, index, load.address + index * 64, table[index*64:(index+1)*64])
                     for index in range(count))
  return tuple(resources)

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
  _require(nsamp == ntex and ntex + nuav <= mesa.IR3_MAX_SHADER_IMAGES, "unsupported IR3 resource counts")
  resources = (("samplers", nsamp, mesa.REG_A6XX_SP_CS_SAMPLER_BASE, 16, nsamp),
               ("textures", ntex, mesa.REG_A6XX_SP_CS_TEXMEMOBJ_BASE, 64, min(16, ntex)),
               ("uavs", nuav, mesa.REG_A6XX_SP_CS_UAV_BASE, 64, nuav))
  for kind,count,base_register,unit_size,load_units in resources:
    if count == 0: continue
    _require(kind in loads and loads[kind].units == load_units, f"missing or inconsistent {kind} state load")
    base_words = _registers(regs, base_register, 2, f"{kind} base")
    base = _address(base_words[0], base_words[1], unit_size if kind != "uavs" else 16, kind)
    _require(base == loads[kind].address, f"{kind} base differs from state load")
    expected_base = constants.address + {"textures":2048, "uavs":2048 + 64*ntex,
                                         "samplers":2048 + 64*(ntex+nuav)}[kind]
    _require(base == expected_base, f"unsupported {kind} descriptor-table placement")
    _require(base % 64 == 0, f"unsupported {kind} descriptor-table alignment")
    ranges.append(A630MemoryRange(base, count * unit_size, read=True, write=False, purpose=f"{kind} descriptors"))
  if nsamp:
    border = _registers(regs, mesa.REG_A6XX_TPL1_CS_BORDER_COLOR_BASE, 2, "border-color base")
    ranges.append(A630MemoryRange(_address(border[0], border[1], 128, "border-color"), 128,
                                  read=True, write=False, purpose="border color"))

  active_loads = tuple(loads[kind] for kind in ("constants", "shader", "samplers", "textures", "uavs") if kind in loads)
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
  read_images:dict[tuple[int, int, str], bytes] = {}
  for memory_range in ranges:
    view = resolver(memory_range.address, memory_range.size)
    _require(len(view) == memory_range.size, f"short resolved {memory_range.purpose} range")
    image = bytes(view) if memory_range.read else None
    resolved_range = replace(memory_range, image=image)
    resolved.append(resolved_range)
    if image is not None: read_images[(memory_range.address, memory_range.size, memory_range.purpose)] = image
  dispatch_instructions = tuple(decode_a630_ir3(read_images[(dispatch.shader_address, dispatch.shader_size, "shader")])
                                for dispatch in dispatches)
  dispatch_resources = tuple(_resources(dispatch, read_images) for dispatch in dispatches)
  nested_ranges = tuple(A630MemoryRange(resource.address, resource.size, read=resource.read, write=resource.write,
                                        purpose=f"{resource.kind} {resource.index} target")
                        for resources in dispatch_resources for resource in resources)
  for memory_range in nested_ranges:
    view = resolver(memory_range.address, memory_range.size)
    _require(len(view) == memory_range.size, f"short resolved {memory_range.purpose} range")
    image = bytes(view) if memory_range.read else None
    resolved.append(replace(memory_range, image=image))
    if image is not None: read_images[(memory_range.address, memory_range.size, memory_range.purpose)] = image

  frozen_dispatches = tuple(replace(dispatch,
    shader_image=read_images[(dispatch.shader_address, dispatch.shader_size, "shader")],
    constants_image=read_images[(dispatch.constants_address, dispatch.constants_size, "constants")],
    resources=tuple(replace(resource,
      image=read_images[(resource.address, resource.size, f"{resource.kind} {resource.index} target")]) for resource in resources),
    instructions=instructions)
    for dispatch,resources,instructions in zip(dispatches, dispatch_resources, dispatch_instructions))
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

def _u32_binary_instruction(instructions:Sequence[A630IR3Instruction], opcode:str) -> A630IR3Instruction|None:
  load_destinations = frozenset(instruction.dst for instruction in instructions if instruction.opcode == "ldg.u32")
  if len(load_destinations) != 2: return None
  matches = tuple(instruction for instruction in instructions if instruction.opcode == opcode and
                  len(instruction.srcs) == 2 and frozenset(instruction.srcs) == load_destinations)
  return matches[0] if len(matches) == 1 else None

def _u32_comparison_instruction(instructions:Sequence[A630IR3Instruction]) -> A630IR3Instruction|None:
  matches:list[A630IR3Instruction] = []
  for opcode in ("cmps.s.lt", "cmps.u.lt", "cmps.s.eq"):
    if (instruction:=_u32_binary_instruction(instructions, opcode)) is not None: matches.append(instruction)
  return matches[0] if len(matches) == 1 else None

def _u32_multiply_sequence(instructions:Sequence[A630IR3Instruction]) \
    -> tuple[A630IR3Instruction, A630IR3Instruction, A630IR3Instruction]|None:
  # Pinned ir3_nir_imul.py lowers imul32 to a low-16 product followed by both low/high cross terms in two MADSH.M16s.
  load_destinations = frozenset(instruction.dst for instruction in instructions if instruction.opcode == "ldg.u32")
  multiplies = tuple(instruction for instruction in instructions if instruction.opcode == "mull.u")
  accumulates = tuple(instruction for instruction in instructions if instruction.opcode == "madsh.m16")
  if len(load_destinations) != 2 or len(multiplies) != 1 or len(accumulates) != 2: return None
  low,first,second = multiplies[0],accumulates[0],accumulates[1]
  if low.dst is None or first.dst is None or second.dst is None or not (low.index < first.index < second.index): return None
  if frozenset(low.srcs) != load_destinations: return None
  if any(len(instruction.srcs) != 3 or frozenset(instruction.srcs[:2]) != load_destinations for instruction in accumulates): return None
  if first.srcs[2] != low.dst or second.srcs[2] != first.dst: return None
  return low,first,second

def _validate_constant_pointer_moves(instructions:Sequence[A630IR3Instruction]) -> bool:
  moves = tuple(instruction for instruction in instructions if instruction.opcode == "mov.u32" and
                instruction.srcs[0].kind == "const")
  if not moves: return False
  loads = tuple(instruction for instruction in instructions if instruction.opcode == "ldg.u32")
  stores = tuple(instruction for instruction in instructions if instruction.opcode in {"stg.u32", "stg.u8"})
  _require(len(stores) == 1 and len(moves) == 2 * (len(loads) + 1), "unsupported A630 constant-pointer move inventory")
  destinations:dict[int, int] = {}
  for instruction in moves:
    assert instruction.dst is not None
    constant = instruction.srcs[0].value
    _require(constant not in destinations and instruction.dst.value not in destinations.values(),
             "duplicate A630 constant-pointer move")
    destinations[constant] = instruction.dst.value
  _require(set(destinations) == set(range(2 * (len(loads) + 1))), "unsupported A630 constant-pointer source")
  bases = (stores[0].srcs[0].value, *(instruction.srcs[0].value for instruction in loads))
  _require(all((destinations[2*index], destinations[2*index+1]) == (base, base+1) for index,base in enumerate(bases)),
           "A630 constant-pointer moves do not match the scalar buffer argument ABI")
  return True

def _execution_dispatch(submission:A630Submission) -> A630Dispatch:
  _require(len(submission.dispatches) == 1, "A630 execution requires exactly one dispatch")
  dispatch = submission.dispatches[0]
  _require(not dispatch.resources, "A630 image execution is not implemented")
  _require(dispatch.groups[1:] == (1, 1) and 1 <= dispatch.groups[0] and
           dispatch.local_size[1:] == (1, 1) and 1 <= dispatch.local_size[0] <= 64 and
           dispatch.global_size == (dispatch.groups[0] * dispatch.local_size[0], 1, 1) and dispatch.global_size[0] <= _MAX_INVOCATIONS,
           "A630 execution currently requires a bounded one-dimensional Thread64 dispatch")
  registers = dict(dispatch.registers)
  wgid,lid = _system_registers(dispatch)
  _require(len(dispatch.constants_image) == 4096, "unsupported A630 constant image size")

  end = next(instruction.index for instruction in dispatch.instructions if instruction.opcode == "end")
  active = dispatch.instructions[:end+1]
  unsupported = next((instruction for instruction in active if instruction.opcode is None), None)
  _require(unsupported is None, f"unsupported A630 semantic at instruction {unsupported.index if unsupported else -1}")
  opcodes = tuple(instruction.opcode for instruction in active)
  input_count = opcodes.count("ldg.u32")
  float_add_count = opcodes.count("add.f")
  _require(input_count in (0, 1, 2), "A630 execution supports at most two global loads")
  _require((input_count, float_add_count) in ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1)), "unsupported A630 scalar kernel shape")
  integer_instruction:A630IR3Instruction|None = None
  integer_kind:str|None = None
  comparison_instruction = _u32_comparison_instruction(active)
  multiply_sequence = _u32_multiply_sequence(active)
  comparison_count = sum(opcodes.count(opcode) for opcode in ("cmps.s.lt", "cmps.u.lt", "cmps.s.eq"))
  if opcodes.count("stg.u8"):
    _require((input_count, float_add_count, comparison_count, opcodes.count("stg.u8")) == (2, 0, 1, 1) and
             comparison_instruction is not None, "u32 comparison does not consume both global loads")
  elif opcodes.count("mull.u") or opcodes.count("madsh.m16"):
    _require((input_count, float_add_count, opcodes.count("mull.u"), opcodes.count("madsh.m16")) == (2, 0, 1, 2) and
             multiply_sequence is not None, "u32 multiplication sequence does not consume both global loads")
    assert multiply_sequence is not None
    integer_instruction,integer_kind = multiply_sequence[-1],"multiply"
  elif opcodes.count("sub.u"):
    integer_instruction = _u32_binary_instruction(active, "sub.u")
    _require((input_count, float_add_count, opcodes.count("sub.u")) == (2, 0, 1) and integer_instruction is not None,
             "u32 subtraction does not consume both global loads")
    integer_kind = "subtraction"
  elif opcodes.count("xor.b"):
    integer_instruction = _u32_binary_instruction(active, "xor.b")
    _require((input_count, float_add_count, opcodes.count("xor.b")) == (2, 0, 1) and integer_instruction is not None,
             "u32 bitwise XOR does not consume both global loads")
    integer_kind = "bitwise XOR"
  elif opcodes.count("and.b"):
    integer_instruction = _u32_binary_instruction(active, "and.b")
    _require((input_count, float_add_count, opcodes.count("and.b")) == (2, 0, 1) and integer_instruction is not None,
             "u32 bitwise AND does not consume both global loads")
    integer_kind = "bitwise AND"
  elif (input_count, float_add_count) == (2, 0):
    integer_instruction = _u32_binary_instruction(active, "add.u")
    _require(integer_instruction is not None, "u32 add does not consume both global loads")
    integer_kind = "add"
  has_integer_add = integer_instruction is not None and integer_instruction.opcode == "add.u"
  shared_uses = tuple(operand.value for instruction in active for operand in instruction.srcs if operand.kind == "shared")
  uses_constant_pointers = _validate_constant_pointer_moves(active)
  if input_count == 0:
    expected_counts = {"shl.b":3, "mov.u32":1, "nop":3, "add.u":4, "ashr.b":1, "shrg":1,
                       "cmps.u.lt":1, "cov.u16s32":1, "stg.u32":1, "end":1}
  elif uses_constant_pointers:
    _require((integer_instruction is not None or comparison_instruction is not None) and not shared_uses,
             "constant-pointer A630 execution supports only scalar u32 arithmetic or comparison")
    _require(dispatch.local_size == dispatch.groups == dispatch.global_size == (1, 1, 1),
             "constant-pointer A630 execution requires one scalar invocation")
    if comparison_instruction is not None:
      assert comparison_instruction.opcode is not None
      expected_counts = {"mov.u32":6, "nop":3, "ldg.u32":2, comparison_instruction.opcode:1, "stg.u8":1, "end":1}
    else:
      assert integer_instruction is not None and integer_instruction.opcode is not None
      expected_counts = {"mov.u32":6, "nop":3, "ldg.u32":2, "stg.u32":1, "end":1}
      if integer_kind == "multiply": expected_counts.update({"mull.u":1, "madsh.m16":2})
      else: expected_counts[integer_instruction.opcode] = 1
  else:
    _require(comparison_instruction is None, "u32 comparison requires the scalar constant-pointer ABI")
    _require(integer_kind in (None, "add"),
             "u32 subtraction, multiplication, and bitwise operations currently require the scalar constant-pointer ABI")
    uses_workgroup_id = bool(shared_uses)
    expected_counts = {"ashr.b":1, "shl.b":2, "shrg":1, "add.u":3 * (input_count + 1) + int(has_integer_add),
                       "cmps.u.lt":input_count + 1, "cov.u16s32":input_count + 1, "nop":3 + int(uses_workgroup_id),
                       "ldg.u32":input_count, "stg.u32":1, "end":1}
    if float_add_count: expected_counts["add.f"] = 1
    if uses_workgroup_id: expected_counts["mov.u32"] = 1
  _require(len(opcodes) == sum(expected_counts.values()) and
           all(opcodes.count(opcode) == count for opcode,count in expected_counts.items()),
           "unsupported A630 scalar instruction inventory")

  for instruction in active:
    dst_kind = instruction.dst.kind if instruction.dst is not None else None
    src_kinds = tuple(operand.kind for operand in instruction.srcs)
    valid = instruction.opcode in {"nop", "end"} and dst_kind is None and not src_kinds
    if instruction.opcode == "ashr.b":
      valid = dst_kind == "gpr" and src_kinds in (("gpr", "iim"), ("shared", "iim")) and instruction.srcs[1].value == 31
    elif instruction.opcode == "shl.b":
      valid = dst_kind == "gpr" and src_kinds in (("gpr", "iim"), ("shared", "iim")) and instruction.srcs[1].value in (1, 2)
    elif instruction.opcode == "shrg": valid = dst_kind == "gpr" and src_kinds == ("iim", "gpr", "gpr") and instruction.srcs[0].value == 30
    elif instruction.opcode == "mov.u32": valid = dst_kind == "gpr" and src_kinds in (("shared",), ("const",), ("uim",))
    elif instruction.opcode == "add.u": valid = dst_kind == "gpr" and (src_kinds == ("gpr", "gpr") or set(src_kinds) == {"const", "gpr"})
    elif instruction.opcode == "sub.u": valid = dst_kind == "gpr" and src_kinds == ("gpr", "gpr")
    elif instruction.opcode == "xor.b": valid = dst_kind == "gpr" and src_kinds == ("gpr", "gpr")
    elif instruction.opcode == "and.b": valid = dst_kind == "gpr" and src_kinds == ("gpr", "gpr")
    elif instruction.opcode == "mull.u": valid = dst_kind == "gpr" and src_kinds == ("gpr", "gpr")
    elif instruction.opcode == "madsh.m16": valid = dst_kind == "gpr" and src_kinds == ("gpr", "gpr", "gpr")
    elif instruction.opcode == "cmps.u.lt":
      valid = dst_kind == "half" and (src_kinds == ("gpr", "const") or
              comparison_instruction is not None and instruction.index == comparison_instruction.index and src_kinds == ("gpr", "gpr"))
    elif instruction.opcode in {"cmps.s.lt", "cmps.s.eq"}:
      valid = dst_kind == "half" and comparison_instruction is not None and instruction.index == comparison_instruction.index and \
              src_kinds == ("gpr", "gpr")
    elif instruction.opcode == "cov.u16s32": valid = dst_kind == "gpr" and src_kinds == ("half",)
    elif instruction.opcode == "ldg.u32": valid = dst_kind == "gpr" and src_kinds == ("gpr",)
    elif instruction.opcode == "add.f":
      valid = dst_kind == "gpr" and (src_kinds == ("gpr", "gpr") or
              (set(src_kinds) == {"gpr", "flut"} and next(x.value for x in instruction.srcs if x.kind == "flut") in (2, 3)))
    elif instruction.opcode == "stg.u32": valid = dst_kind is None and src_kinds == ("gpr", "gpr")
    elif instruction.opcode == "stg.u8": valid = dst_kind is None and src_kinds == ("gpr", "half")
    _require(valid, f"unsupported A630 operand contract at instruction {instruction.index}")

  if integer_instruction is not None:
    assert integer_instruction.dst is not None
    store = next(instruction for instruction in active if instruction.opcode == "stg.u32")
    assert integer_kind is not None
    _require(store.srcs[1] == integer_instruction.dst, f"global store does not consume the u32 {integer_kind}")
  if comparison_instruction is not None:
    assert comparison_instruction.dst is not None
    store = next(instruction for instruction in active if instruction.opcode == "stg.u8")
    _require(store.srcs[1] == comparison_instruction.dst, "global store does not consume the u32 comparison")

  constant_uses = sorted(operand.value for instruction in active for operand in instruction.srcs if operand.kind == "const")
  expected_constants = list(range(2 * (input_count + 1))) if uses_constant_pointers else \
    sorted(value for pointer in range(input_count + 1) for value in (2*pointer, 2*pointer, 2*pointer+1))
  _require(constant_uses == expected_constants, "A630 pointer constants do not match the scalar buffer argument ABI")
  moves = tuple(instruction for instruction in active if instruction.opcode == "mov.u32")
  if input_count == 0:
    _require(len(moves) == 1 and moves[0].srcs == (A630IR3Operand("uim", 0x3f800000),),
             "unsupported A630 fill literal")
  elif uses_constant_pointers:
    _require(len(moves) == 6 and all(move.srcs[0].kind == "const" for move in moves), "unsupported A630 constant-pointer moves")
  else:
    _require((not shared_uses and not moves) or
             (len(moves) == 1 and moves[0].srcs[0].kind == "shared"), "unsupported A630 scalar move contract")
  if shared_uses:
    _require(wgid != 0xfc and all(wgid <= register <= wgid + 2 for register in shared_uses),
             "A630 shared operand is outside the workgroup-id vector")
  full_registers = [] if lid == 0xfc else [lid, lid + 1, lid + 2]
  half_registers:list[int] = []
  for instruction in active:
    operands = ((instruction.dst,) if instruction.dst is not None else ()) + instruction.srcs
    for operand in operands:
      if operand.kind == "gpr": full_registers.append(operand.value)
      elif operand.kind == "half": half_registers.append(operand.value)
    if instruction.opcode in {"ldg.u32", "stg.u32", "stg.u8"}: full_registers.append(instruction.srcs[0].value + 1)
  _require(all(0 <= register < 0xc0 for register in full_registers + half_registers), "unsupported shared or special IR3 register")

  pending_carries:set[int] = set()
  for instruction in active:
    if instruction.opcode == "cmps.u.lt" and (comparison_instruction is None or instruction.index != comparison_instruction.index):
      assert instruction.dst is not None
      _require(instruction.dst.value not in pending_carries, "overwritten A630 carry comparison")
      pending_carries.add(instruction.dst.value)
    elif instruction.opcode == "cov.u16s32":
      source = instruction.srcs[0].value
      _require(source in pending_carries, "A630 carry conversion lacks a comparison")
      pending_carries.remove(source)
  _require(not pending_carries, "A630 carry comparison lacks a conversion")

  control = registers[mesa.REG_A6XX_SP_CS_CNTL_0]
  half_footprint,full_footprint = control >> 1 & 0x3f, control >> 7 & 0x3f
  _require(control == half_footprint << 1 | full_footprint << 7, "unsupported A630 thread or register control flags")
  expected_half = max(half_registers) // 4 + 1 if half_registers else 0
  expected_full = max(full_registers) // 4 + 1
  _require((half_footprint, full_footprint) == (expected_half, expected_full),
           "A630 register footprints do not match decoded operands")
  return dispatch

def execute_a630(submission:A630Submission, resolver:Resolver) -> tuple[A630ExecutionWrite, ...]:
  """Execute the supported A630 scalar images into an immutable write journal; this does not retire the KGSL submission."""
  dispatch = _execution_dispatch(submission)
  constants = struct.unpack("<1024I", dispatch.constants_image)
  lane_count = dispatch.local_size[0]
  invocation_count = dispatch.global_size[0]
  wgid,lid = _system_registers(dispatch)
  active = dispatch.instructions[:next(instruction.index for instruction in dispatch.instructions if instruction.opcode == "end") + 1]
  loads = tuple(instruction for instruction in active if instruction.opcode == "ldg.u32")
  has_float_add = any(instruction.opcode == "add.f" for instruction in active)
  comparison_instruction = _u32_comparison_instruction(active)
  multiply_sequence = _u32_multiply_sequence(active)
  integer_instruction = multiply_sequence[-1] if multiply_sequence is not None else \
    _u32_binary_instruction(active, "sub.u") or _u32_binary_instruction(active, "xor.b") or \
    _u32_binary_instruction(active, "and.b") or _u32_binary_instruction(active, "add.u")
  load_ordinals = {instruction.index:index for index,instruction in enumerate(loads)}
  output_base = constants[0] | constants[1] << 32
  input_bases = tuple(constants[2*index+2] | constants[2*index+3] << 32 for index in range(len(loads)))
  output_itemsize = 1 if comparison_instruction is not None else 4
  _require(output_base != 0 and output_base + invocation_count * output_itemsize <= 1 << 64 and
           all(base != 0 and base % 4 == 0 and base + invocation_count * 4 <= 1 << 64 for base in input_bases),
           "invalid A630 scalar argument range")
  writes:list[A630ExecutionWrite] = []

  for group in range(dispatch.groups[0]):
    full = [({lid:lane, lid+1:0, lid+2:0} if lid != 0xfc else {}) for lane in range(lane_count)]
    half:list[dict[int, int]] = [{} for _ in range(lane_count)]
    origins:list[dict[int, tuple[str, int]]] = [{} for _ in range(lane_count)]
    half_origins:list[dict[int, tuple[str, int]]] = [{} for _ in range(lane_count)]
    shared = {} if wgid == 0xfc else {wgid:group, wgid+1:0, wgid+2:0}
    for instruction in dispatch.instructions:
      if instruction.opcode == "end": break
      if instruction.opcode == "nop": continue
      _require((instruction.opcode is not None and instruction.dst is not None) or instruction.opcode in {"stg.u32", "stg.u8"},
               f"unsupported A630 semantic at instruction {instruction.index}")
      for lane in range(lane_count):
        global_lane = group * lane_count + lane
        try:
          src = tuple(_read_ir3_operand(operand, full[lane], half[lane], shared, constants) for operand in instruction.srcs)
          opcode = instruction.opcode
          origin:tuple[str, int]|None = None
          if opcode == "mov.u32":
            value = src[0]
            if instruction.srcs[0].kind == "uim": origin = ("fill", value)
          elif opcode in {"add.u", "sub.u", "xor.b", "and.b"}:
            if opcode == "add.u": value = src[0] + src[1]
            elif opcode == "sub.u": value = src[0] - src[1]
            elif opcode == "xor.b": value = src[0] ^ src[1]
            else: value = src[0] & src[1]
            if integer_instruction is not None and instruction.index == integer_instruction.index:
              source_origins = tuple(origins[lane].get(operand.value) for operand in instruction.srcs)
              operation = {"add.u":"add", "sub.u":"subtraction", "xor.b":"bitwise XOR", "and.b":"bitwise AND"}[opcode]
              _require(frozenset(source_origins) == frozenset((("load", 0), ("load", 1))),
                       f"u32 {operation} does not consume both global loads")
              origin = ({"add.u":"u32-add", "sub.u":"u32-subtract", "xor.b":"u32-xor", "and.b":"u32-and"}[opcode], 0)
          elif opcode == "mull.u":
            value = (src[0] & 0xffff) * (src[1] & 0xffff)
            source_origins = tuple(origins[lane].get(operand.value) for operand in instruction.srcs)
            _require(multiply_sequence is not None and instruction.index == multiply_sequence[0].index and
                     frozenset(source_origins) == frozenset((("load", 0), ("load", 1))),
                     "u32 multiplication sequence does not consume both global loads")
            origin = ("u32-mul-low", 0)
          elif opcode == "madsh.m16":
            value = ((src[0] & 0xffff) * (src[1] >> 16) << 16) + src[2]
            _require(multiply_sequence is not None, "u32 multiplication sequence does not consume both global loads")
            assert multiply_sequence is not None
            first = instruction.index == multiply_sequence[1].index
            _require(first or instruction.index == multiply_sequence[2].index,
                     "u32 multiplication sequence does not consume both global loads")
            source_origins = tuple(origins[lane].get(operand.value) for operand in instruction.srcs)
            expected_accumulator = ("u32-mul-low", 0) if first else ("u32-mul-cross", 0)
            _require(frozenset(source_origins[:2]) == frozenset((("load", 0), ("load", 1))) and
                     source_origins[2] == expected_accumulator,
                     "u32 multiplication sequence does not consume both global loads")
            origin = ("u32-mul-cross" if first else "u32-multiply", 0)
          elif opcode == "shl.b": value = src[0] << (src[1] & 31)
          elif opcode == "ashr.b":
            signed = src[0] - (1 << 32) if src[0] & 0x80000000 else src[0]
            value = signed >> (src[1] & 31)
          elif opcode == "shrg": value = (src[1] >> (src[0] & 31)) | src[2]
          elif opcode in {"cmps.s.lt", "cmps.u.lt", "cmps.s.eq"}:
            left,right = src
            if opcode == "cmps.s.lt":
              left = left - (1 << 32) if left & 0x80000000 else left
              right = right - (1 << 32) if right & 0x80000000 else right
            value = int(left == right) if opcode == "cmps.s.eq" else int(left < right)
            if comparison_instruction is not None and instruction.index == comparison_instruction.index:
              source_origins = tuple(origins[lane].get(operand.value) for operand in instruction.srcs)
              _require(frozenset(source_origins) == frozenset((("load", 0), ("load", 1))),
                       "u32 comparison does not consume both global loads")
              origin = (("u32-equal", 0) if opcode == "cmps.s.eq" else
                        ("s32-less-than" if opcode == "cmps.s.lt" else "u32-less-than", 0))
          elif opcode == "cov.u16s32": value = src[0] & 0xffff
          elif opcode == "add.f":
            source_origins = tuple(("flut", operand.value) if operand.kind == "flut" else origins[lane].get(operand.value)
                                   if operand.kind == "gpr" else None for operand in instruction.srcs)
            if len(loads) == 1:
              _require(frozenset(source_origins) in (frozenset((("load", 0), ("flut", 2))),
                                                     frozenset((("load", 0), ("flut", 3)))),
                       "f32 add does not consume the global load and supported FLUT immediate")
            else:
              _require(frozenset(source_origins) == frozenset((("load", 0), ("load", 1))),
                       "f32 add does not consume both global loads")
            _require(all((bits >> 23 & 0xff) != 0xff and ((bits >> 23 & 0xff) != 0 or bits & 0x7fffff == 0) for bits in src),
                     "unsupported special or subnormal float input")
            result = ctypes.c_float(struct.unpack("<f", struct.pack("<I", src[0]))[0] +
                                    struct.unpack("<f", struct.pack("<I", src[1]))[0]).value
            value = struct.unpack("<I", struct.pack("<f", result))[0]
            _require(value >> 23 & 0xff != 0xff, "unsupported special float result")
            origin = ("f32-add", 0)
          elif opcode == "ldg.u32":
            base = instruction.srcs[0].value
            address = full[lane][base] | full[lane][base + 1] << 32
            ordinal = load_ordinals[instruction.index]
            _require(address == input_bases[ordinal] + global_lane * 4, f"global load {ordinal} does not address its scalar input")
            _require(address % 4 == 0 and address + 4 <= 1 << 64, "invalid A630 global-load address")
            view = resolver(address, 4)
            _require(len(view) == 4, "short A630 global-load range")
            value = struct.unpack("<I", bytes(view))[0]
            origin = ("load", ordinal)
          elif opcode == "stg.u32":
            base = instruction.srcs[0].value
            address = full[lane][base] | full[lane][base + 1] << 32
            _require(address == output_base + global_lane * 4, "global store does not address the scalar output")
            if not loads: expected_origin,store_source = ("fill", 0x3f800000),"A630 fill"
            elif has_float_add: expected_origin,store_source = ("f32-add", 0),"f32 add"
            elif integer_instruction is not None:
              if integer_instruction.opcode == "add.u": expected_origin,store_source = ("u32-add", 0),"u32 add"
              elif integer_instruction.opcode == "sub.u": expected_origin,store_source = ("u32-subtract", 0),"u32 subtraction"
              elif integer_instruction.opcode == "xor.b": expected_origin,store_source = ("u32-xor", 0),"u32 bitwise XOR"
              elif integer_instruction.opcode == "and.b": expected_origin,store_source = ("u32-and", 0),"u32 bitwise AND"
              else: expected_origin,store_source = ("u32-multiply", 0),"u32 multiplication"
            else: expected_origin,store_source = ("load", 0),"global load"
            _require(origins[lane].get(instruction.srcs[1].value) == expected_origin,
                     f"global store does not consume the {store_source}")
            _require(address % 4 == 0 and address + 4 <= 1 << 64, "invalid A630 global-store address")
            view = resolver(address, 4)
            _require(len(view) == 4, "short A630 global-store range")
            writes.append(A630ExecutionWrite(address, struct.pack("<I", src[1])))
            continue
          elif opcode == "stg.u8":
            _require(comparison_instruction is not None and comparison_instruction.opcode is not None,
                     "u32 comparison store lacks a comparison")
            assert comparison_instruction is not None
            base = instruction.srcs[0].value
            address = full[lane][base] | full[lane][base + 1] << 32
            _require(address == output_base + global_lane, "global store does not address the scalar bool output")
            if comparison_instruction.opcode == "cmps.s.eq": expected_origin = ("u32-equal", 0)
            else: expected_origin = ("s32-less-than" if comparison_instruction.opcode == "cmps.s.lt" else "u32-less-than", 0)
            _require(half_origins[lane].get(instruction.srcs[1].value) == expected_origin,
                     "global store does not consume the u32 comparison")
            _require(src[1] in (0, 1), "u32 comparison result is not boolean")
            view = resolver(address, 1)
            _require(len(view) == 1, "short A630 global-store range")
            writes.append(A630ExecutionWrite(address, bytes((src[1],))))
            continue
          else: raise ValueError(f"unsupported A630 opcode {opcode}")
          assert instruction.dst is not None
          _write_ir3_operand(instruction.dst, value, full[lane], half[lane])
          if instruction.dst.kind == "gpr":
            if origin is None: origins[lane].pop(instruction.dst.value, None)
            else: origins[lane][instruction.dst.value] = origin
          elif instruction.dst.kind == "half":
            if origin is None: half_origins[lane].pop(instruction.dst.value, None)
            else: half_origins[lane][instruction.dst.value] = origin
        except (KeyError, ValueError, RuntimeError) as error:
          raise ValueError(f"A630 instruction {instruction.index} lane {lane} group {group}: {error}") from error

  ordered = sorted(writes, key=lambda write: write.address)
  _require(all(left.address + len(left.data) <= right.address for left,right in zip(ordered, ordered[1:])),
           "overlapping A630 global stores")
  return tuple(writes)

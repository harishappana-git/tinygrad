from __future__ import annotations
import ctypes, os, struct
from dataclasses import dataclass, replace
from typing import Callable, Sequence
from tinygrad.runtime.autogen import libc, mesa
from test.mockgpu.qcom.pm4 import PM4Packet, PM4Type4Packet, PM4Type7Packet

# Payload fields and units follow Mesa 25.2.7 at 461196a1c827769168304ff3f5b36360f16618ca:
# adreno_pm4.xml, a6xx.xml, a6xx_descriptors.xml, tu_shader.cc, tu_cmd_buffer.cc, ir3_shader.h,
# ir3.xml, ir3-common.xml, ir3-cat[0-7].xml, ir3.h, ir3.c, ir3_a6xx.c, ir3_compiler_nir.c,
# ir3_nir_analyze_ubo_ranges.c, ir3_nir_imul.py, ir3_nir_lower_64b.c, ir3_rpt.c, nir_lower_int64.c,
# nir_lower_system_values.c, nir_opcodes.py, isaspec.h, and isaspec_decode_impl.c.

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

# These labels bind scalar admission, producer provenance, and store validation; arithmetic remains explicit in execute_a630.
_SIMPLE_CAT2_INTEGER:dict[str, tuple[str, str]] = {
  "shr.b": ("logical right shift", "u32-logical-right-shift"),
  "sub.u": ("subtraction", "u32-subtract"),
  "xor.b": ("bitwise XOR", "u32-xor"),
  "and.b": ("bitwise AND", "u32-and"),
  "or.b": ("bitwise OR", "u32-or"),
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
  typed_cov_variable = cov_variable | (0x7 << 46) | (0x7 << 50) | (0x3 << 55)
  if category == 1 and name is None and raw & ~typed_cov_variable == 0x2000000000000000:
    src_type = _same_int_field(fields, "SRC_TYPE")
    opcode = {5:"cov.s32f32", 3:"cov.u32f32"}.get(src_type)
    if (_same_int_field(fields, "SRC_TYPE"), _same_int_field(fields, "DST_TYPE"), _same_int_field(fields, "ROUND"),
        _same_int_field(fields, "DST_HALF"), _same_int_field(fields, "HALF")) == (src_type, 1, 1, 0, 0) and \
       opcode is not None and _has_no_repeat(fields) and \
       all(_int_field_is(fields, field, 0) for field in ("JP", "UL", "SRC_R", "LAST")):
      dst,src = _register_operand(_same_int_field(fields, "DST"), True), _register_operand(_same_int_field(fields, "SRC"), True)
      if dst is not None and dst.kind == "gpr" and src is not None and src.kind == "gpr": return opcode, dst, (src,)
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

def _integer_to_f32_instruction(instructions:Sequence[A630IR3Instruction]) -> A630IR3Instruction|None:
  loads = tuple(instruction for instruction in instructions if instruction.opcode == "ldg.u32")
  if len(loads) != 1 or loads[0].dst is None: return None
  matches = tuple(instruction for instruction in instructions if instruction.opcode in {"cov.s32f32", "cov.u32f32"} and
                  instruction.srcs == (loads[0].dst,))
  return matches[0] if len(matches) == 1 else None

def _integer_to_f32_rne_bits(value:int, signed:bool) -> int:
  _require(0 <= value <= 0xffffffff, "integer-to-f32 source is outside 32 bits")
  sign = int(signed and bool(value & 0x80000000))
  magnitude = (1 << 32) - value if sign else value
  if magnitude == 0: return 0
  exponent = magnitude.bit_length() - 1
  if exponent <= 23: significand = magnitude << (23 - exponent)
  else:
    shift = exponent - 23
    significand = magnitude >> shift
    remainder,halfway = magnitude & ((1 << shift) - 1),1 << (shift - 1)
    if remainder > halfway or remainder == halfway and significand & 1:
      significand += 1
      if significand == 1 << 24: significand,exponent = significand >> 1,exponent + 1
  return sign << 31 | (exponent + 127) << 23 | significand & 0x7fffff

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
  loads = tuple(instruction for instruction in instructions if instruction.opcode in {"ldg.u32", "ldg.u32x4"})
  stores = tuple(instruction for instruction in instructions if instruction.opcode in {"stg.u32", "stg.u8", "stg.u32x4"})
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
           "A630 constant-pointer moves do not match the buffer argument ABI")
  return True

def _validate_carry_conversions(instructions:Sequence[A630IR3Instruction], excluded_comparison:int|None=None) -> None:
  pending:set[int] = set()
  for instruction in instructions:
    if instruction.opcode == "cmps.u.lt" and instruction.index != excluded_comparison:
      assert instruction.dst is not None
      _require(instruction.dst.value not in pending, "overwritten A630 carry comparison")
      pending.add(instruction.dst.value)
    elif instruction.opcode == "cov.u16s32":
      source = instruction.srcs[0].value
      _require(source in pending, "A630 carry conversion lacks a comparison")
      pending.remove(source)
  _require(not pending, "A630 carry comparison lacks a conversion")

def _validate_register_footprint(registers:dict[int, int], full:Sequence[int], half:Sequence[int], message:str) -> None:
  _require(all(0 <= register < 0xc0 for register in (*full, *half)), "unsupported shared or special IR3 register")
  control = registers[mesa.REG_A6XX_SP_CS_CNTL_0]
  half_footprint,full_footprint = control >> 1 & 0x3f, control >> 7 & 0x3f
  _require(control == half_footprint << 1 | full_footprint << 7, "unsupported A630 thread or register control flags")
  expected_half = max(half) // 4 + 1 if half else 0
  expected_full = max(full) // 4 + 1
  _require((half_footprint, full_footprint) == (expected_half, expected_full), message)

def _gpr_address(full:dict[int, int], operand:A630IR3Operand) -> int:
  return full[operand.value] | full[operand.value + 1] << 32

def _finish_writes(writes:Sequence[A630ExecutionWrite]) -> tuple[A630ExecutionWrite, ...]:
  ordered = sorted(writes, key=lambda write: write.address)
  _require(all(left.address + len(left.data) <= right.address for left,right in zip(ordered, ordered[1:])),
           "overlapping A630 global stores")
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

def _validate_vector_u32_dispatch(dispatch:A630Dispatch, active:Sequence[A630IR3Instruction], registers:dict[int, int],
                                  wgid:int, lid:int) -> None:
  # Pinned ir3_a6xx.c emits Cat6 SIZE from the collected component count, while ir3_rpt.c merges four consecutive
  # ADD.F leaves into REPEAT=3 with both sources advancing. This admits only that compiler-observed four-U32 slice.
  opcodes = tuple(instruction.opcode for instruction in active)
  fill = opcodes.count("stg.u32x4") == 1 and not any(opcode in {"ldg.u32x4", "add.f.rpt4"} for opcode in opcodes)
  add = (opcodes.count("ldg.u32x4"), opcodes.count("add.f.rpt4"), opcodes.count("stg.u32x4")) == (2, 1, 1)
  _require(fill or add, "unsupported A630 four-component kernel shape")
  # Pinned ir3_nir_analyze_ubo_ranges.c:268-337 promotes analyzable fixed UBO dwords to constant-file loads;
  # the captured N=4 fill/add shaders consume those argument dwords as one/three direct 64-bit pointer pairs and no system value.
  uses_constant_pointers = _validate_constant_pointer_moves(active)
  constant_pointer_x4 = uses_constant_pointers and dispatch.local_size == dispatch.groups == dispatch.global_size == (1, 1, 1)
  one_workgroup = dispatch.groups == (1, 1, 1) and dispatch.global_size == dispatch.local_size and \
    dispatch.local_size[1:] == (1, 1) and 2 <= dispatch.local_size[0] <= 32
  local32_workgroups = add and dispatch.groups == (2, 1, 1) and dispatch.local_size == (32, 1, 1) and \
    dispatch.global_size == (64, 1, 1)
  _require(constant_pointer_x4 or one_workgroup or local32_workgroups,
           "A630 four-component execution requires one workgroup or the exact local-32 two-workgroup add")
  _require((constant_pointer_x4 and (wgid, lid) == (0xfc, 0xfc)) or
           (one_workgroup and wgid == 0xfc and lid != 0xfc) or (local32_workgroups and (wgid, lid) == (0xc0, 0)),
           "unsupported A630 four-component system-value mapping")
  expected_counts = ({"mov.u32":6, "nop":1, "stg.u32x4":1, "end":1} if constant_pointer_x4 and fill else
                     {"mov.u32":6, "nop":3, "ldg.u32x4":2, "add.f.rpt4":1, "stg.u32x4":1, "end":1}
                     if constant_pointer_x4 else
                     {"shl.b":3, "mov.u32":4, "ashr.b":1, "add.u":3, "cmps.u.lt":1, "shrg":1,
                      "cov.u16s32":1, "nop":2, "stg.u32x4":1, "end":1} if fill else
                     {"shl.b":3 + int(local32_workgroups), "ashr.b":1, "add.u":9 + int(local32_workgroups),
                      "cmps.u.lt":3, "shrg":1, "cov.u16s32":3,
                      "nop":3, "ldg.u32x4":2, "add.f.rpt4":1, "stg.u32x4":1, "end":1})
  _require(len(opcodes) == sum(expected_counts.values()) and
           all(opcodes.count(opcode) == count for opcode,count in expected_counts.items()),
           "unsupported A630 four-component instruction inventory")

  for instruction in active:
    dst_kind = instruction.dst.kind if instruction.dst is not None else None
    src_kinds = tuple(operand.kind for operand in instruction.srcs)
    valid = instruction.opcode in {"nop", "end"} and dst_kind is None and not src_kinds
    if instruction.opcode == "ashr.b":
      valid = dst_kind == "gpr" and src_kinds == ("gpr", "iim") and instruction.srcs[1].value == 31
    elif instruction.opcode == "shl.b":
      valid = dst_kind == "gpr" and ((src_kinds == ("gpr", "iim") and instruction.srcs[1].value in (2, 4)) or
              (local32_workgroups and src_kinds == ("shared", "iim") and instruction.srcs[1].value == 7))
    elif instruction.opcode == "shrg":
      valid = dst_kind == "gpr" and src_kinds == ("iim", "gpr", "gpr") and instruction.srcs[0].value == 30
    elif instruction.opcode == "mov.u32":
      move_kinds = (("const",), ("uim",)) if constant_pointer_x4 and fill else \
        (("const",),) if constant_pointer_x4 else (("uim",),)
      valid = dst_kind == "gpr" and src_kinds in move_kinds
    elif instruction.opcode == "add.u":
      valid = dst_kind == "gpr" and (src_kinds == ("gpr", "gpr") or set(src_kinds) == {"const", "gpr"})
    elif instruction.opcode == "cmps.u.lt": valid = dst_kind == "half" and src_kinds == ("gpr", "const")
    elif instruction.opcode == "cov.u16s32": valid = dst_kind == "gpr" and src_kinds == ("half",)
    elif instruction.opcode == "ldg.u32x4": valid = dst_kind == "gpr" and src_kinds == ("gpr",)
    elif instruction.opcode == "add.f.rpt4": valid = dst_kind == "gpr" and src_kinds == ("gpr", "gpr")
    elif instruction.opcode == "stg.u32x4": valid = dst_kind is None and src_kinds == ("gpr", "gpr")
    _require(valid, f"unsupported A630 four-component operand contract at instruction {instruction.index}")

  store = next(instruction for instruction in active if instruction.opcode == "stg.u32x4")
  moves = tuple(instruction for instruction in active if instruction.opcode == "mov.u32")
  loads = tuple(instruction for instruction in active if instruction.opcode == "ldg.u32x4")
  pointer_moves = {instruction.srcs[0].value:instruction for instruction in moves if instruction.srcs[0].kind == "const"}
  if constant_pointer_x4:
    pointer_users = (store, *loads)
    _require(all(max(pointer_moves[2*index].index, pointer_moves[2*index+1].index) < instruction.index
                 for index,instruction in enumerate(pointer_users)),
             "A630 constant-pointer moves do not dominate their buffer access")
  data_blocks:tuple[set[int], ...]
  if fill:
    fill_moves = tuple(move for move in moves if move.srcs[0].kind == "uim")
    _require(len(fill_moves) == 4 and all(move.dst is not None for move in fill_moves), "unsupported A630 four-component fill moves")
    assert all(move.dst is not None for move in fill_moves)
    _require(tuple(sorted(move.dst.value for move in fill_moves if move.dst is not None)) ==
             tuple(range(store.srcs[1].value, store.srcs[1].value + 4)),
             "four-component store does not consume the consecutive fill literals")
    _require(all(move.index < store.index for move in fill_moves), "four-component fill literals do not dominate the store")
    data_blocks = (set(range(store.srcs[1].value, store.srcs[1].value + 4)),)
  else:
    repeated = next(instruction for instruction in active if instruction.opcode == "add.f.rpt4")
    assert repeated.dst is not None and all(load.dst is not None for load in loads)
    load_destinations = tuple(load.dst for load in loads if load.dst is not None)
    _require(len(set(load_destinations)) == len(set(repeated.srcs)) == 2 and
             frozenset(repeated.srcs) == frozenset(load_destinations) and store.srcs[1] == repeated.dst,
             "four-component add does not connect two distinct loads to the store")
    _require(max(load.index for load in loads) < repeated.index < store.index,
             "four-component add producers do not dominate their consumers")
    data_blocks = tuple(set(range(operand.value, operand.value + 4)) for operand in (*load_destinations, repeated.dst))

  if constant_pointer_x4:
    pointer_registers = {register for instruction in (store, *loads)
                         for register in (instruction.srcs[0].value, instruction.srcs[0].value + 1)}
    _require(all(pointer_registers.isdisjoint(block) for block in data_blocks),
             "A630 four-component data registers overlap pointer registers")
    _require(all(left.isdisjoint(right) for index,left in enumerate(data_blocks) for right in data_blocks[index+1:]),
             "A630 four-component data register blocks overlap")

  full_registers = [] if lid == 0xfc else [lid, lid + 1, lid + 2]
  half_registers:list[int] = []
  for instruction in active:
    operands = ((instruction.dst,) if instruction.dst is not None else ()) + instruction.srcs
    for operand in operands:
      if operand.kind == "gpr": full_registers.append(operand.value)
      elif operand.kind == "half": half_registers.append(operand.value)
    if instruction.opcode in {"ldg.u32x4", "stg.u32x4"}: full_registers.append(instruction.srcs[0].value + 1)
    if instruction.opcode == "ldg.u32x4" and instruction.dst is not None:
      full_registers.extend(range(instruction.dst.value + 1, instruction.dst.value + 4))
    if instruction.opcode in {"stg.u32x4", "add.f.rpt4"}:
      for operand in instruction.srcs[1:] if instruction.opcode == "stg.u32x4" else instruction.srcs:
        full_registers.extend(range(operand.value + 1, operand.value + 4))
    if instruction.opcode == "add.f.rpt4" and instruction.dst is not None:
      full_registers.extend(range(instruction.dst.value + 1, instruction.dst.value + 4))

  if constant_pointer_x4:
    constant_uses = sorted(operand.value for instruction in active for operand in instruction.srcs if operand.kind == "const")
    _require(constant_uses == list(range(2 * (len(loads) + 1))),
             "A630 pointer constants do not match the four-component buffer argument ABI")
    _require(not any(operand.kind == "shared" for instruction in active for operand in instruction.srcs),
             "constant-pointer A630 four-component execution uses a system value")
    _validate_register_footprint(registers, full_registers, half_registers,
                                 "A630 register footprints do not match decoded four-component operands")
    return

  constant_uses = sorted(operand.value for instruction in active for operand in instruction.srcs if operand.kind == "const")
  _require(constant_uses == ([0, 0, 1] if fill else [0, 0, 1, 2, 2, 3, 4, 4, 5]),
           "A630 pointer constants do not match the four-component buffer argument ABI")
  local_id = A630IR3Operand("gpr", lid)
  element_shifts = tuple(instruction for instruction in active if instruction.opcode == "shl.b" and
                         instruction.srcs == (local_id, A630IR3Operand("iim", 2)))
  _require(len(element_shifts) == 1 and element_shifts[0].dst is not None,
           "A630 four-component local-id scaling is not the compiler address chain")
  global_element = element_shifts[0].dst
  global_add:A630IR3Instruction|None = None
  shared_uses = tuple(operand.value for instruction in active for operand in instruction.srcs if operand.kind == "shared")
  if local32_workgroups:
    # nir_lower_system_values.c forms WGID*local_size+LID; four components make this exact group term WGID << 7.
    group_shifts = tuple(instruction for instruction in active if instruction.opcode == "shl.b" and
                         instruction.srcs == (A630IR3Operand("shared", wgid), A630IR3Operand("iim", 7)))
    _require(len(group_shifts) == 1 and group_shifts[0].dst is not None and shared_uses == (wgid,),
             "A630 four-component workgroup-id scaling is not the compiler address chain")
    _require(group_shifts[0].dst != element_shifts[0].dst,
             "A630 four-component workgroup and local-id terms alias")
    global_adds = tuple(instruction for instruction in active if instruction.opcode == "add.u" and instruction.dst is not None and
                        len(instruction.srcs) == 2 and instruction.srcs[0] != instruction.srcs[1] and
                        frozenset(instruction.srcs) == frozenset((group_shifts[0].dst, element_shifts[0].dst)))
    _require(len(global_adds) == 1, "A630 four-component global-id add is not the compiler address chain")
    global_add = global_adds[0]
    _require(max(group_shifts[0].index, element_shifts[0].index) < global_add.index,
             "A630 four-component global-id producers do not dominate the add")
    global_element = global_add.dst
  else: _require(not shared_uses, "one-workgroup A630 four-component execution uses a shared system value")
  assert global_element is not None
  byte_source,byte_amount = (global_element, 2) if local32_workgroups else (local_id, 4)
  byte_shifts = tuple(instruction for instruction in active if instruction.opcode == "shl.b" and
                      instruction.srcs == (byte_source, A630IR3Operand("iim", byte_amount)))
  _require(len(byte_shifts) == 1 and byte_shifts[0].dst is not None,
           "A630 four-component byte scaling is not the compiler address chain")
  byte_shift = byte_shifts[0]
  sign_extends = tuple(instruction for instruction in active if instruction.opcode == "ashr.b" and
                       instruction.srcs == (global_element, A630IR3Operand("iim", 31)))
  _require(len(sign_extends) == 1 and sign_extends[0].dst is not None,
           "A630 four-component sign extension is not the compiler address chain")
  if global_add is not None:
    _require(global_add.index < min(byte_shift.index, sign_extends[0].index),
             "A630 four-component global-id add does not dominate its offset consumers")
  high_shifts = tuple(instruction for instruction in active if instruction.opcode == "shl.b" and
                      instruction.srcs == (sign_extends[0].dst, A630IR3Operand("iim", 2)))
  _require(len(high_shifts) == 1 and high_shifts[0].dst is not None,
           "A630 four-component high offset is not the compiler address chain")
  high_offsets = tuple(instruction for instruction in active if instruction.opcode == "shrg" and
                       instruction.srcs == (A630IR3Operand("iim", 30), global_element, high_shifts[0].dst))
  _require(len(high_offsets) == 1 and high_offsets[0].dst is not None,
           "A630 four-component packed offset is not the compiler address chain")
  address_bases = (store.srcs[0], *(load.srcs[0] for load in loads))
  for pointer,address_base in enumerate(address_bases):
    low_constant,high_constant = (A630IR3Operand("const", 2*pointer), A630IR3Operand("const", 2*pointer+1))
    low_adds = tuple(instruction for instruction in active if instruction.opcode == "add.u" and
                     instruction.dst == address_base and frozenset(instruction.srcs) == frozenset((low_constant, byte_shift.dst)))
    carries = tuple(instruction for instruction in active if instruction.opcode == "cmps.u.lt" and
                    instruction.srcs == (address_base, low_constant))
    _require(len(low_adds) == len(carries) == 1 and carries[0].dst is not None,
             "A630 four-component low address is not the compiler pointer chain")
    carry_conversions = tuple(instruction for instruction in active if instruction.opcode == "cov.u16s32" and
                              instruction.srcs == (carries[0].dst,))
    high_adds = tuple(instruction for instruction in active if instruction.opcode == "add.u" and
                      frozenset(instruction.srcs) == frozenset((high_constant, high_offsets[0].dst)))
    _require(len(carry_conversions) == len(high_adds) == 1 and carry_conversions[0].dst is not None and high_adds[0].dst is not None,
             "A630 four-component high address is not the compiler pointer chain")
    final_high = A630IR3Operand("gpr", address_base.value + 1)
    final_adds = tuple(instruction for instruction in active if instruction.opcode == "add.u" and
                       instruction.dst == final_high and
                       frozenset(instruction.srcs) == frozenset((carry_conversions[0].dst, high_adds[0].dst)))
    _require(len(final_adds) == 1, "A630 four-component carry is not the compiler pointer chain")
  _validate_carry_conversions(active)

  _validate_register_footprint(registers, full_registers, half_registers,
                               "A630 register footprints do not match decoded four-component operands")

def _validate_two_segment_u32_copy(dispatch:A630Dispatch, active:Sequence[A630IR3Instruction], registers:dict[int, int],
                                   wgid:int, lid:int) \
    -> tuple[tuple[A630IR3Instruction, A630IR3Instruction], tuple[A630IR3Instruction, A630IR3Instruction]]:
  # Exact equal-eight-word two-segment copy: two scalar loads feed disjoint output segments. Pinned
  # nir_lower_int64.c:296-309 forms each 64-bit pointer as low ADD.U, unsigned carry, and high ADD.U;
  # ir3-cat6.xml:90-123,251-295 and ir3_a6xx.c:411-494 define one-component U32 LDG/STG operands.
  _require(dispatch.local_size == dispatch.global_size == (8, 1, 1) and dispatch.groups == (1, 1, 1),
           "A630 two-segment u32 copy requires one exact eight-lane workgroup")
  _require((wgid, lid) == (0xfc, 0), "unsupported A630 two-segment u32 copy system-value mapping")
  _require(not any(operand.kind == "shared" for instruction in active for operand in instruction.srcs),
           "A630 two-segment u32 copy uses a shared system value")
  opcodes = tuple(instruction.opcode for instruction in active)
  expected_counts = {"ashr.b":2, "shl.b":3, "add.u":14, "shrg":2, "cmps.u.lt":4, "cov.u16s32":4,
                     "ldg.u32":2, "nop":2, "stg.u32":2, "end":1}
  _require(len(opcodes) == sum(expected_counts.values()) and
           all(opcodes.count(opcode) == count for opcode,count in expected_counts.items()),
           "unsupported A630 two-segment u32 copy instruction inventory")

  def one(matches:Sequence[A630IR3Instruction], message:str) -> A630IR3Instruction:
    _require(len(matches) == 1, message)
    return matches[0]

  def live_until(producer:A630IR3Instruction, consumers:Sequence[A630IR3Instruction], message:str) -> None:
    _require(producer.dst is not None and all(producer.index < consumer.index and
             not any(instruction.dst == producer.dst and producer.index < instruction.index < consumer.index
                     for instruction in active) for consumer in consumers), message)

  local_id = A630IR3Operand("gpr", lid)
  lane_bytes = one(tuple(instruction for instruction in active if instruction.opcode == "shl.b" and
                         instruction.srcs == (local_id, A630IR3Operand("iim", 2))),
                   "A630 two-segment u32 copy lacks the lane byte offset")
  second_lane = one(tuple(instruction for instruction in active if instruction.opcode == "add.u" and
                          instruction.srcs == (local_id, A630IR3Operand("iim", 8))),
                    "A630 two-segment u32 copy lacks the second-segment lane offset")
  lane_sign = one(tuple(instruction for instruction in active if instruction.opcode == "ashr.b" and
                        instruction.srcs == (local_id, A630IR3Operand("iim", 31)) and instruction.index < second_lane.index),
                  "A630 two-segment u32 copy lacks the lane sign extension")
  _require(lane_sign.dst is not None and lane_bytes.dst is not None and second_lane.dst is not None,
           "A630 two-segment u32 copy lane offset lacks a destination")
  lane_sign_bytes = one(tuple(instruction for instruction in active if instruction.opcode == "shl.b" and
                               instruction.srcs == (lane_sign.dst, A630IR3Operand("iim", 2))),
                         "A630 two-segment u32 copy lacks the high lane byte offset")
  _require(lane_sign_bytes.dst is not None, "A630 two-segment u32 copy high lane offset lacks a destination")
  lane_high = one(tuple(instruction for instruction in active if instruction.opcode == "shrg" and
                        instruction.srcs == (A630IR3Operand("iim", 30), local_id, lane_sign_bytes.dst)),
                  "A630 two-segment u32 copy lacks the packed high lane offset")
  _require(lane_high.dst is not None, "A630 two-segment u32 copy packed lane offset lacks a destination")

  second_bytes = one(tuple(instruction for instruction in active if instruction.opcode == "add.u" and
                           instruction.srcs == (lane_bytes.dst, A630IR3Operand("iim", 32))),
                     "A630 two-segment u32 copy lacks the second-segment byte offset")
  _require(second_lane.dst is not None and second_bytes.dst is not None,
           "A630 two-segment u32 copy second-segment offset lacks a destination")
  second_sign = one(tuple(instruction for instruction in active if instruction.opcode == "ashr.b" and
                          instruction.srcs == (second_lane.dst, A630IR3Operand("iim", 31)) and
                          instruction.index > second_lane.index),
                    "A630 two-segment u32 copy lacks the second-segment sign extension")
  _require(second_sign.dst is not None, "A630 two-segment u32 copy second-segment sign lacks a destination")
  second_sign_bytes = one(tuple(instruction for instruction in active if instruction.opcode == "shl.b" and
                                 instruction.srcs == (second_sign.dst, A630IR3Operand("iim", 2))),
                           "A630 two-segment u32 copy lacks the second-segment high byte offset")
  _require(second_sign_bytes.dst is not None,
           "A630 two-segment u32 copy second-segment high offset lacks a destination")
  second_high = one(tuple(instruction for instruction in active if instruction.opcode == "shrg" and
                          instruction.srcs == (A630IR3Operand("iim", 30), second_lane.dst, second_sign_bytes.dst)),
                    "A630 two-segment u32 copy lacks the packed second-segment high offset")
  _require(second_high.dst is not None, "A630 two-segment u32 copy packed second-segment offset lacks a destination")
  _require(lane_sign.index < lane_sign_bytes.index < lane_high.index and lane_bytes.index < second_bytes.index and
           second_lane.index < second_sign.index < second_sign_bytes.index < second_high.index,
           "A630 two-segment u32 copy offset producers do not dominate their consumers")
  local_consumers = (lane_bytes, second_lane, lane_sign, lane_high)
  _require(not any(instruction.dst == local_id and instruction.index < consumer.index
                   for instruction in active for consumer in local_consumers),
           "A630 two-segment u32 copy local-id register is overwritten before an offset consumer")
  live_until(lane_sign, (lane_sign_bytes,),
             "A630 two-segment u32 copy lane-sign register is overwritten before its byte shift")
  live_until(lane_sign_bytes, (lane_high,),
             "A630 two-segment u32 copy high lane byte register is overwritten before its packed shift")
  live_until(lane_bytes, (second_bytes,),
             "A630 two-segment u32 copy lane-byte register is overwritten before the second-segment offset")
  live_until(second_lane, (second_sign, second_high),
             "A630 two-segment u32 copy second-segment lane register is overwritten before an offset consumer")
  live_until(second_sign, (second_sign_bytes,),
             "A630 two-segment u32 copy second-segment sign register is overwritten before its byte shift")
  live_until(second_sign_bytes, (second_high,),
             "A630 two-segment u32 copy second-segment high byte register is overwritten before its packed shift")
  assert lane_bytes.dst is not None and lane_high.dst is not None and second_bytes.dst is not None and second_high.dst is not None
  lane_bytes_dst,lane_high_dst = lane_bytes.dst,lane_high.dst
  second_bytes_dst,second_high_dst = second_bytes.dst,second_high.dst

  loads = tuple(instruction for instruction in active if instruction.opcode == "ldg.u32")
  stores = tuple(instruction for instruction in active if instruction.opcode == "stg.u32")
  _require(all(instruction.dst is not None and instruction.dst.kind == "gpr" and
               instruction.srcs[0].kind == "gpr" for instruction in loads) and
           all(instruction.srcs[0].kind == instruction.srcs[1].kind == "gpr" for instruction in stores),
           "unsupported A630 two-segment u32 copy memory operands")

  address_adds:set[int] = set()
  address_pairs:list[set[int]] = []
  def address_chain(low_constant:int, high_constant:int, low_offset:A630IR3Operand, high_offset:A630IR3Operand,
                    low_offset_index:int, high_offset_index:int, consumer:A630IR3Instruction,
                    low:A630IR3Instruction, purpose:str) -> None:
    address = consumer.srcs[0]
    _require(low.opcode == "add.u" and low.dst == address and
             frozenset(low.srcs) == frozenset((A630IR3Operand("const", low_constant), low_offset)),
             f"A630 two-segment u32 copy {purpose} low address is not the compiler pointer chain")
    carry = one(tuple(instruction for instruction in active if instruction.opcode == "cmps.u.lt" and
                      instruction.srcs == (address, A630IR3Operand("const", low_constant))),
                f"A630 two-segment u32 copy {purpose} carry is not the compiler pointer chain")
    _require(carry.dst is not None, f"A630 two-segment u32 copy {purpose} carry lacks a destination")
    carry_conversion = one(tuple(instruction for instruction in active if instruction.opcode == "cov.u16s32" and
                                 instruction.srcs == (carry.dst,)),
                           f"A630 two-segment u32 copy {purpose} carry lacks a conversion")
    high = one(tuple(instruction for instruction in active if instruction.opcode == "add.u" and instruction.dst is not None and
                     frozenset(instruction.srcs) == frozenset((A630IR3Operand("const", high_constant), high_offset))),
               f"A630 two-segment u32 copy {purpose} high address is not the compiler pointer chain")
    _require(carry_conversion.dst is not None and high.dst is not None,
             f"A630 two-segment u32 copy {purpose} high address lacks a destination")
    high_address = A630IR3Operand("gpr", address.value + 1)
    final = one(tuple(instruction for instruction in active if instruction.opcode == "add.u" and instruction.dst == high_address and
                      frozenset(instruction.srcs) == frozenset((carry_conversion.dst, high.dst))),
                f"A630 two-segment u32 copy {purpose} final high address is not the compiler pointer chain")
    _require(low_offset_index < low.index < carry.index < carry_conversion.index < consumer.index and
             high_offset_index < high.index < final.index < consumer.index and
             max(carry_conversion.index, high.index) < final.index,
             f"A630 two-segment u32 copy {purpose} pointer producers do not dominate the memory operation")
    _require(carry_conversion.dst != high.dst,
             f"A630 two-segment u32 copy {purpose} final high-address sources alias")
    low_offset_producer = one(tuple(instruction for instruction in active if instruction.index == low_offset_index and
                                    instruction.dst == low_offset),
                              f"A630 two-segment u32 copy {purpose} low offset lacks its producer")
    high_offset_producer = one(tuple(instruction for instruction in active if instruction.index == high_offset_index and
                                     instruction.dst == high_offset),
                               f"A630 two-segment u32 copy {purpose} high offset lacks its producer")
    live_until(low_offset_producer, (low,),
               f"A630 two-segment u32 copy {purpose} low offset is overwritten before the address add")
    live_until(high_offset_producer, (high,),
               f"A630 two-segment u32 copy {purpose} high offset is overwritten before the address add")
    live_until(carry, (carry_conversion,),
               f"A630 two-segment u32 copy {purpose} carry is overwritten before conversion")
    live_until(carry_conversion, (final,),
               f"A630 two-segment u32 copy {purpose} converted carry is overwritten before the final add")
    live_until(high, (final,),
               f"A630 two-segment u32 copy {purpose} high temporary is overwritten before the final add")
    for register,producer in ((address, low), (high_address, final)):
      _require(not any(instruction.dst == register and producer.index < instruction.index < consumer.index
                       for instruction in active),
               f"A630 two-segment u32 copy {purpose} pointer is overwritten before the memory operation")
    address_adds.update((low.index, high.index, final.index))
    address_pairs.append({address.value, address.value + 1})

  input_loads:list[A630IR3Instruction] = []
  for pointer,low_offset,high_offset in ((1, lane_bytes_dst, lane_high_dst), (2, lane_bytes_dst, lane_high_dst)):
    low_constant = 2 * pointer
    low_add = one(tuple(instruction for instruction in active if instruction.opcode == "add.u" and instruction.dst is not None and
                        frozenset(instruction.srcs) == frozenset((A630IR3Operand("const", low_constant), low_offset))),
                  f"A630 two-segment u32 copy input {pointer-1} lacks a low address")
    assert low_add.dst is not None
    load = one(tuple(instruction for instruction in loads if instruction.srcs == (low_add.dst,)),
               f"A630 two-segment u32 copy input {pointer-1} lacks its global load")
    address_chain(low_constant, low_constant + 1, low_offset, high_offset, lane_bytes.index, lane_high.index,
                  load, low_add, f"input {pointer-1}")
    input_loads.append(load)

  output_stores:list[A630IR3Instruction] = []
  for segment,(low_offset,high_offset) in enumerate(((lane_bytes_dst, lane_high_dst), (second_bytes_dst, second_high_dst))):
    low_add = one(tuple(instruction for instruction in active if instruction.opcode == "add.u" and instruction.dst is not None and
                        frozenset(instruction.srcs) == frozenset((A630IR3Operand("const", 0), low_offset)) and
                        (instruction.index < second_bytes.index if segment == 0 else instruction.index > second_bytes.index)),
                  f"A630 two-segment u32 copy output segment {segment} lacks a low address")
    assert low_add.dst is not None
    store = one(tuple(instruction for instruction in stores if instruction.srcs[0] == low_add.dst),
                f"A630 two-segment u32 copy output segment {segment} lacks its global store")
    address_chain(0, 1, low_offset, high_offset,
                  lane_bytes.index if segment == 0 else second_bytes.index,
                  lane_high.index if segment == 0 else second_high.index,
                  store, low_add, f"output segment {segment}")
    output_stores.append(store)

  load_destinations = tuple(instruction.dst for instruction in input_loads)
  _require(None not in load_destinations and len(set(load_destinations)) == 2 and
           set(store.srcs[1] for store in output_stores) == set(load_destinations),
           "A630 two-segment u32 copy stores do not form a bijection over the global loads")
  for store in output_stores:
    load = next(instruction for instruction in input_loads if instruction.dst == store.srcs[1])
    _require(load.index < store.index and not any(instruction.dst == load.dst and load.index < instruction.index < store.index
                                                  for instruction in active),
             "A630 two-segment u32 copy load does not dominate its store")

  _require(all(left.isdisjoint(right) for index,left in enumerate(address_pairs) for right in address_pairs[index+1:]),
           "A630 two-segment u32 copy address pairs overlap")
  address_registers = set().union(*address_pairs)
  _require(all(instruction.dst is not None and instruction.dst.value not in address_registers for instruction in input_loads),
           "A630 two-segment u32 copy data registers overlap pointer registers")
  _require(address_registers.isdisjoint((lid, lid + 1, lid + 2)) and
           all(instruction.dst is not None and instruction.dst.value not in (lid, lid + 1, lid + 2) for instruction in input_loads),
           "A630 two-segment u32 copy memory registers overlap the local-id vector")
  all_adds = {instruction.index for instruction in active if instruction.opcode == "add.u"}
  _require(all_adds == address_adds | {second_lane.index, second_bytes.index},
           "unsupported A630 two-segment u32 copy address arithmetic")
  _validate_carry_conversions(active)

  constant_uses = sorted(operand.value for instruction in active for operand in instruction.srcs if operand.kind == "const")
  _require(constant_uses == [0, 0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5],
           "A630 two-segment u32 copy constants do not match the buffer argument ABI")
  full_registers = [lid, lid + 1, lid + 2]
  half_registers:list[int] = []
  for instruction in active:
    operands = ((instruction.dst,) if instruction.dst is not None else ()) + instruction.srcs
    for operand in operands:
      if operand.kind == "gpr": full_registers.append(operand.value)
      elif operand.kind == "half": half_registers.append(operand.value)
    if instruction.opcode in {"ldg.u32", "stg.u32"}: full_registers.append(instruction.srcs[0].value + 1)
  _validate_register_footprint(registers, full_registers, half_registers,
                               "A630 register footprints do not match decoded two-segment u32 copy operands")
  return (input_loads[0], input_loads[1]), (output_stores[0], output_stores[1])

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
  if any(opcode in {"ldg.u32x4", "stg.u32x4", "add.f.rpt4"} for opcode in opcodes):
    _validate_vector_u32_dispatch(dispatch, active, registers, wgid, lid)
    return dispatch
  if opcodes.count("stg.u32") == 2:
    _validate_two_segment_u32_copy(dispatch, active, registers, wgid, lid)
    return dispatch
  input_count = opcodes.count("ldg.u32")
  float_add_count = opcodes.count("add.f")
  _require(input_count in (0, 1, 2), "A630 execution supports at most two global loads")
  _require((input_count, float_add_count) in ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1)), "unsupported A630 scalar kernel shape")
  integer_instruction:A630IR3Instruction|None = None
  integer_kind:str|None = None
  integer_value_type = "u32"
  comparison_instruction = _u32_comparison_instruction(active)
  conversion_instruction = _integer_to_f32_instruction(active)
  multiply_sequence = _u32_multiply_sequence(active)
  comparison_count = sum(opcodes.count(opcode) for opcode in ("cmps.s.lt", "cmps.u.lt", "cmps.s.eq"))
  conversion_count = sum(opcodes.count(opcode) for opcode in ("cov.s32f32", "cov.u32f32"))
  _require(conversion_count == int(conversion_instruction is not None), "integer-to-f32 conversion does not consume the global load")
  if conversion_count:
    _require((input_count, float_add_count, conversion_count) == (1, 0, 1),
             "integer-to-f32 conversion requires one global load and no other data operation")
  simple_opcode = next((opcode for opcode in _SIMPLE_CAT2_INTEGER if opcodes.count(opcode)), None)
  if opcodes.count("stg.u8"):
    _require((input_count, float_add_count, comparison_count, opcodes.count("stg.u8")) == (2, 0, 1, 1) and
             comparison_instruction is not None, "u32 comparison does not consume both global loads")
  elif opcodes.count("mull.u") or opcodes.count("madsh.m16"):
    _require((input_count, float_add_count, opcodes.count("mull.u"), opcodes.count("madsh.m16")) == (2, 0, 1, 2) and
             multiply_sequence is not None, "u32 multiplication sequence does not consume both global loads")
    assert multiply_sequence is not None
    integer_instruction,integer_kind = multiply_sequence[-1],"multiply"
  elif simple_opcode is not None:
    integer_kind,_ = _SIMPLE_CAT2_INTEGER[simple_opcode]
    integer_instruction = _u32_binary_instruction(active, simple_opcode)
    _require((input_count, float_add_count, opcodes.count(simple_opcode)) == (2, 0, 1) and integer_instruction is not None,
             f"u32 {integer_kind} does not consume both global loads")
  elif opcodes.count("max.s") or opcodes.count("max.u"):
    integer_instruction = _u32_binary_instruction(active, "max.s") or _u32_binary_instruction(active, "max.u")
    maximum_count = opcodes.count("max.s") + opcodes.count("max.u")
    maximum_kind = "s32 maximum" if opcodes.count("max.s") else "u32 maximum"
    _require((input_count, float_add_count, maximum_count) == (2, 0, 1) and integer_instruction is not None,
             f"{maximum_kind} does not consume both global loads")
    assert integer_instruction is not None
    integer_kind = "maximum"
    integer_value_type = "s32" if integer_instruction.opcode == "max.s" else "u32"
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
    _require((integer_instruction is not None or comparison_instruction is not None or conversion_instruction is not None) and not shared_uses,
             "constant-pointer A630 execution supports only scalar 32-bit integer arithmetic, comparison, or conversion")
    _require(dispatch.local_size == dispatch.groups == dispatch.global_size == (1, 1, 1),
             "constant-pointer A630 execution requires one scalar invocation")
    if conversion_instruction is not None:
      assert conversion_instruction.opcode is not None
      expected_counts = {"mov.u32":4, "nop":2, "ldg.u32":1, conversion_instruction.opcode:1, "stg.u32":1, "end":1}
    elif comparison_instruction is not None:
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
             "32-bit subtraction, multiplication, logical shift, bitwise, and maximum operations currently require the scalar constant-pointer ABI")
    uses_workgroup_id = bool(shared_uses)
    expected_counts = {"ashr.b":1, "shl.b":2, "shrg":1, "add.u":3 * (input_count + 1) + int(has_integer_add),
                       "cmps.u.lt":input_count + 1, "cov.u16s32":input_count + 1, "nop":3 + int(uses_workgroup_id),
                       "ldg.u32":input_count, "stg.u32":1, "end":1}
    if float_add_count: expected_counts["add.f"] = 1
    if conversion_instruction is not None:
      assert conversion_instruction.opcode is not None
      expected_counts[conversion_instruction.opcode] = 1
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
    elif instruction.opcode in _SIMPLE_CAT2_INTEGER: valid = dst_kind == "gpr" and src_kinds == ("gpr", "gpr")
    elif instruction.opcode in {"max.s", "max.u"}: valid = dst_kind == "gpr" and src_kinds == ("gpr", "gpr")
    elif instruction.opcode == "mull.u": valid = dst_kind == "gpr" and src_kinds == ("gpr", "gpr")
    elif instruction.opcode == "madsh.m16": valid = dst_kind == "gpr" and src_kinds == ("gpr", "gpr", "gpr")
    elif instruction.opcode == "cmps.u.lt":
      valid = dst_kind == "half" and (src_kinds == ("gpr", "const") or
              comparison_instruction is not None and instruction.index == comparison_instruction.index and src_kinds == ("gpr", "gpr"))
    elif instruction.opcode in {"cmps.s.lt", "cmps.s.eq"}:
      valid = dst_kind == "half" and comparison_instruction is not None and instruction.index == comparison_instruction.index and \
              src_kinds == ("gpr", "gpr")
    elif instruction.opcode == "cov.u16s32": valid = dst_kind == "gpr" and src_kinds == ("half",)
    elif instruction.opcode in {"cov.s32f32", "cov.u32f32"}: valid = dst_kind == "gpr" and src_kinds == ("gpr",)
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
    _require(store.srcs[1] == integer_instruction.dst, f"global store does not consume the {integer_value_type} {integer_kind}")
  if comparison_instruction is not None:
    assert comparison_instruction.dst is not None
    store = next(instruction for instruction in active if instruction.opcode == "stg.u8")
    _require(store.srcs[1] == comparison_instruction.dst, "global store does not consume the u32 comparison")
  if conversion_instruction is not None:
    assert conversion_instruction.dst is not None
    store = next(instruction for instruction in active if instruction.opcode == "stg.u32")
    _require(store.srcs[1] == conversion_instruction.dst, "global store does not consume the integer-to-f32 conversion")

  constant_uses = sorted(operand.value for instruction in active for operand in instruction.srcs if operand.kind == "const")
  expected_constants = list(range(2 * (input_count + 1))) if uses_constant_pointers else \
    sorted(value for pointer in range(input_count + 1) for value in (2*pointer, 2*pointer, 2*pointer+1))
  _require(constant_uses == expected_constants, "A630 pointer constants do not match the scalar buffer argument ABI")
  moves = tuple(instruction for instruction in active if instruction.opcode == "mov.u32")
  if input_count == 0:
    _require(len(moves) == 1 and moves[0].srcs == (A630IR3Operand("uim", 0x3f800000),),
             "unsupported A630 fill literal")
  elif uses_constant_pointers:
    _require(len(moves) == 2 * (input_count + 1) and all(move.srcs[0].kind == "const" for move in moves),
             "unsupported A630 constant-pointer moves")
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
  _validate_carry_conversions(active, comparison_instruction.index if comparison_instruction is not None else None)
  _validate_register_footprint(registers, full_registers, half_registers, "A630 register footprints do not match decoded operands")
  return dispatch

def _execute_vector_u32(dispatch:A630Dispatch, resolver:Resolver,
                        active:Sequence[A630IR3Instruction]) -> tuple[A630ExecutionWrite, ...]:
  constants = struct.unpack("<1024I", dispatch.constants_image)
  lane_count = dispatch.local_size[0]
  invocation_count = dispatch.global_size[0]
  wgid,lid = _system_registers(dispatch)
  loads = tuple(instruction for instruction in active if instruction.opcode == "ldg.u32x4")
  load_ordinals = {instruction.index:index for index,instruction in enumerate(loads)}
  output_base = constants[0] | constants[1] << 32
  input_bases = tuple(constants[2*index+2] | constants[2*index+3] << 32 for index in range(len(loads)))
  byte_count = invocation_count * 16
  _require(output_base != 0 and output_base % 4 == 0 and output_base + byte_count <= 1 << 64 and
           all(base != 0 and base % 4 == 0 and base + byte_count <= 1 << 64 for base in input_bases),
           "invalid A630 four-component argument range")
  fill = not loads
  writes:list[A630ExecutionWrite] = []
  for group in range(dispatch.groups[0]):
    full = [({} if lid == 0xfc else {lid:lane, lid+1:0, lid+2:0}) for lane in range(lane_count)]
    half:list[dict[int, int]] = [{} for _ in range(lane_count)]
    origins:list[dict[int, tuple[str, int]]] = [{} for _ in range(lane_count)]
    shared = {} if wgid == 0xfc else {wgid:group, wgid+1:0, wgid+2:0}

    for instruction in active:
      if instruction.opcode == "end": break
      if instruction.opcode == "nop": continue
      _require((instruction.opcode is not None and instruction.dst is not None) or instruction.opcode == "stg.u32x4",
               f"unsupported A630 semantic at instruction {instruction.index}")
      for lane in range(lane_count):
        try:
          opcode = instruction.opcode
          global_lane = group * lane_count + lane
          if opcode == "ldg.u32x4":
            assert instruction.dst is not None
            address = _gpr_address(full[lane], instruction.srcs[0])
            ordinal = load_ordinals[instruction.index]
            _require(address == input_bases[ordinal] + global_lane * 16,
                     f"global load {ordinal} does not address its four-component input")
            _require(address % 4 == 0 and address + 16 <= 1 << 64, "invalid A630 four-component global-load address")
            view = resolver(address, 16)
            _require(len(view) == 16, "short A630 four-component global-load range")
            for component,value in enumerate(struct.unpack("<4I", bytes(view))):
              full[lane][instruction.dst.value + component] = value
              origins[lane][instruction.dst.value + component] = ("load", ordinal * 4 + component)
            continue
          if opcode == "add.f.rpt4":
            assert instruction.dst is not None
            for component in range(4):
              left_register,right_register = (operand.value + component for operand in instruction.srcs)
              left,right = full[lane][left_register],full[lane][right_register]
              _require(frozenset((origins[lane].get(left_register), origins[lane].get(right_register))) ==
                       frozenset((("load", component), ("load", 4 + component))),
                       "four-component f32 add does not consume both corresponding global-load components")
              value = _f32_add_bits(left, right)
              full[lane][instruction.dst.value + component] = value
              origins[lane][instruction.dst.value + component] = ("f32-add", component)
            continue
          if opcode == "stg.u32x4":
            _,data_base = (operand.value for operand in instruction.srcs)
            address = _gpr_address(full[lane], instruction.srcs[0])
            _require(address == output_base + global_lane * 16, "global store does not address the four-component output")
            expected_origins = tuple(("fill", full[lane][data_base + component]) if fill else ("f32-add", component)
                                     for component in range(4))
            _require(tuple(origins[lane].get(data_base + component) for component in range(4)) == expected_origins,
                     "four-component store does not consume the fill literals" if fill else
                     "four-component store does not consume the repeated f32 add")
            _require(address % 4 == 0 and address + 16 <= 1 << 64, "invalid A630 four-component global-store address")
            view = resolver(address, 16)
            _require(len(view) == 16, "short A630 four-component global-store range")
            writes.append(A630ExecutionWrite(address, struct.pack("<4I", *(full[lane][data_base+i] for i in range(4)))))
            continue

          src = tuple(_read_ir3_operand(operand, full[lane], half[lane], shared, constants) for operand in instruction.srcs)
          origin:tuple[str, int]|None = None
          if opcode == "mov.u32":
            value = src[0]
            if instruction.srcs[0].kind == "uim": origin = ("fill", value)
          elif opcode == "add.u": value = src[0] + src[1]
          elif opcode == "shl.b": value = src[0] << (src[1] & 31)
          elif opcode == "ashr.b":
            signed = src[0] - (1 << 32) if src[0] & 0x80000000 else src[0]
            value = signed >> (src[1] & 31)
          elif opcode == "shrg": value = (src[1] >> (src[0] & 31)) | src[2]
          elif opcode == "cmps.u.lt": value = int(src[0] < src[1])
          elif opcode == "cov.u16s32": value = src[0] & 0xffff
          else: raise ValueError(f"unsupported A630 four-component opcode {opcode}")
          assert instruction.dst is not None
          _write_ir3_operand(instruction.dst, value, full[lane], half[lane])
          if instruction.dst.kind == "gpr":
            if origin is None: origins[lane].pop(instruction.dst.value, None)
            else: origins[lane][instruction.dst.value] = origin
        except (KeyError, ValueError, RuntimeError) as error:
          raise ValueError(f"A630 instruction {instruction.index} lane {lane} group {group}: {error}") from error

  return _finish_writes(writes)

def execute_a630(submission:A630Submission, resolver:Resolver) -> tuple[A630ExecutionWrite, ...]:
  """Execute supported A630 images into an immutable write journal; this does not retire the KGSL submission."""
  dispatch = _execution_dispatch(submission)
  active = dispatch.instructions[:next(instruction.index for instruction in dispatch.instructions if instruction.opcode == "end") + 1]
  if any(instruction.opcode in {"ldg.u32x4", "stg.u32x4", "add.f.rpt4"} for instruction in active):
    return _execute_vector_u32(dispatch, resolver, active)
  constants = struct.unpack("<1024I", dispatch.constants_image)
  lane_count = dispatch.local_size[0]
  invocation_count = dispatch.global_size[0]
  wgid,lid = _system_registers(dispatch)
  copy_plan = _validate_two_segment_u32_copy(dispatch, active, dict(dispatch.registers), wgid, lid) \
    if sum(instruction.opcode == "stg.u32" for instruction in active) == 2 else None
  loads = copy_plan[0] if copy_plan is not None else tuple(instruction for instruction in active if instruction.opcode == "ldg.u32")
  has_float_add = any(instruction.opcode == "add.f" for instruction in active)
  comparison_instruction = _u32_comparison_instruction(active)
  conversion_instruction = _integer_to_f32_instruction(active)
  multiply_sequence = _u32_multiply_sequence(active)
  integer_instruction = multiply_sequence[-1] if multiply_sequence is not None else \
    _u32_binary_instruction(active, "shr.b") or _u32_binary_instruction(active, "sub.u") or _u32_binary_instruction(active, "xor.b") or \
    _u32_binary_instruction(active, "and.b") or _u32_binary_instruction(active, "or.b") or \
    _u32_binary_instruction(active, "max.s") or _u32_binary_instruction(active, "max.u") or \
    _u32_binary_instruction(active, "add.u")
  load_ordinals = {instruction.index:index for index,instruction in enumerate(loads)}
  output_base = constants[0] | constants[1] << 32
  input_bases = tuple(constants[2*index+2] | constants[2*index+3] << 32 for index in range(len(loads)))
  output_itemsize = 1 if comparison_instruction is not None else 4
  output_count = invocation_count * (2 if copy_plan is not None else 1)
  _require(output_base != 0 and output_base % output_itemsize == 0 and output_base + output_count * output_itemsize <= 1 << 64 and
           all(base != 0 and base % 4 == 0 and base + invocation_count * 4 <= 1 << 64 for base in input_bases),
           "invalid A630 scalar argument range")
  writes:list[A630ExecutionWrite] = []
  copy_store_segments = {instruction.index:segment for segment,instruction in enumerate(copy_plan[1])} if copy_plan is not None else {}
  copy_load_by_destination = {instruction.dst.value:ordinal for ordinal,instruction in enumerate(loads)
                              if copy_plan is not None and instruction.dst is not None}

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
          elif opcode in _SIMPLE_CAT2_INTEGER or opcode in {"add.u", "max.s", "max.u"}:
            if opcode == "add.u": value = src[0] + src[1]
            elif opcode == "sub.u": value = src[0] - src[1]
            elif opcode == "shr.b":
              # Pinned NIR masks SHR.B to five bits, but tinygrad's PYTHON backend returns zero for wider public shift counts.
              _require(src[1] < 32, "u32 logical right shift count is outside the supported 0..31 range")
              value = src[0] >> src[1]
            elif opcode == "max.u": value = max(src)
            elif opcode == "max.s":
              signed_sources = tuple(x - (1 << 32) if x & 0x80000000 else x for x in src)
              value = src[0] if signed_sources[0] >= signed_sources[1] else src[1]
            elif opcode == "xor.b": value = src[0] ^ src[1]
            elif opcode == "and.b": value = src[0] & src[1]
            elif opcode == "or.b": value = src[0] | src[1]
            else: raise ValueError(f"unsupported simple Cat2 integer opcode {opcode}")
            if integer_instruction is not None and instruction.index == integer_instruction.index:
              source_origins = tuple(origins[lane].get(operand.value) for operand in instruction.srcs)
              if opcode in _SIMPLE_CAT2_INTEGER:
                integer_kind,origin_tag = _SIMPLE_CAT2_INTEGER[opcode]
                operation = f"u32 {integer_kind}"
              else:
                operation = {"add.u":"u32 add", "max.s":"s32 maximum", "max.u":"u32 maximum"}[opcode]
                origin_tag = {"add.u":"u32-add", "max.s":"s32-maximum", "max.u":"u32-maximum"}[opcode]
              _require(frozenset(source_origins) == frozenset((("load", 0), ("load", 1))),
                       f"{operation} does not consume both global loads")
              origin = (origin_tag, 0)
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
          elif opcode in {"cov.s32f32", "cov.u32f32"}:
            _require(conversion_instruction is not None and instruction.index == conversion_instruction.index and
                     origins[lane].get(instruction.srcs[0].value) == ("load", 0),
                     "integer-to-f32 conversion does not consume the global load")
            value = _integer_to_f32_rne_bits(src[0], opcode == "cov.s32f32")
            origin = ("s32-to-f32-rne" if opcode == "cov.s32f32" else "u32-to-f32-rne", 0)
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
            value = _f32_add_bits(src[0], src[1])
            origin = ("f32-add", 0)
          elif opcode == "ldg.u32":
            address = _gpr_address(full[lane], instruction.srcs[0])
            ordinal = load_ordinals[instruction.index]
            _require(address == input_bases[ordinal] + global_lane * 4, f"global load {ordinal} does not address its scalar input")
            _require(address % 4 == 0 and address + 4 <= 1 << 64, "invalid A630 global-load address")
            view = resolver(address, 4)
            _require(len(view) == 4, "short A630 global-load range")
            value = struct.unpack("<I", bytes(view))[0]
            origin = ("load", ordinal)
          elif opcode == "stg.u32":
            address = _gpr_address(full[lane], instruction.srcs[0])
            if copy_plan is not None:
              segment = copy_store_segments[instruction.index]
              _require(address == output_base + (segment * invocation_count + global_lane) * 4,
                       f"two-segment u32 copy store {segment} does not address its output segment")
              ordinal = copy_load_by_destination[instruction.srcs[1].value]
              expected_origin,store_source = ("load", ordinal),f"two-segment u32 copy input {ordinal}"
            else:
              _require(address == output_base + global_lane * 4, "global store does not address the scalar output")
              if not loads: expected_origin,store_source = ("fill", 0x3f800000),"A630 fill"
              elif has_float_add: expected_origin,store_source = ("f32-add", 0),"f32 add"
              elif conversion_instruction is not None:
                assert conversion_instruction.opcode is not None
                signed_conversion = conversion_instruction.opcode == "cov.s32f32"
                expected_origin = ("s32-to-f32-rne" if signed_conversion else "u32-to-f32-rne", 0)
                store_source = "s32-to-f32 conversion" if signed_conversion else "u32-to-f32 conversion"
              elif integer_instruction is not None:
                if integer_instruction.opcode in _SIMPLE_CAT2_INTEGER:
                  integer_kind,origin_tag = _SIMPLE_CAT2_INTEGER[integer_instruction.opcode]
                  expected_origin,store_source = (origin_tag, 0),f"u32 {integer_kind}"
                elif integer_instruction.opcode == "add.u": expected_origin,store_source = ("u32-add", 0),"u32 add"
                elif integer_instruction.opcode == "max.s": expected_origin,store_source = ("s32-maximum", 0),"s32 maximum"
                elif integer_instruction.opcode == "max.u": expected_origin,store_source = ("u32-maximum", 0),"u32 maximum"
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
            address = _gpr_address(full[lane], instruction.srcs[0])
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

  return _finish_writes(writes)

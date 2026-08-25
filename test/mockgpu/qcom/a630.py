from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Callable, Sequence
from tinygrad.runtime.autogen import mesa
from test.mockgpu.qcom.pm4 import PM4Packet, PM4Type4Packet, PM4Type7Packet

# Payload fields and units follow Mesa 25.2.7 at 461196a1c827769168304ff3f5b36360f16618ca:
# adreno_pm4.xml, a6xx.xml, tu_shader.cc, tu_cmd_buffer.cc, and ir3_shader.h.

@dataclass(frozen=True)
class A630MemoryRange:
  address:int
  size:int
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

@dataclass(frozen=True)
class A630Submission:
  dispatches:tuple[A630Dispatch, ...]
  memory_ranges:tuple[A630MemoryRange, ...]
  waits:tuple[A630Wait, ...]
  writes:tuple[A630Write, ...]

Resolver = Callable[[int, int], memoryview]

def _require(condition:bool, message:str):
  if not condition: raise ValueError(message)

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

  nsamp, ntex, nuav = config >> 17 & 0x1f, config >> 9 & 0xff, config >> 22 & 0x7f
  resources = (("samplers", nsamp, mesa.REG_A6XX_SP_CS_SAMPLER_BASE, 16, nsamp),
               ("textures", ntex, mesa.REG_A6XX_SP_CS_TEXMEMOBJ_BASE, 64, min(16, ntex)),
               ("uavs", nuav, mesa.REG_A6XX_SP_CS_UAV_BASE, 64, nuav))
  for kind,count,base_register,unit_size,load_units in resources:
    if count == 0: continue
    _require(kind in loads and loads[kind].units == load_units, f"missing or inconsistent {kind} state load")
    base_words = _registers(regs, base_register, 2, f"{kind} base")
    base = _address(base_words[0], base_words[1], unit_size if kind != "uavs" else 16, kind)
    _require(base == loads[kind].address, f"{kind} base differs from state load")
    ranges.append(A630MemoryRange(base, count * unit_size, False, f"{kind} descriptors"))
  if nsamp:
    border = _registers(regs, mesa.REG_A6XX_TPL1_CS_BORDER_COLOR_BASE, 2, "border-color base")
    ranges.append(A630MemoryRange(_address(border[0], border[1], 128, "border-color"), 128, False, "border color"))

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
      ranges.append(A630MemoryRange(address, 4, False, "wait value"))
      waits.append(A630Wait(packet.word_offset, address, values[3], values[4]))
    elif packet.opcode == mesa.CP_EVENT_WRITE:
      if len(values) == 1: _require(values[0] == mesa.CACHE_INVALIDATE, "invalid cache-invalidate event")
      else:
        _require(values[0] == mesa.CACHE_FLUSH_TS, "invalid cache-flush event")
        address = _address(values[1], values[2], 4, "event-write")
        ranges.append(A630MemoryRange(address, 4, True, "event value"))
        writes.append(A630Write(packet.word_offset, address, 4, values[3], "event value"))
    elif packet.opcode == mesa.CP_REG_TO_MEM:
      expected = mesa.REG_A6XX_CP_ALWAYS_ON_COUNTER | 2 << 18 | 1 << 30
      _require(values[0] == expected, "invalid counter-to-memory control")
      address = _address(values[1], values[2], 8, "counter")
      ranges.append(A630MemoryRange(address, 8, True, "counter value"))
      writes.append(A630Write(packet.word_offset, address, 8, None, "counter value"))
    elif packet.opcode in (mesa.CP_WAIT_MEM_WRITES, mesa.CP_WAIT_FOR_IDLE): pass
    elif packet.opcode == mesa.CP_SET_MARKER:
      _require(values == (mesa.RM6_COMPUTE,), "invalid compute marker")
      marker_pending = True
    elif packet.opcode == mesa.CP_LOAD_STATE6_FRAG:
      load = _load_state(values)
      loads[load.kind] = load
      ranges.append(A630MemoryRange(load.address, load.size, False, load.kind))
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
    image = None if memory_range.write else bytes(view)
    resolved_range = replace(memory_range, image=image)
    resolved.append(resolved_range)
    if image is not None: read_images[(memory_range.address, memory_range.size, memory_range.purpose)] = image
  frozen_dispatches = tuple(replace(dispatch,
    shader_image=read_images[(dispatch.shader_address, dispatch.shader_size, "shader")],
    constants_image=read_images[(dispatch.constants_address, dispatch.constants_size, "constants")]) for dispatch in dispatches)
  return A630Submission(frozen_dispatches, tuple(resolved), tuple(waits), tuple(writes))

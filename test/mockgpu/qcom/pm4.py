from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
from typing import Sequence
from tinygrad.runtime.autogen import mesa

# Header layouts and odd-parity bits follow pinned Mesa 25.2.7 freedreno_pm4.h at commit 461196a1c827769168304ff3f5b36360f16618ca.

@dataclass(frozen=True)
class PM4Type4Packet:
  word_offset:int
  register:int
  values:tuple[int, ...]

@dataclass(frozen=True)
class PM4Type7Packet:
  word_offset:int
  opcode:int
  values:tuple[int, ...]

PM4Packet = PM4Type4Packet|PM4Type7Packet

TYPE4_SHAPES = frozenset({
  (mesa.REG_A6XX_SP_UPDATE_CNTL, 1),
  (mesa.REG_A6XX_SP_CS_TSIZE, 1),
  (mesa.REG_A6XX_SP_CS_USIZE, 1),
  (mesa.REG_A6XX_SP_MODE_CNTL, 1),
  (mesa.REG_A6XX_SP_PERFCTR_SHADER_MASK, 1),
  (mesa.REG_A6XX_TPL1_MODE_CNTL, 1),
  (mesa.REG_A6XX_TPL1_DBG_ECO_CNTL, 1),
  (mesa.REG_A6XX_SP_CS_NDRANGE_0, 12),
  (mesa.REG_A6XX_SP_CS_CNTL_0, 10),
  (mesa.REG_A6XX_SP_REG_PROG_ID_0, 5),
  (mesa.REG_A6XX_SP_CS_PVT_MEM_STACK_OFFSET, 1),
  (mesa.REG_A6XX_SP_CS_INSTR_SIZE, 1),
  (mesa.REG_A6XX_SP_CS_SAMPLER_BASE, 2),
  (mesa.REG_A6XX_TPL1_CS_BORDER_COLOR_BASE, 2),
  (mesa.REG_A6XX_SP_CS_TEXMEMOBJ_BASE, 2),
  (mesa.REG_A6XX_SP_CS_UAV_BASE, 2),
  (mesa.REG_A6XX_SP_CS_CONFIG, 1),
  (mesa.REG_A6XX_SP_CS_CONST_CONFIG_0, 2),
})

TYPE7_SHAPES = MappingProxyType({
  mesa.CP_WAIT_REG_MEM: frozenset({6}),
  mesa.CP_EVENT_WRITE: frozenset({1, 4}),
  mesa.CP_REG_TO_MEM: frozenset({3}),
  mesa.CP_WAIT_MEM_WRITES: frozenset({0}),
  mesa.CP_WAIT_FOR_IDLE: frozenset({0}),
  mesa.CP_SET_MARKER: frozenset({1}),
  mesa.CP_LOAD_STATE6_FRAG: frozenset({3}),
  mesa.CP_EXEC_CS: frozenset({4}),
})

def _odd_parity(value:int) -> int:
  value ^= value >> 16
  value ^= value >> 8
  value ^= value >> 4
  return (~0x6996 >> (value & 0xf)) & 1

def _type4_header(register:int, count:int) -> int:
  return mesa.CP_TYPE4_PKT | count | _odd_parity(count) << 7 | register << 8 | _odd_parity(register) << 27

def _type7_header(opcode:int, count:int) -> int:
  return mesa.CP_TYPE7_PKT | count | _odd_parity(count) << 15 | opcode << 16 | _odd_parity(opcode) << 23

def parse_pm4(words:Sequence[int]) -> tuple[PM4Packet, ...]:
  if not words: raise ValueError("empty PM4 stream")
  if any(word < 0 or word > 0xffffffff for word in words): raise ValueError("PM4 word outside uint32 range")

  packets:list[PM4Packet] = []
  offset = 0
  while offset < len(words):
    header = words[offset]
    packet_type = header >> 28
    if packet_type == 4:
      register, count = (header >> 8) & 0x7ffff, header & 0x7f
      if register & (1 << 18): raise ValueError(f"non-production type-4 register at word {offset}")
      if count == 0 or count >= 0x7f: raise ValueError(f"invalid type-4 count {count} at word {offset}")
      if header != _type4_header(register, count): raise ValueError(f"invalid type-4 header at word {offset}")
      end = offset + count + 1
      if end > len(words): raise ValueError(f"truncated type-4 packet at word {offset}")
      if (register, count) not in TYPE4_SHAPES: raise ValueError(f"unsupported type-4 packet shape {register:#x}/{count} at word {offset}")
      packets.append(PM4Type4Packet(offset, register, tuple(words[offset+1:end])))
    elif packet_type == 7:
      opcode, count = (header >> 16) & 0x7f, header & 0x3fff
      if header != _type7_header(opcode, count): raise ValueError(f"invalid type-7 header at word {offset}")
      end = offset + count + 1
      if end > len(words): raise ValueError(f"truncated type-7 packet at word {offset}")
      if opcode not in TYPE7_SHAPES: raise ValueError(f"unsupported type-7 opcode {opcode:#x} at word {offset}")
      if count not in TYPE7_SHAPES[opcode]: raise ValueError(f"unsupported type-7 packet shape {opcode:#x}/{count} at word {offset}")
      packets.append(PM4Type7Packet(offset, opcode, tuple(words[offset+1:end])))
    else: raise ValueError(f"unsupported packet header {header:#010x} at word {offset}")
    offset = end
  return tuple(packets)

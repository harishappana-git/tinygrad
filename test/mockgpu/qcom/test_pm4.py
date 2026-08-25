import unittest
from typing import Callable, cast
from tinygrad.runtime.autogen import mesa
from tinygrad.runtime.ops_qcom import parity, pkt4_hdr, pkt7_hdr
from test.mockgpu.qcom.pm4 import PM4Type4Packet, PM4Type7Packet, parse_pm4


TYPE4_SHAPES = (
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
)

TYPE7_SHAPES = (
  (mesa.CP_WAIT_REG_MEM, 6),
  (mesa.CP_EVENT_WRITE, 1),
  (mesa.CP_EVENT_WRITE, 4),
  (mesa.CP_REG_TO_MEM, 3),
  (mesa.CP_WAIT_MEM_WRITES, 0),
  (mesa.CP_WAIT_FOR_IDLE, 0),
  (mesa.CP_SET_MARKER, 1),
  (mesa.CP_LOAD_STATE6_FRAG, 3),
  (mesa.CP_EXEC_CS, 4),
)

def packet_stream(shapes:tuple[tuple[int, int], ...], header:Callable[[int, int], int]) -> list[int]:
  words:list[int] = []
  for field,count in shapes: words.extend((header(field, count), *([0] * count)))
  return words


class TestA630PM4Parser(unittest.TestCase):
  def test_accepts_production_compute_packet_shapes(self):
    packets = parse_pm4(packet_stream(TYPE4_SHAPES, pkt4_hdr) + packet_stream(TYPE7_SHAPES, pkt7_hdr))
    self.assertEqual(len(packets), len(TYPE4_SHAPES) + len(TYPE7_SHAPES))
    for packet,(register,count) in zip(packets, TYPE4_SHAPES):
      self.assertIsInstance(packet, PM4Type4Packet)
      packet = cast(PM4Type4Packet, packet)
      self.assertEqual((packet.register, len(packet.values)), (register, count))
    for packet,(opcode,count) in zip(packets[len(TYPE4_SHAPES):], TYPE7_SHAPES):
      self.assertIsInstance(packet, PM4Type7Packet)
      packet = cast(PM4Type7Packet, packet)
      self.assertEqual((packet.opcode, len(packet.values)), (opcode, count))

  def test_rejects_noncanonical_headers(self):
    nonproduction_register = (1 << 18) | mesa.REG_A6XX_SP_UPDATE_CNTL
    wide_register_header = mesa.CP_TYPE4_PKT | 1 | parity(1) << 7 | nonproduction_register << 8 | parity(nonproduction_register) << 27
    cases:tuple[tuple[list[int], str], ...] = (
      ([], "empty PM4 stream"),
      ([0], "unsupported packet header"),
      ([mesa.CP_TYPE2_PKT], "unsupported packet header"),
      ([mesa.CP_TYPE3_PKT], "unsupported packet header"),
      ([0x50000000], "unsupported packet header"),
      ([pkt4_hdr(mesa.REG_A6XX_SP_UPDATE_CNTL, 1) ^ (1 << 7), 0], "invalid type-4 header"),
      ([pkt4_hdr(mesa.REG_A6XX_SP_UPDATE_CNTL, 1) ^ (1 << 27), 0], "invalid type-4 header"),
      ([pkt7_hdr(mesa.CP_WAIT_FOR_IDLE, 0) ^ (1 << 15)], "invalid type-7 header"),
      ([pkt7_hdr(mesa.CP_WAIT_FOR_IDLE, 0) ^ (1 << 23)], "invalid type-7 header"),
      ([pkt7_hdr(mesa.CP_WAIT_FOR_IDLE, 0) | (1 << 14)], "invalid type-7 header"),
      ([pkt7_hdr(mesa.CP_WAIT_FOR_IDLE, 0) | (1 << 24)], "invalid type-7 header"),
      ([wide_register_header, 0], "non-production type-4 register"),
      ([pkt4_hdr(mesa.REG_A6XX_SP_UPDATE_CNTL, 0)], "invalid type-4 count"),
      ([pkt4_hdr(mesa.REG_A6XX_SP_UPDATE_CNTL, 0x7f), *([0] * 0x7f)], "invalid type-4 count"),
    )
    for words,message in cases:
      with self.subTest(message=message), self.assertRaisesRegex(ValueError, message): parse_pm4(words)

  def test_rejects_unknown_shapes_and_truncation(self):
    cases = (
      ([pkt7_hdr(0x7e, 0)], "unsupported type-7 opcode"),
      ([pkt7_hdr(mesa.CP_RUN_OPENCL, 1), 0], "unsupported type-7 opcode"),
      ([pkt7_hdr(mesa.CP_WAIT_REG_MEM, 5), *([0] * 5)], "unsupported type-7 packet shape"),
      ([pkt4_hdr(0x1234, 1), 0], "unsupported type-4 packet shape"),
      ([pkt4_hdr(mesa.REG_A6XX_SP_UPDATE_CNTL, 2), 0, 0], "unsupported type-4 packet shape"),
      ([pkt4_hdr(mesa.REG_A6XX_SP_CS_NDRANGE_0 + 1, 11), *([0] * 11)], "unsupported type-4 packet shape"),
      ([pkt7_hdr(mesa.CP_EXEC_CS, 4), 0, 0, 0], "truncated type-7 packet"),
      ([pkt4_hdr(mesa.REG_A6XX_SP_CS_NDRANGE_0, 12), *([0] * 11)], "truncated type-4 packet"),
    )
    for words,message in cases:
      with self.subTest(message=message), self.assertRaisesRegex(ValueError, message): parse_pm4(words)

  def test_returns_immutable_structured_packets(self):
    words = [pkt7_hdr(mesa.CP_EVENT_WRITE, 1), 0x12345678]
    packets = parse_pm4(words)
    words[1] = 0
    self.assertEqual(packets, (PM4Type7Packet(0, mesa.CP_EVENT_WRITE, (0x12345678,)),))
    with self.assertRaises((AttributeError, TypeError)): setattr(packets[0], "values", (0,))


if __name__ == '__main__':
  unittest.main()

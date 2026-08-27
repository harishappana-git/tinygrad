import unittest
from unittest import mock
from test.mockgpu.qcom.qcom_test_base import _QCOMTestBase

class TestA630Contracts(_QCOMTestBase):
  def test_ir3_decoder_rejects_invalid_and_private_encodings(self):
    from test.mockgpu.qcom import a630 as a630_module
    from test.mockgpu.qcom.a630 import decode_a630_ir3

    end = 6 << 55
    decoded = decode_a630_ir3(end.to_bytes(8, "little"))
    self.assertEqual(len(decoded), 1)
    self.assertEqual((decoded[0].index, decoded[0].category, decoded[0].raw, decoded[0].name), (0, 0, end, "end"))
    self.assertEqual(decoded[0].fields,
                     (("SY", 0), ("SS", 0), ("EQ", 0), ("JP", 0), ("REPEAT", 0), ("NAME", "end")))
    with mock.patch.object(a630_module, "_MAX_SHADER_INSTRUCTIONS", 1):
      self.assertEqual(decode_a630_ir3(end.to_bytes(8, "little")), decoded)
      with self.assertRaisesRegex(ValueError, "emulator instruction limit"):
        decode_a630_ir3(end.to_bytes(8, "little") + bytes(8))

    for image,message in (
      (b"", "empty IR3 image"),
      (bytes(7), "multiple of 8"),
      (bytes(8), "missing end instruction"),
      ((end.to_bytes(8, "little") + (1 << 40).to_bytes(8, "little")), "nonzero instruction after end"),
      ((end | 1 << 32).to_bytes(8, "little"), "invalid or reserved IR3 encoding"),
      ((0xb87f84e841a312a8).to_bytes(8, "little"), "invalid or reserved IR3 encoding"),
      ((0xecb0ba8427a4164a).to_bytes(8, "little"), "unmatched IR3 encoding"),
    ):
      with self.subTest(message=message), self.assertRaisesRegex(ValueError, message): decode_a630_ir3(image)

    private_or_stack = {
      "ret":4 << 55,
      "call":3 << 55,
      "ldp":(6 << 61) | (2 << 54) | (1 << 23) | 1,
      "stp":(6 << 61) | (5 << 54) | (1 << 40) | (1 << 23),
    }
    for name,word in private_or_stack.items():
      with self.subTest(name=name), self.assertRaisesRegex(ValueError, f"unsupported IR3 instruction {name}"):
        decode_a630_ir3(word.to_bytes(8, "little") + end.to_bytes(8, "little"))

  def test_ir3_image_limit_precedes_shader_resolution(self):
    from tinygrad.runtime.autogen import mesa
    from test.mockgpu.qcom import a630 as a630_module
    from test.mockgpu.qcom.a630 import _load_state, stage_a630
    from test.mockgpu.qcom.pm4 import PM4Type7Packet

    max_groups = a630_module._MAX_SHADER_INSTRUCTIONS * 8 // 128
    self.assertEqual(max_groups, 256)
    def shader_values(units):
      control = mesa.ST_SHADER << 14 | mesa.SS6_INDIRECT << 16 | mesa.SB6_CS_SHADER << 18 | units << 22
      return control, 0x1000, 0

    admitted = _load_state(shader_values(max_groups))
    self.assertEqual((admitted.kind, admitted.size, admitted.units), ("shader", max_groups * 128, max_groups))
    resolver = mock.Mock(side_effect=AssertionError("oversized shader must not be resolved"))
    oversized = PM4Type7Packet(0, mesa.CP_LOAD_STATE6_FRAG, shader_values(max_groups + 1))
    with self.assertRaisesRegex(ValueError, "IR3 image exceeds the emulator instruction limit"):
      stage_a630((oversized,), resolver)
    resolver.assert_not_called()

  def test_state_snapshot_budget_deduplicates_and_bounds_amplification(self):
    from tinygrad.runtime.autogen import mesa
    from test.mockgpu.qcom.a630 import stage_a630
    from test.mockgpu.qcom.pm4 import PM4Type7Packet

    units = 0x3ff
    control = mesa.ST_CONSTANTS << 14 | mesa.SS6_INDIRECT << 16 | mesa.SB6_CS_TEX << 18 | units << 22
    load_size = units * 64
    load = PM4Type7Packet(0, mesa.CP_LOAD_STATE6_FRAG, (control, 0x10000, 0))
    resolver = mock.Mock(return_value=memoryview(bytearray(load_size)))
    submission = stage_a630((load,) * (1 << 14), resolver)
    self.assertEqual((submission.dispatches, len(submission.memory_ranges), resolver.call_count), ((), 1, 1))

    unique = tuple(PM4Type7Packet(index * 4, mesa.CP_LOAD_STATE6_FRAG,
                                 (control, 0x10000 + index * 0x10000, 0)) for index in range(17))
    resolver.reset_mock()
    self.assertEqual(len(stage_a630(unique[:16], resolver).memory_ranges), 16)
    self.assertEqual(resolver.call_count, 16)
    resolver.reset_mock()
    with self.assertRaisesRegex(ValueError, "state snapshots exceed the emulator byte limit"):
      stage_a630(unique, resolver)
    self.assertEqual(resolver.call_count, 16)

  def test_memory_and_fixed_delay_scoreboards_track_register_dependencies(self):
    from dataclasses import replace
    from test.mockgpu.qcom import a630 as a630_module
    from test.mockgpu.qcom.a630 import A630Dispatch, A630IR3Instruction, A630IR3Operand, _validate_control_flow

    def gpr(value): return A630IR3Operand("gpr", value)
    def half(value): return A630IR3Operand("half", value)
    def iim(value): return A630IR3Operand("iim", value)
    def pred(value): return A630IR3Operand("pred", value)
    def instruction(index, opcode, dst=None, srcs=(), *, category=0, ss=0, sy=0, repeat=0, nop=0):
      fields = (("SS", ss), ("SY", sy), ("REPEAT", repeat), ("NOP", nop))
      return A630IR3Instruction(index, category, 0, opcode, fields, opcode, dst, srcs)
    dispatch = A630Dispatch(0, (), (), 0, 0, b"", 0, 0, b"", 0, 0, (1, 1, 1), (1, 1, 1), (1, 1, 1))
    load = instruction(0, "ldl.u32x4", gpr(8), (gpr(0),))
    end = instruction(2, "end")
    hazards = {
      "full RAW upper component":instruction(1, "add.u", gpr(30), (gpr(11), gpr(1))),
      "full WAW upper component":instruction(1, "add.u", gpr(11), (gpr(30), gpr(1))),
    }
    for name,consumer in hazards.items():
      with self.subTest(hazard=name), self.assertRaisesRegex(ValueError, "local-load dependency lacks SS"):
        _validate_control_flow((load, consumer, end), dispatch)
      with self.subTest(synchronized=name):
        _validate_control_flow((load, replace(consumer, fields=(("SS", 1), ("SY", 0))), end), dispatch)

    # The supported CS_CNTL_0 mode has MERGEDREGS clear, so even numerically adjacent half registers use a separate file.
    for value in (15, 16, 23, 24):
      for consumer in (instruction(1, "cmps.s.lt", half(value), (gpr(30), gpr(31))),
                       instruction(1, "cov.u16s32", gpr(30), (half(value),))):
        with self.subTest(non_alias=value, opcode=consumer.opcode): _validate_control_flow((load, consumer, end), dispatch)

    control_program = (
      instruction(0, "mov.u32", gpr(8), (gpr(0),)),
      instruction(1, "add.u", gpr(30), (gpr(8), gpr(1))),
      instruction(2, "ldl.u32x4", gpr(8), (gpr(0),)),
      instruction(3, "jump", srcs=(iim(-2),)),
      instruction(4, "end"),
    )
    with self.assertRaisesRegex(ValueError, "local memory with control flow is unsupported"):
      _validate_control_flow(control_program, dispatch)

    barrier = instruction(0, "bar.g")
    ss_nop = instruction(1, "nop", ss=1)
    unsynchronized_load = instruction(2, "ldl.u32x4", gpr(8), (gpr(0),))
    with self.assertRaisesRegex(ValueError, "barrier synchronization lacks SY"):
      _validate_control_flow((barrier, ss_nop, unsynchronized_load, instruction(3, "end")), dispatch)
    synchronized_load = replace(unsynchronized_load, fields=(("SS", 0), ("SY", 1)))
    _validate_control_flow((barrier, ss_nop, synchronized_load, instruction(3, "nop", ss=1),
                            instruction(4, "add.u", gpr(30), (gpr(8), gpr(1))), instruction(5, "end")), dispatch)

    global_load = instruction(0, "ldg.u32x4", gpr(8), (gpr(0),))
    for consumer in (instruction(1, "add.u", gpr(30), (gpr(11), gpr(1))), instruction(1, "add.u", gpr(11), (gpr(30), gpr(1)))):
      with self.subTest(global_load_hazard=consumer.dst), self.assertRaisesRegex(ValueError, "global-load dependency lacks SY"):
        _validate_control_flow((global_load, consumer, end), dispatch)
      _validate_control_flow((global_load, replace(consumer, fields=(("SS", 0), ("SY", 1))), end), dispatch)

    global_address_overwrite = instruction(1, "add.u", gpr(1), (gpr(30), gpr(31)))
    with self.assertRaisesRegex(ValueError, "global-load source overwrite lacks SS or SY"):
      _validate_control_flow((global_load, global_address_overwrite, end), dispatch)
    for ss,sy in ((1, 0), (0, 1)):
      _validate_control_flow((global_load, replace(global_address_overwrite, fields=(("SS", ss), ("SY", sy))), end), dispatch)

    local_address_overwrite = instruction(1, "add.u", gpr(0), (gpr(30), gpr(31)))
    with self.assertRaisesRegex(ValueError, "memory-source overwrite lacks SS"):
      _validate_control_flow((load, local_address_overwrite, end), dispatch)
    _validate_control_flow((load, replace(local_address_overwrite, fields=(("SS", 1), ("SY", 0))), end), dispatch)
    with self.assertRaisesRegex(ValueError, "memory-source overwrite lacks SS"):
      _validate_control_flow((load, replace(local_address_overwrite, fields=(("SS", 0), ("SY", 1))), end), dispatch)

    skipped_sy = (instruction(0, "ldg.u32", gpr(8), (gpr(0),)),
                  instruction(1, "br.p0", srcs=(pred(0), iim(2))), instruction(2, "nop", sy=1),
                  instruction(3, "add.u", gpr(30), (gpr(8), gpr(1))), instruction(4, "end"))
    with self.assertRaisesRegex(ValueError, "global-load dependency lacks SY"): _validate_control_flow(skipped_sy, dispatch)
    backedge = (instruction(0, "add.u", gpr(0), (gpr(30), gpr(31))), instruction(1, "ldg.u32", gpr(8), (gpr(0),)),
                instruction(2, "jump", srcs=(iim(-2),)), instruction(3, "end"))
    with self.assertRaisesRegex(ValueError, "global-load source overwrite lacks SS or SY"):
      _validate_control_flow(backedge, dispatch)

    byte_store = instruction(0, "stg.u8", srcs=(gpr(4), half(6)))
    store_overwrites = (instruction(1, "add.u", gpr(5), (gpr(30), gpr(31))),
                        instruction(1, "cmps.s.lt", half(6), (gpr(30), gpr(31))))
    for overwrite in store_overwrites:
      with self.subTest(store_war=overwrite.dst), self.assertRaisesRegex(ValueError, "memory-source overwrite lacks SS"):
        _validate_control_flow((byte_store, overwrite, end), dispatch)
      _validate_control_flow((byte_store, replace(overwrite, fields=(("SS", 1), ("SY", 0))), end), dispatch)

    # A branch lattice can grow a different store-source mask at every merge. The emulator rejects boundedly instead
    # of allowing a maximum-size adversarial shader to monopolize validation before any execution budget applies.
    complex_cfg = tuple(item for pair in ((instruction(2*index, "br.p0", srcs=(pred(0), iim(2))),
                                           instruction(2*index+1, "stg.u8", srcs=(gpr(4), half(index % 0xc0))))
                                          for index in range(32)) for item in pair) + (instruction(64, "end"),)
    with mock.patch.object(a630_module, "_MAX_SCHEDULE_VALIDATION_STEPS", 32), \
         self.assertRaisesRegex(ValueError, "schedule validation exceeds the emulator work limit"):
      _validate_control_flow(complex_cfg, dispatch)

    rpt4_overlap = (instruction(0, "add.f.rpt4", gpr(20), (gpr(19), gpr(40)), category=2, repeat=3, nop=3),
                    instruction(1, "end"))
    rpt2_overlap = (instruction(0, "add.u.rpt2", gpr(20), (gpr(40), gpr(19)), category=2, repeat=1, nop=3),
                    instruction(1, "end"))
    for program in (rpt4_overlap, rpt2_overlap):
      with self.subTest(intra_repeat=program[0].opcode), \
           self.assertRaisesRegex(ValueError, "fixed ALU dependency lacks delay slots"):
        _validate_control_flow(program, dispatch)

    aligned_repeats = (instruction(0, "add.f.rpt4", gpr(20), (gpr(0), gpr(4)), category=2, repeat=3),
                       instruction(1, "add.f.rpt4", gpr(40), (gpr(20), gpr(24)), category=2, repeat=3, nop=3),
                       instruction(2, "end"))
    _validate_control_flow(aligned_repeats, dispatch)
    immediate_late_read = (instruction(0, "add.u", gpr(20), (gpr(0), gpr(1)), category=2),
                           instruction(1, "madsh.m16", gpr(21), (gpr(30), gpr(31), gpr(20)), category=3),
                           instruction(2, "end"))
    with self.assertRaisesRegex(ValueError, "fixed ALU dependency lacks delay slots"):
      _validate_control_flow(immediate_late_read, dispatch)
    delayed_late_read = (immediate_late_read[0], instruction(1, "nop"),
                         replace(immediate_late_read[1], index=2), instruction(3, "end"))
    _validate_control_flow(delayed_late_read, dispatch)

  def test_max_group_memory_free_dispatch_uses_no_dense_local_backing(self):
    from tinygrad.runtime.autogen import mesa
    from test.mockgpu.qcom import a630 as a630_module
    from test.mockgpu.qcom.a630 import A630Dispatch, A630ExecutionBudget, _execute_a630_dispatch, decode_a630_ir3

    end = decode_a630_ir3((6 << 55).to_bytes(8, "little"))[0]
    registers = (
      (mesa.REG_A6XX_SP_CS_CONST_CONFIG_0, 0xfcfcfcc0),
      (mesa.REG_A6XX_SP_CS_WGE_CNTL, 0xfc),
      (mesa.REG_A6XX_SP_CS_CNTL_1, mesa.CONSTLEN_256 << mesa.A6XX_SP_CS_CNTL_1_CONSTANTRAMMODE__SHIFT),
    )
    dispatch = A630Dispatch(0, registers, (), 0, 8, end.raw.to_bytes(8, "little"), 0, 4096, bytes(4096), 0, 0,
                            (1, 1, 1), (a630_module._MAX_INVOCATIONS, 1, 1),
                            (a630_module._MAX_INVOCATIONS, 1, 1), (end,))
    resolver = mock.Mock(side_effect=AssertionError("memory-free dispatch resolved memory"))
    budget = A630ExecutionBudget()
    with mock.patch.object(a630_module, "bytearray", side_effect=AssertionError("dense A630 local backing allocated"), create=True):
      writes = _execute_a630_dispatch(dispatch, resolver, (end,), read_observer=None, budget=budget)
    self.assertEqual((writes, budget.lane_instruction_steps, budget.memory_events),
                     ((), a630_module._MAX_INVOCATIONS, 0))
    resolver.assert_not_called()

  def test_ir3_typed_instruction_and_modifier_contracts(self):
    from dataclasses import replace
    from tinygrad.runtime.autogen import mesa
    from test.mockgpu.qcom.a630 import A630Dispatch, A630IR3Operand, _full_gpr_accesses, _validate_register_footprint, decode_a630_ir3

    end = 6 << 55
    words = (
      0x47180803201f0000,  # ashr.b
      0x46d8080320020003,  # shl.b
      0x650004030003301e,  # shrg
      0x421000050008000a,  # add.u
      0x4290400010020004,  # cmps.u.lt
      0x2009400a00000000,  # cov.u16s32
      0x10000000000,       # nop with one repeated no-op
      0xc006000b01810001,  # ldg.u32
      0x5018080b2802000b,  # add.f with FLUT[2]
      0xc0c60d0001800016,  # stg.u32
      0x5650080800070002,  # mull.u
      0x6183880800080002,  # madsh.m16 with one scheduling nop
      0x6181080700088007,  # madsh.m16 with three scheduling nops
      0x42b400f820000000,  # cmps.s.eq p0.x, r0, #0
      0xc1064f000180001a,  # stl.u32 local[r39], r13
      0xe042000000000000,  # bar.g
      0xd046002c04800001,  # (sy) ldl.u32x4 r44:r47, local[r0]
      0x0682000000000000,  # predt p0.x
      0x0782000000000000,  # prede
    )
    decoded = decode_a630_ir3(b"".join(word.to_bytes(8, "little") for word in words + (end,)))
    self.assertEqual(tuple(instruction.opcode for instruction in decoded),
                     ("ashr.b", "shl.b", "shrg", "add.u", "cmps.u.lt", "cov.u16s32", "nop",
                      "ldg.u32", "add.f", "stg.u32", "mull.u", "madsh.m16", "madsh.m16",
                      "cmps.s.eq.p0", "stl.u32", "bar.g", "ldl.u32x4", "predt.p0", "prede", "end"))

    def decode_one(word): return decode_a630_ir3(word.to_bytes(8, "little") + end.to_bytes(8, "little"))[0]

    addressing_words = {
      "shared multiply":0x46500005200c00c0, "register multiply":0x4650080320030000,
      "constant accumulate":0x6182000400051010, "scheduled constant accumulate":0x6180080000039011,
      "repeated address add":0x42180110000b1003,
    }
    addressing_expected = {
      "shared multiply":("mull.u", A630IR3Operand("gpr", 5),
                         (A630IR3Operand("shared", 192), A630IR3Operand("iim", 12))),
      "register multiply":("mull.u", A630IR3Operand("gpr", 3),
                           (A630IR3Operand("gpr", 0), A630IR3Operand("iim", 3))),
      "constant accumulate":("madsh.m16", A630IR3Operand("gpr", 4),
                             (A630IR3Operand("const", 16), A630IR3Operand("gpr", 4), A630IR3Operand("gpr", 5))),
      "scheduled constant accumulate":("madsh.m16", A630IR3Operand("gpr", 0),
                                       (A630IR3Operand("const", 17), A630IR3Operand("gpr", 0), A630IR3Operand("gpr", 3))),
      "repeated address add":("add.u.rpt2", A630IR3Operand("gpr", 16),
                              (A630IR3Operand("const", 3), A630IR3Operand("gpr", 11))),
    }
    for name,word in addressing_words.items():
      instruction = decode_one(word)
      self.assertEqual((instruction.opcode, instruction.dst, instruction.srcs), addressing_expected[name])
    repeated_address = decode_one(addressing_words["repeated address add"])
    self.assertEqual(tuple(value for field,value in repeated_address.fields if field == "SRC_R"), (0, 1))
    self.assertTrue({("REPEAT", 1), ("DST_HALF", 0)} <= set(repeated_address.fields))
    self.assertIn(("NOP", 1), decode_one(addressing_words["register multiply"]).fields)
    self.assertIn(("NOP", 3), decode_one(addressing_words["scheduled constant accumulate"]).fields)
    boundary_repeat = replace(repeated_address, dst=A630IR3Operand("gpr", 15),
                              srcs=(A630IR3Operand("const", 3), A630IR3Operand("gpr", 7)))
    self.assertEqual(_full_gpr_accesses(boundary_repeat), {7, 8, 15, 16})
    empty_dispatch = A630Dispatch(0, (), (), 0, 0, b"", 0, 0, b"", 0, 0, (1, 1, 1), (1, 1, 1), (1, 1, 1))
    footprint_shift = mesa.A6XX_SP_CS_CNTL_0_FULLREGFOOTPRINT__SHIFT
    admitted = replace(empty_dispatch, registers=((mesa.REG_A6XX_SP_CS_CNTL_0, 5 << footprint_shift),))
    _validate_register_footprint(admitted, (boundary_repeat,), 0xfc, 0xfc)
    with self.assertRaisesRegex(ValueError, "register footprints do not match"):
      _validate_register_footprint(replace(admitted, registers=((mesa.REG_A6XX_SP_CS_CNTL_0, 4 << footprint_shift),)),
                                   (boundary_repeat,), 0xfc, 0xfc)
    self.assertIsNone(decode_one(repeated_address.raw & ~((1 << 43) | (1 << 51))).opcode)
    constant_madsh = decode_one(addressing_words["constant accumulate"])
    with self.assertRaisesRegex(ValueError, "unmatched IR3 encoding at instruction 0"):
      decode_one(constant_madsh.raw | 1 << 13)

    workgroup_words = {
      "compare":0x42b400f820000000, "store-local":0xc1064f000180001a, "barrier":0xe042000000000000,
      "load-local":0xd046002c04800001, "predicate-true":0x0682000000000000, "predicate-end":0x0782000000000000,
    }
    workgroup_expected = {
      "compare":("cmps.s.eq.p0", A630IR3Operand("pred", 0),
                 (A630IR3Operand("gpr", 0), A630IR3Operand("iim", 0))),
      "store-local":("stl.u32", None, (A630IR3Operand("gpr", 39), A630IR3Operand("gpr", 13))),
      "barrier":("bar.g", None, ()),
      "load-local":("ldl.u32x4", A630IR3Operand("gpr", 44), (A630IR3Operand("gpr", 0),)),
      "predicate-true":("predt.p0", None, (A630IR3Operand("pred", 0),)),
      "predicate-end":("prede", None, ()),
    }
    for name,word in workgroup_words.items():
      instruction = decode_one(word)
      self.assertEqual((instruction.opcode, instruction.dst, instruction.srcs), workgroup_expected[name])
    # Scheduling is observationally inert in this synchronous model, but only the Cat6 layouts expose a legal SY bit.
    self.assertEqual(decode_one(workgroup_words["load-local"] & ~(1 << 60)).opcode, "ldl.u32x4")
    self.assertEqual(decode_one(workgroup_words["store-local"] | 1 << 60).opcode, "stl.u32")
    self.assertIsNone(decode_one(workgroup_words["compare"] & ~(0x7 << 48) | 5 << 48).opcode)
    self.assertIsNone(decode_one(workgroup_words["barrier"] & ~(1 << 54)).opcode)

    mov_shared = decode_one(0x200cc001000000c0)
    self.assertEqual((mov_shared.opcode, mov_shared.dst, mov_shared.srcs),
                     ("mov.u32", A630IR3Operand("gpr", 1), (A630IR3Operand("shared", 0xc0),)))
    mov_immediate = decode_one(0x204cc0033f800000)
    self.assertEqual((mov_immediate.opcode, mov_immediate.dst, mov_immediate.srcs),
                     ("mov.u32", A630IR3Operand("gpr", 3), (A630IR3Operand("uim", 0x3f800000),)))
    constant_moves = (
      (0x202cc00000000002, 0, 2), (0x202cc00100000003, 1, 3),
    )
    for word,dst,src in constant_moves:
      with self.subTest(word=f"{word:#x}"):
        self.assertEqual((decode_one(word).opcode, decode_one(word).dst, decode_one(word).srcs),
                         ("mov.u32", A630IR3Operand("gpr", dst), (A630IR3Operand("const", src),)))
    scheduled_constant = decode_one(0x202cc0bf000007ff | 1 << 44 | 1 << 60)
    self.assertEqual((scheduled_constant.opcode, scheduled_constant.dst, scheduled_constant.srcs),
                     ("mov.u32", A630IR3Operand("gpr", 0xbf), (A630IR3Operand("const", 0x7ff),)))
    self.assertTrue({("SY", 1), ("SS", 1)} <= set(scheduled_constant.fields))
    def gpr(value): return A630IR3Operand("gpr", value)
    binary_cases = {
      "add.u":(0x5218080b0010000b, gpr(11), (gpr(11), gpr(16))),
      "sub.u":(0x5258080200070002, gpr(2), (gpr(2), gpr(7))),
      "xor.b":(0x53f8080200070002, gpr(2), (gpr(2), gpr(7))),
      "and.b":(0x5398080200070002, gpr(2), (gpr(2), gpr(7))),
      "or.b":(0x53b8080200070002, gpr(2), (gpr(2), gpr(7))),
      "shr.b":(0x56f8080200070002, gpr(2), (gpr(2), gpr(7))),
      "max.s":(0x5338080200070002, gpr(2), (gpr(2), gpr(7))),
      "max.u":(0x5318080200070002, gpr(2), (gpr(2), gpr(7))),
    }
    binaries = {opcode:decode_one(word) for opcode,(word,_,_) in binary_cases.items()}
    for opcode,(_,dst,srcs) in binary_cases.items():
      self.assertEqual((binaries[opcode].opcode, binaries[opcode].dst, binaries[opcode].srcs), (opcode, dst, srcs))
      self.assertTrue({("SY", 1), ("NOP", 3)} <= set(binaries[opcode].fields))
    signed_compare = decode_one(0x52b8480000070002)
    unsigned_compare = decode_one(0x5298480000070002)
    equality_compare = decode_one(0x52bc480000070002)
    for compare,opcode in ((signed_compare, "cmps.s.lt"), (unsigned_compare, "cmps.u.lt")):
      self.assertEqual((compare.opcode, compare.dst, compare.srcs),
                       (opcode, A630IR3Operand("half", 0),
                        (A630IR3Operand("gpr", 2), A630IR3Operand("gpr", 7))))
      self.assertTrue({("SY", 1), ("NOP", 3), ("COND", 0), ("DST_HALF", 1)} <= set(compare.fields))
    self.assertEqual((equality_compare.opcode, equality_compare.dst, equality_compare.srcs),
                     ("cmps.s.eq", A630IR3Operand("half", 0),
                      (A630IR3Operand("gpr", 2), A630IR3Operand("gpr", 7))))
    self.assertTrue({("SY", 1), ("NOP", 3), ("COND", 4), ("DST_HALF", 1)} <= set(equality_compare.fields))
    predicate_compare = decode_one(0x42bb00f810040002)
    self.assertEqual((predicate_compare.opcode, predicate_compare.dst, predicate_compare.srcs),
                     ("cmps.s.ge.p0", A630IR3Operand("pred", 0),
                      (A630IR3Operand("gpr", 2), A630IR3Operand("const", 4))))
    self.assertTrue({("NAME", "cmps.s"), ("COND", 3), ("DST_HALF", 0), ("DST", 0xf8)} <= set(predicate_compare.fields))
    entry_branch,back_jump = (decode_one(word) for word in (0x0080000000000011, 0x01000000ffffffed))
    self.assertEqual((entry_branch.opcode, entry_branch.dst, entry_branch.srcs),
                     ("br.p0", None, (A630IR3Operand("pred", 0), A630IR3Operand("iim", 17))))
    self.assertEqual((back_jump.opcode, back_jump.dst, back_jump.srcs),
                     ("jump", None, (A630IR3Operand("iim", -19),)))
    self.assertIsNone(decode_one(entry_branch.raw | 1 << 52).opcode)
    unsupported_compare = decode_one(0x529c480000070002)
    self.assertIsNone(unsupported_compare.opcode)
    self.assertTrue({("NAME", "cmps.u"), ("COND", 4)} <= set(unsupported_compare.fields))
    byte_store = decode_one(0xc0cc0b0001800000)
    self.assertEqual((byte_store.opcode, byte_store.dst, byte_store.srcs),
                     ("stg.u8", None, (A630IR3Operand("gpr", 5), A630IR3Operand("half", 0))))
    self.assertTrue({("TYPE", 6), ("TYPE_HALF", 1), ("OFF", 0), ("SIZE", 1)} <= set(byte_store.fields))
    self.assertEqual(decode_one(byte_store.raw | 1 << 60).opcode, "stg.u8")
    vector_load = decode_one(0xc006000f04814001)
    vector_store = decode_one(0xc0c6130004800008)
    repeated_add = decode_one(0x50180b180014000f)
    self.assertEqual((vector_load.opcode, vector_load.dst, vector_load.srcs),
                     ("ldg.u32x4", A630IR3Operand("gpr", 15), (A630IR3Operand("gpr", 5),)))
    self.assertEqual((vector_store.opcode, vector_store.dst, vector_store.srcs),
                     ("stg.u32x4", None, (A630IR3Operand("gpr", 9), A630IR3Operand("gpr", 4))))
    self.assertEqual((repeated_add.opcode, repeated_add.dst, repeated_add.srcs),
                     ("add.f.rpt4", A630IR3Operand("gpr", 24),
                      (A630IR3Operand("gpr", 15), A630IR3Operand("gpr", 20))))
    self.assertTrue({("TYPE", 3), ("TYPE_HALF", 0), ("OFF", 0), ("SIZE", 4)} <= set(vector_load.fields))
    self.assertTrue({("TYPE", 3), ("TYPE_HALF", 0), ("OFF", 0), ("SIZE", 4)} <= set(vector_store.fields))
    self.assertEqual(tuple(value for field,value in repeated_add.fields if field == "SRC_R"), (1, 1))
    self.assertTrue({("REPEAT", 3), ("DST_HALF", 0)} <= set(repeated_add.fields))
    repeated_integer_add = decode_one(0x52180b0e000a0002)
    self.assertIsNone(repeated_integer_add.opcode)
    self.assertTrue({("NAME", "add.u"), ("REPEAT", 3)} <= set(repeated_integer_add.fields))
    self.assertEqual(tuple(value for field,value in repeated_integer_add.fields if field == "SRC_R"), (1, 1))
    self.assertIsNone(decode_one(vector_load.raw & ~(0x7 << 24)).opcode)
    self.assertIsNone(decode_one(repeated_add.raw & ~(1 << 51)).opcode)
    integer_multiply,first_cross_term,second_cross_term = decoded[10:13]
    self.assertEqual((integer_multiply.dst, integer_multiply.srcs),
                     (A630IR3Operand("gpr", 8), (A630IR3Operand("gpr", 2), A630IR3Operand("gpr", 7))))
    self.assertEqual((first_cross_term.dst, first_cross_term.srcs),
                     (A630IR3Operand("gpr", 8),
                      (A630IR3Operand("gpr", 2), A630IR3Operand("gpr", 7), A630IR3Operand("gpr", 8))))
    self.assertEqual((second_cross_term.dst, second_cross_term.srcs),
                     (A630IR3Operand("gpr", 7),
                      (A630IR3Operand("gpr", 7), A630IR3Operand("gpr", 2), A630IR3Operand("gpr", 8))))
    self.assertTrue({("SY", 1), ("NOP", 1)} <= set(integer_multiply.fields))
    self.assertIn(("NOP", 1), first_cross_term.fields)
    self.assertIn(("NOP", 3), second_cross_term.fields)

    self.assertEqual(decode_one(words[8] | 1 << 16).srcs[1], A630IR3Operand("flut", 3))
    representative_rejections = (
      words[3] | 1 << 42,                       # Cat2 saturate
      words[5] | 1 << 55,                       # Cat1 conversion rounding
      first_cross_term.raw | 1 << 40,           # Cat3 repeat
      entry_branch.raw | 1 << 52,                # Cat0 inverse branch
      words[7] | 1 << 59,                        # Cat6 jump target
      byte_store.raw & ~(0x7 << 24) | 2 << 24,   # Cat6 byte-store size
      constant_moves[0][0] | 0xe0 << 32,         # Cat1 special destination
    )
    for word in representative_rejections: self.assertIsNone(decode_one(word).opcode)
    with self.assertRaisesRegex(ValueError, "unmatched IR3 encoding at instruction 0"):
      decode_one(first_cross_term.raw | 1 << 13)
    with self.assertRaisesRegex(ValueError, "invalid or reserved IR3 encoding at instruction 0"):
      decode_one(words[7] | 1 << 41)

if __name__ == '__main__':
  unittest.main()

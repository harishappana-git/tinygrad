import unittest
from typing import Any, cast
from unittest import mock
from tinygrad.helpers import DEV
from test.mockgpu.qcom.qcom_test_base import _QCOMTestBase

class TestA630Execution(_QCOMTestBase):
  def test_production_add_machine_execution_and_retirement(self):
    import struct
    from tinygrad import Device, Tensor
    from tinygrad.codegen import to_program
    from tinygrad.engine.realize import get_runtime
    from tinygrad.runtime.ops_qcom import QCOMComputeQueue, QCOMProgram
    from tinygrad.runtime.autogen import kgsl, mesa
    from test.mockgpu.qcom import a630 as a630_module
    from test.mockgpu.qcom.a630 import execute_a630, stage_a630
    from test.mockgpu.qcom.pm4 import PM4Type7Packet, parse_pm4
    last_command = self.device.last_cmd
    source_values = [-7.5, -0.0, 1024.5]
    source = Tensor(source_values, device=Device.DEFAULT).realize()
    result = source + 1
    schedule_item = result.schedule_linear().src[-1]
    program_spec = to_program(schedule_item.src[0], self.device.renderer)
    runtime = get_runtime(self.device.device, program_spec)
    result_buffer, source_buffer = cast(Any, result.uop.buffer), cast(Any, source.uop.buffer)
    result_buffer.allocate()
    args = runtime.fill_kernargs([result_buffer._buf, source_buffer._buf])
    queue = self.device.hw_compute_queue_t()
    queue.wait(self.device.timeline_signal, self.device.timeline_value - 1)
    queue.memory_barrier()
    queue.exec(runtime, args, program_spec.arg.global_size, program_spec.arg.local_size)
    queue.timestamp(self.device.timeline_signal)
    queue.signal(self.device.timeline_signal, self.device.timeline_value)
    words = tuple(queue._q)
    packets = parse_pm4(words)
    submission = stage_a630(packets, lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))
    self.assertIs(type(runtime), QCOMProgram)
    self.assertIs(type(queue), QCOMComputeQueue)
    self.assertGreater(runtime.image_size, 0)
    self.assertEqual(runtime.image_size % 128, 0)
    self.assertTrue(any(isinstance(packet, PM4Type7Packet) and packet.opcode == mesa.CP_EXEC_CS for packet in packets))
    self.assertEqual(len(submission.dispatches), 1)
    dispatch = submission.dispatches[0]
    self.assertEqual((dispatch.shader_address, dispatch.shader_size), (int(runtime.lib_gpu.va_addr), runtime.image_size))
    self.assertEqual(dispatch.shader_image, bytes(self.driver.resolve_owned(self.device.fd.fd, dispatch.shader_address, dispatch.shader_size)))
    self.assertEqual((dispatch.constants_address, dispatch.constants_size), (int(args.buf.va_addr), 4096))
    self.assertEqual((dispatch.local_size, dispatch.groups), (tuple(program_spec.arg.local_size), tuple(program_spec.arg.global_size)))
    self.assertEqual(dispatch.global_size, tuple(g*l for g,l in zip(dispatch.groups, dispatch.local_size)))
    self.assertEqual(len(dispatch.instructions), runtime.image_size // 8)
    end = next(instruction for instruction in dispatch.instructions if instruction.opcode == "end")
    active = dispatch.instructions[:end.index+1]
    float_add = next(instruction for instruction in active if instruction.opcode == "add.f" and
                     any((operand.kind, operand.value) == ("flut", 2) for operand in instruction.srcs))
    flut_operand = next(operand for operand in float_add.srcs if (operand.kind, operand.value) == ("flut", 2))
    range_sizes = {(memory_range.purpose, memory_range.size) for memory_range in submission.memory_ranges}
    self.assertTrue({("wait value", 4), ("event value", 4), ("counter value", 8), ("constants", 4096),
                     ("shader", runtime.image_size)} <= range_sizes)
    self.assertEqual((submission.waits[0].address, submission.waits[0].mask),
                     (self.device.timeline_signal.value_addr, 0xffffffff))
    self.assertTrue(any(write.size == 8 and write.value is None for write in submission.writes))
    self.assertTrue(any(write.size == 4 and write.value == self.device.timeline_value for write in submission.writes))
    self.assertFalse(any(name in dispatch.__dataclass_fields__ for name in ("program", "uops", "python_program")))

    self.assertEqual(Device.DEFAULT, "QCOM")
    result_size, result_format = len(source_values) * 4, f"<{len(source_values)}f"
    result_view = self.driver.resolve_owned(self.device.fd.fd, int(result_buffer._buf.va_addr), result_size)
    result_view[:] = bytes(result_size)
    journal = execute_a630(submission, lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))
    self.assertEqual(bytes(result_view), bytes(result_size))
    self.assertEqual(tuple((write.address, len(write.data)) for write in journal),
                     tuple((int(result_buffer._buf.va_addr) + offset, 4) for offset in range(0, result_size, 4)))
    with mock.patch.object(a630_module, "_MAX_MEMORY_EVENTS", 0), \
         self.assertRaisesRegex(ValueError, "bounded memory-event limit"):
      execute_a630(submission, lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))
    self.assertEqual(bytes(result_view), bytes(result_size))
    lane_steps = len(active) * dispatch.groups[0] * dispatch.local_size[0]
    with mock.patch.object(a630_module, "_MAX_LANE_INSTRUCTION_STEPS", lane_steps):
      self.assertEqual(execute_a630(submission, self._resolve_owned), journal)
    with mock.patch.object(a630_module, "_MAX_LANE_INSTRUCTION_STEPS", lane_steps - 1), \
         self.assertRaisesRegex(ValueError, "bounded lane-instruction limit"):
      execute_a630(submission, self._resolve_owned)
    self.assertEqual(bytes(result_view), bytes(result_size))
    self.assertEqual(execute_a630(submission, self._resolve_owned), journal)
    for write in journal: self.driver.resolve_owned(self.device.fd.fd, write.address, len(write.data))[:] = write.data
    reference = cast(list[float], (Tensor(source_values, device="PYTHON") + 1).tolist())
    self.assertEqual(list(struct.unpack(result_format, result_view)), reference)

    shader_view = self.driver.resolve_owned(self.device.fd.fd, dispatch.shader_address, dispatch.shader_size)
    flut_source_index = float_add.srcs.index(flut_operand)
    flut_shift = 16 * flut_source_index
    # Pinned ir3-common.xml encodes FLUT in one 16-bit Cat2 source; FLUT[2]=1.0 and FLUT[3]=2.0.
    self.assertEqual(float_add.raw >> flut_shift & 0xffff, 0x2802)
    flut3_raw = float_add.raw & ~(0xffff << flut_shift) | 0x2803 << flut_shift

    constants_view = self.driver.resolve_owned(self.device.fd.fd, dispatch.constants_address, dispatch.constants_size)
    original_output_pointer = bytes(constants_view[:8])
    input_view = self.driver.resolve_owned(self.device.fd.fd, int(source_buffer._buf.va_addr), result_size)
    try:
      constants_view[:8] = constants_view[8:16]
      alias_submission = stage_a630(packets, self._resolve_owned)
      with self.assertRaisesRegex(ValueError, r"A630 global store aliases snapshotted A630 global load I\d+"):
        execute_a630(alias_submission, self._resolve_owned)
      alias_input_buffer, _, alias_input_request = self.gpu_command(words)
      alias_input_request.timestamp = 0x42424242
      alias_input_signal = self.driver.resolve_owned(self.device.fd.fd, int(self.device.timeline_signal.value_addr), 16)
      alias_input_dummy = self.driver.resolve_owned(self.device.fd.fd, self.device.dummy_addr, 4)
      alias_input_before = (bytes(result_view), bytes(input_view), bytes(alias_input_signal), bytes(alias_input_dummy),
                            self.driver.context_timestamps[self.device.ctx], self.driver.always_on_counter, self.device.last_cmd)
      with self.assertRaisesRegex(RuntimeError, r"A630 global store aliases snapshotted A630 global load I\d+"):
        kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=alias_input_request)
      self.assertEqual(alias_input_request.timestamp, 0x42424242)
      self.assertEqual((bytes(result_view), bytes(input_view), bytes(alias_input_signal), bytes(alias_input_dummy),
                        self.driver.context_timestamps[self.device.ctx], self.driver.always_on_counter, self.device.last_cmd), alias_input_before)
      self.device._gpu_free(alias_input_buffer)
    finally: constants_view[:8] = original_output_pointer

    # Pinned ir3_legalize.c records a non-local load destination in needs_sy and requires a consuming instruction to carry SY.
    self.assertEqual(tuple(value for field,value in float_add.fields if field == "SY"), (1,))
    missing_sy_raw = float_add.raw & ~(1 << 60)
    def schedule_state():
      return (bytes(result_view), bytes(input_view), bytes(constants_view), bytes(shader_view),
              bytes(self._resolve_owned(int(self.device.timeline_signal.value_addr), 16)), self.device.timeline_value,
              self.driver.context_timestamps[self.device.ctx], self.driver.always_on_counter, self.device.last_cmd,
              self.device.error_state, tuple(self.device.sig_prof_records), self.device.prof_exec_counter)
    with self._mutate_a630_replay(submission, words, ((float_add, missing_sy_raw),), timestamp=0x53594c44) as \
         (missing_sy_submission,missing_sy_dispatch,missing_sy_request):
      missing_sy = missing_sy_dispatch.instructions[float_add.index]
      self.assertEqual((missing_sy.opcode, missing_sy.dst, missing_sy.srcs), (float_add.opcode, float_add.dst, float_add.srcs))
      self.assertEqual(tuple(value for field,value in missing_sy.fields if field == "SY"), (0,))
      self._assert_a630_transactional_rejection(execute=execute_a630, submission=missing_sy_submission,
        request=missing_sy_request, message="global-load dependency lacks SY", marker=0x53594c44, state=schedule_state)

    store = next(instruction for instruction in active if instruction.opcode == "stg.u32")
    pointer_write = next(instruction for instruction in active if instruction.dst == store.srcs[0])
    padding = dispatch.instructions[end.index + 1]
    self.assertEqual((padding.opcode, padding.raw), ("nop", 0))
    with self._mutate_a630_replay(submission, words, ((end, pointer_write.raw), (padding, end.raw)), timestamp=0x53535752) as \
         (missing_ss_submission,missing_ss_dispatch,missing_ss_request):
      overwrite,moved_end = missing_ss_dispatch.instructions[end.index:end.index+2]
      self.assertEqual((overwrite.opcode, overwrite.dst, overwrite.srcs),
                       (pointer_write.opcode, pointer_write.dst, pointer_write.srcs))
      self.assertEqual(moved_end.opcode, "end")
      self._assert_a630_transactional_rejection(execute=execute_a630, submission=missing_ss_submission,
        request=missing_ss_request, message="memory-source overwrite lacks SS", marker=0x53535752, state=schedule_state)

    # A630's pinned compiler configuration requires six cycles from an ALU destination to a Cat6 store source.
    # The ADD.F NOP field contributes three cycles and the following repeated Cat0 NOP contributes the other three.
    delay_nop = active[float_add.index + 1]
    self.assertEqual((tuple(value for field,value in float_add.fields if field == "NOP"),
                      delay_nop.opcode, tuple(value for field,value in delay_nop.fields if field == "REPEAT")),
                     ((3,), "nop", (2,)))
    missing_delay_raw = float_add.raw & ~((1 << 43) | (1 << 51))
    with self._mutate_a630_replay(submission, words, ((float_add, missing_delay_raw),), timestamp=0x444c4159) as \
         (missing_delay_submission,missing_delay_dispatch,missing_delay_request):
      missing_delay = missing_delay_dispatch.instructions[float_add.index]
      self.assertEqual((missing_delay.opcode, missing_delay.dst, missing_delay.srcs),
                       (float_add.opcode, float_add.dst, float_add.srcs))
      self.assertEqual(tuple(value for field,value in missing_delay.fields if field == "NOP"), ())
      self._assert_a630_transactional_rejection(execute=execute_a630, submission=missing_delay_submission,
        request=missing_delay_request, message="fixed ALU dependency lacks delay slots", marker=0x444c4159, state=schedule_state)

    with self._edit_a630_shader(submission, ((float_add, flut3_raw),)):
      self.assertNotEqual(bytes(shader_view), dispatch.shader_image)
    # END's raw bit 32 is reserved by pinned ir3-cat0.xml and has no structured callback field.
    with self.assertRaisesRegex(ValueError, rf"invalid or reserved IR3 encoding at instruction {end.index}"):
      self._stage_a630_edits(submission, packets, ((end, end.raw ^ 1 << 32),))

    result_view[:] = bytes(result_size)
    signal_view = self.driver.resolve_owned(self.device.fd.fd, int(self.device.timeline_signal.value_addr), 16)
    self.assertEqual(self.device.timeline_signal.value, self.device.timeline_value - 1)
    signal_view[8:] = bytes(8)
    dummy_view = self.driver.resolve_owned(self.device.fd.fd, self.device.dummy_addr, 4)
    dummy_view[:] = b"A630"
    queue.bind(self.device)
    queue.submit_req.timestamp = 0x87654321
    previous_context_timestamp = self.driver.context_timestamps[self.device.ctx]
    queue.submit(self.device)
    retired_command = self.device.last_cmd
    self.assertGreater(retired_command, last_command)
    self.assertEqual((queue.submit_req.timestamp, retired_command, self.driver.context_timestamps[self.device.ctx]),
                     (previous_context_timestamp + 1,) * 3)
    self.assertEqual(list(struct.unpack(result_format, result_view)), reference)
    self.assertEqual(self.device.timeline_signal.value, self.device.timeline_value)
    first_counter = struct.unpack_from("<Q", signal_view, 8)[0]
    self.assertGreater(first_counter, 0)
    self.assertEqual(bytes(dummy_view), bytes(4))

    with self._edit_a630_shader(submission, ((float_add, flut3_raw),)):
      result_view[:] = bytes(result_size)
      queue.submit(self.device)
      self.assertEqual(self.device.last_cmd, retired_command + 1)
      self.assertEqual(list(struct.unpack(result_format, result_view)), [value + 1 for value in reference])
      mutated_counter = struct.unpack_from("<Q", signal_view, 8)[0]
      self.assertGreater(mutated_counter, first_counter)
      retired_command = self.device.last_cmd

    result_view[:] = bytes(result_size)
    queue.submit(self.device)
    self.assertEqual(self.device.last_cmd, retired_command + 1)
    self.assertGreater(struct.unpack_from("<Q", signal_view, 8)[0], mutated_counter)
    self.assertEqual(list(struct.unpack(result_format, result_view)), reference)
    retired_command = self.device.last_cmd

    end_to_end = (Tensor(source_values, device=Device.DEFAULT) + 1).realize()
    self.assertEqual(cast(list[float], end_to_end.tolist()), reference)
    self.assertGreater(self.device.last_cmd, retired_command)

  def test_production_two_segment_u32_copy_uses_mapped_machine_bytes(self):
    import struct
    from tinygrad import Device, Tensor
    from tinygrad.runtime.autogen import kgsl

    left_values = [1.25, -2.5, 3.75, -4.0, 5.5, -6.25, 7.0, -8.75]
    right_values = [9.5, -10.0, 11.25, -12.5, 13.0, -14.75, 15.5, -16.0]
    left,right = Tensor(left_values, device=Device.DEFAULT).realize(), Tensor(right_values, device=Device.DEFAULT).realize()
    python_reference = cast(list[float], Tensor.cat(Tensor(left_values, device="PYTHON"),
                                                    Tensor(right_values, device="PYTHON")).tolist())
    cpu_reference = cast(list[float], Tensor.cat(Tensor(left_values, device="CPU"), Tensor(right_values, device="CPU")).tolist())
    timeline_before = self.device.timeline_value
    with self._capture_a630_execution() as (submissions,command_images,real_execute):
      result = Tensor.cat(left, right).realize()
    actual = cast(list[float], result.tolist())

    self.assertEqual((Device.DEFAULT, (DEV.interface, DEV.device, DEV.renderer, DEV.arch)),
                     ("QCOM", ("MOCK", "QCOM", "IR3", "a630")))
    self.assertEqual((actual, python_reference, cpu_reference), (left_values + right_values,) * 3)
    self.assertEqual((self.device.timeline_value, self.device.timeline_signal.value), (timeline_before + 1, timeline_before))
    self.assertIsNone(self.device.error_state)
    self.assertEqual((len(submissions), len(command_images)), (1, 1))
    submission,dispatch = submissions[0],submissions[0].dispatches[0]
    self.assertEqual((dispatch.local_size, dispatch.groups, dispatch.global_size),
                     ((8, 1, 1), (1, 1, 1), (8, 1, 1)))
    stores = tuple(instruction for instruction in dispatch.instructions if instruction.opcode == "stg.u32")
    self.assertEqual(len(stores), 2)
    self.assertEqual(bytes(self._resolve_owned(dispatch.shader_address, dispatch.shader_size)), dispatch.shader_image)

    output_base,input0_base,input1_base = struct.unpack_from("<3Q", dispatch.constants_image)
    output = self._resolve_owned(output_base, 64)
    inputs = (self._resolve_owned(input0_base, 32), self._resolve_owned(input1_base, 32))
    self.assertEqual({tuple(struct.unpack("<8f", view)) for view in inputs}, {tuple(left_values), tuple(right_values)})
    original_output = bytes(output)
    output[:] = bytes([0x93]) * len(output)
    try:
      journal = real_execute(submission, self._resolve_owned)
      self.assertEqual(bytes(output), bytes([0x93]) * len(output))
      self.assertEqual(tuple(len(write.data) for write in journal), (4,) * 16)
      self.assertEqual(b"".join(write.data for write in sorted(journal, key=lambda write: write.address)),
                       struct.pack("<16f", *python_reference))
    finally: output[:] = original_output

    command_words = struct.unpack(f"<{len(command_images[0]) // 4}I", command_images[0])
    first_store,second_store = stores
    first_data,second_data = first_store.srcs[1],second_store.srcs[1]
    store_data_mask = 0xff << 1
    def store_data_edit(store, source):
      raw = store.raw & ~store_data_mask | source.value << 1
      self.assertEqual((raw ^ store.raw) & ~store_data_mask, 0)
      return store,raw
    swap_edits = (store_data_edit(first_store, second_data), store_data_edit(second_store, first_data))

    # Swapping only the two mapped store data operands preserves the validated copy bijection and reverses the segments.
    with self._mutate_a630_replay(submission, command_words, swap_edits) as (_,mutated_dispatch,request):
      mutated_stores = tuple(instruction for instruction in mutated_dispatch.instructions if instruction.opcode == "stg.u32")
      self.assertEqual(tuple(store.srcs[1] for store in mutated_stores), (second_data, first_data))
      self.assertTrue(all((mutated.opcode, mutated.srcs[0]) == (original.opcode, original.srcs[0]) and
                          (mutated.raw ^ original.raw) & ~store_data_mask == 0
                          for original,mutated in zip(stores, mutated_stores)))
      output[:] = bytes([0x94]) * len(output)
      timestamp_before,timeline_value_before,last_command = (self.driver.context_timestamps[self.device.ctx],
                                                              self.device.timeline_value, self.device.last_cmd)
      kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=request)
      self.assertEqual(list(struct.unpack("<16f", output)), right_values + left_values)
      retired_timestamp = (timestamp_before + 1) & 0xffffffff
      self.assertEqual((request.timestamp, self.driver.context_timestamps[self.device.ctx]),
                       (retired_timestamp, retired_timestamp))
      self.assertEqual((self.device.timeline_value, self.device.timeline_signal.value, self.device.last_cmd,
                        self.device.error_state), (timeline_value_before, timeline_value_before - 1, last_command, None))
    # Direct UAPI replay bypasses QCOMComputeQueue's assignment; keep the shared-device command invariant aligned.
    self.device.last_cmd = self.driver.context_timestamps[self.device.ctx]

  def test_production_vector_fill_and_add_use_mapped_machine_bytes(self):
    import struct
    from tinygrad import Device, Tensor

    self.assertEqual((Device.DEFAULT, DEV.interface, DEV.device, DEV.renderer, DEV.arch),
                     ("QCOM", "MOCK", "QCOM", "IR3", "a630"))

    # One grouped fill proves mapped workgroup execution, immutable journaling, and literal dependence.
    grouped_fill_size = 256
    with self._capture_a630_execution() as (grouped_fill_submissions,grouped_fill_commands,grouped_fill_real_execute):
      grouped_fill_tensor = Tensor.ones(grouped_fill_size, device=Device.DEFAULT).contiguous().realize()
    self.assertEqual((len(grouped_fill_submissions), len(grouped_fill_commands)), (1, 1))
    grouped_fill_submission = grouped_fill_submissions[0]
    grouped_fill_dispatch = grouped_fill_submission.dispatches[0]
    self.assertEqual((grouped_fill_dispatch.local_size, grouped_fill_dispatch.groups, grouped_fill_dispatch.global_size),
                     ((32, 1, 1), (2, 1, 1), (64, 1, 1)))
    grouped_fill_base = struct.unpack_from("<Q", grouped_fill_dispatch.constants_image)[0]
    grouped_fill_output = self._resolve_owned(grouped_fill_base, grouped_fill_size * 4)
    grouped_fill_python = Tensor.ones(grouped_fill_size, device="PYTHON").tolist()
    grouped_fill_cpu = Tensor.ones(grouped_fill_size, device="CPU").tolist()
    self.assertEqual(grouped_fill_tensor.tolist(), grouped_fill_python)
    self.assertEqual(grouped_fill_tensor.tolist(), grouped_fill_cpu)
    self.assertEqual(list(struct.unpack(f"<{grouped_fill_size}f", grouped_fill_output)), grouped_fill_python)
    grouped_fill_original = bytes(grouped_fill_output)
    grouped_fill_output[:] = bytes([0xa7]) * len(grouped_fill_output)
    try:
      grouped_fill_journal = grouped_fill_real_execute(grouped_fill_submission, self._resolve_owned)
      self.assertEqual(bytes(grouped_fill_output), bytes([0xa7]) * len(grouped_fill_output))
      self.assertEqual((len(grouped_fill_journal), tuple(len(write.data) for write in grouped_fill_journal)),
                       (64, (16,) * 64))
      self.assertEqual(b"".join(write.data for write in sorted(grouped_fill_journal, key=lambda write: write.address)),
                       struct.pack(f"<{grouped_fill_size}f", *([1.0] * grouped_fill_size)))
    finally: grouped_fill_output[:] = grouped_fill_original

    grouped_fill_words = struct.unpack(f"<{len(grouped_fill_commands[0]) // 4}I", grouped_fill_commands[0])
    grouped_fill_store = next(instruction for instruction in grouped_fill_dispatch.instructions if instruction.opcode == "stg.u32x4")
    grouped_fill_literal = next(instruction for instruction in grouped_fill_dispatch.instructions
      if instruction.opcode == "mov.u32" and instruction.srcs[0].kind == "uim" and instruction.dst is not None and
      instruction.dst.value == grouped_fill_store.srcs[1].value + 2)
    grouped_fill_literal_raw = grouped_fill_literal.raw & ~0xffffffff | 0x40000000
    grouped_fill_output[:] = bytes([0xb6]) * len(grouped_fill_output)
    try:
      with self._mutate_a630_replay(grouped_fill_submission, grouped_fill_words,
                                    ((grouped_fill_literal, grouped_fill_literal_raw),)) as \
           (mutated_submission,mutated_dispatch,_):
        mutated_literal = mutated_dispatch.instructions[grouped_fill_literal.index]
        self.assertEqual((mutated_literal.opcode, mutated_literal.srcs[0].kind, mutated_literal.srcs[0].value),
                         ("mov.u32", "uim", 0x40000000))
        mutated_journal = grouped_fill_real_execute(mutated_submission, self._resolve_owned)
        self.assertEqual(bytes(grouped_fill_output), bytes([0xb6]) * len(grouped_fill_output))
        self.assertEqual(b"".join(write.data for write in sorted(mutated_journal, key=lambda write: write.address)),
                         struct.pack(f"<{grouped_fill_size}f", *([1.0, 1.0, 2.0, 1.0] * (grouped_fill_size // 4))))
    finally: grouped_fill_output[:] = grouped_fill_original

    # One wide vector add covers the production numerical route and immutable 16-byte-per-lane writes.
    control_size = 128
    left_values = [(index - control_size//2) * 0.5 for index in range(control_size)]
    right_values = [(((index * 7) % 19) - 9) * 0.25 for index in range(control_size)]
    left_values[:4],right_values[:4] = ([1.0, 16777216.0, -1.0, -0.0], [2**-24, 1.0, 2**-24, -0.0])
    left,right = Tensor(left_values, device=Device.DEFAULT).realize(),Tensor(right_values, device=Device.DEFAULT).realize()
    python_reference = cast(list[float], (Tensor(left_values, device="PYTHON") + Tensor(right_values, device="PYTHON")).tolist())
    cpu_reference = cast(list[float], (Tensor(left_values, device="CPU") + Tensor(right_values, device="CPU")).tolist())
    with self._capture_a630_execution() as (submissions,command_images,real_execute): output_tensor = (left + right).realize()
    self.assertEqual((output_tensor.tolist(), python_reference, cpu_reference), (python_reference,) * 3)
    self.assertEqual((len(submissions), len(command_images)), (1, 1))
    submission,dispatch = submissions[0],submissions[0].dispatches[0]
    self.assertEqual((dispatch.global_size[0] * 4, dispatch.global_size[1:], dispatch.groups),
                     (control_size, (1, 1), (1, 1, 1)))
    output_base,input0_base,input1_base = struct.unpack_from("<3Q", dispatch.constants_image)
    output = self.driver.resolve_owned(self.device.fd.fd, output_base, control_size * 4)
    inputs = (self.driver.resolve_owned(self.device.fd.fd, input0_base, control_size * 4),
              self.driver.resolve_owned(self.device.fd.fd, input1_base, control_size * 4))
    original_output = bytes(output)
    output[:] = bytes([0x90]) * len(output)
    try:
      journal = real_execute(submission, self._resolve_owned)
      self.assertEqual(bytes(output), bytes([0x90]) * len(output))
      self.assertEqual(tuple(len(write.data) for write in journal), (16,) * dispatch.global_size[0])
      self.assertEqual(b"".join(write.data for write in sorted(journal, key=lambda write: write.address)),
                       struct.pack(f"<{control_size}f", *python_reference))
    finally: output[:] = original_output

    # The shader does not request FP32 denormal preservation or flushing, so an otherwise normal-input add that
    # cancels to a subnormal remains explicitly fail-closed.
    original_inputs = tuple(bytes(input_view) for input_view in inputs)
    try:
      inputs[0][:] = struct.pack(f"<{control_size}I", *([0x00800001] * control_size))
      inputs[1][:] = struct.pack(f"<{control_size}I", *([0x80800000] * control_size))
      with self.assertRaisesRegex(ValueError, "unsupported special or subnormal float result"):
        real_execute(submission, self._resolve_owned)
    finally:
      for input_view,original in zip(inputs, original_inputs): input_view[:] = original

    # Four elements use one invocation; overlapping RPT4 registers prove component writes execute in order.
    constant_left_values,constant_right_values = ([1.0, -4.0, 16.0, 0.5], [2.0, 1.5, -8.0, 0.25])
    constant_left = Tensor(constant_left_values, device=Device.DEFAULT).realize()
    constant_right = Tensor(constant_right_values, device=Device.DEFAULT).realize()
    constant_python = cast(list[float], (Tensor(constant_left_values, device="PYTHON") +
                                         Tensor(constant_right_values, device="PYTHON")).tolist())
    constant_cpu = cast(list[float], (Tensor(constant_left_values, device="CPU") +
                                      Tensor(constant_right_values, device="CPU")).tolist())
    with self._capture_a630_execution() as (constant_submissions,constant_commands,constant_real_execute):
      constant_result = (constant_left + constant_right).realize()
    self.assertEqual((constant_result.tolist(), constant_python, constant_cpu), (constant_python,) * 3)
    self.assertEqual((len(constant_submissions), len(constant_commands)), (1, 1))
    constant_submission = constant_submissions[0]
    constant_dispatch = constant_submission.dispatches[0]
    self.assertEqual((constant_dispatch.local_size, constant_dispatch.groups, constant_dispatch.global_size), ((1, 1, 1),) * 3)
    constant_output_base = struct.unpack_from("<Q", constant_dispatch.constants_image)[0]
    constant_output = self.driver.resolve_owned(self.device.fd.fd, constant_output_base, 16)
    constant_output[:] = bytes([0x9a]) * 16
    constant_journal = constant_real_execute(constant_submission, self._resolve_owned)
    self.assertEqual((bytes(constant_output), len(constant_journal), len(constant_journal[0].data)),
                     (bytes([0x9a]) * 16, 1, 16))
    for write in constant_journal: self._resolve_owned(write.address, len(write.data))[:] = write.data
    self.assertEqual(list(struct.unpack("<4f", constant_output)), constant_python)

    constant_words = struct.unpack(f"<{len(constant_commands[0]) // 4}I", constant_commands[0])
    # Pinned ir3_delay.c treats RPT as sequential component cycles. Shifting an advancing source one register below
    # the destination makes component 1 read component 0's result after one cycle, before its three-cycle ALU latency.
    repeated_add = next(instruction for instruction in constant_dispatch.instructions if instruction.opcode == "add.f.rpt4")
    assert repeated_add.dst is not None and repeated_add.srcs[0].kind == "gpr"
    overlap_source = repeated_add.dst.value - 1
    overlap_add_raw = repeated_add.raw & ~0xffff | overlap_source
    constant_input_bases = struct.unpack_from("<2Q", constant_dispatch.constants_image, 8)
    constant_inputs = tuple(self._resolve_owned(base, 16) for base in constant_input_bases)
    constant_shader = self._resolve_owned(constant_dispatch.shader_address, constant_dispatch.shader_size)
    constant_constants = self._resolve_owned(constant_dispatch.constants_address, constant_dispatch.constants_size)
    def repeat_state():
      return (bytes(constant_output), tuple(bytes(view) for view in constant_inputs), bytes(constant_shader), bytes(constant_constants),
              bytes(self._resolve_owned(int(self.device.timeline_signal.value_addr), 16)), self.device.timeline_value,
              self.driver.context_timestamps[self.device.ctx], self.driver.always_on_counter, self.device.last_cmd,
              self.device.error_state, tuple(self.device.sig_prof_records), self.device.prof_exec_counter)
    with self._mutate_a630_replay(constant_submission, constant_words, ((repeated_add, overlap_add_raw),), timestamp=0x52505434) as \
         (overlap_submission,overlap_dispatch,overlap_request):
      mutated_add = overlap_dispatch.instructions[repeated_add.index]
      self.assertEqual((mutated_add.srcs[0].value, mutated_add.dst), (overlap_source, repeated_add.dst))
      self._assert_a630_transactional_rejection(execute=constant_real_execute, submission=overlap_submission,
        request=overlap_request, message="fixed ALU dependency lacks delay slots", marker=0x52505434, state=repeat_state)

  def test_repeated_integer_vector_add_remains_fail_closed(self):
    from tinygrad import Device, Tensor, dtypes
    from tinygrad.runtime.support.hcq import HCQSubmissionRejected

    self.assertEqual((Device.DEFAULT, DEV.interface, DEV.device, DEV.renderer, DEV.arch),
                     ("QCOM", "MOCK", "QCOM", "IR3", "a630"))
    for dtype in (dtypes.int, dtypes.uint):
      left = Tensor([1, 2, 3, 4], dtype=dtype, device=Device.DEFAULT).realize()
      right = Tensor([5, 6, 7, 8], dtype=dtype, device=Device.DEFAULT).realize()
      def rejection_state():
        signal = self._resolve_owned(int(self.device.timeline_signal.value_addr), 16)
        return (bytes(signal), self.device.timeline_value, self.driver.context_timestamps[self.device.ctx],
                self.device.last_cmd, self.device.error_state, tuple(self.device.sig_prof_records))
      state_before = rejection_state()
      with self.subTest(dtype=dtype), self._capture_a630_execution() as (submissions,_,_), \
           self.assertRaisesRegex(HCQSubmissionRejected, "unsupported A630 semantic"):
        (left + right).realize()
      self.assertEqual((len(submissions), rejection_state(), left.tolist(), right.tolist()),
                       (1, state_before, [1, 2, 3, 4], [5, 6, 7, 8]))
      repeated = next(instruction for instruction in submissions[0].dispatches[0].instructions
                      if dict(instruction.fields).get("NAME") == "add.u" and dict(instruction.fields).get("REPEAT") == 3)
      self.assertIsNone(repeated.opcode)
      self.assertEqual(tuple(value for field,value in repeated.fields if field == "SRC_R"), (1, 1))
    self.assertEqual((Tensor([1.0, 2.0, 3.0, 4.0], device=Device.DEFAULT) +
                      Tensor([5.0, 6.0, 7.0, 8.0], device=Device.DEFAULT)).tolist(), [6.0, 8.0, 10.0, 12.0])

  def test_production_vector_add_two_workgroups_use_mapped_system_values(self):
    import struct
    from tinygrad import Device, Tensor

    size = 256
    left_values = [(index - 97) * 0.5 for index in range(size)]
    right_values = [(((index * 7) % 31) - 15) * 0.25 for index in range(size)]
    left = Tensor(left_values, device=Device.DEFAULT).realize()
    right = Tensor(right_values, device=Device.DEFAULT).realize()
    python_reference = (Tensor(left_values, device="PYTHON") + Tensor(right_values, device="PYTHON")).tolist()
    cpu_reference = (Tensor(left_values, device="CPU") + Tensor(right_values, device="CPU")).tolist()
    with self._capture_a630_execution() as (submissions,command_images,real_execute):
      result = (left + right).realize()
    actual = cast(list[float], result.tolist())

    self.assertEqual((Device.DEFAULT, DEV.interface, DEV.device, DEV.renderer, DEV.arch),
                     ("QCOM", "MOCK", "QCOM", "IR3", "a630"))
    self.assertEqual(actual, python_reference)
    self.assertEqual(actual, cpu_reference)
    self.assertNotEqual(actual[127], actual[128])
    self.assertEqual((len(submissions), len(command_images)), (1, 1))
    submission,dispatch = submissions[0],submissions[0].dispatches[0]
    self.assertEqual((dispatch.local_size, dispatch.groups, dispatch.global_size),
                     ((32, 1, 1), (2, 1, 1), (64, 1, 1)))
    self.assertEqual(bytes(self.driver.resolve_owned(self.device.fd.fd, dispatch.shader_address, dispatch.shader_size)),
                     dispatch.shader_image)
    output_base = struct.unpack_from("<Q", dispatch.constants_image)[0]
    byte_count = size * 4
    output = self.driver.resolve_owned(self.device.fd.fd, output_base, byte_count)
    output_original = bytes(output)
    output[:] = bytes([0xa6]) * byte_count
    try:
      journal = real_execute(submission, lambda address,length: self.driver.resolve_owned(self.device.fd.fd, address, length))
      self.assertEqual(bytes(output), bytes([0xa6]) * byte_count)
      self.assertEqual((len(journal), tuple(len(write.data) for write in journal)), (64, (16,) * 64))
      self.assertEqual(b"".join(write.data for write in sorted(journal, key=lambda write: write.address)),
                       struct.pack(f"<{size}f", *cast(list[float], python_reference)))
    finally: output[:] = output_original

  def test_production_integer_add_wraps_from_mapped_machine_bytes(self):
    from tinygrad import Device, Tensor, dtypes

    cases = ((dtypes.int, [dtypes.int.max, dtypes.int.min, -7], [1, -1, 3], [dtypes.int.min, dtypes.int.max, -4]),
             (dtypes.uint, [dtypes.uint.max, 0x80000000, 7], [1, 0x80000000, 5], [0, 0, 12]))
    actual = []
    with self._capture_a630_execution() as (submissions,command_images,_):
      for dtype,left,right,_ in cases:
        actual.append((Tensor(left, dtype=dtype, device=Device.DEFAULT) + Tensor(right, dtype=dtype, device=Device.DEFAULT)).tolist())
    reference = [(Tensor(left, dtype=dtype, device="PYTHON") + Tensor(right, dtype=dtype, device="PYTHON")).tolist()
                 for dtype,left,right,_ in cases]

    self.assertEqual((Device.DEFAULT, (DEV.interface, DEV.device, DEV.renderer, DEV.arch)),
                     ("QCOM", ("MOCK", "QCOM", "IR3", "a630")))
    self.assertEqual(actual, reference)
    self.assertEqual(actual, [expected for *_,expected in cases])
    self.assertEqual((len(submissions), len(command_images)), (len(cases), len(cases)))
    for (_,left,_,_),submission in zip(cases, submissions):
      dispatch = submission.dispatches[0]
      size = len(left)
      self.assertEqual((dispatch.local_size, dispatch.groups, dispatch.global_size), ((size, 1, 1), (1, 1, 1), (size, 1, 1)))
    self.assertEqual((Tensor([9], dtype=dtypes.int, device=Device.DEFAULT) +
                      Tensor([-4], dtype=dtypes.int, device=Device.DEFAULT)).tolist(), [5])

  def test_production_integer_subtract_wraps_from_mapped_machine_bytes(self):
    import operator
    from tinygrad import Device, Tensor, dtypes
    cases = ((dtypes.int, dtypes.int.min, 1, dtypes.int.max),
             (dtypes.uint, 0, 1, dtypes.uint.max),
             (dtypes.int, 9, 4, 5))
    # Reversing only the mapped sources changes 9 - 4 to 4 - 9 while preserving the valid SUB.U encoding.
    self._assert_production_integer_binary_uses_mapped_machine_bytes(
      tensor_operator=operator.sub, opcode="sub.u", opcode_bits=0x12, cases=cases,
      mutation_opcode="sub.u", mutation_opcode_bits=0x12, mutation_expected=(-5) & 0xffffffff, swap_mutation_sources=True)
    self.assertEqual((Tensor([9], dtype=dtypes.int, device=Device.DEFAULT) -
                      Tensor([4], dtype=dtypes.int, device=Device.DEFAULT)).tolist(), [5])

  def _assert_production_integer_binary_uses_mapped_machine_bytes(self, *, tensor_operator, opcode, opcode_bits, cases,
                                                                  mutation_opcode, mutation_opcode_bits, mutation_expected,
                                                                  swap_mutation_sources=False, unsupported_shift_counts=()):
    import struct
    from tinygrad import Device, Tensor
    from tinygrad.runtime.autogen import kgsl

    actual,live_tensors = [],[]
    with self._capture_a630_execution() as (submissions,command_images,real_execute):
      for dtype,left,right,_ in cases:
        lhs,rhs = Tensor([left], dtype=dtype, device=Device.DEFAULT).realize(), Tensor([right], dtype=dtype, device=Device.DEFAULT).realize()
        result = tensor_operator(lhs, rhs).realize()
        live_tensors.append((lhs, rhs, result))
        actual.append(result.tolist())
    reference = [tensor_operator(Tensor([left], dtype=dtype, device="PYTHON"), Tensor([right], dtype=dtype, device="PYTHON")).tolist()
                 for dtype,left,right,_ in cases]

    self.assertEqual((Device.DEFAULT, (DEV.interface, DEV.device, DEV.renderer, DEV.arch)),
                     ("QCOM", ("MOCK", "QCOM", "IR3", "a630")))
    self.assertEqual(actual, reference)
    self.assertEqual(actual, [[expected] for *_,expected in cases])
    self.assertEqual((len(submissions), len(command_images)), (len(cases), len(cases)))
    decoded_dispatches = []
    for submission in submissions:
      dispatch = submission.dispatches[0]
      self.assertEqual((dispatch.local_size, dispatch.groups, dispatch.global_size), ((1, 1, 1),) * 3)
      binary_instructions = tuple(instruction for instruction in dispatch.instructions if instruction.opcode == opcode)
      self.assertEqual(len(binary_instructions), 1)
      binary = binary_instructions[0]
      self.assertEqual((binary.raw >> 61, binary.raw >> 53 & 0x3f), (2, opcode_bits))
      self.assertIsNotNone(binary.dst)
      assert binary.dst is not None
      self.assertEqual(binary.dst.kind, "gpr")
      decoded_dispatches.append((dispatch, binary))

    # The newest command avoids replaying a stale historical EVENT_WRITE after the numerical matrix.
    case_index = len(cases) - 1
    self.assertNotEqual(mutation_expected, cases[case_index][3] & 0xffffffff)
    submission = submissions[case_index]
    dispatch,binary = decoded_dispatches[case_index]
    output_base = struct.unpack_from("<Q", dispatch.constants_image)[0]
    output = self.driver.resolve_owned(self.device.fd.fd, output_base, 4)
    if swap_mutation_sources:
      src1,src2 = binary.raw & 0xffff, binary.raw >> 16 & 0xffff
      mutation_raw = binary.raw & ~0xffffffff | src2 | src1 << 16
      mutation_srcs = (binary.srcs[1], binary.srcs[0])
      self.assertEqual(mutation_raw & ~0xffffffff, binary.raw & ~0xffffffff)
    else:
      mutation_raw = binary.raw & ~(0x3f << 53) | mutation_opcode_bits << 53
      mutation_srcs = binary.srcs
      self.assertEqual((mutation_raw >> 53 & 0x3f, mutation_raw & ~(0x3f << 53)),
                       (mutation_opcode_bits, binary.raw & ~(0x3f << 53)))
    command_words = struct.unpack(f"<{len(command_images[case_index]) // 4}I", command_images[case_index])
    timestamp_before = self.driver.context_timestamps[self.device.ctx]
    with self._mutate_a630_replay(submission, command_words, ((binary, mutation_raw),)) as (_,mutated_dispatch,request):
      mutated = mutated_dispatch.instructions[binary.index]
      self.assertEqual((mutated.opcode, mutated.dst, mutated.srcs), (mutation_opcode, binary.dst, mutation_srcs))
      output[:] = struct.pack("<I", mutation_expected ^ 0xffffffff)
      kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=request)
      self.assertEqual(struct.unpack("<I", output)[0], mutation_expected)
      self.assertEqual((request.timestamp, self.driver.context_timestamps[self.device.ctx]), ((timestamp_before + 1) & 0xffffffff,) * 2)

    if unsupported_shift_counts:
      rhs_base = struct.unpack_from("<3Q", dispatch.constants_image)[2]
      rhs = self.driver.resolve_owned(self.device.fd.fd, rhs_base, 4)
      original_rhs = bytes(rhs)
      try:
        for index,count in enumerate(unsupported_shift_counts):
          with self.subTest(unsupported_shift_count=count):
            rhs[:] = struct.pack("<I", count)
            request_buffer,_,request = self.gpu_command(struct.unpack(f"<{len(command_images[case_index]) // 4}I", command_images[case_index]))
            request.timestamp = marker = 0x16180339 + index
            signal = self.driver.resolve_owned(self.device.fd.fd, int(self.device.timeline_signal.value_addr), 16)
            output[:] = struct.pack("<I", count ^ 0xffffffff)
            state_before = (bytes(output), bytes(signal), self.driver.context_timestamps[self.device.ctx],
                            self.driver.always_on_counter, self.device.last_cmd, self.device.error_state)
            try:
              with self.assertRaisesRegex(ValueError, "shift count is outside the supported 0..31 range"):
                real_execute(submission, lambda address,length: self.driver.resolve_owned(self.device.fd.fd, address, length))
              with self.assertRaisesRegex(RuntimeError, "shift count is outside the supported 0..31 range"):
                kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=request)
              self.assertEqual((request.timestamp, bytes(output), bytes(signal), self.driver.context_timestamps[self.device.ctx],
                                self.driver.always_on_counter, self.device.last_cmd, self.device.error_state), (marker, *state_before))
            finally: self.device._gpu_free(request_buffer)
      finally: rhs[:] = original_rhs

  def test_production_integer_xor_uses_mapped_machine_bytes(self):
    import operator
    from tinygrad import Device, Tensor, dtypes
    cases = ((dtypes.int, -1431655766, 252645135, -1515870811),
             (dtypes.uint, 0x80000000, 0x7fffffff, dtypes.uint.max),
             (dtypes.uint, dtypes.uint.max, 1, 0xfffffffe))
    # The opcode-only ADD.U mutation changes uint.max XOR 1 from 0xfffffffe to wrapped zero.
    self._assert_production_integer_binary_uses_mapped_machine_bytes(
      tensor_operator=operator.xor, opcode="xor.b", opcode_bits=0x1f, cases=cases,
      mutation_opcode="add.u", mutation_opcode_bits=0x10, mutation_expected=0)
    self.assertEqual(operator.xor(Tensor([0xaaaaaaaa], dtype=dtypes.uint, device=Device.DEFAULT),
                                  Tensor([0x0f0f0f0f], dtype=dtypes.uint, device=Device.DEFAULT)).tolist(), [0xa5a5a5a5])

  def test_production_integer_and_uses_mapped_machine_bytes(self):
    import operator
    from tinygrad import Device, Tensor, dtypes
    cases = ((dtypes.int, dtypes.int.min, dtypes.int.max, 0),
             (dtypes.int, dtypes.int.min, -1, dtypes.int.min),
             (dtypes.uint, 0xaaaaaaaa, 0x0f0f0f0f, 0x0a0a0a0a),
             (dtypes.uint, dtypes.uint.max, 0x80000001, 0x80000001))
    # The opcode-only XOR.B mutation changes uint.max AND 0x80000001 from 0x80000001 to 0x7ffffffe.
    self._assert_production_integer_binary_uses_mapped_machine_bytes(
      tensor_operator=operator.and_, opcode="and.b", opcode_bits=0x1c, cases=cases,
      mutation_opcode="xor.b", mutation_opcode_bits=0x1f, mutation_expected=0x7ffffffe)
    self.assertEqual(operator.and_(Tensor([0xaaaaaaaa], dtype=dtypes.uint, device=Device.DEFAULT),
                                   Tensor([0x0f0f0f0f], dtype=dtypes.uint, device=Device.DEFAULT)).tolist(), [0x0a0a0a0a])

  def test_production_integer_or_uses_mapped_machine_bytes(self):
    import operator
    from tinygrad import Device, Tensor, dtypes
    cases = ((dtypes.int, dtypes.int.min, dtypes.int.max, -1),
             (dtypes.int, dtypes.int.min, 0, dtypes.int.min),
             (dtypes.uint, 0xaaaaaaaa, 0x0f0f0f0f, 0xafafafaf),
             (dtypes.uint, dtypes.uint.max, 0x80000001, dtypes.uint.max))
    # The opcode-only AND.B mutation changes uint.max OR 0x80000001 from uint.max to 0x80000001.
    self._assert_production_integer_binary_uses_mapped_machine_bytes(
      tensor_operator=operator.or_, opcode="or.b", opcode_bits=0x1d, cases=cases,
      mutation_opcode="and.b", mutation_opcode_bits=0x1c, mutation_expected=0x80000001)
    self.assertEqual(operator.or_(Tensor([0xaaaaaaaa], dtype=dtypes.uint, device=Device.DEFAULT),
                                  Tensor([0x0f0f0f0f], dtype=dtypes.uint, device=Device.DEFAULT)).tolist(), [0xafafafaf])

  def test_production_unsigned_right_shift_uses_mapped_machine_bytes(self):
    import operator
    from tinygrad import Device, Tensor, dtypes
    cases = ((dtypes.uint, 0x80000001, 1, 0x40000000),
             (dtypes.uint, dtypes.uint.max, 16, 0x0000ffff),
             (dtypes.uint, 16, 2, 4))
    unsupported_counts = (32, dtypes.uint.max)
    self.assertEqual([operator.rshift(Tensor([0x80000001], dtype=dtypes.uint, device="PYTHON"),
                                      Tensor([count], dtype=dtypes.uint, device="PYTHON")).tolist() for count in unsupported_counts],
                     [[0], [0]])
    # Swapping only the mapped SHR.B sources changes 16 >> 2 to 2 >> 16, while keeping both counts in the supported range.
    self._assert_production_integer_binary_uses_mapped_machine_bytes(
      tensor_operator=operator.rshift, opcode="shr.b", opcode_bits=0x37, cases=cases,
      mutation_opcode="shr.b", mutation_opcode_bits=0x37, mutation_expected=0, swap_mutation_sources=True,
      unsupported_shift_counts=unsupported_counts)
    self.assertEqual(operator.rshift(Tensor([0x80000001], dtype=dtypes.uint, device=Device.DEFAULT),
                                     Tensor([1], dtype=dtypes.uint, device=Device.DEFAULT)).tolist(), [0x40000000])

  def test_production_signed_maximum_uses_mapped_machine_bytes(self):
    from tinygrad import Device, Tensor, dtypes
    cases = ((dtypes.int, dtypes.int.min, dtypes.int.max, dtypes.int.max),
             (dtypes.int, -1431655766, 252645135, 252645135),
             (dtypes.int, -7, 3, 3))
    # The opcode-only OR.B mutation changes signed max(-7, 3) from 3 to the raw bit pattern for -5.
    self._assert_production_integer_binary_uses_mapped_machine_bytes(
      tensor_operator=Tensor.maximum, opcode="max.s", opcode_bits=0x19, cases=cases,
      mutation_opcode="or.b", mutation_opcode_bits=0x1d, mutation_expected=0xfffffffb)
    self.assertEqual(Tensor([-7], dtype=dtypes.int, device=Device.DEFAULT).maximum(
      Tensor([3], dtype=dtypes.int, device=Device.DEFAULT)).tolist(), [3])

  def test_production_unsigned_maximum_uses_mapped_machine_bytes(self):
    from tinygrad import Device, Tensor, dtypes
    cases = ((dtypes.uint, 0, dtypes.uint.max, dtypes.uint.max),
             (dtypes.uint, 0xaaaaaaaa, 0x0f0f0f0f, 0xaaaaaaaa),
             (dtypes.uint, 0x80000000, 0x7fffffff, 0x80000000))
    # The opcode-only OR.B mutation changes unsigned max(0x80000000, 0x7fffffff) to uint.max.
    self._assert_production_integer_binary_uses_mapped_machine_bytes(
      tensor_operator=Tensor.maximum, opcode="max.u", opcode_bits=0x18, cases=cases,
      mutation_opcode="or.b", mutation_opcode_bits=0x1d, mutation_expected=dtypes.uint.max)
    self.assertEqual(Tensor([0x80000000], dtype=dtypes.uint, device=Device.DEFAULT).maximum(
      Tensor([0x7fffffff], dtype=dtypes.uint, device=Device.DEFAULT)).tolist(), [0x80000000])

  def test_production_integer_multiply_wraps_from_mapped_machine_bytes(self):
    import struct
    from tinygrad import Device, Tensor, dtypes
    from tinygrad.runtime.autogen import kgsl

    cases = ((dtypes.int, dtypes.int.min, -1, dtypes.int.min),
             (dtypes.int, dtypes.int.max, 2, -2),
             (dtypes.uint, 0x00010002, 0x00030004, 0x000a0008))
    actual,live_tensors = [],[]
    with self._capture_a630_execution() as (submissions,command_images,real_execute):
      for dtype,left,right,_ in cases:
        lhs,rhs = Tensor([left], dtype=dtype, device=Device.DEFAULT).realize(), Tensor([right], dtype=dtype, device=Device.DEFAULT).realize()
        result = (lhs * rhs).realize()
        live_tensors.append((lhs, rhs, result))
        actual.append(result.tolist())
    reference = [(Tensor([left], dtype=dtype, device="PYTHON") * Tensor([right], dtype=dtype, device="PYTHON")).tolist()
                 for dtype,left,right,_ in cases]

    self.assertEqual((Device.DEFAULT, (DEV.interface, DEV.device, DEV.renderer, DEV.arch)),
                     ("QCOM", ("MOCK", "QCOM", "IR3", "a630")))
    self.assertEqual(actual, reference)
    self.assertEqual(actual, [[expected] for *_,expected in cases])
    self.assertEqual((len(submissions), len(command_images)), (len(cases), len(cases)))
    decoded_dispatches = []
    for submission in submissions:
      dispatch = submission.dispatches[0]
      self.assertEqual((dispatch.local_size, dispatch.groups, dispatch.global_size), ((1, 1, 1),) * 3)
      accumulates = tuple(instruction for instruction in dispatch.instructions if instruction.opcode == "madsh.m16")
      self.assertGreaterEqual(len(accumulates), 1)
      decoded_dispatches.append((dispatch, accumulates[0]))

    case_index = len(cases) - 1
    submission = submissions[case_index]
    dispatch,first = decoded_dispatches[case_index]
    output_base = struct.unpack_from("<Q", dispatch.constants_image)[0]
    output = self.driver.resolve_owned(self.device.fd.fd, output_base, 4)

    # Swapping the first MADSH inputs preserves the legal dataflow but changes which high-half cross term is accumulated.
    src1,src2 = first.raw & 0x1fff, first.raw >> 47 & 0xff
    swapped_raw = first.raw & ~(0x1fff | (0xff << 47)) | src2 | src1 << 47
    command_words = struct.unpack(f"<{len(command_images[case_index]) // 4}I", command_images[case_index])
    timestamp_before = self.driver.context_timestamps[self.device.ctx]
    with self._mutate_a630_replay(submission, command_words, ((first, swapped_raw),)) as (_,mutated_dispatch,request):
      mutated = mutated_dispatch.instructions[first.index]
      self.assertEqual((mutated.opcode, mutated.srcs), ("madsh.m16", (first.srcs[1], first.srcs[0], first.srcs[2])))
      output[:] = bytes(4)
      kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=request)
      self.assertEqual(struct.unpack("<I", output)[0], 0x00080008)
      self.assertEqual((request.timestamp, self.driver.context_timestamps[self.device.ctx]), ((timestamp_before + 1) & 0xffffffff,) * 2)

    self.assertEqual((Tensor([9], dtype=dtypes.int, device=Device.DEFAULT) *
                      Tensor([4], dtype=dtypes.int, device=Device.DEFAULT)).tolist(), [36])

  def _assert_production_integer_comparison_uses_mapped_machine_bytes(self, *, tensor_operator, cases, opcode_by_dtype, condition,
                                                                      mutation_mask, mutation_value, mutation_opcode, mutation_expected):
    import struct
    from tinygrad import Device, Tensor
    from tinygrad.runtime.autogen import kgsl

    actual,live_tensors = [],[]
    with self._capture_a630_execution() as (submissions,command_images,_):
      for dtype,left,right,_ in cases:
        lhs,rhs = Tensor([left], dtype=dtype, device=Device.DEFAULT).realize(), Tensor([right], dtype=dtype, device=Device.DEFAULT).realize()
        result = tensor_operator(lhs, rhs).realize()
        live_tensors.append((lhs, rhs, result))
        actual.append(result.tolist())
    reference = [tensor_operator(Tensor([left], dtype=dtype, device="PYTHON"), Tensor([right], dtype=dtype, device="PYTHON")).tolist()
                 for dtype,left,right,_ in cases]

    self.assertEqual((Device.DEFAULT, (DEV.interface, DEV.device, DEV.renderer, DEV.arch)),
                     ("QCOM", ("MOCK", "QCOM", "IR3", "a630")))
    self.assertEqual(actual, reference)
    self.assertEqual(actual, [[expected] for *_,expected in cases])
    self.assertEqual((len(submissions), len(command_images)), (len(cases), len(cases)))
    decoded_dispatches = []
    for (dtype,_,_,_),submission in zip(cases, submissions):
      dispatch = submission.dispatches[0]
      self.assertEqual((dispatch.local_size, dispatch.groups, dispatch.global_size), ((1, 1, 1),) * 3)
      expected_opcode = opcode_by_dtype[dtype]
      comparisons = tuple(instruction for instruction in dispatch.instructions if instruction.opcode == expected_opcode)
      self.assertEqual(len(comparisons), 1)
      self.assertIn(("COND", condition), comparisons[0].fields)
      decoded_dispatches.append((dispatch, comparisons[0]))

    # Replaying only the newest capture avoids a stale historical EVENT_WRITE after the numerical matrix.
    case_index = len(cases) - 1
    self.assertIn(mutation_expected, (0, 1))
    self.assertNotEqual(bool(mutation_expected), cases[case_index][3])
    submission = submissions[case_index]
    dispatch,comparison = decoded_dispatches[case_index]
    output_base = struct.unpack_from("<Q", dispatch.constants_image)[0]
    output = self.driver.resolve_owned(self.device.fd.fd, output_base, 1)
    self.assertEqual(mutation_value & ~mutation_mask, 0)
    mutation_raw = comparison.raw & ~mutation_mask | mutation_value
    self.assertEqual(mutation_raw & ~mutation_mask, comparison.raw & ~mutation_mask)
    self.assertNotEqual(mutation_raw, comparison.raw)
    command_words = struct.unpack(f"<{len(command_images[case_index]) // 4}I", command_images[case_index])
    timestamp_before = self.driver.context_timestamps[self.device.ctx]
    with self._mutate_a630_replay(submission, command_words, ((comparison, mutation_raw),)) as (_,mutated_dispatch,request):
      mutated = mutated_dispatch.instructions[comparison.index]
      self.assertEqual((mutated.opcode, mutated.dst, mutated.srcs), (mutation_opcode, comparison.dst, comparison.srcs))
      output[:] = bytes([mutation_expected ^ 0xff])
      kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=request)
      self.assertEqual(bytes(output), bytes([mutation_expected]))
      self.assertEqual((request.timestamp, self.driver.context_timestamps[self.device.ctx]), ((timestamp_before + 1) & 0xffffffff,) * 2)

  def test_production_integer_less_than_uses_mapped_comparison_bytes(self):
    import operator
    from tinygrad import Device, Tensor, dtypes
    cases = ((dtypes.int, dtypes.int.min, dtypes.int.max, True),
             (dtypes.int, dtypes.int.max, dtypes.int.min, False),
             (dtypes.uint, 0x7fffffff, 0x80000000, True))
    # Setting only the mapped signedness bit changes the unsigned boundary comparison from true to false.
    self._assert_production_integer_comparison_uses_mapped_machine_bytes(
      tensor_operator=operator.lt, cases=cases, opcode_by_dtype={dtypes.int:"cmps.s.lt", dtypes.uint:"cmps.u.lt"}, condition=0,
      mutation_mask=1 << 53, mutation_value=1 << 53, mutation_opcode="cmps.s.lt", mutation_expected=0)
    self.assertEqual((Tensor([-1], dtype=dtypes.int, device=Device.DEFAULT) <
                      Tensor([0], dtype=dtypes.int, device=Device.DEFAULT)).tolist(), [True])

  def test_production_integer_equality_uses_mapped_comparison_bytes(self):
    import operator
    from tinygrad import Device, Tensor, dtypes
    cases = ((dtypes.int, dtypes.int.min, dtypes.int.min, True),
             (dtypes.uint, dtypes.uint.max, 0, False),
             (dtypes.uint, dtypes.uint.max, dtypes.uint.max, True))
    # Equality is signedness-independent; changing only COND from EQ to LT changes the latest equal words to false.
    self._assert_production_integer_comparison_uses_mapped_machine_bytes(
      tensor_operator=operator.eq, cases=cases, opcode_by_dtype={dtypes.int:"cmps.s.eq", dtypes.uint:"cmps.s.eq"}, condition=4,
      mutation_mask=0x7 << 48, mutation_value=0, mutation_opcode="cmps.s.lt", mutation_expected=0)
    self.assertEqual((Tensor([-1], dtype=dtypes.int, device=Device.DEFAULT) ==
                      Tensor([-1], dtype=dtypes.int, device=Device.DEFAULT)).tolist(), [True])

  def test_production_symbolic_workgroups_execute_mapped_system_values(self):
    import struct
    from dataclasses import replace
    from tinygrad import Device, Tensor, Variable
    from tinygrad.runtime.autogen import mesa
    from test.mockgpu.qcom.a630 import decode_a630_ir3

    values = [-7.5, 0.25, 1024.0, -0.0, 3.5, 19.0, -2.0, 8.25, 11.0, -4.5]
    size = Variable("qcom_symbolic_size", 1, len(values))
    source = Tensor(values, device=Device.DEFAULT).realize()
    with self._capture_a630_execution() as (submissions,_,real_execute):
      actual = (source[:size.bind(5)] + 1).contiguous()[:5].tolist()
    reference = cast(list[float], (Tensor(values[:5], device="PYTHON") + 1).tolist())

    self.assertEqual((Device.DEFAULT, (DEV.interface, DEV.device, DEV.renderer, DEV.arch)),
                     ("QCOM", ("MOCK", "QCOM", "IR3", "a630")))
    self.assertEqual(actual, reference)
    self.assertEqual(len(submissions), 2)
    multi_add = next(submission for submission in submissions
                     if any(instruction.opcode == "add.f" for instruction in submission.dispatches[0].instructions))
    dispatch = multi_add.dispatches[0]
    self.assertEqual((dispatch.local_size, dispatch.groups, dispatch.global_size),
                     ((1, 1, 1), (5, 1, 1), (5, 1, 1)))
    system = dict(dispatch.registers)[mesa.REG_A6XX_SP_CS_CONST_CONFIG_0]
    self.assertEqual(system & 0xff, 0xc0)
    def with_register(register, value):
      return replace(dispatch, registers=tuple((reg, value if reg == register else current) for reg,current in dispatch.registers))
    with self.assertRaisesRegex(ValueError, "lacks a workgroup-id mapping"):
      real_execute(replace(multi_add, dispatches=(with_register(mesa.REG_A6XX_SP_CS_CONST_CONFIG_0, system & ~0xff | 0xfc),)),
                   self._resolve_owned)
    with self.assertRaisesRegex(ValueError, "bounded one-dimensional"):
      real_execute(replace(multi_add, dispatches=(replace(dispatch, groups=(0, 1, 1), global_size=(0, 1, 1)),)),
                   self._resolve_owned)

    shader = self.driver.resolve_owned(self.device.fd.fd, dispatch.shader_address, dispatch.shader_size)
    shared_instructions = tuple(instruction for instruction in dispatch.instructions
                                if any(operand.kind == "shared" for operand in instruction.srcs))
    shared_original = bytes(shader)
    try:
      for shared_instruction in shared_instructions:
        self.assertEqual(shader[shared_instruction.index*8], 0xc0)
        shader[shared_instruction.index*8] = 0xc4
      image = bytes(shader)
      renamed_registers = tuple((reg, system & ~0xff | 0xc4 if reg == mesa.REG_A6XX_SP_CS_CONST_CONFIG_0 else current)
                                for reg,current in dispatch.registers)
      renamed_dispatch = replace(dispatch, shader_image=image, instructions=decode_a630_ir3(image), registers=renamed_registers)
      output_base = struct.unpack_from("<Q", dispatch.constants_image)[0]
      output = self._resolve_owned(output_base, 20)
      output_before = bytes(output)
      renamed_journal = real_execute(replace(multi_add, dispatches=(renamed_dispatch,)), self._resolve_owned)
      self.assertEqual(bytes(output), output_before)
      self.assertEqual(tuple(struct.unpack("<f", write.data)[0] for write in renamed_journal), tuple(reference))
    finally: shader[:] = shared_original

  def test_production_symbolic_reduce_executes_mapped_scalar_control_flow(self):
    import struct
    from dataclasses import replace
    from tinygrad import Device, Tensor, Variable
    from tinygrad.runtime.autogen import mesa
    from test.mockgpu.qcom.a630 import decode_a630_ir3

    values = [1.5, -0.5, 2.0, 4.0, -1.0, 3.0, 0.25, 8.0, -4.0, 16.0]
    bounds = (2, 5)
    symbolic = Variable("qcom_symbolic_reduce_bound", 1, 10)
    source = Tensor(values, device=Device.DEFAULT).realize()
    with self._capture_a630_execution() as (submissions,_,real_execute):
      actual = [source[:symbolic.bind(bound)].sum().item() for bound in bounds]
    python_reference = [Tensor(values[:bound], device="PYTHON").sum().item() for bound in bounds]
    cpu_reference = [Tensor(values[:bound], device="CPU").sum().item() for bound in bounds]

    self.assertEqual((Device.DEFAULT, (DEV.interface, DEV.device, DEV.renderer, DEV.arch)),
                     ("QCOM", ("MOCK", "QCOM", "IR3", "a630")))
    self.assertEqual(actual, python_reference)
    self.assertEqual(actual, cpu_reference)
    self.assertEqual(len(submissions), len(bounds))
    dispatches = tuple(submission.dispatches[0] for submission in submissions)
    self.assertTrue(all((dispatch.local_size, dispatch.groups, dispatch.global_size) ==
                        ((1, 1, 1), (1, 1, 1), (1, 1, 1)) for dispatch in dispatches))
    self.assertEqual(len({dispatch.shader_image for dispatch in dispatches}), 1)
    self.assertEqual(tuple(struct.unpack_from("<I", dispatch.constants_image, 16)[0] for dispatch in dispatches), bounds)
    for dispatch in dispatches:
      control = dict(dispatch.registers)[mesa.REG_A6XX_SP_CS_CNTL_0]
      system = dict(dispatch.registers)[mesa.REG_A6XX_SP_CS_CONST_CONFIG_0]
      self.assertEqual((control >> 14 & 0x3f, system & 0xff, system >> 24 & 0xff), (0, 0xfc, 0xfc))

    for submission,dispatch,bound,reference in zip(submissions, dispatches, bounds, python_reference):
      observed_reads:list[tuple[int, int, str]] = []
      journal = real_execute(submission, self._resolve_owned,
                             read_observer=lambda address,size,purpose: observed_reads.append((address, size, purpose)))
      input_base = struct.unpack_from("<Q", dispatch.constants_image, 8)[0]
      self.assertEqual([(address, size) for address,size,_ in observed_reads],
                       [(input_base + index * 4, 4) for index in range(bound)])
      self.assertTrue(all(purpose.startswith("A630 global load I") for _,_,purpose in observed_reads))
      self.assertEqual((len(journal), struct.unpack("<f", journal[0].data)[0]), (1, reference))

    selected = 1
    submission,dispatch = submissions[selected],dispatches[selected]
    shader = self._resolve_owned(dispatch.shader_address, dispatch.shader_size)

    accumulator_add = next(instruction for instruction in dispatch.instructions if instruction.opcode == "add.f")
    accumulator_seed = next(instruction for instruction in dispatch.instructions if instruction.opcode == "mov.u32" and
                            instruction.dst == accumulator_add.dst and instruction.srcs[0].kind == "uim")
    seed_raw = accumulator_seed.raw & ~0xffffffff | 0x3f800000
    with self._edit_a630_shader(submission, ((accumulator_seed, seed_raw),)):
      image = bytes(shader)
      mutated_dispatch = replace(dispatch, shader_image=image, instructions=decode_a630_ir3(image))
      mutated_submission = replace(submission, dispatches=(mutated_dispatch,))
      mutated_journal = real_execute(mutated_submission, self._resolve_owned)
      self.assertEqual(struct.unpack("<f", mutated_journal[0].data)[0], python_reference[selected] + 1.0)
      self.assertEqual(source[:symbolic.bind(bounds[selected])].sum().item(), python_reference[selected] + 1.0)
    self.assertEqual(bytes(shader), dispatch.shader_image)
    self.assertEqual(source[:symbolic.bind(bounds[selected])].sum().item(), python_reference[selected])

  def test_production_sum_executes_mapped_workgroup_local_reduction(self):
    import struct
    from tinygrad import Device, Tensor
    from tinygrad.runtime.autogen import kgsl

    values = [float(index) for index in range(1, 257)]
    source = Tensor(values, device=Device.DEFAULT).realize()
    with self._capture_a630_execution() as (submissions,command_images,real_execute):
      actual = source.sum().item()
    python_reference = Tensor(values, device="PYTHON").sum().item()
    cpu_reference = Tensor(values, device="CPU").sum().item()

    self.assertEqual((Device.DEFAULT, (DEV.interface, DEV.device, DEV.renderer, DEV.arch)),
                     ("QCOM", ("MOCK", "QCOM", "IR3", "a630")))
    self.assertEqual((actual, python_reference, cpu_reference), (32896.0,) * 3)
    self.assertEqual((len(submissions), len(command_images)), (1, 1))
    submission,dispatch = submissions[0],submissions[0].dispatches[0]
    self.assertEqual((dispatch.local_size, dispatch.groups, dispatch.global_size),
                     ((16, 1, 1), (1, 1, 1), (16, 1, 1)))
    output_base,input_base = struct.unpack_from("<2Q", dispatch.constants_image)
    output = self._resolve_owned(output_base, 4)
    input_view = self._resolve_owned(input_base, 1024)
    constants = self._resolve_owned(dispatch.constants_address, dispatch.constants_size)
    shader = self._resolve_owned(dispatch.shader_address, dispatch.shader_size)
    self.assertEqual((bytes(shader), tuple(struct.unpack("<256f", input_view))), (dispatch.shader_image, tuple(values)))
    observed_reads:list[tuple[int, int, str]] = []
    output_before = bytes(output)
    journal = real_execute(submission, self._resolve_owned,
                           read_observer=lambda address,size,purpose: observed_reads.append((address, size, purpose)))
    self.assertEqual(bytes(output), output_before)
    self.assertEqual(sorted((address, size) for address,size,_ in observed_reads),
                     [(input_base + index * 4, 4) for index in range(256)])
    self.assertTrue(all(purpose.startswith("A630 global load I") for _,_,purpose in observed_reads))
    self.assertEqual(tuple((write.address, struct.unpack("<f", write.data)[0]) for write in journal), ((output_base, 32896.0),))

    command_words = struct.unpack(f"<{len(command_images[0]) // 4}I", command_images[0])
    signal = self._resolve_owned(int(self.device.timeline_signal.value_addr), 16)
    local_offset = next(instruction for instruction in dispatch.instructions if instruction.opcode == "mov.u32" and
                        len(instruction.srcs) == 1 and instruction.srcs[0].kind == "uim" and instruction.srcs[0].value == 16)
    offset_mutation = local_offset.raw & ~0xffffffff | 12
    with self._mutate_a630_replay(submission, command_words, ((local_offset, offset_mutation),)) as \
         (mutated_submission,mutated_dispatch,request):
      mutated = mutated_dispatch.instructions[local_offset.index]
      self.assertEqual((mutated.opcode, mutated.dst, tuple((source.kind, source.value) for source in mutated.srcs)),
                       ("mov.u32", local_offset.dst, (("uim", 12),)))
      mutated_journal = real_execute(mutated_submission, self._resolve_owned)
      self.assertEqual(struct.unpack("<f", mutated_journal[0].data)[0], 31872.0)
      output[:] = bytes(4)
      timestamp_before = self.driver.context_timestamps[self.device.ctx]
      signal_before = bytes(signal)
      try:
        kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=request)
        self.assertEqual((struct.unpack("<f", output)[0], self.driver.context_timestamps[self.device.ctx]),
                         (31872.0, timestamp_before + 1))
      finally: signal[:] = signal_before
    self.device.last_cmd = self.driver.context_timestamps[self.device.ctx]
    self.assertEqual(bytes(shader), dispatch.shader_image)
    self.assertEqual(source.sum().item(), 32896.0)

    output[:] = struct.pack("<f", 32896.0)
    def retirement_state():
      return (bytes(output), bytes(input_view), bytes(constants), bytes(shader), bytes(signal), self.device.timeline_value,
              self.driver.context_timestamps[self.device.ctx], self.driver.always_on_counter, self.device.last_cmd,
              self.device.error_state, tuple(self.device.sig_prof_records), self.device.prof_exec_counter)

    barrier = next(instruction for instruction in dispatch.instructions if instruction.opcode == "bar.g")
    with self._mutate_a630_replay(submission, command_words, ((barrier, 0),), timestamp=0x57524241) as \
         (barrierless_submission,barrierless_dispatch,barrierless_request):
      self.assertEqual(barrierless_dispatch.instructions[barrier.index].opcode, "nop")
      self._assert_a630_transactional_rejection(execute=real_execute, submission=barrierless_submission,
        request=barrierless_request, message="multi-lane local memory requires one store/barrier/load phase",
        marker=0x57524241, state=retirement_state)

    def schedule_flag(instruction, name):
      values = tuple(value for field,value in instruction.fields if field == name)
      self.assertTrue(not values or values in ((0,), (1,)))
      return values[0] if values else 0
    first_local_load = next(instruction for instruction in dispatch.instructions if instruction.opcode == "ldl.u32x4")
    post_barrier_sync = next(instruction for instruction in dispatch.instructions[barrier.index+1:first_local_load.index]
                             if schedule_flag(instruction, "SS"))
    self.assertEqual((post_barrier_sync.opcode, schedule_flag(first_local_load, "SY")), ("nop", 1))
    sync_mutations = (
      ("barrier SS", post_barrier_sync, post_barrier_sync.raw & ~(1 << 44), "barrier synchronization lacks SS"),
      ("barrier SY", first_local_load, first_local_load.raw & ~(1 << 60), "barrier synchronization lacks SY"),
    )
    for ordinal,(name,instruction,raw,message) in enumerate(sync_mutations):
      marker = 0x57525310 + ordinal
      with self.subTest(sync_mutation=name), \
           self._mutate_a630_replay(submission, command_words, ((instruction, raw),), timestamp=marker) as \
           (mutated_submission,mutated_dispatch,mutated_request):
        mutated = mutated_dispatch.instructions[instruction.index]
        self.assertEqual((mutated.opcode, mutated.dst, mutated.srcs),
                         (instruction.opcode, instruction.dst, instruction.srcs))
        self._assert_a630_transactional_rejection(execute=real_execute, submission=mutated_submission,
          request=mutated_request, message=message, marker=marker, state=retirement_state)
    self.assertEqual((bytes(shader), source.sum().item()), (dispatch.shader_image, 32896.0))

if __name__ == '__main__':
  unittest.main()

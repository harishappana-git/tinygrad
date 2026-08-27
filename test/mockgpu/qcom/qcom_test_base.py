import contextlib, ctypes, unittest
from typing import Any
from unittest import mock
from tinygrad.helpers import DEV

QCOM_MOCK = DEV.interface == "MOCK" and DEV.device == "QCOM" and DEV.renderer == "IR3" and DEV.arch == "a630"

@unittest.skipUnless(QCOM_MOCK, "requires DEV=MOCK+QCOM:IR3:a630")
class _QCOMTestBase(unittest.TestCase):
  device:Any
  driver:Any

  @classmethod
  def setUpClass(cls):
    from tinygrad.device import Device
    cls.device = Device["QCOM"]
    from test.mockgpu import mockgpu
    from test.mockgpu.qcom.qcomdriver import QCOMDriver
    cls.driver = next(driver for driver in mockgpu.drivers if isinstance(driver, QCOMDriver))

  def allocation_for(self, addr:int, size:int):
    return next((allocation for allocation in self.driver.allocations.values()
                 if allocation.addr is not None and allocation.addr <= addr and addr + size <= allocation.addr + allocation.size), None)

  def _resolve_owned(self, address, size):
    return self.driver.resolve_owned(self.device.fd.fd, address, size)

  def gpu_command(self, words):
    from tinygrad.runtime.autogen import kgsl
    buffer = self.device._gpu_alloc(len(words) * 4, fill_zeroes=True)
    (ctypes.c_uint32 * len(words)).from_address(int(buffer.va_addr))[:] = words
    command = kgsl.struct_kgsl_command_object(gpuaddr=int(buffer.va_addr), size=len(words) * 4, flags=kgsl.KGSL_CMDLIST_IB)
    request = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(command), cmdsize=ctypes.sizeof(command), numcmds=1,
                                           context_id=self.device.ctx)
    return buffer, command, request

  @contextlib.contextmanager
  def _capture_a630_execution(self):
    from test.mockgpu.qcom import qcomdriver
    submissions,command_images = [],[]
    real_execute = qcomdriver.execute_a630
    real_plan = self.driver._plan_a630_retirement
    def capture_execution(submission, resolver, *, read_observer=None, budget=None):
      submissions.append(submission)
      return real_execute(submission, resolver, read_observer=read_observer, budget=budget)
    def capture_plan(fd, submission, command_address, command_size):
      command_images.append(bytes(self.driver.resolve_owned(fd, command_address, command_size)))
      return real_plan(fd, submission, command_address, command_size)
    with mock.patch.object(qcomdriver, "execute_a630", side_effect=capture_execution), \
         mock.patch.object(self.driver, "_plan_a630_retirement", side_effect=capture_plan):
      yield submissions,command_images,real_execute

  @contextlib.contextmanager
  def _edit_a630_shader(self, submission, edits):
    import struct
    edits = tuple(edits)
    self.assertEqual(len(submission.dispatches), 1)
    self.assertTrue(edits)
    dispatch = submission.dispatches[0]
    shader = self._resolve_owned(dispatch.shader_address, dispatch.shader_size)
    self.assertEqual(bytes(shader), dispatch.shader_image)
    self.assertEqual(len({instruction.index for instruction,_ in edits}), len(edits))
    for instruction,raw in edits:
      self.assertTrue(0 <= instruction.index < len(dispatch.instructions))
      self.assertLessEqual((instruction.index + 1) * 8, len(shader))
      self.assertEqual(dispatch.instructions[instruction.index], instruction)
      self.assertNotEqual(raw, instruction.raw)
      self.assertTrue(0 <= raw < 1 << 64)
    try:
      for instruction,raw in edits: struct.pack_into("<Q", shader, instruction.index * 8, raw)
      yield
    finally:
      for instruction,_ in edits: struct.pack_into("<Q", shader, instruction.index * 8, instruction.raw)
      self.assertEqual(bytes(shader), dispatch.shader_image)

  def _stage_a630_edits(self, submission, packets, edits):
    from test.mockgpu.qcom.a630 import stage_a630
    with self._edit_a630_shader(submission, edits): return stage_a630(packets, self._resolve_owned)

  @contextlib.contextmanager
  def _mutate_a630_replay(self, submission, command_words, edits, *, timestamp=None):
    from dataclasses import replace
    from test.mockgpu.qcom.a630 import decode_a630_ir3
    dispatch = submission.dispatches[0]
    shader = self._resolve_owned(dispatch.shader_address, dispatch.shader_size)
    request_buffer,_,request = self.gpu_command(command_words)
    if timestamp is not None: request.timestamp = timestamp
    try:
      with self._edit_a630_shader(submission, edits):
        image = bytes(shader)
        mutated_dispatch = replace(dispatch, shader_image=image, instructions=decode_a630_ir3(image))
        yield replace(submission, dispatches=(mutated_dispatch,)),mutated_dispatch,request
    finally:
      self.device._gpu_free(request_buffer)

  def _assert_a630_transactional_rejection(self, *, execute, submission, request, message, marker, state):
    from tinygrad.runtime.autogen import kgsl
    before = state()
    with self.assertRaisesRegex(ValueError, message): execute(submission, self._resolve_owned)
    with self.assertRaisesRegex(RuntimeError, message): kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=request)
    self.assertEqual((request.timestamp, state()), (marker, before))

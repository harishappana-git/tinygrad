import contextlib, ctypes, functools, mmap, os, unittest
from typing import Any, cast
from unittest import mock
from tinygrad.helpers import DEV, mv_address

QCOM_MOCK = DEV.interface == "MOCK" and DEV.device == "QCOM" and DEV.renderer == "IR3" and DEV.arch == "a630"

@unittest.skipUnless(QCOM_MOCK, "requires DEV=MOCK+QCOM:IR3:a630")
class TestQCOMDriver(unittest.TestCase):
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

  def test_production_backend_identity_and_initialization(self):
    from tinygrad.device import Device
    from tinygrad.renderer.nir import IR3Renderer
    from tinygrad.runtime.ops_qcom import QCOMAllocator, QCOMComputeQueue, QCOMDevice, QCOMProgram
    from test.mockgpu.mockgpu import MockFileIOInterface, tracked_fds
    from test.mockgpu.qcom.qcomdriver import KGSLFileDesc, QCOMDriver
    self.assertEqual(Device.DEFAULT, "QCOM")
    self.assertEqual((DEV.interface, DEV.device, DEV.renderer, DEV.arch), ("MOCK", "QCOM", "IR3", "a630"))
    self.assertIs(type(self.device), QCOMDevice)
    self.assertIs(type(self.device.renderer), IR3Renderer)
    self.assertIs(type(self.device.allocator), QCOMAllocator)
    self.assertIs(self.device.runtime_t, QCOMProgram)
    self.assertIsInstance(self.driver, QCOMDriver)
    self.assertIs(type(self.device.fd), MockFileIOInterface)
    self.assertIs(type(tracked_fds[self.device.fd.fd]), KGSLFileDesc)
    self.assertEqual(self.device.fd.path, "/dev/kgsl-3d0")
    self.assertEqual(self.device.gpu_id, (6, 3, 0))
    self.assertEqual(self.device.arch, "a630")
    self.assertEqual((self.device.renderer.target.interface, self.device.renderer.target.device,
                      self.device.renderer.target.renderer, self.device.renderer.target.arch), ("MOCK", "QCOM", "IR3", "a630"))
    self.assertIn(self.device.ctx, self.driver.contexts)
    self.assertEqual(self.driver.power_levels[self.device.ctx], 1)
    self.assertIsInstance(self.device.hw_compute_queue_t, functools.partial)
    self.assertIs(self.device.hw_compute_queue_t.func, QCOMComputeQueue)
    for buffer in (self.device.cmd_buf, self.device.border_color_buf, self.device.kernargs_buf, self.device.timeline_signal.base_buf):
      self.assertIsNotNone(self.allocation_for(int(buffer.va_addr), buffer.size))
    self.assertIsNotNone(self.allocation_for(self.device.dummy_addr, 0x1000))

  def test_allocate_map_and_free(self):
    buffer = self.device._gpu_alloc(0x1234, fill_zeroes=True)
    allocation_id = buffer.meta[0].id
    self.assertEqual(self.driver.allocations[allocation_id].size, 0x2000)
    self.assertEqual(self.driver.allocations[allocation_id].addr, buffer.va_addr)
    self.assertEqual(bytes(buffer.cpu_view().mv[:buffer.size]), bytes(buffer.size))
    buffer.cpu_view().mv[:4] = b"A630"
    self.assertEqual(bytes(buffer.cpu_view().mv[:4]), b"A630")
    self.assertEqual(bytes(self.driver.resolve(int(buffer.va_addr), 4)), b"A630")
    self.assertEqual(len(self.driver.resolve(int(buffer.va_addr) + buffer.meta[0].mmapsize - 1, 1)), 1)
    with self.assertRaisesRegex(RuntimeError, "unmapped or ambiguous GPU range"):
      self.driver.resolve(int(buffer.va_addr) + buffer.meta[0].mmapsize, 1)
    with self.assertRaisesRegex(RuntimeError, "invalid GPU range"):
      self.driver.resolve(int(buffer.va_addr), 0)
    self.device._gpu_free(buffer)
    self.assertNotIn(allocation_id, self.driver.allocations)
    with self.assertRaisesRegex(RuntimeError, "unmapped or ambiguous GPU range"):
      self.driver.resolve(int(buffer.va_addr), 1)
    with self.assertRaisesRegex(RuntimeError, "unknown allocation"):
      self.device._gpu_free(buffer)

  def test_external_map_and_free(self):
    from tinygrad.runtime.autogen import kgsl
    from test.mockgpu.qcom.qcomdriver import ioctl_code
    backing = bytearray(0x3000)
    mapped_start = (mv_address(memoryview(backing)) + 0xfff) & ~0xfff
    mapping_count = len(self.driver.user_mappings)
    with self.assertRaisesRegex(RuntimeError, "unsupported user-memory fields"):
      kgsl.IOCTL_KGSL_MAP_USER_MEM(self.device.fd, fd=1, hostptr=mapped_start, len=0x1000, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR)
    with self.assertRaisesRegex(RuntimeError, "unsupported user-memory type"):
      kgsl.IOCTL_KGSL_MAP_USER_MEM(self.device.fd, hostptr=mapped_start, len=0x1000, memtype=kgsl.KGSL_USER_MEM_TYPE_PMEM)
    with self.assertRaisesRegex(RuntimeError, "unaligned user-memory range"):
      kgsl.IOCTL_KGSL_MAP_USER_MEM(self.device.fd, hostptr=mapped_start + 1, len=0x1000, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR)
    with self.assertRaisesRegex(RuntimeError, "overflowing user-memory range"):
      kgsl.IOCTL_KGSL_MAP_USER_MEM(self.device.fd, hostptr=(1 << 64) - 0x1000, len=0x2000, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR)
    self.assertEqual(len(self.driver.user_mappings), mapping_count)

    mapping = kgsl.IOCTL_KGSL_MAP_USER_MEM(self.device.fd, hostptr=mapped_start, len=0x1000, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR)
    self.assertEqual(mapping.gpuaddr, mapped_start)
    self.assertIn(mapped_start, self.driver.user_mappings)
    self.assertEqual(len(self.driver.resolve(mapped_start, 0x1000)), 0x1000)
    with self.assertRaisesRegex(RuntimeError, "overlapping user-memory range"):
      kgsl.IOCTL_KGSL_MAP_USER_MEM(self.device.fd, hostptr=mapped_start, len=0x1000, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR)
    with self.assertRaisesRegex(RuntimeError, "unknown user mapping"):
      kgsl.IOCTL_KGSL_SHAREDMEM_FREE(self.device.fd, gpuaddr=mapped_start + 1)
    other_fd = self.driver.open('/dev/kgsl-3d0', os.O_RDWR, 0, self.driver.tracked_files[0])
    shared_free = kgsl.struct_kgsl_sharedmem_free(gpuaddr=mapped_start)
    with self.assertRaisesRegex(RuntimeError, "belongs to another descriptor"):
      other_fd.ioctl(other_fd.fd, ioctl_code(kgsl.IOCTL_KGSL_SHAREDMEM_FREE), ctypes.addressof(shared_free))
    other_fd.close(other_fd.fd)
    kgsl.IOCTL_KGSL_SHAREDMEM_FREE(self.device.fd, gpuaddr=mapped_start)
    self.assertNotIn(mapped_start, self.driver.user_mappings)
    with self.assertRaisesRegex(RuntimeError, "unknown user mapping"):
      kgsl.IOCTL_KGSL_SHAREDMEM_FREE(self.device.fd, gpuaddr=mapped_start)

  def test_context_lifecycle(self):
    from tinygrad.runtime.autogen import kgsl
    context = kgsl.IOCTL_KGSL_DRAWCTXT_CREATE(self.device.fd, flags=self.driver.contexts[self.device.ctx][1])
    self.assertNotEqual(context.drawctxt_id, self.device.ctx)
    self.assertIn(context.drawctxt_id, self.driver.contexts)
    self.assertEqual(self.driver.context_timestamps[context.drawctxt_id], 0)
    kgsl.IOCTL_KGSL_DRAWCTXT_DESTROY(self.device.fd, drawctxt_id=context.drawctxt_id)
    self.assertNotIn(context.drawctxt_id, self.driver.contexts)
    self.assertNotIn(context.drawctxt_id, self.driver.context_timestamps)
    with self.assertRaisesRegex(RuntimeError, "unknown context"):
      kgsl.IOCTL_KGSL_DRAWCTXT_DESTROY(self.device.fd, drawctxt_id=context.drawctxt_id)

  def test_waittimestamp_context_and_wrap_contract(self):
    import struct
    from tinygrad.runtime.autogen import kgsl, mesa
    from tinygrad.runtime.ops_qcom import pkt7_hdr
    from test.mockgpu.qcom.qcomdriver import ioctl_code
    wait_code = ioctl_code(kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID)
    self.assertEqual((ctypes.sizeof(kgsl.struct_kgsl_device_waittimestamp_ctxtid), wait_code), (12, 0x400c0907))
    layout = kgsl.struct_kgsl_device_waittimestamp_ctxtid(context_id=0x11223344, timestamp=0x55667788, timeout=0x99aabbcc)
    self.assertEqual(ctypes.string_at(ctypes.addressof(layout), ctypes.sizeof(layout)), struct.pack("<III", 0x11223344, 0x55667788, 0x99aabbcc))
    state = (dict(self.driver.contexts), dict(self.driver.context_timestamps), dict(self.driver.allocations),
             dict(self.driver.user_mappings), dict(self.driver.power_levels), self.driver.always_on_counter, self.device.last_cmd)
    for malformed in (wait_code ^ (1 << 30), wait_code ^ (1 << 16), wait_code ^ (1 << 8), wait_code ^ 1):
      with self.assertRaisesRegex(RuntimeError, "unsupported KGSL ioctl"): self.device.fd.ioctl(malformed, layout)
    self.assertEqual((ctypes.string_at(ctypes.addressof(layout), ctypes.sizeof(layout)),
                      self.driver.contexts, self.driver.context_timestamps, self.driver.allocations,
                      self.driver.user_mappings, self.driver.power_levels, self.driver.always_on_counter, self.device.last_cmd),
                     (struct.pack("<III", 0x11223344, 0x55667788, 0x99aabbcc), *state))

    comparisons = ((0, 0, 0), (7, 6, 1), (7, 8, -1), (0, 0xffffffff, 1),
                   (0xffffffff, 0, -1), (0, 0x80000000, 1), (0x80000000, 0, -1))
    self.assertEqual([self.driver._timestamp_cmp(a, b) for a,b,_ in comparisons], [result for _,_,result in comparisons])

    context = kgsl.IOCTL_KGSL_DRAWCTXT_CREATE(self.device.fd, flags=self.driver.contexts[self.device.ctx][1])
    other_fd = self.driver.open('/dev/kgsl-3d0', os.O_RDWR, 0, self.driver.tracked_files[0])
    try:
      for retired,requested,comparison in comparisons:
        self.driver.context_timestamps[context.drawctxt_id] = retired
        for timeout in (0, 1, 0xffffffff):
          wait = kgsl.struct_kgsl_device_waittimestamp_ctxtid(context_id=context.drawctxt_id, timestamp=requested, timeout=timeout)
          before = (ctypes.string_at(ctypes.addressof(wait), ctypes.sizeof(wait)), dict(self.driver.contexts),
                    dict(self.driver.context_timestamps), self.driver.always_on_counter, self.device.last_cmd)
          if comparison >= 0: self.assertEqual(self.device.fd.ioctl(wait_code, wait), 0)
          else:
            with self.assertRaisesRegex(RuntimeError, f"future timestamp {requested:#x} cannot progress synchronously"):
              self.device.fd.ioctl(wait_code, wait)
          self.assertEqual((ctypes.string_at(ctypes.addressof(wait), ctypes.sizeof(wait)), self.driver.contexts,
                            self.driver.context_timestamps, self.driver.always_on_counter, self.device.last_cmd), before)

      foreign = kgsl.struct_kgsl_device_waittimestamp_ctxtid(context_id=context.drawctxt_id, timestamp=0, timeout=0)
      foreign_before = ctypes.string_at(ctypes.addressof(foreign), ctypes.sizeof(foreign))
      with self.assertRaisesRegex(RuntimeError, "belongs to another descriptor"):
        other_fd.ioctl(other_fd.fd, wait_code, ctypes.addressof(foreign))
      self.assertEqual(ctypes.string_at(ctypes.addressof(foreign), ctypes.sizeof(foreign)), foreign_before)

      self.driver.context_timestamps[context.drawctxt_id] = 0xfffffffe
      buffer, _, request = self.gpu_command([pkt7_hdr(mesa.CP_WAIT_FOR_IDLE, 0)])
      request.context_id = context.drawctxt_id
      try:
        kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=request)
        self.assertEqual((request.timestamp, self.driver.context_timestamps[context.drawctxt_id]), (0xffffffff, 0xffffffff))
        kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID(self.device.fd, context_id=context.drawctxt_id,
                                                    timestamp=0xffffffff, timeout=0xffffffff)
        with self.assertRaisesRegex(RuntimeError, "future timestamp 0x0 cannot progress synchronously"):
          kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID(self.device.fd, context_id=context.drawctxt_id, timestamp=0, timeout=1)

        kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=request)
        self.assertEqual((request.timestamp, self.driver.context_timestamps[context.drawctxt_id]), (0, 0))
        for timestamp in (0xffffffff, 0):
          kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID(self.device.fd, context_id=context.drawctxt_id,
                                                      timestamp=timestamp, timeout=0xffffffff)
      finally:
        self.device._gpu_free(buffer)
    finally:
      other_fd.close(other_fd.fd)
      kgsl.IOCTL_KGSL_DRAWCTXT_DESTROY(self.device.fd, drawctxt_id=context.drawctxt_id)

    with self.assertRaisesRegex(RuntimeError, "unknown context"):
      kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID(self.device.fd, context_id=context.drawctxt_id, timestamp=0, timeout=0)

  def test_descriptor_binding_and_close_cleanup(self):
    from tinygrad.runtime.autogen import kgsl
    from test.mockgpu.qcom.qcomdriver import ioctl_code
    descriptor = self.driver.open('/dev/kgsl-3d0', os.O_RDWR, 0, self.driver.tracked_files[0])
    flags = kgsl.KGSL_MEMFLAGS_USE_CPU_MAP | (12 << kgsl.KGSL_MEMALIGN_SHIFT)
    allocation = kgsl.struct_kgsl_gpuobj_alloc(size=0x1000, mmapsize=0x1000, flags=flags)
    descriptor.ioctl(descriptor.fd, ioctl_code(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC), ctypes.addressof(allocation))
    address = descriptor.mmap(0, 0x1000, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, descriptor.fd, allocation.id * 0x1000)

    context = kgsl.struct_kgsl_drawctxt_create(flags=self.driver.contexts[self.device.ctx][1])
    descriptor.ioctl(descriptor.fd, ioctl_code(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE), ctypes.addressof(context))
    level = kgsl.struct_kgsl_device_constraint_pwrlevel(level=kgsl.KGSL_CONSTRAINT_PWR_MAX)
    constraint = kgsl.struct_kgsl_device_constraint(type=kgsl.KGSL_CONSTRAINT_PWRLEVEL, context_id=context.drawctxt_id,
                                                     data=ctypes.addressof(level), size=ctypes.sizeof(level))
    prop = kgsl.struct_kgsl_device_getproperty(type=kgsl.KGSL_PROP_PWR_CONSTRAINT,
                                               value=ctypes.addressof(constraint), sizebytes=ctypes.sizeof(constraint))
    descriptor.ioctl(descriptor.fd, ioctl_code(kgsl.IOCTL_KGSL_SETPROPERTY), ctypes.addressof(prop))

    backing = bytearray(0x3000)
    external_address = (mv_address(memoryview(backing)) + 0xfff) & ~0xfff
    external = kgsl.struct_kgsl_map_user_mem(hostptr=external_address, len=0x1000, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR)
    descriptor.ioctl(descriptor.fd, ioctl_code(kgsl.IOCTL_KGSL_MAP_USER_MEM), ctypes.addressof(external))
    self.assertEqual(len(self.driver.resolve(address, 1)), 1)
    self.assertEqual(len(self.driver.resolve(external_address, 1)), 1)

    free = kgsl.struct_kgsl_gpuobj_free(id=allocation.id)
    with self.assertRaisesRegex(RuntimeError, "invalid KGSL descriptor"):
      descriptor.ioctl(self.device.fd.fd, ioctl_code(kgsl.IOCTL_KGSL_GPUOBJ_FREE), ctypes.addressof(free))
    with self.assertRaisesRegex(RuntimeError, "invalid KGSL descriptor"):
      descriptor.mmap(0, 0x1000, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, self.device.fd.fd, allocation.id * 0x1000)
    with self.assertRaisesRegex(RuntimeError, "invalid KGSL descriptor"):
      descriptor.close(self.device.fd.fd)

    descriptor.close(descriptor.fd)
    self.assertNotIn(allocation.id, self.driver.allocations)
    self.assertNotIn(context.drawctxt_id, self.driver.contexts)
    self.assertNotIn(context.drawctxt_id, self.driver.context_timestamps)
    self.assertNotIn(context.drawctxt_id, self.driver.power_levels)
    self.assertNotIn(external_address, self.driver.user_mappings)
    for stale_address in (address, external_address):
      with self.assertRaisesRegex(RuntimeError, "unmapped or ambiguous GPU range"):
        self.driver.resolve(stale_address, 1)
    with self.assertRaisesRegex(RuntimeError, "closed descriptor"):
      descriptor.close(descriptor.fd)

  def test_malformed_lifecycle_requests_do_not_mutate_state(self):
    from tinygrad.runtime.autogen import kgsl
    from tinygrad.helpers import Target
    from test.mockgpu.qcom.qcomdriver import QCOMDriver
    for target in ("MOCK+QCOM:IR3:a643", "MOCK+QCOM:QCOMCL:a630", "MOCK+QCOM"):
      with self.assertRaisesRegex(RuntimeError, "unsupported QCOM mock target"):
        QCOMDriver(Target.parse(target))

    flags = kgsl.KGSL_MEMFLAGS_USE_CPU_MAP | (12 << kgsl.KGSL_MEMALIGN_SHIFT)
    allocation_count, next_allocation_id = len(self.driver.allocations), self.driver.next_allocation_id
    with self.assertRaisesRegex(RuntimeError, "invalid allocation sizes"):
      kgsl.IOCTL_KGSL_GPUOBJ_ALLOC(self.device.fd, size=0, mmapsize=0, flags=flags)
    with self.assertRaisesRegex(RuntimeError, "unsupported allocation metadata"):
      kgsl.IOCTL_KGSL_GPUOBJ_ALLOC(self.device.fd, size=0x1000, mmapsize=0x1000, flags=flags, metadata=1)
    self.assertEqual((len(self.driver.allocations), self.driver.next_allocation_id), (allocation_count, next_allocation_id))

    context_count, next_context_id = len(self.driver.contexts), self.driver.next_context_id
    with self.assertRaisesRegex(RuntimeError, "missing required compute-context flags"):
      kgsl.IOCTL_KGSL_DRAWCTXT_CREATE(self.device.fd, flags=0)
    self.assertEqual((len(self.driver.contexts), self.driver.next_context_id), (context_count, next_context_id))

    with self.assertRaisesRegex(RuntimeError, "invalid device-info payload"):
      kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY(self.device.fd, type=kgsl.KGSL_PROP_DEVICE_INFO,
                                         value=0, sizebytes=ctypes.sizeof(kgsl.struct_kgsl_devinfo))
    with self.assertRaisesRegex(RuntimeError, "unsupported property"):
      kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY(self.device.fd, type=0)
    with self.assertRaisesRegex(RuntimeError, "unsupported property"):
      kgsl.IOCTL_KGSL_SETPROPERTY(self.device.fd, type=0)
    level = kgsl.struct_kgsl_device_constraint_pwrlevel(level=kgsl.KGSL_CONSTRAINT_PWR_MAX)
    constraint = kgsl.struct_kgsl_device_constraint(type=kgsl.KGSL_CONSTRAINT_PWRLEVEL, context_id=self.device.ctx,
                                                     data=ctypes.addressof(level), size=ctypes.sizeof(level) - 1)
    with self.assertRaisesRegex(RuntimeError, "invalid power-level payload"):
      kgsl.IOCTL_KGSL_SETPROPERTY(self.device.fd, type=kgsl.KGSL_PROP_PWR_CONSTRAINT,
                                  value=ctypes.addressof(constraint), sizebytes=ctypes.sizeof(constraint))
    self.assertEqual(self.driver.power_levels[self.device.ctx], kgsl.KGSL_CONSTRAINT_PWR_MAX)

  def test_gpu_command_validation_and_control_retirement(self):
    from tinygrad.runtime.autogen import kgsl, mesa
    from tinygrad.runtime.ops_qcom import pkt7_hdr
    from test.mockgpu.qcom.qcomdriver import KGSLJournalWrite, ioctl_code
    words = [pkt7_hdr(mesa.CP_WAIT_FOR_IDLE, 0)]
    buffer, command, request = self.gpu_command(words)
    request.timestamp = 0x12345678
    before_bytes = bytes(buffer.cpu_view().mv[:4])
    before_state = (dict(self.driver.contexts), dict(self.driver.allocations), dict(self.driver.user_mappings))
    previous_timestamp = self.driver.context_timestamps[self.device.ctx]
    kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=request)
    self.assertEqual((request.timestamp, self.driver.context_timestamps[self.device.ctx]),
                     (previous_timestamp + 1, previous_timestamp + 1))
    self.assertEqual(bytes(buffer.cpu_view().mv[:4]), before_bytes)
    self.assertEqual((self.driver.contexts, self.driver.allocations, self.driver.user_mappings), before_state)
    self.assertEqual(bytes(self.driver.resolve_owned(self.device.fd.fd, int(buffer.va_addr), 4, internal_only=True)), before_bytes)

    signal = self.device.new_signal(value=5)
    control_queue = self.device.hw_compute_queue_t().wait(signal, 5).signal(signal, 6).signal(signal, 7)
    control_timestamp = self.driver.context_timestamps[self.device.ctx]
    control_queue.submit(self.device)
    self.assertEqual(signal.value, 7)
    self.assertEqual((self.device.last_cmd, self.driver.context_timestamps[self.device.ctx]),
                     (control_timestamp + 1, control_timestamp + 1))
    wait_before = (dict(self.driver.context_timestamps), self.driver.always_on_counter, self.device.last_cmd)
    self.device.timeline_signal._sleep(0)
    self.assertEqual((self.driver.context_timestamps, self.driver.always_on_counter, self.device.last_cmd), wait_before)

    writable, readonly = memoryview(bytearray(b"left")), memoryview(b"right")
    preflight_journal = (KGSLJournalWrite(0, 0, 1, b"LEFT", "first test", False),
                         KGSLJournalWrite(1, 0, 5, b"RIGHT", "second test", False))
    with mock.patch.object(self.driver, "resolve_owned", side_effect=(writable, readonly)), \
         self.assertRaisesRegex(RuntimeError, "read-only second test range"):
      self.driver._commit_a630_journal(self.device.fd.fd, preflight_journal)
    self.assertEqual(bytes(writable), b"left")
    request.timestamp = 0x12345678

    outer_cases = (
      ("flags", 1, "unsupported GPU command flags"),
      ("cmdlist", 0, "invalid command-list pointer"),
      ("cmdlist", ctypes.addressof(command) + 1, "invalid command-list pointer"),
      ("cmdsize", ctypes.sizeof(command) - 1, "invalid command-list shape"),
      ("cmdsize", ctypes.sizeof(command) + 1, "invalid command-list shape"),
      ("numcmds", 0, "invalid command-list shape"),
      ("numcmds", 2, "invalid command-list shape"),
      ("objlist", 1, "unsupported GPU object list"),
      ("objsize", 1, "unsupported GPU object list"),
      ("numobjs", 1, "unsupported GPU object list"),
      ("synclist", 1, "unsupported GPU sync list"),
      ("syncsize", 1, "unsupported GPU sync list"),
      ("numsyncs", 1, "unsupported GPU sync list"),
      ("context_id", 0xffffffff, "unknown context"),
    )
    for field,value,message in outer_cases:
      original = getattr(request, field)
      setattr(request, field, value)
      with self.subTest(field=field, value=value), self.assertRaisesRegex(RuntimeError, message):
        kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=request)
      setattr(request, field, original)
      self.assertEqual(request.timestamp, 0x12345678)

    request_code = ioctl_code(kgsl.IOCTL_KGSL_GPU_COMMAND)
    for malformed in (request_code ^ (1 << 30), request_code ^ (1 << 16), request_code ^ (1 << 8), request_code ^ 1):
      with self.assertRaisesRegex(RuntimeError, "unsupported KGSL ioctl"):
        self.device.fd.ioctl(malformed, request)

    object_cases = (
      ("offset", 4, "unsupported command-object offset or id"),
      ("id", 1, "unsupported command-object offset or id"),
      ("flags", 0, "unsupported command-object flags"),
      ("flags", kgsl.KGSL_CMDLIST_IB | 2, "unsupported command-object flags"),
      ("gpuaddr", int(buffer.va_addr) + 1, "unaligned command address"),
      ("size", 0, "invalid command size"),
      ("size", 2, "invalid command size"),
      ("gpuaddr", (1 << 64) - 4, "invalid GPU range"),
    )
    for field,value,message in object_cases:
      original = getattr(command, field)
      if field == "gpuaddr" and value == (1 << 64) - 4:
        original_size, command.size = command.size, 8
      setattr(command, field, value)
      with self.subTest(field=field, value=value), self.assertRaisesRegex(RuntimeError, message):
        kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=request)
      setattr(command, field, original)
      if field == "gpuaddr" and value == (1 << 64) - 4: command.size = original_size
      self.assertEqual(request.timestamp, 0x12345678)

    command.gpuaddr = int(buffer.va_addr) + buffer.meta[0].mmapsize
    with self.assertRaisesRegex(RuntimeError, "not in one owned mapping"):
      kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=request)
    command.gpuaddr = int(buffer.va_addr)

    other_fd = self.driver.open('/dev/kgsl-3d0', os.O_RDWR, 0, self.driver.tracked_files[0])
    with self.assertRaisesRegex(RuntimeError, "belongs to another descriptor"):
      other_fd.ioctl(other_fd.fd, request_code, ctypes.addressof(request))
    flags = kgsl.KGSL_MEMFLAGS_USE_CPU_MAP | (12 << kgsl.KGSL_MEMALIGN_SHIFT)
    foreign = kgsl.struct_kgsl_gpuobj_alloc(size=0x1000, mmapsize=0x1000, flags=flags)
    other_fd.ioctl(other_fd.fd, ioctl_code(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC), ctypes.addressof(foreign))
    foreign_address = other_fd.mmap(0, 0x1000, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, other_fd.fd, foreign.id * 0x1000)
    command.gpuaddr = foreign_address
    with self.assertRaisesRegex(RuntimeError, "not in one owned mapping"):
      kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=request)
    command.gpuaddr = int(buffer.va_addr)
    other_fd.close(other_fd.fd)

    backing = bytearray(0x3000)
    external_address = (mv_address(memoryview(backing)) + 0xfff) & ~0xfff
    kgsl.IOCTL_KGSL_MAP_USER_MEM(self.device.fd, hostptr=external_address, len=0x1000, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR)
    command.gpuaddr = external_address
    with self.assertRaisesRegex(RuntimeError, "not in one owned mapping"):
      kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=request)
    kgsl.IOCTL_KGSL_SHAREDMEM_FREE(self.device.fd, gpuaddr=external_address)
    command.gpuaddr = int(buffer.va_addr)

    freed_buffer = self.device._gpu_alloc(4, fill_zeroes=True)
    freed_address = int(freed_buffer.va_addr)
    self.device._gpu_free(freed_buffer)
    command.gpuaddr = freed_address
    with self.assertRaisesRegex(RuntimeError, "not in one owned mapping"):
      kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=request)
    command.gpuaddr = int(buffer.va_addr)

    malformed_words = [pkt7_hdr(mesa.CP_WAIT_FOR_IDLE, 0), 0]
    malformed_buffer, malformed_command, malformed_request = self.gpu_command(malformed_words)
    malformed_request.timestamp = 77
    malformed_before = bytes(malformed_buffer.cpu_view().mv[:8])
    with self.assertRaisesRegex(RuntimeError, "invalid KGSL request: unsupported packet header"):
      kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=malformed_request)
    self.assertEqual((malformed_request.timestamp, bytes(malformed_buffer.cpu_view().mv[:8])), (77, malformed_before))
    self.device._gpu_free(malformed_buffer)
    self.device._gpu_free(buffer)

    recovered_timestamp = self.driver.context_timestamps[self.device.ctx]
    self.device.hw_compute_queue_t().wait(signal, 7).signal(signal, 8).submit(self.device)
    self.assertEqual((signal.value, self.device.last_cmd, self.driver.context_timestamps[self.device.ctx]),
                     (8, recovered_timestamp + 1, recovered_timestamp + 1))
    self.device.timeline_signal._sleep(0)
    self.device.synchronize()

  def test_retirement_interval_sweep_preserves_order_and_overlap_edges(self):
    from test.mockgpu.qcom import qcomdriver
    from test.mockgpu.qcom.a630 import A630ExecutionWrite, A630Submission, A630Write

    count,write_base = 512,0x10000000
    writes = tuple(A630ExecutionWrite(write_base + index * 8, index.to_bytes(4, "little")) for index in range(count))
    submission = A630Submission((cast(Any, mock.Mock(word_offset=7)),), (), (), ())

    def execute_writes(_submission, _resolver, *, read_observer=None, budget=None): return writes

    with mock.patch.object(qcomdriver, "execute_a630", side_effect=execute_writes), \
         mock.patch.object(self.driver, "_overlaps", wraps=self.driver._overlaps) as overlaps:
      journal,_ = self.driver._plan_a630_retirement(self.device.fd.fd, submission, 0x30000000, 4)
    self.assertEqual(tuple((write.address, write.data) for write in journal),
                     tuple((write.address, write.data) for write in writes))
    self.assertLessEqual(overlaps.call_count, 4 * count + 8)

    repeated = A630Submission((), (), (),
      (A630Write(1, 0x40000000, 4, 1, "first PM4 write"), A630Write(2, 0x40000000, 4, 2, "second PM4 write")))
    self.assertEqual(len(self.driver._plan_a630_retirement(self.device.fd.fd, repeated, 0x50000000, 4)[0]), 2)
    partial = A630Submission((), (), (),
      (A630Write(1, 0x40000000, 8, None, "first PM4 write"), A630Write(2, 0x40000004, 4, 2, "partial PM4 write")))
    with self.assertRaisesRegex(RuntimeError, "overlapping first PM4 write and partial PM4 write"):
      self.driver._plan_a630_retirement(self.device.fd.fd, partial, 0x50000000, 4)
    two_dispatches = A630Submission((cast(Any, mock.Mock(word_offset=7)), cast(Any, mock.Mock(word_offset=8))), (), (), ())
    with mock.patch.object(qcomdriver, "execute_a630", side_effect=execute_writes), \
         mock.patch.object(qcomdriver, "_MAX_A630_OVERLAY_INTERVAL_WORK", count - 1), \
         self.assertRaisesRegex(RuntimeError, "bounded prior-effect overlay limit"):
      self.driver._plan_a630_retirement(self.device.fd.fd, two_dispatches, 0x50000000, 4)

  def test_rejected_hcq_program_recovers_timeline_for_next_kernel(self):
    import struct
    from tinygrad import Device, Tensor
    from tinygrad.codegen import to_program
    from tinygrad.engine.realize import get_runtime
    from tinygrad.helpers import Context
    from tinygrad.runtime.support.hcq import HCQSubmissionRejected

    left_values, right_values = [1.0, -2.5, 1024.0], [4.0, 0.5, -24.0]
    left, right = Tensor(left_values, device=Device.DEFAULT).realize(), Tensor(right_values, device=Device.DEFAULT).realize()
    result = left + right
    program_spec = to_program(result.schedule_linear().src[-1].src[0], self.device.renderer)
    runtime = get_runtime(self.device.device, program_spec)
    result_buffer, left_buffer, right_buffer = (cast(Any, tensor.uop.buffer) for tensor in (result, left, right))
    result_buffer.allocate()
    runtime_args = (result_buffer._buf, left_buffer._buf, right_buffer._buf)
    runtime_sizes = {"global_size": program_spec.arg.global_size, "local_size": program_spec.arg.local_size}

    result_size = len(left_values) * 4
    result_view = self.driver.resolve_owned(self.device.fd.fd, int(result_buffer._buf.va_addr), result_size)
    timeline_view = self.device.timeline_signal.base_buf.cpu_view().mv[:16]
    dummy_view = self.driver.resolve_owned(self.device.fd.fd, self.device.dummy_addr, 4)
    original_dummy = bytes(dummy_view)
    blocker = self.device.new_signal(value=0)
    result_view[:] = bytes(result_size)
    dummy_view[:] = b"A630"

    def state():
      return {"timeline_value": self.device.timeline_value, "timeline_image": bytes(timeline_view),
              "context_timestamp": self.driver.context_timestamps[self.device.ctx], "last_cmd": self.device.last_cmd,
              "counter": self.driver.always_on_counter, "error_state": self.device.error_state, "blocker": blocker.value,
              "result": bytes(result_view), "dummy": bytes(dummy_view), "profile_records": tuple(self.device.sig_prof_records)}

    before, before_prof_exec = state(), self.device.prof_exec_counter
    rejected_state, completed = None, False
    def restore_before():
      self.device.timeline_value, timeline_view[:] = before["timeline_value"], before["timeline_image"]
      self.driver.context_timestamps[self.device.ctx], self.device.last_cmd = before["context_timestamp"], before["last_cmd"]
      self.driver.always_on_counter, self.device.error_state = before["counter"], before["error_state"]
      blocker.value, result_view[:], dummy_view[:] = before["blocker"], before["result"], before["dummy"]
      self.device.sig_prof_records[:] = before["profile_records"]
      self.device.prof_exec_counter = before_prof_exec

    try:
      queue_type = self.device.hw_compute_queue_t
      with Context(PROFILE=1), mock.patch.object(self.device, "hw_compute_queue_t", side_effect=lambda: queue_type().wait(blocker, 1)), \
           self.assertRaisesRegex(HCQSubmissionRejected, "unsatisfied memory wait"):
        runtime(*runtime_args, **runtime_sizes)
      rejected_state = state()
      if rejected_state != before: restore_before()
      self.assertEqual(rejected_state, before)
      self.assertEqual(self.device.prof_exec_counter, before_prof_exec + 1)

      runtime(*runtime_args, **runtime_sizes)
      self.device.synchronize()
      reference = cast(list[float], (Tensor(left_values, device="PYTHON") + Tensor(right_values, device="PYTHON")).tolist())
      self.assertEqual(list(struct.unpack(f"<{len(left_values)}f", result_view)), reference)
      self.assertEqual((self.device.timeline_value, self.device.timeline_signal.value),
                       (before["timeline_value"] + 1, before["timeline_value"]))
      self.assertEqual((self.driver.context_timestamps[self.device.ctx], self.device.last_cmd),
                       (before["context_timestamp"] + 1, before["last_cmd"] + 1))
      self.assertEqual((self.driver.always_on_counter, self.device.error_state, blocker.value, bytes(dummy_view)),
                       (before["counter"], None, 0, bytes(4)))
      completed = True
    finally:
      if not completed: restore_before()
      dummy_view[:] = original_dummy

  def test_multi_dispatch_overlay_is_transactional_and_recovers(self):
    import numpy as np, struct
    from tinygrad import Device, Tensor
    from tinygrad.codegen import to_program
    from tinygrad.engine.realize import get_runtime
    from tinygrad.runtime.autogen import kgsl
    from tinygrad.runtime.support.hcq import HCQSubmissionRejected
    from test.mockgpu.qcom import a630 as a630_module, qcomdriver
    from test.mockgpu.qcom.a630 import stage_a630
    from test.mockgpu.qcom.pm4 import parse_pm4

    shape,count = (45, 68),45 * 68
    a = Tensor(np.arange(count, dtype=np.float32).reshape(shape), device=Device.DEFAULT).realize()
    b = a + 1
    program_spec = to_program(b.schedule_linear().src[-1].src[0], self.device.renderer)
    runtime = get_runtime(self.device.device, program_spec)
    a_buffer,b_buffer = cast(Any, a.uop.buffer),cast(Any, b.uop.buffer)
    b_buffer.allocate()
    a_address,b_address = int(a_buffer._buf.va_addr),int(b_buffer._buf.va_addr)
    b_args = runtime.fill_kernargs([b_buffer._buf, a_buffer._buf])
    a_args = runtime.fill_kernargs([a_buffer._buf, b_buffer._buf])
    queue = self.device.hw_compute_queue_t().wait(self.device.timeline_signal, self.device.timeline_value - 1) \
      .exec(runtime, b_args, program_spec.arg.global_size, program_spec.arg.local_size) \
      .exec(runtime, a_args, program_spec.arg.global_size, program_spec.arg.local_size) \
      .signal(self.device.timeline_signal, self.device.timeline_value)
    words = tuple(queue._q)
    submission = stage_a630(parse_pm4(words), self._resolve_owned)
    self.assertEqual(len(submission.dispatches), 2)
    self.assertLess(submission.dispatches[0].word_offset, submission.dispatches[1].word_offset)
    self.assertEqual(tuple(struct.unpack("<2Q", dispatch.constants_image[:16]) for dispatch in submission.dispatches),
                     ((b_address, a_address), (a_address, b_address)))

    command_buffer,_,request = self.gpu_command(words)
    request.timestamp = marker = 0x4d554c54
    a_view,b_view = self._resolve_owned(a_address, count * 4),self._resolve_owned(b_address, count * 4)
    signal_view = self._resolve_owned(int(self.device.timeline_signal.value_addr), 16)
    dummy_view = self._resolve_owned(self.device.dummy_addr, 4)
    before = (bytes(a_view), bytes(b_view), bytes(signal_view), bytes(dummy_view),
              self.driver.context_timestamps[self.device.ctx], self.driver.always_on_counter,
              self.device.last_cmd, self.device.error_state)
    first = submission.dispatches[0]
    active_count = next(instruction.index + 1 for instruction in first.instructions if instruction.opcode == "end")
    first_dispatch_steps = active_count * first.groups[0] * first.local_size[0]
    real_execute,calls = qcomdriver.execute_a630,0
    def count_execution(*args, **kwargs):
      nonlocal calls
      calls += 1
      return real_execute(*args, **kwargs)
    try:
      with mock.patch.object(a630_module, "_MAX_LANE_INSTRUCTION_STEPS", first_dispatch_steps), \
           mock.patch.object(qcomdriver, "execute_a630", side_effect=count_execution), \
           self.assertRaisesRegex(HCQSubmissionRejected, "bounded lane-instruction limit"):
        kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=request)
      self.assertEqual((calls, request.timestamp), (2, marker))
      self.assertEqual((bytes(a_view), bytes(b_view), bytes(signal_view), bytes(dummy_view),
                        self.driver.context_timestamps[self.device.ctx], self.driver.always_on_counter,
                        self.device.last_cmd, self.device.error_state), before)
    finally: self.device._gpu_free(command_buffer)

    queue.submit(self.device)
    self.device.timeline_signal.wait(self.device.timeline_value)
    self.device.timeline_value += 1
    self.assertEqual((struct.unpack(f"<{count}f", a_view), struct.unpack(f"<{count}f", b_view)),
                     (tuple(float(index + 2) for index in range(count)), tuple(float(index + 1) for index in range(count))))

  def test_timeline_rollback_requires_latest_definite_rejection(self):
    from tinygrad.runtime.support.hcq import HCQSubmissionRejected

    before = self.device.timeline_value
    ambiguous_timeline = later_timeline = None
    try:
      ambiguous = self.device.hw_compute_queue_t()
      with mock.patch.object(ambiguous, "_submit", side_effect=RuntimeError("ambiguous acceptance")), \
           self.assertRaisesRegex(RuntimeError, "ambiguous acceptance") as raised:
        self.device.submit_timeline(ambiguous)
      self.assertIs(type(raised.exception), RuntimeError)
      ambiguous_timeline = self.device.timeline_value
      self.device.timeline_value = before

      def reject_after_later_reservation(_):
        self.device.next_timeline()
        raise HCQSubmissionRejected("definite rejection after a later reservation")
      non_lifo = self.device.hw_compute_queue_t()
      with mock.patch.object(non_lifo, "_submit", side_effect=reject_after_later_reservation), \
           self.assertRaisesRegex(HCQSubmissionRejected, "after a later reservation"):
        self.device.submit_timeline(non_lifo)
      later_timeline = self.device.timeline_value
    finally: self.device.timeline_value = before

    self.assertEqual((ambiguous_timeline, later_timeline), (before + 1, before + 2))

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

  def test_local_load_scoreboard_tracks_full_register_dependencies(self):
    from dataclasses import replace
    from test.mockgpu.qcom.a630 import A630Dispatch, A630IR3Instruction, A630IR3Operand, _validate_control_flow

    def gpr(value): return A630IR3Operand("gpr", value)
    def half(value): return A630IR3Operand("half", value)
    def iim(value): return A630IR3Operand("iim", value)
    def instruction(index, opcode, dst=None, srcs=(), *, ss=0, sy=0):
      return A630IR3Instruction(index, 0, 0, opcode, (("SS", ss), ("SY", sy)), opcode, dst, srcs)
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
    # Pinned ir3_legalize executes RPT component cycles in order. Shift the decoded ADD.F first source to one register
    # below its destination: each later component must observe the preceding component's mapped destination write.
    repeated_add = next(instruction for instruction in constant_dispatch.instructions if instruction.opcode == "add.f.rpt4")
    assert repeated_add.dst is not None and repeated_add.srcs[0].kind == "gpr"
    overlap_source = repeated_add.dst.value - 1
    overlap_add_raw = repeated_add.raw & ~0xffff | overlap_source
    overlap_values = (1.0, 3.0, 7.0, 15.0)
    overlap_expected = (16.0, 19.0, 26.0, 41.0)
    constant_input_bases = struct.unpack_from("<2Q", constant_dispatch.constants_image, 8)
    constant_inputs = tuple(self._resolve_owned(base, 16) for base in constant_input_bases)
    constant_input_originals = tuple(bytes(view) for view in constant_inputs)
    constant_output_original = bytes(constant_output)
    try:
      for view in constant_inputs: view[:] = struct.pack("<4f", *overlap_values)
      constant_output[:] = bytes([0x9b]) * 16
      with self._mutate_a630_replay(constant_submission, constant_words, ((repeated_add, overlap_add_raw),)) as \
           (overlap_submission,overlap_dispatch,_):
        mutated_add = overlap_dispatch.instructions[repeated_add.index]
        self.assertEqual((mutated_add.srcs[0].value, mutated_add.dst), (overlap_source, repeated_add.dst))
        overlap_journal = constant_real_execute(overlap_submission, self._resolve_owned)
        self.assertEqual(bytes(constant_output), bytes([0x9b]) * 16)
        self.assertEqual((len(overlap_journal), struct.unpack("<4f", overlap_journal[0].data)),
                         (1, overlap_expected))
    finally:
      for view,original in zip(constant_inputs, constant_input_originals): view[:] = original
      constant_output[:] = constant_output_original

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

  def test_ioctl_and_mmap_fail_closed(self):
    from tinygrad.runtime.autogen import kgsl
    from test.mockgpu.qcom import qcomdriver
    from test.mockgpu.qcom.qcomdriver import A630_CHIP_ID, ioctl_code
    flags = kgsl.KGSL_MEMFLAGS_USE_CPU_MAP | (12 << kgsl.KGSL_MEMALIGN_SHIFT)
    allocation = kgsl.IOCTL_KGSL_GPUOBJ_ALLOC(self.device.fd, size=0x1000, mmapsize=0x1000, flags=flags)
    mmap_args = (0, 0x1000, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, allocation.id * 0x1000)
    invalid_mmaps = [
      ((1, *mmap_args[1:]), "fixed mmap address"),
      (((mmap_args[0], mmap_args[1], mmap.PROT_READ, *mmap_args[3:])), "mmap protection"),
      (((*mmap_args[:3], mmap.MAP_PRIVATE, mmap_args[4])), "mmap flags"),
      (((*mmap_args[:4], mmap_args[4] + 1)), "unaligned mmap offset"),
      (((mmap_args[0], 0x2000, *mmap_args[2:])), "mmap size"),
    ]
    for args,message in invalid_mmaps:
      with self.assertRaisesRegex(RuntimeError, message): self.device.fd.mmap(*args)
    with self.assertRaisesRegex(RuntimeError, "invalid KGSL request: unknown allocation"):
      self.device.fd.mmap(0, 0x1000, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, (allocation.id + 1) * 0x1000)
    with mock.patch.object(qcomdriver.libc, "mmap", return_value=qcomdriver.MAP_FAILED):
      with self.assertRaisesRegex(RuntimeError, "anonymous mmap failed"): self.device.fd.mmap(*mmap_args)
    self.assertIsNone(self.driver.allocations[allocation.id].addr)
    mapped_address = self.device.fd.mmap(*mmap_args)
    with self.assertRaisesRegex(RuntimeError, "already mapped"): self.device.fd.mmap(*mmap_args)
    kgsl.IOCTL_KGSL_GPUOBJ_FREE(self.device.fd, id=allocation.id)
    self.device.fd.munmap(mapped_address, 0x1000)

    allocation_count = len(self.driver.allocations)
    request = ioctl_code(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC)
    for malformed in (request ^ (1 << 30), request ^ (1 << 16), request ^ (1 << 8)):
      with self.assertRaisesRegex(RuntimeError, "unsupported KGSL ioctl"):
        self.device.fd.ioctl(malformed, kgsl.struct_kgsl_gpuobj_alloc())
    with self.assertRaisesRegex(RuntimeError, "null ioctl payload"):
      self.driver.ioctl(self.device.fd.fd, ioctl_code(kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY), 0)
    self.assertEqual(len(self.driver.allocations), allocation_count)
    info = kgsl.struct_kgsl_devinfo()
    with self.assertRaisesRegex(RuntimeError, "invalid device-info payload"):
      kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY(self.device.fd, type=kgsl.KGSL_PROP_DEVICE_INFO,
                                         value=ctypes.addressof(info), sizebytes=ctypes.sizeof(info) - 1)
    kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY(self.device.fd, type=kgsl.KGSL_PROP_DEVICE_INFO,
                                       value=ctypes.addressof(info), sizebytes=ctypes.sizeof(info))
    self.assertEqual((info.chip_id, info.gpu_id, info.gmem_gpubaseaddr, info.gmem_sizebytes), (A630_CHIP_ID, 630, 0, 0))

    other_fd = self.driver.open('/dev/kgsl-3d0', os.O_RDWR, 0, self.driver.tracked_files[0])
    owned = kgsl.IOCTL_KGSL_GPUOBJ_ALLOC(self.device.fd, size=0x1000, mmapsize=0x1000, flags=flags)
    free = kgsl.struct_kgsl_gpuobj_free(id=owned.id)
    with self.assertRaisesRegex(RuntimeError, "belongs to another descriptor"):
      other_fd.ioctl(other_fd.fd, ioctl_code(kgsl.IOCTL_KGSL_GPUOBJ_FREE), ctypes.addressof(free))
    with self.assertRaisesRegex(RuntimeError, "belongs to another descriptor"):
      other_fd.mmap(0, owned.mmapsize, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED, other_fd.fd, owned.id * 0x1000)
    self.assertIsNone(self.driver.allocations[owned.id].addr)
    destroy = kgsl.struct_kgsl_drawctxt_destroy(drawctxt_id=self.device.ctx)
    with self.assertRaisesRegex(RuntimeError, "belongs to another descriptor"):
      other_fd.ioctl(other_fd.fd, ioctl_code(kgsl.IOCTL_KGSL_DRAWCTXT_DESTROY), ctypes.addressof(destroy))
    level = kgsl.struct_kgsl_device_constraint_pwrlevel(level=kgsl.KGSL_CONSTRAINT_PWR_MAX)
    constraint = kgsl.struct_kgsl_device_constraint(type=kgsl.KGSL_CONSTRAINT_PWRLEVEL, context_id=self.device.ctx,
                                                     data=ctypes.addressof(level), size=ctypes.sizeof(level))
    prop = kgsl.struct_kgsl_device_getproperty(type=kgsl.KGSL_PROP_PWR_CONSTRAINT,
                                               value=ctypes.addressof(constraint), sizebytes=ctypes.sizeof(constraint))
    with self.assertRaisesRegex(RuntimeError, "belongs to another descriptor"):
      other_fd.ioctl(other_fd.fd, ioctl_code(kgsl.IOCTL_KGSL_SETPROPERTY), ctypes.addressof(prop))
    self.assertIn(self.device.ctx, self.driver.contexts)
    self.assertEqual(self.driver.power_levels[self.device.ctx], kgsl.KGSL_CONSTRAINT_PWR_MAX)
    other_fd.close(other_fd.fd)
    kgsl.IOCTL_KGSL_GPUOBJ_FREE(self.device.fd, id=owned.id)
    with self.assertRaisesRegex(RuntimeError, "unsupported open flags"):
      self.driver.open('/dev/kgsl-3d0', os.O_RDONLY, 0, self.driver.tracked_files[0])
    with self.assertRaisesRegex(RuntimeError, "invalid command-list pointer"):
      kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, context_id=self.device.ctx)

if __name__ == '__main__':
  unittest.main()

import ctypes, functools, mmap, os, unittest
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

  def gpu_command(self, words):
    from tinygrad.runtime.autogen import kgsl
    buffer = self.device._gpu_alloc(len(words) * 4, fill_zeroes=True)
    (ctypes.c_uint32 * len(words)).from_address(int(buffer.va_addr))[:] = words
    command = kgsl.struct_kgsl_command_object(gpuaddr=int(buffer.va_addr), size=len(words) * 4, flags=kgsl.KGSL_CMDLIST_IB)
    request = kgsl.struct_kgsl_gpu_command(cmdlist=ctypes.addressof(command), cmdsize=ctypes.sizeof(command), numcmds=1,
                                           context_id=self.device.ctx)
    return buffer, command, request

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
    from test.mockgpu.qcom.a630 import decode_a630_ir3

    end = 6 << 55
    decoded = decode_a630_ir3(end.to_bytes(8, "little"))
    self.assertEqual(len(decoded), 1)
    self.assertEqual((decoded[0].index, decoded[0].category, decoded[0].raw, decoded[0].name), (0, 0, end, "end"))
    self.assertEqual(decoded[0].fields,
                     (("SY", 0), ("SS", 0), ("EQ", 0), ("JP", 0), ("REPEAT", 0), ("NAME", "end")))

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

  def test_ir3_typed_instruction_and_modifier_contracts(self):
    from test.mockgpu.qcom.a630 import A630IR3Operand, decode_a630_ir3

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
    )
    decoded = decode_a630_ir3(b"".join(word.to_bytes(8, "little") for word in words + (end,)))
    self.assertEqual(tuple(instruction.opcode for instruction in decoded),
                     ("ashr.b", "shl.b", "shrg", "add.u", "cmps.u.lt", "cov.u16s32", "nop",
                      "ldg.u32", "add.f", "stg.u32", "mull.u", "madsh.m16", "madsh.m16", "end"))

    def decode_one(word): return decode_a630_ir3(word.to_bytes(8, "little") + end.to_bytes(8, "little"))[0]

    mov_shared = decode_one(0x200cc001000000c0)
    self.assertEqual((mov_shared.opcode, mov_shared.dst, mov_shared.srcs),
                     ("mov.u32", A630IR3Operand("gpr", 1), (A630IR3Operand("shared", 0xc0),)))
    mov_immediate = decode_one(0x204cc0033f800000)
    self.assertEqual((mov_immediate.opcode, mov_immediate.dst, mov_immediate.srcs),
                     ("mov.u32", A630IR3Operand("gpr", 3), (A630IR3Operand("uim", 0x3f800000),)))
    constant_moves = (
      (0x202cc00000000002, 0, 2), (0x202cc00100000003, 1, 3),
      (0x202cc00300000004, 3, 4), (0x202cc00400000005, 4, 5),
      (0x202cc00500000000, 5, 0), (0x202cc00600000001, 6, 1),
    )
    for word,dst,src in constant_moves:
      with self.subTest(word=f"{word:#x}"):
        self.assertEqual((decode_one(word).opcode, decode_one(word).dst, decode_one(word).srcs),
                         ("mov.u32", A630IR3Operand("gpr", dst), (A630IR3Operand("const", src),)))
    scheduled_constant = decode_one(0x202cc0bf000007ff | 1 << 44 | 1 << 60)
    self.assertEqual((scheduled_constant.opcode, scheduled_constant.dst, scheduled_constant.srcs),
                     ("mov.u32", A630IR3Operand("gpr", 0xbf), (A630IR3Operand("const", 0x7ff),)))
    self.assertTrue({("SY", 1), ("SS", 1)} <= set(scheduled_constant.fields))
    integer_add = decode_one(0x5218080b0010000b)
    self.assertEqual((integer_add.opcode, integer_add.dst, integer_add.srcs),
                     ("add.u", A630IR3Operand("gpr", 11),
                      (A630IR3Operand("gpr", 11), A630IR3Operand("gpr", 16))))
    self.assertTrue({("SY", 1), ("NOP", 3)} <= set(integer_add.fields))
    integer_sub = decode_one(0x5258080200070002)
    self.assertEqual((integer_sub.opcode, integer_sub.dst, integer_sub.srcs),
                     ("sub.u", A630IR3Operand("gpr", 2),
                      (A630IR3Operand("gpr", 2), A630IR3Operand("gpr", 7))))
    self.assertTrue({("SY", 1), ("NOP", 3)} <= set(integer_sub.fields))
    integer_xor = decode_one(0x53f8080200070002)
    self.assertEqual((integer_xor.opcode, integer_xor.dst, integer_xor.srcs),
                     ("xor.b", A630IR3Operand("gpr", 2),
                      (A630IR3Operand("gpr", 2), A630IR3Operand("gpr", 7))))
    self.assertTrue({("SY", 1), ("NOP", 3)} <= set(integer_xor.fields))
    integer_and = decode_one(0x5398080200070002)
    self.assertEqual((integer_and.opcode, integer_and.dst, integer_and.srcs),
                     ("and.b", A630IR3Operand("gpr", 2),
                      (A630IR3Operand("gpr", 2), A630IR3Operand("gpr", 7))))
    self.assertTrue({("SY", 1), ("NOP", 3)} <= set(integer_and.fields))
    integer_or = decode_one(0x53b8080200070002)
    self.assertEqual((integer_or.opcode, integer_or.dst, integer_or.srcs),
                     ("or.b", A630IR3Operand("gpr", 2),
                      (A630IR3Operand("gpr", 2), A630IR3Operand("gpr", 7))))
    self.assertTrue({("SY", 1), ("NOP", 3)} <= set(integer_or.fields))
    logical_shift = decode_one(0x56f8080200070002)
    self.assertEqual((logical_shift.opcode, logical_shift.dst, logical_shift.srcs),
                     ("shr.b", A630IR3Operand("gpr", 2),
                      (A630IR3Operand("gpr", 2), A630IR3Operand("gpr", 7))))
    self.assertTrue({("NAME", "shr.b"), ("SY", 1), ("NOP", 3)} <= set(logical_shift.fields))
    for nop_count in range(4):
      scheduled_or = decode_one(integer_or.raw & ~((1 << 43) | (1 << 51)) |
                                (nop_count & 1) << 43 | (nop_count >> 1) << 51)
      self.assertEqual((scheduled_or.opcode, scheduled_or.dst, scheduled_or.srcs),
                       (integer_or.opcode, integer_or.dst, integer_or.srcs))
      self.assertIn(("REPEAT", 0) if nop_count == 0 else ("NOP", nop_count), scheduled_or.fields)
    self.assertEqual(decode_one(integer_or.raw | 1 << 44).opcode, "or.b")
    for reserved_bit in range(48, 51):
      with self.subTest(or_reserved_bit=reserved_bit), \
           self.assertRaisesRegex(ValueError, "invalid or reserved IR3 encoding at instruction 0"):
        decode_one(integer_or.raw | 1 << reserved_bit)
    signed_maximum = decode_one(0x5338080200070002)
    unsigned_maximum = decode_one(0x5318080200070002)
    for maximum,opcode in ((signed_maximum, "max.s"), (unsigned_maximum, "max.u")):
      self.assertEqual((maximum.opcode, maximum.dst, maximum.srcs),
                       (opcode, A630IR3Operand("gpr", 2),
                        (A630IR3Operand("gpr", 2), A630IR3Operand("gpr", 7))))
      self.assertTrue({("NAME", opcode), ("SY", 1), ("NOP", 3)} <= set(maximum.fields))
    scheduled_sub = decode_one(integer_sub.raw | 1 << 44)
    self.assertEqual((scheduled_sub.opcode, scheduled_sub.dst, scheduled_sub.srcs),
                     (integer_sub.opcode, integer_sub.dst, integer_sub.srcs))
    self.assertTrue({("SY", 1), ("SS", 1), ("NOP", 3)} <= set(scheduled_sub.fields))
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
    for unsupported_word,name,condition in ((0x529c480000070002, "cmps.u", 4),
                                            (0x52bd480000070002, "cmps.s", 5),
                                            (0x529d480000070002, "cmps.u", 5)):
      unsupported_compare = decode_one(unsupported_word)
      self.assertIsNone(unsupported_compare.opcode)
      self.assertTrue({("NAME", name), ("COND", condition)} <= set(unsupported_compare.fields))
    byte_store = decode_one(0xc0cc0b0001800000)
    self.assertEqual((byte_store.opcode, byte_store.dst, byte_store.srcs),
                     ("stg.u8", None, (A630IR3Operand("gpr", 5), A630IR3Operand("half", 0))))
    self.assertTrue({("TYPE", 6), ("TYPE_HALF", 1), ("OFF", 0), ("SIZE", 1)} <= set(byte_store.fields))
    self.assertEqual(decode_one(byte_store.raw | 1 << 60).opcode, "stg.u8")
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

    cov = words[5] & ~((0xff << 32) | 0xff) | 9 << 32 | 3
    self.assertEqual((decode_one(cov).dst, decode_one(cov).srcs),
                     (A630IR3Operand("gpr", 9), (A630IR3Operand("half", 3),)))
    ldg = words[7] & ~((0xff << 32) | (0xff << 14)) | 12 << 32 | 8 << 14
    self.assertEqual((decode_one(ldg).dst, decode_one(ldg).srcs),
                     (A630IR3Operand("gpr", 12), (A630IR3Operand("gpr", 8),)))
    stg = words[9] & ~((0xff << 41) | (0xff << 1)) | 8 << 41 | 12 << 1
    self.assertEqual(decode_one(stg).srcs, (A630IR3Operand("gpr", 8), A630IR3Operand("gpr", 12)))
    self.assertEqual(decode_one(words[8] | 1 << 16).srcs[1], A630IR3Operand("flut", 3))

    # SY/SS serialize dependencies in hardware; the lane interpreter is already sequential.
    scheduled = decode_one(words[3] | 1 << 60 | 1 << 44)
    self.assertEqual((scheduled.opcode, scheduled.srcs),
                     ("add.u", (A630IR3Operand("gpr", 10), A630IR3Operand("gpr", 8))))
    self.assertTrue({("SY", 1), ("SS", 1)} <= set(scheduled.fields))

    rejected_modifiers = {
      "saturate": words[3] | 1 << 42,
      "signed-add opcode": 0x5238080b0010000b,
      "jump-target": words[3] | 1 << 59,
      "repeat": words[3] | 1 << 40,
      "absolute/negate": words[3] | 1 << 14,
      "A7xx last-use": words[3] | 1 << 10,
      "half sources": words[3] ^ 1 << 52,
      "converted destination": words[3] | 1 << 46,
      "conversion-rounding": words[5] | 1 << 55,
      "shrg wrong precision": words[2] ^ 1 << 42,
      "global-load jump-target": words[7] | 1 << 59,
      "global-store destination offset": words[9] ^ 1 << 40,
      "shared-move repeat": 0x200cc001000000c0 | 1 << 40,
      "shared-move relative destination": 0x200cc001000000c0 | 1 << 49,
      "shared-move shared destination": 0x200cc0c0000000c0,
      "shared-move special source": 0x200cc001000000e0,
      "immediate-move jump-target": 0x204cc0033f800000 | 1 << 59,
      "immediate-move repeat": 0x204cc0033f800000 | 1 << 40,
      "immediate-move unsigned-low": 0x204cc0033f800000 | 1 << 45,
      "immediate-move rounding": 0x204cc0033f800000 | 1 << 55,
      "constant-move repeat": constant_moves[0][0] | 1 << 40,
      "constant-move relative source": constant_moves[0][0] | 1 << 43,
      "constant-move relative destination": constant_moves[0][0] | 1 << 49,
      "constant-move half source": constant_moves[0][0] ^ 1 << 50,
      "constant-move half destination": constant_moves[0][0] ^ 1 << 46,
      "constant-move jump-target": constant_moves[0][0] | 1 << 59,
      "constant-move unsigned-low": constant_moves[0][0] | 1 << 45,
      "constant-move rounding": constant_moves[0][0] | 1 << 55,
      "constant-move shared destination": constant_moves[0][0] | 0xc0 << 32,
      "constant-move special destination": constant_moves[0][0] | 0xe0 << 32,
    }
    for modifier,word in rejected_modifiers.items():
      with self.subTest(modifier=modifier): self.assertIsNone(decode_one(word).opcode)

    rejected_subtract = {
      "signed opcode": integer_sub.raw | 1 << 53,
      "saturate": integer_sub.raw | 1 << 42,
      "repeat": integer_sub.raw | 1 << 40,
      "unsigned-low": integer_sub.raw | 1 << 45,
      "converted destination": integer_sub.raw | 1 << 46,
      "early input": integer_sub.raw | 1 << 47,
      "jump-target": integer_sub.raw | 1 << 59,
      "half sources": integer_sub.raw ^ 1 << 52,
      "source 1 last-use": integer_sub.raw | 1 << 10,
      "source 1 absolute/negate": integer_sub.raw | 1 << 14,
      "source 2 last-use": integer_sub.raw | 1 << 26,
      "source 2 absolute/negate": integer_sub.raw | 1 << 30,
    }
    for modifier,word in rejected_subtract.items():
      with self.subTest(subtract_modifier=modifier): self.assertIsNone(decode_one(word).opcode)
    for binary in (integer_xor, integer_and, integer_or, logical_shift, signed_maximum, unsigned_maximum):
      rejected_binary = {
        "saturate": binary.raw | 1 << 42,
        "repeat": binary.raw | 1 << 40,
        "unsigned-low": binary.raw | 1 << 45,
        "converted destination": binary.raw | 1 << 46,
        "early input": binary.raw | 1 << 47,
        "jump-target": binary.raw | 1 << 59,
        "half sources": binary.raw ^ 1 << 52,
        "source 1 last-use": binary.raw | 1 << 10,
        "source 1 modifier": binary.raw | 1 << 14,
        "source 2 last-use": binary.raw | 1 << 26,
        "source 2 modifier": binary.raw | 1 << 30,
        "shared destination": binary.raw & ~(0xff << 32) | 0xc0 << 32,
        "special destination": binary.raw & ~(0xff << 32) | 0xe0 << 32,
        "constant source 1": binary.raw & ~0xffff | 0x1000,
        "immediate source 1": binary.raw & ~0xffff | 0x2000,
        "relative GPR source 1": binary.raw & ~0xffff | 0x0800,
        "relative constant source 1": binary.raw & ~0xffff | 0x0c00,
        "FLUT source 1": binary.raw & ~0xffff | 0x2802,
        "shared source 1": binary.raw & ~0xffff | 0xc0,
        "special source 1": binary.raw & ~0xffff | 0xe0,
        "constant source 2": binary.raw & ~(0xffff << 16) | 0x1000 << 16,
        "immediate source 2": binary.raw & ~(0xffff << 16) | 0x2000 << 16,
        "relative GPR source 2": binary.raw & ~(0xffff << 16) | 0x0800 << 16,
        "relative constant source 2": binary.raw & ~(0xffff << 16) | 0x0c00 << 16,
        "FLUT source 2": binary.raw & ~(0xffff << 16) | 0x2802 << 16,
        "shared source 2": binary.raw & ~(0xffff << 16) | 0xc0 << 16,
        "special source 2": binary.raw & ~(0xffff << 16) | 0xe0 << 16,
      }
      for modifier,word in rejected_binary.items():
        with self.subTest(binary_opcode=binary.name, binary_modifier=modifier): self.assertIsNone(decode_one(word).opcode)
    for compare in (signed_compare, unsigned_compare, equality_compare):
      rejected_compare = {
        "condition": compare.raw | 1 << 48,
        "saturate": compare.raw | 1 << 42,
        "repeat": compare.raw | 1 << 40,
        "unsigned-low": compare.raw | 1 << 45,
        "early input": compare.raw | 1 << 47,
        "jump-target": compare.raw | 1 << 59,
        "full destination": compare.raw ^ 1 << 46,
        "half sources": compare.raw ^ 1 << 52,
        "source 1 last-use": compare.raw | 1 << 10,
        "source 1 absolute/negate": compare.raw | 1 << 14,
        "source 2 last-use": compare.raw | 1 << 26,
        "source 2 absolute/negate": compare.raw | 1 << 30,
        "shared destination": compare.raw & ~(0xff << 32) | 0xc0 << 32,
        "special destination": compare.raw & ~(0xff << 32) | 0xe0 << 32,
        "special source 1": compare.raw & ~0xffff | 0xe0,
        "special source 2": compare.raw & ~(0xffff << 16) | 0xe0 << 16,
      }
      for modifier,word in rejected_compare.items():
        with self.subTest(compare=compare.opcode, compare_modifier=modifier): self.assertIsNone(decode_one(word).opcode)
    rejected_byte_store = {
      "jump-target": byte_store.raw | 1 << 59,
      "size": byte_store.raw & ~(0x7 << 24) | 2 << 24,
      "offset": byte_store.raw | 1 << 9,
      "destination offset": byte_store.raw ^ 1 << 40,
      "special data": byte_store.raw & ~(0xff << 1) | 0xe0 << 1,
      "address pair crosses GPR file": byte_store.raw & ~(0xff << 41) | 0xbf << 41,
      "special address": byte_store.raw & ~(0xff << 41) | 0xe0 << 41,
    }
    for modifier,word in rejected_byte_store.items():
      with self.subTest(byte_store_modifier=modifier): self.assertIsNone(decode_one(word).opcode)
    rejected_multiply = {
      "jump-target": integer_multiply.raw | 1 << 59,
      "saturate": integer_multiply.raw | 1 << 42,
      "unsigned-low": integer_multiply.raw | 1 << 45,
      "early input": integer_multiply.raw | 1 << 47,
      "converted destination": integer_multiply.raw | 1 << 46,
      "half sources": integer_multiply.raw ^ 1 << 52,
      "repeat": integer_multiply.raw | 1 << 40,
      "source 1 last-use": integer_multiply.raw | 1 << 10,
      "source 1 absolute/negate": integer_multiply.raw | 1 << 14,
      "source 2 last-use": integer_multiply.raw | 1 << 26,
      "source 2 absolute/negate": integer_multiply.raw | 1 << 30,
    }
    for modifier,word in rejected_multiply.items():
      with self.subTest(multiply_modifier=modifier): self.assertIsNone(decode_one(word).opcode)
    rejected_madsh = {
      "different opcode": first_cross_term.raw ^ 1 << 55,
      "jump-target": first_cross_term.raw | 1 << 59,
      "saturate": first_cross_term.raw | 1 << 42,
      "unsigned-low": first_cross_term.raw | 1 << 45,
      "converted destination": first_cross_term.raw | 1 << 46,
      "repeat": first_cross_term.raw | 1 << 40,
      "source 1 last-use": first_cross_term.raw | 1 << 10,
      "source 1 relative encoding": first_cross_term.raw | 1 << 11,
      "source 1 negate": first_cross_term.raw | 1 << 14,
      "source 2 negate": first_cross_term.raw | 1 << 30,
      "source 3 last-use": first_cross_term.raw | 1 << 26,
      "source 3 relative encoding": first_cross_term.raw | 1 << 27,
      "source 3 negate": first_cross_term.raw | 1 << 31,
      "source 3 R flag": first_cross_term.raw | 1 << 29,
      "shared source 1": first_cross_term.raw & ~0x1fff | 0xc0,
      "special source 2": first_cross_term.raw & ~(0xff << 47) | 0xe0 << 47,
      "shared source 3": first_cross_term.raw & ~(0x1fff << 16) | 0xc0 << 16,
      "shared destination": first_cross_term.raw & ~(0xff << 32) | 0xc0 << 32,
    }
    for modifier,word in rejected_madsh.items():
      with self.subTest(madsh_modifier=modifier): self.assertIsNone(decode_one(word).opcode)
    with self.assertRaisesRegex(ValueError, "unmatched IR3 encoding at instruction 0"):
      decode_one(first_cross_term.raw | 1 << 13)
    with self.assertRaisesRegex(ValueError, "unmatched IR3 encoding at instruction 0"):
      decode_one(0x200cc001000000c0 | 1 << 8)
    with self.assertRaisesRegex(ValueError, "unmatched IR3 encoding at instruction 0"):
      decode_one(constant_moves[0][0] | 1 << 11)
    with self.assertRaisesRegex(ValueError, "invalid or reserved IR3 encoding at instruction 0"):
      decode_one(words[7] | 1 << 41)

  def test_production_add_machine_execution_and_retirement(self):
    import struct
    from tinygrad import Device, Tensor
    from tinygrad.codegen import to_program
    from tinygrad.engine.realize import get_runtime
    from tinygrad.runtime.ops_qcom import QCOMComputeQueue, QCOMProgram
    from tinygrad.runtime.autogen import kgsl, mesa
    from test.mockgpu.qcom import qcomdriver
    from test.mockgpu.qcom.a630 import execute_a630, stage_a630
    from test.mockgpu.qcom.pm4 import PM4Type4Packet, PM4Type7Packet, parse_pm4
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
    self.assertEqual((dispatch.instructions[0].raw, dispatch.instructions[0].category, dispatch.instructions[0].name),
                     (0x47180803201f0000, 2, "ashr.b"))
    self.assertEqual((dispatch.instructions[10].category, dispatch.instructions[10].name), (1, None))
    self.assertIn(("SRC_TYPE", 2), dispatch.instructions[10].fields)
    self.assertIn(("DST_TYPE", 5), dispatch.instructions[10].fields)
    cov = dispatch.instructions[10]
    assert cov.dst is not None
    self.assertEqual((cov.opcode, cov.dst.kind, cov.srcs[0].kind), ("cov.u16s32", "gpr", "half"))
    self.assertEqual((dispatch.instructions[16].opcode, dispatch.instructions[17].opcode, dispatch.instructions[19].opcode),
                     ("ldg.u32", "add.f", "stg.u32"))
    self.assertEqual((dispatch.instructions[17].srcs[1].kind, dispatch.instructions[17].srcs[1].value), ("flut", 2))
    self.assertEqual((dispatch.instructions[20].raw, dispatch.instructions[20].name), (6 << 55, "end"))
    self.assertTrue(all(instruction.raw == 0 and instruction.name == "nop" for instruction in dispatch.instructions[21:]))
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
    for write in journal: self.driver.resolve_owned(self.device.fd.fd, write.address, len(write.data))[:] = write.data
    reference = cast(list[float], (Tensor(source_values, device="PYTHON") + 1).tolist())
    self.assertEqual(list(struct.unpack(result_format, result_view)), reference)

    # FLUT immediate 2 is 1.0 and immediate 3 is 2.0 in pinned Mesa ir3-common.xml.
    shader_view = self.driver.resolve_owned(self.device.fd.fd, dispatch.shader_address, dispatch.shader_size)
    add_immediate_byte = 17 * 8 + 2
    self.assertEqual(shader_view[add_immediate_byte], 2)
    try:
      result_view[:] = bytes(result_size)
      shader_view[add_immediate_byte] = 3
      mutated_submission = stage_a630(packets, lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))
      mutated_journal = execute_a630(mutated_submission, lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))
      self.assertEqual(bytes(result_view), bytes(result_size))
      for write in mutated_journal: self.driver.resolve_owned(self.device.fd.fd, write.address, len(write.data))[:] = write.data
      mutated = list(struct.unpack(result_format, result_view))
      self.assertEqual(mutated, [value + 1 for value in reference])
      self.assertNotEqual(mutated, reference)
    finally:
      shader_view[add_immediate_byte] = 2
      result_view[:] = bytes(result_size)

    # SY changes the mapped machine image but only serializes dependencies already ordered by the lane interpreter.
    schedule_byte = 13 * 8 + 7
    self.assertEqual(shader_view[schedule_byte] & 0x10, 0)
    try:
      shader_view[schedule_byte] |= 0x10
      scheduled_submission = stage_a630(packets, lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))
      self.assertIn(("SY", 1), scheduled_submission.dispatches[0].instructions[13].fields)
      scheduled_journal = execute_a630(scheduled_submission, lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))
      for write in scheduled_journal: self.driver.resolve_owned(self.device.fd.fd, write.address, len(write.data))[:] = write.data
      self.assertEqual(list(struct.unpack(result_format, result_view)), reference)
    finally:
      shader_view[schedule_byte] &= ~0x10
      result_view[:] = bytes(result_size)

    # Rename a temporary within the declared full-register footprint and prove execution follows the decoded register ids.
    cov_dst_byte,add_src_byte = 10 * 8 + 4, 13 * 8
    self.assertEqual((shader_view[cov_dst_byte], shader_view[add_src_byte]), (10, 10))
    try:
      shader_view[cov_dst_byte] = shader_view[add_src_byte] = 13
      renamed_submission = stage_a630(packets, lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))
      self.assertEqual((renamed_submission.dispatches[0].instructions[10].dst,
                        renamed_submission.dispatches[0].instructions[13].srcs[0].value),
                       (renamed_submission.dispatches[0].instructions[13].srcs[0], 13))
      renamed_journal = execute_a630(renamed_submission, lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))
      for write in renamed_journal: self.driver.resolve_owned(self.device.fd.fd, write.address, len(write.data))[:] = write.data
      self.assertEqual(list(struct.unpack(result_format, result_view)), reference)
    finally:
      shader_view[cov_dst_byte] = shader_view[add_src_byte] = 10
      result_view[:] = bytes(result_size)

    def stage_instruction(index, raw):
      original = struct.unpack_from("<Q", shader_view, index * 8)[0]
      struct.pack_into("<Q", shader_view, index * 8, raw)
      try: return stage_a630(packets, lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))
      finally: struct.pack_into("<Q", shader_view, index * 8, original)

    redirected_load = dispatch.instructions[16].raw & ~(0xff << 14) | 6 << 14
    with self.assertRaisesRegex(ValueError, "global load 0 does not address its scalar input"):
      execute_a630(stage_instruction(16, redirected_load), lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))
    redirected_add = dispatch.instructions[17].raw & ~0xff | 6
    with self.assertRaisesRegex(ValueError, "f32 add does not consume the global load"):
      execute_a630(stage_instruction(17, redirected_add), lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))
    redirected_store = dispatch.instructions[19].raw & ~(0xff << 1) | 6 << 1
    with self.assertRaisesRegex(ValueError, "global store does not consume the f32 add"):
      execute_a630(stage_instruction(19, redirected_store), lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))
    with self.assertRaisesRegex(ValueError, "unsupported A630 scalar instruction inventory"):
      execute_a630(stage_instruction(15, dispatch.instructions[13].raw),
                   lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))
    for kind,index,message in (("full", 13, "unsupported shared or special IR3 register"),
                               ("half", 8, "unsupported A630 semantic at instruction 8")):
      with self.subTest(shared_register=kind), self.assertRaisesRegex(ValueError, message):
        execute_a630(stage_instruction(index, dispatch.instructions[index].raw & ~(0xff << 32) | 0xc0 << 32),
                     lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))

    constants_view = self.driver.resolve_owned(self.device.fd.fd, dispatch.constants_address, dispatch.constants_size)
    original_output_pointer = bytes(constants_view[:8])
    output_allocation = self.allocation_for(int(result_buffer._buf.va_addr), result_size)
    self.assertIsNotNone(output_allocation)
    final_word = output_allocation.addr + output_allocation.size - 4
    final_view = self.driver.resolve_owned(self.device.fd.fd, final_word, 4)
    final_before = bytes(final_view)
    try:
      constants_view[:8] = struct.pack("<Q", final_word)
      late_failure = stage_a630(packets, lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))
      with self.assertRaisesRegex(ValueError, "instruction 19 lane 1"):
        execute_a630(late_failure, lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))
      self.assertEqual(bytes(final_view), final_before)

      late_buffer, _, late_request = self.gpu_command(words)
      late_request.timestamp = 0x31415926
      late_signal = self.driver.resolve_owned(self.device.fd.fd, int(self.device.timeline_signal.value_addr), 16)
      late_dummy = self.driver.resolve_owned(self.device.fd.fd, self.device.dummy_addr, 4)
      late_before = (bytes(result_view), bytes(final_view), bytes(late_signal), bytes(late_dummy),
                     self.driver.context_timestamps[self.device.ctx], self.driver.always_on_counter, self.device.last_cmd)
      with self.assertRaisesRegex(RuntimeError, "instruction 19 lane 1"):
        kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=late_request)
      self.assertEqual(late_request.timestamp, 0x31415926)
      self.assertEqual((bytes(result_view), bytes(final_view), bytes(late_signal), bytes(late_dummy),
                        self.driver.context_timestamps[self.device.ctx], self.driver.always_on_counter, self.device.last_cmd), late_before)
      self.device._gpu_free(late_buffer)
    finally: constants_view[:8] = original_output_pointer

    input_view = self.driver.resolve_owned(self.device.fd.fd, int(source_buffer._buf.va_addr), result_size)
    try:
      constants_view[:8] = constants_view[8:16]
      alias_input_buffer, _, alias_input_request = self.gpu_command(words)
      alias_input_request.timestamp = 0x42424242
      alias_input_signal = self.driver.resolve_owned(self.device.fd.fd, int(self.device.timeline_signal.value_addr), 16)
      alias_input_dummy = self.driver.resolve_owned(self.device.fd.fd, self.device.dummy_addr, 4)
      alias_input_before = (bytes(result_view), bytes(input_view), bytes(alias_input_signal), bytes(alias_input_dummy),
                            self.driver.context_timestamps[self.device.ctx], self.driver.always_on_counter, self.device.last_cmd)
      with self.assertRaisesRegex(RuntimeError, "A630 global store aliases snapshotted A630 global input"):
        kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=alias_input_request)
      self.assertEqual(alias_input_request.timestamp, 0x42424242)
      self.assertEqual((bytes(result_view), bytes(input_view), bytes(alias_input_signal), bytes(alias_input_dummy),
                        self.driver.context_timestamps[self.device.ctx], self.driver.always_on_counter, self.device.last_cmd), alias_input_before)
      self.device._gpu_free(alias_input_buffer)
    finally: constants_view[:8] = original_output_pointer

    first_byte = shader_view[0]
    try:
      shader_view[0] ^= 1
      self.assertEqual(dispatch.shader_image[0], first_byte)
    finally: shader_view[0] = first_byte

    end_reserved_byte = 20 * 8 + 4
    try:
      shader_view[end_reserved_byte] ^= 1
      with self.assertRaisesRegex(ValueError, "invalid or reserved IR3 encoding at instruction 20"):
        stage_a630(packets, lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))
    finally: shader_view[end_reserved_byte] ^= 1

    def mutate(packet, payload_index, value):
      mutated = list(words)
      mutated[packet.word_offset + 1 + payload_index] = value
      return mutated

    def remove(packet): return words[:packet.word_offset] + words[packet.word_offset+len(packet.values)+1:]

    wait_packet = next(packet for packet in packets if isinstance(packet, PM4Type7Packet) and packet.opcode == mesa.CP_WAIT_REG_MEM)
    final_event = [packet for packet in packets if isinstance(packet, PM4Type7Packet) and
                   packet.opcode == mesa.CP_EVENT_WRITE and len(packet.values) == 4][-1]
    marker_packet = next(packet for packet in packets if isinstance(packet, PM4Type7Packet) and packet.opcode == mesa.CP_SET_MARKER)
    load_packets = [packet for packet in packets if isinstance(packet, PM4Type7Packet) and packet.opcode == mesa.CP_LOAD_STATE6_FRAG]
    constants_packet = next(packet for packet in load_packets if packet.values[0] >> 14 & 0x3 == mesa.ST_CONSTANTS)
    shader_packet = next(packet for packet in load_packets if packet.values[0] >> 14 & 0x3 == mesa.ST_SHADER)
    exec_packet = next(packet for packet in packets if isinstance(packet, PM4Type7Packet) and packet.opcode == mesa.CP_EXEC_CS)
    ndrange_packet = next(packet for packet in packets if isinstance(packet, PM4Type4Packet) and packet.register == mesa.REG_A6XX_SP_CS_NDRANGE_0)
    cntl_packet = next(packet for packet in packets if isinstance(packet, PM4Type4Packet) and packet.register == mesa.REG_A6XX_SP_CS_CNTL_0)
    instr_packet = next(packet for packet in packets if isinstance(packet, PM4Type4Packet) and packet.register == mesa.REG_A6XX_SP_CS_INSTR_SIZE)
    stack_packet = next(packet for packet in packets if isinstance(packet, PM4Type4Packet) and
                        packet.register == mesa.REG_A6XX_SP_CS_PVT_MEM_STACK_OFFSET)
    config_packet = next(packet for packet in packets if isinstance(packet, PM4Type4Packet) and packet.register == mesa.REG_A6XX_SP_CS_CONFIG)
    mode_packet = next(packet for packet in packets if isinstance(packet, PM4Type4Packet) and packet.register == mesa.REG_A6XX_SP_MODE_CNTL)
    update_clear = [packet for packet in packets if isinstance(packet, PM4Type4Packet) and
                    packet.register == mesa.REG_A6XX_SP_UPDATE_CNTL][1]
    semantic_cases = (
      mutate(wait_packet, 0, wait_packet.values[0] | 1 << 31),
      mutate(marker_packet, 0, marker_packet.values[0] | 1 << 8),
      mutate(constants_packet, 0, constants_packet.values[0] + (1 << 22)),
      mutate(shader_packet, 1, shader_packet.values[1] | 1),
      mutate(exec_packet, 1, 0),
      mutate(ndrange_packet, 9, ndrange_packet.values[9] + 1),
      mutate(cntl_packet, 4, cntl_packet.values[4] + 128),
      mutate(instr_packet, 0, instr_packet.values[0] + 1),
      mutate(stack_packet, 0, stack_packet.values[0] + 1),
      mutate(config_packet, 0, config_packet.values[0] | 1 << 17),
      mutate(update_clear, 0, 1),
      remove(mode_packet),
      remove(exec_packet),
    )
    for mutated in semantic_cases:
      resolver = mock.Mock(side_effect=AssertionError("semantic failure reached resolver"))
      with self.subTest(word=next(i for i,(left,right) in enumerate(zip(words, mutated)) if left != right)), self.assertRaises(ValueError):
        stage_a630(parse_pm4(mutated), resolver)
      resolver.assert_not_called()

    overdeclared = mutate(cntl_packet, 0, cntl_packet.values[0] + 0x80)
    overdeclared_submission = stage_a630(parse_pm4(overdeclared),
                                        lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))
    with self.assertRaisesRegex(ValueError, "register footprints do not match decoded operands"):
      execute_a630(overdeclared_submission, lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))

    repeated_queue = self.device.hw_compute_queue_t()
    repeated_queue.exec(runtime, args, program_spec.arg.global_size, program_spec.arg.local_size)
    repeated_queue.exec(runtime, args, program_spec.arg.global_size, program_spec.arg.local_size)
    repeated_packets = parse_pm4(tuple(repeated_queue._q))
    repeated = stage_a630(repeated_packets, lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))
    self.assertEqual(len(repeated.dispatches), 2)
    self.assertLess(repeated.dispatches[0].word_offset, repeated.dispatches[1].word_offset)
    self.assertEqual(tuple(item.shader_image for item in repeated.dispatches), (dispatch.shader_image, dispatch.shader_image))
    second_exec = [packet for packet in repeated_packets if isinstance(packet, PM4Type7Packet) and packet.opcode == mesa.CP_EXEC_CS][1]
    invalid_second = list(repeated_queue._q)
    invalid_second[second_exec.word_offset+2] = 0
    resolver = mock.Mock(side_effect=AssertionError("invalid second dispatch reached resolver"))
    with self.assertRaises(ValueError): stage_a630(parse_pm4(invalid_second), resolver)
    resolver.assert_not_called()

    repeated_buffer, _, repeated_request = self.gpu_command(tuple(repeated_queue._q))
    repeated_request.timestamp = 0x27182818
    repeated_signal = self.driver.resolve_owned(self.device.fd.fd, int(self.device.timeline_signal.value_addr), 16)
    repeated_dummy = self.driver.resolve_owned(self.device.fd.fd, self.device.dummy_addr, 4)
    repeated_before = (bytes(result_view), bytes(repeated_signal), bytes(repeated_dummy),
                       self.driver.context_timestamps[self.device.ctx], self.driver.always_on_counter, self.device.last_cmd)
    with self.assertRaisesRegex(RuntimeError, "at most one dispatch"):
      kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=repeated_request)
    self.assertEqual(repeated_request.timestamp, 0x27182818)
    self.assertEqual((bytes(result_view), bytes(repeated_signal), bytes(repeated_dummy),
                      self.driver.context_timestamps[self.device.ctx], self.driver.always_on_counter, self.device.last_cmd), repeated_before)
    self.device._gpu_free(repeated_buffer)

    alias_words = list(words)
    output_address = int(result_buffer._buf.va_addr)
    alias_words[final_event.word_offset+2:final_event.word_offset+4] = (output_address & 0xffffffff, output_address >> 32)
    alias_buffer, _, alias_request = self.gpu_command(alias_words)
    alias_request.timestamp = 0x16180339
    alias_before = (bytes(result_view), bytes(repeated_signal), bytes(repeated_dummy),
                    self.driver.context_timestamps[self.device.ctx], self.driver.always_on_counter, self.device.last_cmd)
    with self.assertRaisesRegex(RuntimeError, "overlapping A630 global store and event value"):
      kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=alias_request)
    self.assertEqual(alias_request.timestamp, 0x16180339)
    self.assertEqual((bytes(result_view), bytes(repeated_signal), bytes(repeated_dummy),
                      self.driver.context_timestamps[self.device.ctx], self.driver.always_on_counter, self.device.last_cmd), alias_before)
    self.device._gpu_free(alias_buffer)

    result_view[:] = bytes(result_size)
    signal_view = self.driver.resolve_owned(self.device.fd.fd, int(self.device.timeline_signal.value_addr), 16)
    signal_view[:] = bytes(16)
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

    try:
      result_view[:] = bytes(result_size)
      shader_view[add_immediate_byte] = 3
      queue.submit(self.device)
      self.assertEqual(self.device.last_cmd, retired_command + 1)
      self.assertEqual(list(struct.unpack(result_format, result_view)), [value + 1 for value in reference])
      mutated_counter = struct.unpack_from("<Q", signal_view, 8)[0]
      self.assertGreater(mutated_counter, first_counter)
      retired_command = self.device.last_cmd
    finally: shader_view[add_immediate_byte] = 2

    result_view[:] = bytes(result_size)
    queue.submit(self.device)
    self.assertEqual(self.device.last_cmd, retired_command + 1)
    self.assertGreater(struct.unpack_from("<Q", signal_view, 8)[0], mutated_counter)
    self.assertEqual(list(struct.unpack(result_format, result_view)), reference)
    retired_command = self.device.last_cmd

    profile_start, profile_end = self.device.new_signal(), self.device.new_signal()
    profile_queue = self.device.hw_compute_queue_t().wait(self.device.timeline_signal, self.device.timeline_value - 1).memory_barrier() \
      .timestamp(profile_start).exec(runtime, args, program_spec.arg.global_size, program_spec.arg.local_size) \
      .timestamp(profile_end).signal(self.device.timeline_signal, self.device.timeline_value)
    result_view[:] = bytes(result_size)
    effect_order:list[str] = []
    real_counter = qcomdriver.time.perf_counter_ns
    real_execute = qcomdriver.execute_a630
    def ordered_counter():
      effect_order.append("counter")
      return real_counter()
    def ordered_execute(*execute_args, **execute_kwargs):
      effect_order.append("execute")
      return real_execute(*execute_args, **execute_kwargs)
    with mock.patch.object(qcomdriver.time, "perf_counter_ns", side_effect=ordered_counter), \
         mock.patch.object(qcomdriver, "execute_a630", side_effect=ordered_execute):
      profile_queue.submit(self.device)
    self.assertEqual(effect_order, ["counter", "execute", "counter"])
    self.assertEqual(self.device.last_cmd, retired_command + 1)
    start_counter = profile_start.base_buf.cpu_view().view(8, 8, "Q")[0]
    end_counter = profile_end.base_buf.cpu_view().view(8, 8, "Q")[0]
    self.assertGreater(end_counter, start_counter)
    self.assertGreater(profile_end.timestamp - profile_start.timestamp, 0)
    self.assertEqual(list(struct.unpack(result_format, result_view)), reference)
    retired_command = self.device.last_cmd

    end_to_end = (Tensor(source_values, device=Device.DEFAULT) + 1).realize()
    self.assertEqual(cast(list[float], end_to_end.tolist()), reference)
    self.assertGreater(self.device.last_cmd, retired_command)
    retired_command = self.device.last_cmd

    unsatisfied_words = mutate(wait_packet, 3, self.device.timeline_signal.value + 1)
    unsatisfied_buffer, _, unsatisfied_request = self.gpu_command(unsatisfied_words)
    unsatisfied_request.timestamp = 0x11223344
    result_view[:] = bytes(result_size)
    before_wait_failure = (bytes(result_view), bytes(signal_view), bytes(dummy_view),
                           self.driver.context_timestamps[self.device.ctx], self.driver.always_on_counter)
    with self.assertRaisesRegex(RuntimeError, "unsatisfied memory wait"):
      kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=unsatisfied_request)
    self.assertEqual((unsatisfied_request.timestamp, self.device.last_cmd), (0x11223344, retired_command))
    self.assertEqual((bytes(result_view), bytes(signal_view), bytes(dummy_view),
                      self.driver.context_timestamps[self.device.ctx], self.driver.always_on_counter), before_wait_failure)
    self.device._gpu_free(unsatisfied_buffer)

    invalid_tail = list(words)
    dummy_allocation = self.allocation_for(self.device.dummy_addr, 1)
    self.assertIsNotNone(dummy_allocation)
    invalid_address = dummy_allocation.addr + dummy_allocation.size
    invalid_tail[final_event.word_offset+2:final_event.word_offset+4] = (invalid_address & 0xffffffff, invalid_address >> 32)
    invalid_buffer, _, invalid_request = self.gpu_command(invalid_tail)
    invalid_request.timestamp = 0x13572468
    before = (dict(self.driver.contexts), dict(self.driver.context_timestamps), dict(self.driver.user_mappings),
              dict(self.driver.power_levels), self.driver.always_on_counter, bytes(result_view), bytes(signal_view), bytes(dummy_view),
              bytes(invalid_buffer.cpu_view().mv[:len(invalid_tail)*4]),
              bytes(self.driver.resolve_owned(self.device.fd.fd, dispatch.shader_address, dispatch.shader_size)),
              bytes(self.driver.resolve_owned(self.device.fd.fd, dispatch.constants_address, dispatch.constants_size)))
    with self.assertRaisesRegex(RuntimeError, "not in one owned mapping"):
      kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=invalid_request)
    self.assertEqual(invalid_request.timestamp, 0x13572468)
    self.assertEqual((self.driver.contexts, self.driver.context_timestamps, self.driver.user_mappings,
                      self.driver.power_levels, self.driver.always_on_counter, bytes(result_view), bytes(signal_view), bytes(dummy_view),
                      bytes(invalid_buffer.cpu_view().mv[:len(invalid_tail)*4]),
                      bytes(self.driver.resolve_owned(self.device.fd.fd, dispatch.shader_address, dispatch.shader_size)),
                      bytes(self.driver.resolve_owned(self.device.fd.fd, dispatch.constants_address, dispatch.constants_size))), before)
    self.assertEqual(self.device.last_cmd, retired_command)
    self.device._gpu_free(invalid_buffer)

  def test_production_two_input_add_machine_execution_and_retirement(self):
    import struct
    from tinygrad import Device, Tensor
    from tinygrad.codegen import to_program
    from tinygrad.engine.realize import get_runtime
    from tinygrad.runtime.autogen import kgsl, mesa
    from test.mockgpu.qcom.a630 import execute_a630, stage_a630
    from test.mockgpu.qcom.pm4 import parse_pm4

    left_values, right_values = [1.0, -2.5, 1024.0], [4.0, 0.5, -24.0]
    left, right = Tensor(left_values, device=Device.DEFAULT).realize(), Tensor(right_values, device=Device.DEFAULT).realize()
    result = left + right
    program_spec = to_program(result.schedule_linear().src[-1].src[0], self.device.renderer)
    runtime = get_runtime(self.device.device, program_spec)
    result_buffer, left_buffer, right_buffer = (cast(Any, tensor.uop.buffer) for tensor in (result, left, right))
    result_buffer.allocate()
    args = runtime.fill_kernargs([result_buffer._buf, left_buffer._buf, right_buffer._buf])
    queue = self.device.hw_compute_queue_t().wait(self.device.timeline_signal, self.device.timeline_value - 1).memory_barrier()
    queue.exec(runtime, args, program_spec.arg.global_size, program_spec.arg.local_size)
    queue.signal(self.device.timeline_signal, self.device.next_timeline())
    words = tuple(queue._q)
    submission = stage_a630(parse_pm4(words), lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))
    dispatch = submission.dispatches[0]

    self.assertEqual(Device.DEFAULT, "QCOM")
    self.assertEqual((runtime.image_size, dispatch.local_size, dispatch.groups), (256, (3, 1, 1), (1, 1, 1)))
    self.assertEqual(dict(dispatch.registers)[mesa.REG_A6XX_SP_CS_CNTL_0], 0x282)
    self.assertEqual(tuple(instruction.raw for instruction in dispatch.instructions[18:27]),
                     (0xc006000b01810001, 0x2009400c00000001, 0xc006001001834001, 0x20000000000,
                      0x421000070009000c, 0x5018080b0010000b, 0x20000000000, 0xc0c60d0001800016, 0x300000000000000))
    self.assertEqual(tuple(instruction.opcode for instruction in dispatch.instructions[18:27]),
                     ("ldg.u32", "cov.u16s32", "ldg.u32", "nop", "add.u", "add.f", "nop", "stg.u32", "end"))
    self.assertEqual(struct.unpack_from("<3Q", dispatch.constants_image),
                     (int(result_buffer._buf.va_addr), int(left_buffer._buf.va_addr), int(right_buffer._buf.va_addr)))

    shader_view = self.driver.resolve_owned(self.device.fd.fd, dispatch.shader_address, dispatch.shader_size)
    second_load = dispatch.instructions[20].raw
    struct.pack_into("<Q", shader_view, 20 * 8, second_load & ~(0xff << 14) | 4 << 14)
    try:
      redirected = stage_a630(parse_pm4(words), lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))
    finally: struct.pack_into("<Q", shader_view, 20 * 8, second_load)
    with self.assertRaisesRegex(ValueError, "global load 1 does not address its scalar input"):
      execute_a630(redirected, lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))

    result_size = len(left_values) * 4
    result_view = self.driver.resolve_owned(self.device.fd.fd, int(result_buffer._buf.va_addr), result_size)
    result_view[:] = bytes(result_size)
    last_command = self.device.last_cmd
    queue.submit(self.device)
    reference = cast(list[float], (Tensor(left_values, device="PYTHON") + Tensor(right_values, device="PYTHON")).tolist())
    self.assertGreater(self.device.last_cmd, last_command)
    self.assertEqual(list(struct.unpack(f"<{len(left_values)}f", result_view)), reference)

    constants_view = self.driver.resolve_owned(self.device.fd.fd, dispatch.constants_address, dispatch.constants_size)
    original_output_pointer = bytes(constants_view[:8])
    right_view = self.driver.resolve_owned(self.device.fd.fd, int(right_buffer._buf.va_addr), result_size)
    try:
      constants_view[:8] = constants_view[16:24]
      alias_buffer, _, alias_request = self.gpu_command(words)
      alias_request.timestamp = 0x23456789
      before = (bytes(result_view), bytes(right_view), self.driver.context_timestamps[self.device.ctx], self.device.last_cmd)
      with self.assertRaisesRegex(RuntimeError, "global store aliases snapshotted A630 global input 1"):
        kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=alias_request)
      self.assertEqual(alias_request.timestamp, 0x23456789)
      self.assertEqual((bytes(result_view), bytes(right_view), self.driver.context_timestamps[self.device.ctx], self.device.last_cmd), before)
      self.device._gpu_free(alias_buffer)
    finally: constants_view[:8] = original_output_pointer

  def test_production_integer_add_wraps_from_mapped_machine_bytes(self):
    import struct
    from dataclasses import replace
    from tinygrad import Device, Tensor, dtypes
    from tinygrad.runtime.autogen import kgsl
    from test.mockgpu.qcom import qcomdriver
    from test.mockgpu.qcom.a630 import A630IR3Operand, decode_a630_ir3

    submissions,command_images = [],[]
    real_execute = qcomdriver.execute_a630
    real_plan = self.driver._plan_a630_retirement
    def capture_execution(submission, resolver):
      submissions.append(submission)
      return real_execute(submission, resolver)
    def capture_plan(fd, submission, command_address, command_size):
      command_images.append(bytes(self.driver.resolve_owned(fd, command_address, command_size)))
      return real_plan(fd, submission, command_address, command_size)

    cases = ((dtypes.int, [dtypes.int.max, dtypes.int.min, -7], [1, -1, 3], [dtypes.int.min, dtypes.int.max, -4]),
             (dtypes.uint, [dtypes.uint.max, 0x80000000, 7], [1, 0x80000000, 5], [0, 0, 12]),
             (dtypes.int, [dtypes.int.max], [1], [dtypes.int.min]),
             (dtypes.uint, [dtypes.uint.max], [1], [0]))
    actual = []
    with mock.patch.object(qcomdriver, "execute_a630", side_effect=capture_execution), \
         mock.patch.object(self.driver, "_plan_a630_retirement", side_effect=capture_plan):
      for dtype,left,right,_ in cases:
        actual.append((Tensor(left, dtype=dtype, device=Device.DEFAULT) + Tensor(right, dtype=dtype, device=Device.DEFAULT)).tolist())
    reference = [(Tensor(left, dtype=dtype, device="PYTHON") + Tensor(right, dtype=dtype, device="PYTHON")).tolist()
                 for dtype,left,right,_ in cases]

    self.assertEqual((Device.DEFAULT, (DEV.interface, DEV.device, DEV.renderer, DEV.arch)),
                     ("QCOM", ("MOCK", "QCOM", "IR3", "a630")))
    self.assertEqual(actual, reference)
    self.assertEqual(actual, [expected for *_,expected in cases])
    self.assertEqual((len(submissions), len(command_images)), (4, 4))
    decoded_dispatches = []
    for (_,left,_,_),submission in zip(cases, submissions):
      dispatch = submission.dispatches[0]
      size = len(left)
      self.assertEqual((dispatch.local_size, dispatch.groups, dispatch.global_size), ((size, 1, 1), (1, 1, 1), (size, 1, 1)))
      loads = tuple(instruction for instruction in dispatch.instructions if instruction.opcode == "ldg.u32")
      stores = tuple(instruction for instruction in dispatch.instructions if instruction.opcode == "stg.u32")
      pointer_moves = tuple(instruction for instruction in dispatch.instructions if instruction.opcode == "mov.u32" and
                            instruction.srcs[0].kind == "const")
      load_destinations = frozenset(instruction.dst for instruction in loads)
      data_adds = tuple(instruction for instruction in dispatch.instructions if instruction.opcode == "add.u" and
                        frozenset(instruction.srcs) == load_destinations)
      self.assertEqual((len(loads), len(stores), len(data_adds)), (2, 1, 1))
      self.assertIn(("SY", 1), data_adds[0].fields)
      self.assertTrue(all(("TYPE", 3) in instruction.fields for instruction in loads + stores))
      self.assertEqual(tuple(sorted(instruction.srcs[0].value for instruction in pointer_moves)), tuple(range(6)) if size == 1 else ())
      if pointer_moves:
        destinations = {}
        for instruction in pointer_moves:
          self.assertIsNotNone(instruction.dst)
          destinations[instruction.srcs[0].value] = instruction.dst.value
        bases = (stores[0].srcs[0].value, *(instruction.srcs[0].value for instruction in loads))
        self.assertEqual(tuple((destinations[2*i], destinations[2*i+1]) for i in range(3)),
                         tuple((base, base+1) for base in bases))
      decoded_dispatches.append((dispatch, data_adds[0]))

    def reject_mapped_mutation(case_index, instruction, mutated_raw, message, expected_srcs):
      submission = submissions[case_index]
      dispatch = submission.dispatches[0]
      shader = self.driver.resolve_owned(self.device.fd.fd, dispatch.shader_address, dispatch.shader_size)
      original = bytes(shader[instruction.index*8:(instruction.index+1)*8])
      output_base = struct.unpack_from("<Q", dispatch.constants_image)[0]
      output = self.driver.resolve_owned(self.device.fd.fd, output_base, len(cases[case_index][1]) * 4)
      output_before = bytes(output)
      request_buffer,_,request = self.gpu_command(struct.unpack(f"<{len(command_images[case_index]) // 4}I", command_images[case_index]))
      request.timestamp = 0x10293847 + case_index
      signal = self.driver.resolve_owned(self.device.fd.fd, int(self.device.timeline_signal.value_addr), 16)
      state_before = (bytes(output), bytes(signal), self.driver.context_timestamps[self.device.ctx],
                      self.driver.always_on_counter, self.device.last_cmd)
      try:
        struct.pack_into("<Q", shader, instruction.index * 8, mutated_raw)
        image = bytes(shader)
        mutated_dispatch = replace(dispatch, shader_image=image, instructions=decode_a630_ir3(image))
        self.assertEqual(mutated_dispatch.instructions[instruction.index].srcs, expected_srcs)
        with self.assertRaisesRegex(ValueError, message):
          real_execute(replace(submission, dispatches=(mutated_dispatch,)),
                       lambda address,length: self.driver.resolve_owned(self.device.fd.fd, address, length))
        with self.assertRaisesRegex(RuntimeError, message): kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=request)
        self.assertEqual((request.timestamp, bytes(output), bytes(signal), self.driver.context_timestamps[self.device.ctx],
                          self.driver.always_on_counter, self.device.last_cmd), (0x10293847 + case_index, *state_before))
      finally:
        shader[instruction.index*8:(instruction.index+1)*8] = original
        self.device._gpu_free(request_buffer)
      self.assertEqual((bytes(shader[instruction.index*8:(instruction.index+1)*8]), bytes(output)), (original, output_before))

    _,data_add = decoded_dispatches[0]
    reject_mapped_mutation(0, data_add, data_add.raw & ~(0xffff << 16) | data_add.srcs[0].value << 16,
                           "u32 add does not consume both global loads", (data_add.srcs[0], data_add.srcs[0]))
    scalar_dispatch,_ = decoded_dispatches[2]
    pointer_move = next(instruction for instruction in scalar_dispatch.instructions if instruction.opcode == "mov.u32" and
                        instruction.srcs == (A630IR3Operand("const", 2),))
    reject_mapped_mutation(2, pointer_move, pointer_move.raw & ~0x7ff | 6, "unsupported A630 constant-pointer source",
                           (A630IR3Operand("const", 6),))
    self.assertEqual((Tensor([9], dtype=dtypes.int, device=Device.DEFAULT) +
                      Tensor([-4], dtype=dtypes.int, device=Device.DEFAULT)).tolist(), [5])

  def test_production_integer_subtract_wraps_from_mapped_machine_bytes(self):
    import operator
    from tinygrad import Device, Tensor, dtypes
    cases = ((dtypes.int, dtypes.int.min, 1, dtypes.int.max),
             (dtypes.int, dtypes.int.max, -1, dtypes.int.min),
             (dtypes.uint, 0, 1, dtypes.uint.max),
             (dtypes.uint, 0x80000000, 0xffffffff, 0x80000001),
             (dtypes.int, 9, 4, 5))
    # Reversing only the mapped sources changes 9 - 4 to 4 - 9 while preserving the valid SUB.U encoding.
    self._assert_production_integer_binary_uses_mapped_machine_bytes(
      tensor_operator=operator.sub, opcode="sub.u", opcode_bits=0x12, operation="subtraction", cases=cases,
      mutation_opcode="sub.u", mutation_opcode_bits=0x12, mutation_expected=(-5) & 0xffffffff, swap_mutation_sources=True)
    self.assertEqual((Tensor([9], dtype=dtypes.int, device=Device.DEFAULT) -
                      Tensor([4], dtype=dtypes.int, device=Device.DEFAULT)).tolist(), [5])

  def _assert_production_integer_binary_uses_mapped_machine_bytes(self, *, tensor_operator, opcode, opcode_bits, operation,
                                                                  cases, mutation_opcode, mutation_opcode_bits, mutation_expected,
                                                                  unsupported_opcode_bits=None, unsupported_name=None, value_type="u32",
                                                                  swap_mutation_sources=False, unsupported_shift_counts=()):
    import struct
    from dataclasses import replace
    from tinygrad import Device, Tensor
    from tinygrad.runtime.autogen import kgsl
    from test.mockgpu.qcom import qcomdriver
    from test.mockgpu.qcom.a630 import decode_a630_ir3

    submissions,command_images = [],[]
    real_execute = qcomdriver.execute_a630
    real_plan = self.driver._plan_a630_retirement
    def capture_execution(submission, resolver):
      submissions.append(submission)
      return real_execute(submission, resolver)
    def capture_plan(fd, submission, command_address, command_size):
      command_images.append(bytes(self.driver.resolve_owned(fd, command_address, command_size)))
      return real_plan(fd, submission, command_address, command_size)

    actual,live_tensors = [],[]
    with mock.patch.object(qcomdriver, "execute_a630", side_effect=capture_execution), \
         mock.patch.object(self.driver, "_plan_a630_retirement", side_effect=capture_plan):
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
      loads = tuple(instruction for instruction in dispatch.instructions if instruction.opcode == "ldg.u32")
      stores = tuple(instruction for instruction in dispatch.instructions if instruction.opcode == "stg.u32")
      binary_instructions = tuple(instruction for instruction in dispatch.instructions if instruction.opcode == opcode)
      pointer_moves = tuple(instruction for instruction in dispatch.instructions if instruction.opcode == "mov.u32" and
                            instruction.srcs[0].kind == "const")
      self.assertEqual((len(loads), len(stores), len(binary_instructions), len(pointer_moves)), (2, 1, 1, 6))
      binary = binary_instructions[0]
      self.assertEqual((binary.raw >> 61, binary.raw >> 53 & 0x3f, binary.srcs),
                       (2, opcode_bits, (loads[0].dst, loads[1].dst)))
      self.assertIsNotNone(binary.dst)
      assert binary.dst is not None
      self.assertEqual(binary.dst.kind, "gpr")
      self.assertEqual(stores[0].srcs[1], binary.dst)
      self.assertTrue({("NAME", opcode), ("SY", 1), ("SS", 0), ("NOP", 3), ("DST_HALF", 0),
                       ("JP", 0), ("SAT", 0), ("UL", 0), ("EI", 0)} <= set(binary.fields))
      self.assertTrue(all(("TYPE", 3) in instruction.fields for instruction in loads + stores))
      destinations = {}
      for instruction in pointer_moves:
        assert instruction.dst is not None
        destinations[instruction.srcs[0].value] = instruction.dst.value
      bases = (stores[0].srcs[0].value, *(instruction.srcs[0].value for instruction in loads))
      self.assertEqual(tuple((destinations[2*i], destinations[2*i+1]) for i in range(3)),
                       tuple((base, base+1) for base in bases))
      decoded_dispatches.append((dispatch, binary, stores[0]))

    # The newest command avoids replaying a stale historical EVENT_WRITE after the numerical matrix.
    case_index = len(cases) - 1
    self.assertNotEqual(mutation_expected, cases[case_index][3] & 0xffffffff)
    submission = submissions[case_index]
    dispatch,binary,store = decoded_dispatches[case_index]
    shader = self.driver.resolve_owned(self.device.fd.fd, dispatch.shader_address, dispatch.shader_size)
    original = bytes(shader[binary.index*8:(binary.index+1)*8])
    output_base = struct.unpack_from("<Q", dispatch.constants_image)[0]
    output = self.driver.resolve_owned(self.device.fd.fd, output_base, 4)
    request_buffer,_,request = self.gpu_command(struct.unpack(f"<{len(command_images[case_index]) // 4}I", command_images[case_index]))
    timestamp_before = self.driver.context_timestamps[self.device.ctx]
    try:
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
      struct.pack_into("<Q", shader, binary.index * 8, mutation_raw)
      mutated = decode_a630_ir3(bytes(shader))[binary.index]
      self.assertEqual((mutated.opcode, mutated.dst, mutated.srcs), (mutation_opcode, binary.dst, mutation_srcs))
      output[:] = struct.pack("<I", mutation_expected ^ 0xffffffff)
      kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=request)
      self.assertEqual(struct.unpack("<I", output)[0], mutation_expected)
      self.assertEqual((request.timestamp, self.driver.context_timestamps[self.device.ctx]), ((timestamp_before + 1) & 0xffffffff,) * 2)
    finally:
      shader[binary.index*8:(binary.index+1)*8] = original
      self.device._gpu_free(request_buffer)
    self.assertEqual(bytes(shader[binary.index*8:(binary.index+1)*8]), original)

    def reject_mapped_mutation(instruction, raw, expected_opcode, expected_srcs, message, *, expected_name=None):
      original_instruction = bytes(shader[instruction.index*8:(instruction.index+1)*8])
      request_buffer,_,request = self.gpu_command(struct.unpack(f"<{len(command_images[case_index]) // 4}I", command_images[case_index]))
      request.timestamp = marker = 0x27182818 + instruction.index
      signal = self.driver.resolve_owned(self.device.fd.fd, int(self.device.timeline_signal.value_addr), 16)
      state_before = (bytes(output), bytes(signal), self.driver.context_timestamps[self.device.ctx],
                      self.driver.always_on_counter, self.device.last_cmd, self.device.error_state)
      try:
        struct.pack_into("<Q", shader, instruction.index * 8, raw)
        image = bytes(shader)
        mutated_dispatch = replace(dispatch, shader_image=image, instructions=decode_a630_ir3(image))
        mutated = mutated_dispatch.instructions[instruction.index]
        self.assertEqual((mutated.opcode, mutated.srcs), (expected_opcode, expected_srcs))
        if expected_name is not None: self.assertEqual(mutated.name, expected_name)
        with self.assertRaisesRegex(ValueError, message):
          real_execute(replace(submission, dispatches=(mutated_dispatch,)),
                       lambda address,length: self.driver.resolve_owned(self.device.fd.fd, address, length))
        with self.assertRaisesRegex(RuntimeError, message): kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=request)
        self.assertEqual((request.timestamp, bytes(output), bytes(signal), self.driver.context_timestamps[self.device.ctx],
                          self.driver.always_on_counter, self.device.last_cmd, self.device.error_state), (marker, *state_before))
      finally:
        shader[instruction.index*8:(instruction.index+1)*8] = original_instruction
        self.device._gpu_free(request_buffer)
      self.assertEqual(bytes(shader[instruction.index*8:(instruction.index+1)*8]), original_instruction)

    dataflow_message = f"{value_type} {operation} does not consume both global loads"
    src1 = binary.raw & 0xffff
    reject_mapped_mutation(binary, binary.raw & ~(0xffff << 16) | src1 << 16,
                           opcode, (binary.srcs[0], binary.srcs[0]), dataflow_message)
    nop = next(instruction for instruction in dispatch.instructions if instruction.opcode == "nop")
    reject_mapped_mutation(nop, binary.raw, opcode, binary.srcs, dataflow_message)
    redirected = next(operand for operand in binary.srcs if operand != binary.dst)
    reject_mapped_mutation(store, store.raw & ~(0xff << 1) | redirected.value << 1, "stg.u32",
                           (store.srcs[0], redirected), f"global store does not consume the {value_type} {operation}")
    reject_mapped_mutation(binary, binary.raw | 1 << 14, None, (),
                           f"unsupported A630 semantic at instruction {binary.index}", expected_name=opcode)
    if unsupported_opcode_bits is not None:
      self.assertIsNotNone(unsupported_name)
      unsupported_raw = binary.raw & ~(0x3f << 53) | unsupported_opcode_bits << 53
      reject_mapped_mutation(binary, unsupported_raw, None, (),
                             f"unsupported A630 semantic at instruction {binary.index}", expected_name=unsupported_name)

    if unsupported_shift_counts:
      loads = tuple(instruction for instruction in dispatch.instructions if instruction.opcode == "ldg.u32")
      pointer_moves = tuple(instruction for instruction in dispatch.instructions if instruction.opcode == "mov.u32" and
                            instruction.srcs[0].kind == "const")
      rhs_load = next(instruction for instruction in loads if instruction.dst == binary.srcs[1])
      constants_by_register = {instruction.dst.value:instruction.srcs[0].value for instruction in pointer_moves if instruction.dst is not None}
      rhs_constant = constants_by_register[rhs_load.srcs[0].value]
      self.assertEqual(rhs_constant & 1, 0)
      rhs_base = struct.unpack_from("<Q", dispatch.constants_image, rhs_constant * 4)[0]
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
    cases = ((dtypes.int, 0, 0, 0),
             (dtypes.int, -1, 0, -1),
             (dtypes.int, dtypes.int.min, dtypes.int.max, -1),
             (dtypes.int, dtypes.int.min, -1, dtypes.int.max),
             (dtypes.int, -1431655766, 252645135, -1515870811),
             (dtypes.uint, 0, dtypes.uint.max, dtypes.uint.max),
             (dtypes.uint, 0x80000000, 0x7fffffff, dtypes.uint.max),
             (dtypes.uint, 0xaaaaaaaa, 0x0f0f0f0f, 0xa5a5a5a5),
             (dtypes.uint, dtypes.uint.max, dtypes.uint.max, 0),
             (dtypes.uint, dtypes.uint.max, 1, 0xfffffffe))
    # The opcode-only ADD.U mutation changes uint.max XOR 1 from 0xfffffffe to wrapped zero.
    self._assert_production_integer_binary_uses_mapped_machine_bytes(
      tensor_operator=operator.xor, opcode="xor.b", opcode_bits=0x1f, operation="bitwise XOR", cases=cases,
      mutation_opcode="add.u", mutation_opcode_bits=0x10, mutation_expected=0)
    self.assertEqual(operator.xor(Tensor([0xaaaaaaaa], dtype=dtypes.uint, device=Device.DEFAULT),
                                  Tensor([0x0f0f0f0f], dtype=dtypes.uint, device=Device.DEFAULT)).tolist(), [0xa5a5a5a5])

  def test_production_integer_and_uses_mapped_machine_bytes(self):
    import operator
    from tinygrad import Device, Tensor, dtypes
    cases = ((dtypes.int, 0, 0, 0),
             (dtypes.int, -1, 0, 0),
             (dtypes.int, -1, 1, 1),
             (dtypes.int, dtypes.int.min, dtypes.int.max, 0),
             (dtypes.int, dtypes.int.min, -1, dtypes.int.min),
             (dtypes.int, -1431655766, 252645135, 168430090),
             (dtypes.uint, 0, dtypes.uint.max, 0),
             (dtypes.uint, 0x80000000, 0x7fffffff, 0),
             (dtypes.uint, 0xaaaaaaaa, 0x0f0f0f0f, 0x0a0a0a0a),
             (dtypes.uint, dtypes.uint.max, 0x80000001, 0x80000001))
    # The opcode-only XOR.B mutation changes uint.max AND 0x80000001 from 0x80000001 to 0x7ffffffe.
    self._assert_production_integer_binary_uses_mapped_machine_bytes(
      tensor_operator=operator.and_, opcode="and.b", opcode_bits=0x1c, operation="bitwise AND", cases=cases,
      mutation_opcode="xor.b", mutation_opcode_bits=0x1f, mutation_expected=0x7ffffffe)
    self.assertEqual(operator.and_(Tensor([0xaaaaaaaa], dtype=dtypes.uint, device=Device.DEFAULT),
                                   Tensor([0x0f0f0f0f], dtype=dtypes.uint, device=Device.DEFAULT)).tolist(), [0x0a0a0a0a])

  def test_production_integer_or_uses_mapped_machine_bytes(self):
    import operator
    from tinygrad import Device, Tensor, dtypes
    cases = ((dtypes.int, 0, 0, 0),
             (dtypes.int, -1, 0, -1),
             (dtypes.int, -1, 1, -1),
             (dtypes.int, dtypes.int.min, dtypes.int.max, -1),
             (dtypes.int, dtypes.int.min, 0, dtypes.int.min),
             (dtypes.int, -1431655766, 252645135, -1347440721),
             (dtypes.uint, 0, dtypes.uint.max, dtypes.uint.max),
             (dtypes.uint, 0x80000000, 0x7fffffff, dtypes.uint.max),
             (dtypes.uint, 0xaaaaaaaa, 0x0f0f0f0f, 0xafafafaf),
             (dtypes.uint, dtypes.uint.max, 0x80000001, dtypes.uint.max))
    # The opcode-only AND.B mutation changes uint.max OR 0x80000001 from uint.max to 0x80000001.
    self._assert_production_integer_binary_uses_mapped_machine_bytes(
      tensor_operator=operator.or_, opcode="or.b", opcode_bits=0x1d, operation="bitwise OR", cases=cases,
      mutation_opcode="and.b", mutation_opcode_bits=0x1c, mutation_expected=0x80000001)
    self.assertEqual(operator.or_(Tensor([0xaaaaaaaa], dtype=dtypes.uint, device=Device.DEFAULT),
                                  Tensor([0x0f0f0f0f], dtype=dtypes.uint, device=Device.DEFAULT)).tolist(), [0xafafafaf])

  def test_production_unsigned_right_shift_uses_mapped_machine_bytes(self):
    import operator
    from tinygrad import Device, Tensor, dtypes
    cases = ((dtypes.uint, 0, 0, 0),
             (dtypes.uint, dtypes.uint.max, 0, dtypes.uint.max),
             (dtypes.uint, 0x80000001, 1, 0x40000000),
             (dtypes.uint, 0x12345678, 4, 0x01234567),
             (dtypes.uint, dtypes.uint.max, 16, 0x0000ffff),
             (dtypes.uint, 0x80000000, 31, 1),
             (dtypes.uint, 16, 2, 4))
    unsupported_counts = (32, 33, dtypes.uint.max)
    self.assertEqual([operator.rshift(Tensor([0x80000001], dtype=dtypes.uint, device="PYTHON"),
                                      Tensor([count], dtype=dtypes.uint, device="PYTHON")).tolist() for count in unsupported_counts],
                     [[0], [0], [0]])
    # Swapping only the mapped SHR.B sources changes 16 >> 2 to 2 >> 16, while keeping both counts in the supported range.
    self._assert_production_integer_binary_uses_mapped_machine_bytes(
      tensor_operator=operator.rshift, opcode="shr.b", opcode_bits=0x37, operation="logical right shift", cases=cases,
      mutation_opcode="shr.b", mutation_opcode_bits=0x37, mutation_expected=0, swap_mutation_sources=True,
      unsupported_shift_counts=unsupported_counts)
    self.assertEqual(operator.rshift(Tensor([0x80000001], dtype=dtypes.uint, device=Device.DEFAULT),
                                     Tensor([1], dtype=dtypes.uint, device=Device.DEFAULT)).tolist(), [0x40000000])

  def test_production_signed_maximum_uses_mapped_machine_bytes(self):
    from tinygrad import Device, Tensor, dtypes
    cases = ((dtypes.int, 0, 0, 0),
             (dtypes.int, -1, 0, 0),
             (dtypes.int, 0, -1, 0),
             (dtypes.int, dtypes.int.min, dtypes.int.max, dtypes.int.max),
             (dtypes.int, dtypes.int.min, -1, -1),
             (dtypes.int, dtypes.int.max, dtypes.int.min, dtypes.int.max),
             (dtypes.int, -1431655766, 252645135, 252645135),
             (dtypes.int, -7, 3, 3))
    # The opcode-only OR.B mutation changes signed max(-7, 3) from 3 to the raw bit pattern for -5.
    self._assert_production_integer_binary_uses_mapped_machine_bytes(
      tensor_operator=Tensor.maximum, opcode="max.s", opcode_bits=0x19, operation="maximum", cases=cases,
      mutation_opcode="or.b", mutation_opcode_bits=0x1d, mutation_expected=0xfffffffb,
      unsupported_opcode_bits=0x17, unsupported_name="min.s", value_type="s32")
    self.assertEqual(Tensor([-7], dtype=dtypes.int, device=Device.DEFAULT).maximum(
      Tensor([3], dtype=dtypes.int, device=Device.DEFAULT)).tolist(), [3])

  def test_production_unsigned_maximum_uses_mapped_machine_bytes(self):
    from tinygrad import Device, Tensor, dtypes
    cases = ((dtypes.uint, 0, 0, 0),
             (dtypes.uint, 0, dtypes.uint.max, dtypes.uint.max),
             (dtypes.uint, dtypes.uint.max, 0, dtypes.uint.max),
             (dtypes.uint, 1, 2, 2),
             (dtypes.uint, 0xaaaaaaaa, 0x0f0f0f0f, 0xaaaaaaaa),
             (dtypes.uint, 0x80000000, 0x7fffffff, 0x80000000))
    # The opcode-only OR.B mutation changes unsigned max(0x80000000, 0x7fffffff) to uint.max.
    self._assert_production_integer_binary_uses_mapped_machine_bytes(
      tensor_operator=Tensor.maximum, opcode="max.u", opcode_bits=0x18, operation="maximum", cases=cases,
      mutation_opcode="or.b", mutation_opcode_bits=0x1d, mutation_expected=dtypes.uint.max,
      unsupported_opcode_bits=0x16, unsupported_name="min.u")
    self.assertEqual(Tensor([0x80000000], dtype=dtypes.uint, device=Device.DEFAULT).maximum(
      Tensor([0x7fffffff], dtype=dtypes.uint, device=Device.DEFAULT)).tolist(), [0x80000000])

  def test_production_integer_multiply_wraps_from_mapped_machine_bytes(self):
    import struct
    from dataclasses import replace
    from tinygrad import Device, Tensor, dtypes
    from tinygrad.runtime.autogen import kgsl
    from test.mockgpu.qcom import qcomdriver
    from test.mockgpu.qcom.a630 import decode_a630_ir3

    submissions,command_images = [],[]
    real_execute = qcomdriver.execute_a630
    real_plan = self.driver._plan_a630_retirement
    def capture_execution(submission, resolver):
      submissions.append(submission)
      return real_execute(submission, resolver)
    def capture_plan(fd, submission, command_address, command_size):
      command_images.append(bytes(self.driver.resolve_owned(fd, command_address, command_size)))
      return real_plan(fd, submission, command_address, command_size)

    cases = ((dtypes.int, dtypes.int.min, -1, dtypes.int.min),
             (dtypes.int, dtypes.int.max, 2, -2),
             (dtypes.int, -7, 3, -21),
             (dtypes.uint, dtypes.uint.max, dtypes.uint.max, 1),
             (dtypes.uint, 0x00010002, 0x00030004, 0x000a0008))
    actual,live_tensors = [],[]
    with mock.patch.object(qcomdriver, "execute_a630", side_effect=capture_execution), \
         mock.patch.object(self.driver, "_plan_a630_retirement", side_effect=capture_plan):
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
      loads = tuple(instruction for instruction in dispatch.instructions if instruction.opcode == "ldg.u32")
      stores = tuple(instruction for instruction in dispatch.instructions if instruction.opcode == "stg.u32")
      multiplies = tuple(instruction for instruction in dispatch.instructions if instruction.opcode == "mull.u")
      accumulates = tuple(instruction for instruction in dispatch.instructions if instruction.opcode == "madsh.m16")
      pointer_moves = tuple(instruction for instruction in dispatch.instructions if instruction.opcode == "mov.u32" and
                            instruction.srcs[0].kind == "const")
      self.assertEqual((len(loads), len(stores), len(multiplies), len(accumulates), len(pointer_moves)), (2, 1, 1, 2, 6))
      self.assertEqual(multiplies[0].srcs, (loads[0].dst, loads[1].dst))
      self.assertEqual(accumulates[0].srcs, (loads[0].dst, loads[1].dst, multiplies[0].dst))
      self.assertEqual(accumulates[1].srcs, (loads[1].dst, loads[0].dst, accumulates[0].dst))
      self.assertEqual(stores[0].srcs[1], accumulates[1].dst)
      self.assertTrue({("SY", 1), ("NOP", 1)} <= set(multiplies[0].fields))
      self.assertIn(("NOP", 1), accumulates[0].fields)
      self.assertIn(("NOP", 3), accumulates[1].fields)
      self.assertTrue(all(("TYPE", 3) in instruction.fields for instruction in loads + stores))
      destinations = {}
      for instruction in pointer_moves:
        self.assertIsNotNone(instruction.dst)
        destinations[instruction.srcs[0].value] = instruction.dst.value
      bases = (stores[0].srcs[0].value, *(instruction.srcs[0].value for instruction in loads))
      self.assertEqual(tuple((destinations[2*i], destinations[2*i+1]) for i in range(3)),
                       tuple((base, base+1) for base in bases))
      decoded_dispatches.append((dispatch, loads, multiplies[0], accumulates))

    case_index = len(cases) - 1
    submission = submissions[case_index]
    dispatch,_,_,accumulates = decoded_dispatches[case_index]
    shader = self.driver.resolve_owned(self.device.fd.fd, dispatch.shader_address, dispatch.shader_size)
    output_base = struct.unpack_from("<Q", dispatch.constants_image)[0]
    output = self.driver.resolve_owned(self.device.fd.fd, output_base, 4)

    # Swapping the first MADSH inputs preserves the legal dataflow but changes which high-half cross term is accumulated.
    first = accumulates[0]
    first_original = bytes(shader[first.index*8:(first.index+1)*8])
    src1,src2 = first.raw & 0x1fff, first.raw >> 47 & 0xff
    swapped_raw = first.raw & ~(0x1fff | (0xff << 47)) | src2 | src1 << 47
    request_buffer,_,request = self.gpu_command(struct.unpack(f"<{len(command_images[case_index]) // 4}I", command_images[case_index]))
    timestamp_before = self.driver.context_timestamps[self.device.ctx]
    try:
      struct.pack_into("<Q", shader, first.index * 8, swapped_raw)
      mutated = decode_a630_ir3(bytes(shader))[first.index]
      self.assertEqual((mutated.opcode, mutated.srcs), ("madsh.m16", (first.srcs[1], first.srcs[0], first.srcs[2])))
      output[:] = bytes(4)
      kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=request)
      self.assertEqual(struct.unpack("<I", output)[0], 0x00080008)
      self.assertEqual((request.timestamp, self.driver.context_timestamps[self.device.ctx]), ((timestamp_before + 1) & 0xffffffff,) * 2)
    finally:
      shader[first.index*8:(first.index+1)*8] = first_original
      self.device._gpu_free(request_buffer)

    # Duplicating a final MADSH source breaks the two-load chain and must not publish any retirement state.
    second = accumulates[1]
    second_original = bytes(shader[second.index*8:(second.index+1)*8])
    duplicate_raw = second.raw & ~(0xff << 47) | second.srcs[0].value << 47
    request_buffer,_,request = self.gpu_command(struct.unpack(f"<{len(command_images[case_index]) // 4}I", command_images[case_index]))
    request.timestamp = 0x27182818
    signal = self.driver.resolve_owned(self.device.fd.fd, int(self.device.timeline_signal.value_addr), 16)
    state_before = (bytes(output), bytes(signal), self.driver.context_timestamps[self.device.ctx],
                    self.driver.always_on_counter, self.device.last_cmd)
    try:
      struct.pack_into("<Q", shader, second.index * 8, duplicate_raw)
      image = bytes(shader)
      mutated_dispatch = replace(dispatch, shader_image=image, instructions=decode_a630_ir3(image))
      self.assertEqual(mutated_dispatch.instructions[second.index].srcs, (second.srcs[0], second.srcs[0], second.srcs[2]))
      with self.assertRaisesRegex(ValueError, "u32 multiplication sequence does not consume both global loads"):
        real_execute(replace(submission, dispatches=(mutated_dispatch,)),
                     lambda address,length: self.driver.resolve_owned(self.device.fd.fd, address, length))
      with self.assertRaisesRegex(RuntimeError, "u32 multiplication sequence does not consume both global loads"):
        kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=request)
      self.assertEqual((request.timestamp, bytes(output), bytes(signal), self.driver.context_timestamps[self.device.ctx],
                        self.driver.always_on_counter, self.device.last_cmd), (0x27182818, *state_before))
    finally:
      shader[second.index*8:(second.index+1)*8] = second_original
      self.device._gpu_free(request_buffer)
    self.assertEqual(bytes(shader[second.index*8:(second.index+1)*8]), second_original)
    self.assertEqual((Tensor([9], dtype=dtypes.int, device=Device.DEFAULT) *
                      Tensor([4], dtype=dtypes.int, device=Device.DEFAULT)).tolist(), [36])

  def _assert_production_integer_comparison_uses_mapped_machine_bytes(self, *, tensor_operator, cases, opcode_by_dtype, condition,
                                                                      mutation_mask, mutation_value, mutation_opcode, mutation_expected,
                                                                      unsupported_condition=None, check_duplicate_inventory=False):
    import struct
    from dataclasses import replace
    from tinygrad import Device, Tensor
    from tinygrad.runtime.autogen import kgsl
    from test.mockgpu.qcom import qcomdriver
    from test.mockgpu.qcom.a630 import A630IR3Operand, decode_a630_ir3

    submissions,command_images = [],[]
    real_execute = qcomdriver.execute_a630
    real_plan = self.driver._plan_a630_retirement
    def capture_execution(submission, resolver):
      submissions.append(submission)
      return real_execute(submission, resolver)
    def capture_plan(fd, submission, command_address, command_size):
      command_images.append(bytes(self.driver.resolve_owned(fd, command_address, command_size)))
      return real_plan(fd, submission, command_address, command_size)

    actual,live_tensors = [],[]
    with mock.patch.object(qcomdriver, "execute_a630", side_effect=capture_execution), \
         mock.patch.object(self.driver, "_plan_a630_retirement", side_effect=capture_plan):
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
    comparison_opcodes = frozenset(opcode_by_dtype.values())
    for (dtype,_,_,_),submission in zip(cases, submissions):
      dispatch = submission.dispatches[0]
      self.assertEqual((dispatch.local_size, dispatch.groups, dispatch.global_size), ((1, 1, 1),) * 3)
      loads = tuple(instruction for instruction in dispatch.instructions if instruction.opcode == "ldg.u32")
      stores = tuple(instruction for instruction in dispatch.instructions if instruction.opcode == "stg.u8")
      comparisons = tuple(instruction for instruction in dispatch.instructions if instruction.opcode in comparison_opcodes and
                          instruction.srcs and all(operand.kind == "gpr" for operand in instruction.srcs))
      pointer_moves = tuple(instruction for instruction in dispatch.instructions if instruction.opcode == "mov.u32" and
                            instruction.srcs[0].kind == "const")
      expected_opcode = opcode_by_dtype[dtype]
      self.assertEqual((len(loads), len(stores), len(comparisons), len(pointer_moves)), (2, 1, 1, 6))
      self.assertEqual((comparisons[0].opcode, comparisons[0].dst, comparisons[0].srcs),
                       (expected_opcode, A630IR3Operand("half", 0), (loads[0].dst, loads[1].dst)))
      self.assertEqual(stores[0].srcs[1], comparisons[0].dst)
      self.assertTrue({("COND", condition), ("DST_HALF", 1), ("SY", 1), ("NOP", 3)} <= set(comparisons[0].fields))
      self.assertTrue({("TYPE", 6), ("TYPE_HALF", 1), ("OFF", 0), ("SIZE", 1)} <= set(stores[0].fields))
      destinations = {}
      for instruction in pointer_moves:
        assert instruction.dst is not None
        destinations[instruction.srcs[0].value] = instruction.dst.value
      bases = (stores[0].srcs[0].value, *(instruction.srcs[0].value for instruction in loads))
      self.assertEqual(tuple((destinations[2*i], destinations[2*i+1]) for i in range(3)),
                       tuple((base, base+1) for base in bases))
      decoded_dispatches.append((dispatch, comparisons[0], stores[0]))

    # Replaying only the newest capture avoids a stale historical EVENT_WRITE after the numerical matrix.
    case_index = len(cases) - 1
    self.assertIn(mutation_expected, (0, 1))
    self.assertNotEqual(bool(mutation_expected), cases[case_index][3])
    submission = submissions[case_index]
    dispatch,comparison,store = decoded_dispatches[case_index]
    shader = self.driver.resolve_owned(self.device.fd.fd, dispatch.shader_address, dispatch.shader_size)
    original = bytes(shader[comparison.index*8:(comparison.index+1)*8])
    output_base = struct.unpack_from("<Q", dispatch.constants_image)[0]
    output = self.driver.resolve_owned(self.device.fd.fd, output_base, 1)
    request_buffer,_,request = self.gpu_command(struct.unpack(f"<{len(command_images[case_index]) // 4}I", command_images[case_index]))
    timestamp_before = self.driver.context_timestamps[self.device.ctx]
    try:
      self.assertEqual(mutation_value & ~mutation_mask, 0)
      mutation_raw = comparison.raw & ~mutation_mask | mutation_value
      self.assertEqual(mutation_raw & ~mutation_mask, comparison.raw & ~mutation_mask)
      self.assertNotEqual(mutation_raw, comparison.raw)
      struct.pack_into("<Q", shader, comparison.index * 8, mutation_raw)
      mutated = decode_a630_ir3(bytes(shader))[comparison.index]
      self.assertEqual((mutated.opcode, mutated.dst, mutated.srcs), (mutation_opcode, comparison.dst, comparison.srcs))
      output[:] = bytes([mutation_expected ^ 0xff])
      kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=request)
      self.assertEqual(bytes(output), bytes([mutation_expected]))
      self.assertEqual((request.timestamp, self.driver.context_timestamps[self.device.ctx]), ((timestamp_before + 1) & 0xffffffff,) * 2)
    finally:
      shader[comparison.index*8:(comparison.index+1)*8] = original
      self.device._gpu_free(request_buffer)
    self.assertEqual(bytes(shader[comparison.index*8:(comparison.index+1)*8]), original)

    def reject_mapped_mutation(instruction, raw, expected_opcode, expected_srcs, message):
      original_instruction = bytes(shader[instruction.index*8:(instruction.index+1)*8])
      request_buffer,_,request = self.gpu_command(struct.unpack(f"<{len(command_images[case_index]) // 4}I", command_images[case_index]))
      request.timestamp = marker = 0x14142135 + instruction.index
      signal = self.driver.resolve_owned(self.device.fd.fd, int(self.device.timeline_signal.value_addr), 16)
      state_before = (bytes(output), bytes(signal), self.driver.context_timestamps[self.device.ctx],
                      self.driver.always_on_counter, self.device.last_cmd, self.device.error_state)
      try:
        struct.pack_into("<Q", shader, instruction.index * 8, raw)
        image = bytes(shader)
        mutated_dispatch = replace(dispatch, shader_image=image, instructions=decode_a630_ir3(image))
        mutated = mutated_dispatch.instructions[instruction.index]
        self.assertEqual((mutated.opcode, mutated.srcs), (expected_opcode, expected_srcs))
        with self.assertRaisesRegex(ValueError, message):
          real_execute(replace(submission, dispatches=(mutated_dispatch,)),
                       lambda address,length: self.driver.resolve_owned(self.device.fd.fd, address, length))
        with self.assertRaisesRegex(RuntimeError, message): kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=request)
        self.assertEqual((request.timestamp, bytes(output), bytes(signal), self.driver.context_timestamps[self.device.ctx],
                          self.driver.always_on_counter, self.device.last_cmd, self.device.error_state), (marker, *state_before))
      finally:
        shader[instruction.index*8:(instruction.index+1)*8] = original_instruction
        self.device._gpu_free(request_buffer)
      self.assertEqual(bytes(shader[instruction.index*8:(instruction.index+1)*8]), original_instruction)

    if unsupported_condition is not None:
      unsupported_raw = comparison.raw & ~(0x7 << 48) | unsupported_condition << 48
      reject_mapped_mutation(comparison, unsupported_raw, None, (), f"unsupported A630 semantic at instruction {comparison.index}")
    src1 = comparison.raw & 0xffff
    reject_mapped_mutation(comparison, comparison.raw & ~(0xffff << 16) | src1 << 16,
                           comparison.opcode, (comparison.srcs[0], comparison.srcs[0]),
                           "u32 comparison does not consume both global loads")
    if check_duplicate_inventory:
      nop = next(instruction for instruction in dispatch.instructions if instruction.opcode == "nop")
      reject_mapped_mutation(nop, comparison.raw, comparison.opcode, comparison.srcs,
                             "u32 comparison does not consume both global loads")
    reject_mapped_mutation(store, store.raw & ~(0xff << 1) | 1 << 1, "stg.u8",
                           (store.srcs[0], A630IR3Operand("half", 1)), "global store does not consume the u32 comparison")

  def test_production_integer_less_than_uses_mapped_comparison_bytes(self):
    import operator
    from tinygrad import Device, Tensor, dtypes
    cases = ((dtypes.int, dtypes.int.min, dtypes.int.max, True),
             (dtypes.int, dtypes.int.max, dtypes.int.min, False),
             (dtypes.int, -1, 0, True),
             (dtypes.int, 0, -1, False),
             (dtypes.int, 0, 0, False),
             (dtypes.uint, 0, dtypes.uint.max, True),
             (dtypes.uint, dtypes.uint.max, 0, False),
             (dtypes.uint, dtypes.uint.max, dtypes.uint.max, False),
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
    cases = ((dtypes.int, 0, 0, True),
             (dtypes.int, 0, 1, False),
             (dtypes.int, -1, -1, True),
             (dtypes.int, dtypes.int.min, dtypes.int.min, True),
             (dtypes.int, dtypes.int.min, dtypes.int.max, False),
             (dtypes.uint, 0, 0, True),
             (dtypes.uint, dtypes.uint.max, 0, False),
             (dtypes.uint, 0x80000000, 0x80000000, True),
             (dtypes.uint, 0x7fffffff, 0x80000000, False),
             (dtypes.uint, dtypes.uint.max, dtypes.uint.max, True))
    # Equality is signedness-independent; changing only COND from EQ to LT changes the latest equal words to false.
    self._assert_production_integer_comparison_uses_mapped_machine_bytes(
      tensor_operator=operator.eq, cases=cases, opcode_by_dtype={dtypes.int:"cmps.s.eq", dtypes.uint:"cmps.s.eq"}, condition=4,
      mutation_mask=0x7 << 48, mutation_value=0, mutation_opcode="cmps.s.lt", mutation_expected=0,
      unsupported_condition=5, check_duplicate_inventory=True)
    self.assertEqual((Tensor([-1], dtype=dtypes.int, device=Device.DEFAULT) ==
                      Tensor([-1], dtype=dtypes.int, device=Device.DEFAULT)).tolist(), [True])

  def test_production_symbolic_workgroups_execute_mapped_system_values(self):
    import struct
    from dataclasses import replace
    from tinygrad import Device, Tensor, Variable
    from tinygrad.runtime.autogen import kgsl, mesa
    from test.mockgpu.qcom import qcomdriver
    from test.mockgpu.qcom.a630 import decode_a630_ir3

    submissions,command_images = [],[]
    real_execute = qcomdriver.execute_a630
    real_plan = self.driver._plan_a630_retirement
    def capture_execution(submission, resolver):
      submissions.append(submission)
      return real_execute(submission, resolver)
    def capture_plan(fd, submission, command_address, command_size):
      command_images.append(bytes(self.driver.resolve_owned(fd, command_address, command_size)))
      return real_plan(fd, submission, command_address, command_size)

    size = Variable("qcom_symbolic_size", 1, 10)
    ones = Tensor.ones(10, device=Device.DEFAULT).contiguous()
    actual = []
    with mock.patch.object(qcomdriver, "execute_a630", side_effect=capture_execution), \
         mock.patch.object(self.driver, "_plan_a630_retirement", side_effect=capture_plan):
      for value in (2, 5): actual.append((ones[:size.bind(value)] + 1).contiguous()[:value].tolist())
    reference = [(Tensor.ones(value, device="PYTHON") + 1).tolist() for value in (2, 5)]

    self.assertEqual((Device.DEFAULT, (DEV.interface, DEV.device, DEV.renderer, DEV.arch)),
                     ("QCOM", ("MOCK", "QCOM", "IR3", "a630")))
    self.assertEqual(actual, reference)

    distinct_values = [-7.5, 0.25, 1024.0, -0.0, 3.5, 19.0, -2.0, 8.25, 11.0, -4.5]
    distinct_size = Variable("qcom_distinct_size", 1, 10)
    distinct = Tensor(distinct_values, device=Device.DEFAULT).realize()
    distinct_actual = [(distinct[:distinct_size.bind(value)] + 1).contiguous()[:value].tolist() for value in (2, 5)]
    distinct_reference = [(Tensor(distinct_values[:value], device="PYTHON") + 1).tolist() for value in (2, 5)]
    self.assertEqual(distinct_actual, distinct_reference)

    self.assertEqual((len(submissions), len(command_images)), (5, 5))
    dispatches = [submission.dispatches[0] for submission in submissions]
    shapes = [(dispatch.local_size, dispatch.groups, dispatch.global_size) for dispatch in dispatches]
    self.assertEqual(shapes, [((2, 1, 1), (5, 1, 1), (10, 1, 1)),
                              ((1, 1, 1), (2, 1, 1), (2, 1, 1)),
                              ((2, 1, 1), (1, 1, 1), (2, 1, 1)),
                              ((1, 1, 1), (5, 1, 1), (5, 1, 1)),
                              ((1, 1, 1), (5, 1, 1), (5, 1, 1))])
    kernel_kinds = [(sum(instruction.opcode == "ldg.u32" for instruction in dispatch.instructions),
                     sum(instruction.opcode == "add.f" for instruction in dispatch.instructions)) for dispatch in dispatches]
    self.assertEqual(kernel_kinds, [(0, 0), (1, 1), (1, 0), (1, 1), (1, 0)])
    for dispatch in dispatches:
      system = dict(dispatch.registers)[mesa.REG_A6XX_SP_CS_CONST_CONFIG_0]
      if dispatch.groups[0] > 1: self.assertEqual(system & 0xff, 0xc0)

    multi_add = submissions[3]
    dispatch = multi_add.dispatches[0]
    system = dict(dispatch.registers)[mesa.REG_A6XX_SP_CS_CONST_CONFIG_0]
    def with_register(register, value):
      return replace(dispatch, registers=tuple((reg, value if reg == register else current) for reg,current in dispatch.registers))
    invalid_system_mappings = (
      (with_register(mesa.REG_A6XX_SP_CS_CONST_CONFIG_0, system & ~0xff | 0xfc), "lacks a workgroup-id mapping"),
      (with_register(mesa.REG_A6XX_SP_CS_CONST_CONFIG_0, system & ~0xff00), "unsupported A630 system-value register mapping"),
      (with_register(mesa.REG_A6XX_SP_CS_CONST_CONFIG_0, system & ~(0xff << 24) | 0xc0 << 24), "invalid A630 local-id"),
      (with_register(mesa.REG_A6XX_SP_CS_WGE_CNTL, 0), "unsupported A630 system-value register mapping"),
    )
    for mutated_dispatch,message in invalid_system_mappings:
      with self.subTest(system_mapping=message), self.assertRaisesRegex(ValueError, message):
        real_execute(replace(multi_add, dispatches=(mutated_dispatch,)),
                     lambda address,length: self.driver.resolve_owned(self.device.fd.fd, address, length))
    for mutated_dispatch in (replace(dispatch, groups=(0, 1, 1), global_size=(0, 1, 1)),
                             replace(dispatch, groups=(0x10001, 1, 1), global_size=(0x10001, 1, 1)),
                             replace(dispatch, groups=(5, 2, 1), global_size=(5, 2, 1))):
      with self.subTest(dispatch_shape=mutated_dispatch.global_size), self.assertRaisesRegex(ValueError, "bounded one-dimensional"):
        real_execute(replace(multi_add, dispatches=(mutated_dispatch,)),
                     lambda address,length: self.driver.resolve_owned(self.device.fd.fd, address, length))

    constants_view = self.driver.resolve_owned(self.device.fd.fd, dispatch.constants_address, dispatch.constants_size)
    constants_before = bytes(constants_view)
    input_base = struct.unpack_from("<Q", constants_before, 8)[0]
    input_view = self.driver.resolve_owned(self.device.fd.fd, input_base, dispatch.global_size[0] * 4 + 4)
    output_base = struct.unpack_from("<Q", constants_before)[0]
    output_view = self.driver.resolve_owned(self.device.fd.fd, output_base, dispatch.global_size[0] * 4)
    alias_words = struct.unpack(f"<{len(command_images[3]) // 4}I", command_images[3])
    alias_buffer,_,alias_request = self.gpu_command(alias_words)
    alias_request.timestamp = 0x24681357
    state_before = (bytes(input_view), bytes(output_view), self.driver.context_timestamps[self.device.ctx],
                    self.driver.always_on_counter, self.device.last_cmd)
    try:
      struct.pack_into("<Q", constants_view, 0, input_base + 4)
      with self.assertRaisesRegex(RuntimeError, "global store aliases snapshotted A630 global input 0"):
        kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=alias_request)
      self.assertEqual((alias_request.timestamp, bytes(input_view), bytes(output_view), self.driver.context_timestamps[self.device.ctx],
                        self.driver.always_on_counter, self.device.last_cmd), (0x24681357, *state_before))
    finally:
      constants_view[:] = constants_before
      self.device._gpu_free(alias_buffer)

    instruction = next(instruction for instruction in dispatch.instructions if instruction.opcode == "shl.b" and
                       any(operand.kind == "shared" for operand in instruction.srcs))
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
      renamed_journal = real_execute(replace(multi_add, dispatches=(renamed_dispatch,)),
                                     lambda address,length: self.driver.resolve_owned(self.device.fd.fd, address, length))
      self.assertEqual(tuple(struct.unpack("<f", write.data)[0] for write in renamed_journal), (2.0,) * 5)
    finally: shader[:] = shared_original

    original = bytes(shader[instruction.index*8:(instruction.index+1)*8])
    output_base = struct.unpack_from("<Q", dispatch.constants_image)[0]
    output = self.driver.resolve_owned(self.device.fd.fd, output_base, dispatch.global_size[0] * 4)
    output_before = bytes(output)
    self.assertEqual(original[0], 0xc0)
    # Redirect the compiled workgroup-X source to workgroup-Y. Decoding the changed mapped bytes must break address generation.
    try:
      shader[instruction.index*8] = 0xc1
      image = bytes(shader)
      mutated_dispatch = replace(dispatch, shader_image=image, instructions=decode_a630_ir3(image))
      mutated_submission = replace(multi_add, dispatches=(mutated_dispatch,))
      with self.assertRaisesRegex(ValueError, "global load 0 does not address its scalar input"):
        real_execute(mutated_submission, lambda address,length: self.driver.resolve_owned(self.device.fd.fd, address, length))
    finally: shader[instruction.index*8:(instruction.index+1)*8] = original
    self.assertEqual((bytes(shader[instruction.index*8:(instruction.index+1)*8]), bytes(output)), (original, output_before))

    fill = submissions[0]
    fill_dispatch = fill.dispatches[0]
    fill_move = next(instruction for instruction in fill_dispatch.instructions
                     if instruction.opcode == "mov.u32" and instruction.srcs[0].kind == "uim")
    fill_shader = self.driver.resolve_owned(self.device.fd.fd, fill_dispatch.shader_address, fill_dispatch.shader_size)
    fill_original = bytes(fill_shader[fill_move.index*8:(fill_move.index+1)*8])
    fill_output_base = struct.unpack_from("<Q", fill_dispatch.constants_image)[0]
    fill_output = self.driver.resolve_owned(self.device.fd.fd, fill_output_base, fill_dispatch.global_size[0] * 4)
    fill_output_before = bytes(fill_output)
    try:
      struct.pack_into("<I", fill_shader, fill_move.index*8, 0x40000000)
      image = bytes(fill_shader)
      mutated_fill = replace(fill, dispatches=(replace(fill_dispatch, shader_image=image, instructions=decode_a630_ir3(image)),))
      with self.assertRaisesRegex(ValueError, "unsupported A630 fill literal"):
        real_execute(mutated_fill, lambda address,length: self.driver.resolve_owned(self.device.fd.fd, address, length))
    finally: fill_shader[fill_move.index*8:(fill_move.index+1)*8] = fill_original
    self.assertEqual((bytes(fill_shader[fill_move.index*8:(fill_move.index+1)*8]), bytes(fill_output)),
                     (fill_original, fill_output_before))

  def test_production_image_descriptor_path_preflights_nested_ranges(self):
    import struct
    from tinygrad import Tensor, dtypes
    from tinygrad.codegen import to_program
    from tinygrad.device import Buffer, Device
    from tinygrad.engine.realize import get_runtime
    from tinygrad.helpers import Context, Target
    from tinygrad.renderer.nir import IR3Renderer
    from tinygrad.runtime.autogen import kgsl
    from test.mockgpu.qcom.a630 import execute_a630, stage_a630
    from test.mockgpu.qcom.pm4 import parse_pm4
    from test.mockgpu.qcom.qcomdriver import ioctl_code

    last_command = self.device.last_cmd
    # Exact DEV routing stays a630; image coalescing additionally requires the production renderer's pitch capability.
    image_arch = self.device.arch if "IMAGE_PITCH_ALIGNMENT=" in self.device.arch else f"{self.device.arch},IMAGE_PITCH_ALIGNMENT=64"
    renderer = IR3Renderer(Target.parse(f"MOCK+QCOM:IR3:{image_arch}"))
    def compile_image(dtype):
      with Context(IMAGE=2):
        source = Tensor.empty(16, 4, 4, device="QCOM", dtype=dtype).contiguous()
        result = (source + 1).contiguous()
        schedule_item = result.schedule_linear().src[-1]
        return to_program(schedule_item.src[0], renderer)
    program_spec = compile_image(dtypes.float)
    runtime = get_runtime(self.device.device, program_spec)
    output_buffer = Buffer("QCOM", 256, dtypes.float).ensure_allocated()
    input_buffer = Buffer("QCOM", 256, dtypes.float).ensure_allocated()
    input_buffer._buf.cpu_view().mv[:8] = b"A630TEX!"
    args = runtime.fill_kernargs((output_buffer._buf, input_buffer._buf))
    queue = self.device.hw_compute_queue_t()
    queue.exec(runtime, args, program_spec.arg.global_size, program_spec.arg.local_size)
    words = tuple(queue._q)
    submission = stage_a630(parse_pm4(words), lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))

    self.assertEqual(Device.DEFAULT, "QCOM")
    self.assertEqual(self.device.renderer.target.arch, "a630")
    self.assertEqual(renderer.target.arch, image_arch)
    self.assertEqual(program_spec.to_elf().signature,
                     ((None, 0, dtypes.float, (1, 64, 4)), (None, 1, dtypes.float, (1, 64, 4))))
    self.assertEqual((runtime.image_size, runtime.pvtmem, runtime.samp_cnt, runtime.tex_cnt, runtime.ibo_cnt), (128, 0, 1, 1, 1))
    self.assertEqual((runtime.tex_off, runtime.ibo_off, runtime.samp_off, runtime.kernargs_alloc_size), (2048, 2112, 2176, 2304))
    self.assertEqual((program_spec.arg.global_size, program_spec.arg.local_size), ((2, 1, 1), (32, 1, 1)))
    self.assertEqual((len(words), len(submission.dispatches)), (100, 1))
    self.assertIsNone(queue.binded_device)
    self.assertEqual(self.device.last_cmd, last_command)
    image_instructions = submission.dispatches[0].instructions
    self.assertEqual(tuple(instruction.name for instruction in image_instructions[:10]),
                     ("shl.b", None, "nop", "add.u", "nop", "isam", "add.f", "nop", "stib.b", "end"))
    self.assertTrue(all(instruction.raw == 0 for instruction in image_instructions[10:]))
    with self.assertRaisesRegex(ValueError, "image execution is not implemented"):
      execute_a630(submission, lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))

    texture_words = struct.unpack_from("<16I", bytes(args.buf.cpu_view().mv), runtime.tex_off)
    uav_words = struct.unpack_from("<16I", bytes(args.buf.cpu_view().mv), runtime.ibo_off)
    sampler_words = struct.unpack_from("<4I", bytes(args.buf.cpu_view().mv), runtime.samp_off)
    input_address, output_address = int(input_buffer._buf.va_addr), int(output_buffer._buf.va_addr)
    self.assertEqual(texture_words[:4] + texture_words[6:], (0x20806888, 0x8040, 0x20020004, 0) +
                     (0x40000000, 13) + (0,) * 8)
    self.assertEqual(uav_words[:4] + uav_words[6:], (0x20800000, 0x8040, 0x20020004, 0) + (0x40000000, 13) + (0,) * 8)
    self.assertEqual(sampler_words, (0x1b60, 0x30, 0, 0))
    self.assertEqual(texture_words[4] | texture_words[5] << 32, input_address)
    self.assertEqual(uav_words[4] | uav_words[5] << 32, output_address)

    resources = submission.dispatches[0].resources
    self.assertEqual([(resource.kind, resource.descriptor_address, resource.address, resource.size, resource.read, resource.write,
                       resource.width, resource.height, resource.pitch, resource.itemsize) for resource in resources],
                     [("texture", int(args.buf.va_addr) + runtime.tex_off, input_address, 1024, True, False, 64, 1, 1024, 4),
                      ("uav", int(args.buf.va_addr) + runtime.ibo_off, output_address, 1024, True, True, 64, 1, 1024, 4)])
    self.assertEqual(resources[0].image, bytes(input_buffer._buf.cpu_view().mv[:1024]))
    self.assertEqual(resources[1].image, bytes(output_buffer._buf.cpu_view().mv[:1024]))
    nested_ranges = {(memory_range.purpose, memory_range.address, memory_range.size, memory_range.read, memory_range.write)
                     for memory_range in submission.memory_ranges if memory_range.purpose.endswith(" target")}
    self.assertEqual(nested_ranges, {("texture 0 target", input_address, 1024, True, False),
                                     ("uav 0 target", output_address, 1024, True, True)})

    descriptor_view = args.buf.cpu_view().mv
    descriptor_cases = ((runtime.samp_off, 0), (runtime.tex_off, 0), (runtime.tex_off+4, 0), (runtime.tex_off+8, 0),
                        (runtime.tex_off+12, 1), (runtime.tex_off+16, texture_words[4] | 1),
                        (runtime.tex_off+20, texture_words[5] | 1 << 17), (runtime.tex_off+24, 0),
                        (runtime.tex_off+28, 0), (runtime.tex_off+32, 1), (runtime.ibo_off, 0))
    for offset,value in descriptor_cases:
      original_word = bytes(descriptor_view[offset:offset+4])
      nested_calls:list[tuple[int, int]] = []
      def tracking_resolver(address:int, size:int):
        nested_calls.append((address, size))
        return self.driver.resolve_owned(self.device.fd.fd, address, size)
      try:
        struct.pack_into("<I", descriptor_view, offset, value)
        with self.subTest(descriptor_offset=offset), self.assertRaises(ValueError):
          stage_a630(parse_pm4(words), tracking_resolver)
        self.assertFalse(any(address in (input_address, output_address) for address,_ in nested_calls))
      finally: descriptor_view[offset:offset+4] = original_word

    border_view = self.device.border_color_buf.cpu_view().mv
    original_border = border_view[0]
    try:
      border_view[0] = 1
      with self.assertRaisesRegex(ValueError, "unsupported border color"):
        stage_a630(parse_pm4(words), lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))
    finally: border_view[0] = original_border

    original_address = bytes(descriptor_view[runtime.ibo_off+16:runtime.ibo_off+24])
    invalid_address = (1 << 48) - 0x1000
    try:
      struct.pack_into("<Q", descriptor_view, runtime.ibo_off+16, invalid_address)
      with self.assertRaisesRegex(RuntimeError, "not in one owned mapping"):
        stage_a630(parse_pm4(words), lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))
    finally: descriptor_view[runtime.ibo_off+16:runtime.ibo_off+24] = original_address

    external_backing = bytearray(0x3000)
    external_address = (mv_address(memoryview(external_backing)) + 0xfff) & ~0xfff
    kgsl.IOCTL_KGSL_MAP_USER_MEM(self.device.fd, hostptr=external_address, len=0x1000, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR)
    try:
      struct.pack_into("<Q", descriptor_view, runtime.ibo_off+16, external_address)
      external_submission = stage_a630(parse_pm4(words),
                                        lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))
      self.assertEqual(external_submission.dispatches[0].resources[1].address, external_address)
      struct.pack_into("<Q", descriptor_view, runtime.ibo_off+16, external_address + 0xe00)
      with self.assertRaisesRegex(RuntimeError, "not in one owned mapping"):
        stage_a630(parse_pm4(words), lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))
    finally:
      descriptor_view[runtime.ibo_off+16:runtime.ibo_off+24] = original_address
      kgsl.IOCTL_KGSL_SHAREDMEM_FREE(self.device.fd, gpuaddr=external_address)

    other_fd = self.driver.open('/dev/kgsl-3d0', os.O_RDWR, 0, self.driver.tracked_files[0])
    try:
      foreign = kgsl.struct_kgsl_map_user_mem(hostptr=external_address, len=0x1000, memtype=kgsl.KGSL_USER_MEM_TYPE_ADDR)
      other_fd.ioctl(other_fd.fd, ioctl_code(kgsl.IOCTL_KGSL_MAP_USER_MEM), ctypes.addressof(foreign))
      struct.pack_into("<Q", descriptor_view, runtime.ibo_off+16, external_address)
      with self.assertRaisesRegex(RuntimeError, "not in one owned mapping"):
        stage_a630(parse_pm4(words), lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))
    finally:
      descriptor_view[runtime.ibo_off+16:runtime.ibo_off+24] = original_address
      other_fd.close(other_fd.fd)
    try:
      struct.pack_into("<Q", descriptor_view, runtime.ibo_off+16, external_address)
      with self.assertRaisesRegex(RuntimeError, "not in one owned mapping"):
        stage_a630(parse_pm4(words), lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))
    finally: descriptor_view[runtime.ibo_off+16:runtime.ibo_off+24] = original_address

    half_program = compile_image(dtypes.half)
    half_runtime = get_runtime(self.device.device, half_program)
    half_output = Buffer("QCOM", 256, dtypes.half).ensure_allocated()
    half_input = Buffer("QCOM", 256, dtypes.half).ensure_allocated()
    half_args = half_runtime.fill_kernargs((half_output._buf, half_input._buf))
    half_queue = self.device.hw_compute_queue_t()
    half_queue.exec(half_runtime, half_args, half_program.arg.global_size, half_program.arg.local_size)
    half_submission = stage_a630(parse_pm4(tuple(half_queue._q)),
                                 lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))
    half_texture = struct.unpack_from("<16I", bytes(half_args.buf.cpu_view().mv), half_runtime.tex_off)
    half_uav = struct.unpack_from("<16I", bytes(half_args.buf.cpu_view().mv), half_runtime.ibo_off)
    self.assertEqual((half_texture[0], half_uav[0], half_texture[2], half_uav[2]),
                     (0x18806888, 0x18800000, 0x20010003, 0x20010003))
    self.assertEqual([(resource.kind, resource.size, resource.pitch, resource.itemsize)
                      for resource in half_submission.dispatches[0].resources],
                     [("texture", 512, 512, 2), ("uav", 512, 512, 2)])
    self.assertEqual(tuple(instruction.raw for instruction in half_submission.dispatches[0].instructions),
                     tuple(instruction.raw for instruction in image_instructions))
    self.assertIsNone(half_queue.binded_device)
    self.assertEqual(self.device.last_cmd, last_command)

    with Context(IMAGE=2):
      left, right = Tensor.empty(16, 4, 4, device="QCOM"), Tensor.empty(16, 4, 4, device="QCOM")
      multi_item = (left + right).contiguous().schedule_linear().src[-1]
      multi_program = to_program(multi_item.src[0], renderer)
    multi_runtime = get_runtime(self.device.device, multi_program)
    multi_queue, multi_args, multi_buffers = self.device.hw_compute_queue_t(), [], []
    for _ in range(2):
      buffers = tuple(Buffer("QCOM", 256, dtypes.float).ensure_allocated() for _ in range(3))
      args_state = multi_runtime.fill_kernargs(tuple(buffer._buf for buffer in buffers))
      multi_buffers.append(buffers)
      multi_args.append(args_state)
      multi_queue.exec(multi_runtime, args_state, multi_program.arg.global_size, multi_program.arg.local_size)
    multi_submission = stage_a630(parse_pm4(tuple(multi_queue._q)),
                                  lambda address,size: self.driver.resolve_owned(self.device.fd.fd, address, size))
    self.assertEqual((multi_runtime.samp_cnt, multi_runtime.tex_cnt, multi_runtime.ibo_cnt), (2, 2, 1))
    self.assertEqual(len(multi_submission.dispatches), 2)
    self.assertNotEqual(int(multi_args[0].buf.va_addr), int(multi_args[1].buf.va_addr))
    for dispatch,args_state,buffers in zip(multi_submission.dispatches, multi_args, multi_buffers):
      args_address = int(args_state.buf.va_addr)
      self.assertEqual([(resource.kind, resource.descriptor_address, resource.address) for resource in dispatch.resources],
                       [("texture", args_address + multi_runtime.tex_off, int(buffers[1]._buf.va_addr)),
                        ("texture", args_address + multi_runtime.tex_off + 64, int(buffers[2]._buf.va_addr)),
                        ("uav", args_address + multi_runtime.ibo_off, int(buffers[0]._buf.va_addr))])
      self.assertEqual(tuple(instruction.name for instruction in dispatch.instructions[:11]),
                       ("shl.b", None, "nop", "add.u", "nop", "isam", "isam", "add.f", "nop", "stib.b", "end"))
      image_samples = [instruction for instruction in dispatch.instructions if instruction.name == "isam"]
      self.assertEqual([[field for field in instruction.fields if field[0] in ("SAMP", "TEX")] for instruction in image_samples],
                       [[("SAMP", 0), ("SAMP", 0), ("TEX", 0), ("TEX", 0)],
                        [("SAMP", 1), ("SAMP", 1), ("TEX", 1), ("TEX", 1)]])
    self.assertIsNone(multi_queue.binded_device)
    self.assertEqual(self.device.last_cmd, last_command)

    command_buffer, _, request = self.gpu_command(words)
    request.timestamp = 0x24681357
    with self.assertRaisesRegex(RuntimeError, "image execution is not implemented"):
      kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=request)
    self.assertEqual((request.timestamp, self.device.last_cmd), (0x24681357, last_command))
    self.device._gpu_free(command_buffer)

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

import ctypes, functools, mmap, os, subprocess, sys, unittest
from typing import Any, cast
from unittest import mock
from tinygrad.helpers import DEV, mv_address
from test.mockgpu.qcom.qcom_test_base import _QCOMTestBase

class TestQCOMDriver(_QCOMTestBase):
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
    self.assertIsNone(self.device.graph)
    for buffer in (self.device.cmd_buf, self.device.border_color_buf, self.device.kernargs_buf, self.device.timeline_signal.base_buf):
      self.assertIsNotNone(self.allocation_for(int(buffer.va_addr), buffer.size))
    self.assertIsNotNone(self.allocation_for(self.device.dummy_addr, 0x1000))

  def test_unselected_qcom_driver_does_not_require_mesa(self):
    code = """
import importlib.abc, sys
class BlockMesa(importlib.abc.MetaPathFinder):
  def find_spec(self, fullname, path=None, target=None):
    if fullname == 'tinygrad.runtime.autogen.mesa': raise ImportError('tinymesa intentionally unavailable')
    return None
sys.meta_path.insert(0, BlockMesa())
import tinygrad.runtime.support.hcq
assert 'test.mockgpu.qcom.qcomdriver' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", code], env={**os.environ, "DEV":"MOCK+AMD", "PYTHONDONTWRITEBYTECODE":"1"},
                            capture_output=True, text=True, timeout=10)
    self.assertEqual(result.returncode, 0, result.stderr)

  def test_mock_qcom_jit_falls_back_to_individual_submissions(self):
    from tinygrad import Tensor, TinyJit

    self.assertIsNone(self.device.graph)
    @TinyJit
    def two_kernels(value):
      intermediate = (value + 1).realize()
      return (intermediate + 2).realize()

    for offset in range(3):
      self.assertEqual(two_kernels(Tensor([offset], device=self.device.device)).item(), offset + 3)

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
    from test.mockgpu.qcom.qcomdriver import KGSLJournalWrite, _MAX_A630_COMMAND_WORDS, ioctl_code
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
      ("size", (_MAX_A630_COMMAND_WORDS + 1) * 4, "command exceeds the A630 emulator word limit"),
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

  def test_retirement_commit_classifies_only_complete_rollback_as_definite(self):
    from tinygrad.runtime.autogen import kgsl
    from test.mockgpu.qcom.qcomdriver import A630SubmissionAmbiguous, KGSLJournalWrite, ioctl_code

    class FaultingView:
      readonly = False
      def __init__(self, image:bytes, failures:set[int]|None=None, failure:type[BaseException]=RuntimeError):
        self.image,self.failures,self.failure,self.assignments = bytearray(image),failures or set(),failure,0
      def __len__(self): return len(self.image)
      def __bytes__(self): return bytes(self.image)
      def __setitem__(self, key, value):
        self.assignments += 1
        if self.assignments in self.failures: raise self.failure(f"injected write failure {self.assignments}")
        self.image[key] = value

    journal = (KGSLJournalWrite(0, 0, 1, b"LEFT", "first test", False),
               KGSLJournalWrite(1, 0, 5, b"RIGHT", "second test", False))
    first,second = FaultingView(b"left"),FaultingView(b"right", {1})
    with mock.patch.object(self.driver, "resolve_owned", side_effect=(first, second)), \
         self.assertRaisesRegex(RuntimeError, "failed to commit A630 retirement") as definite:
      self.driver._commit_a630_journal(self.device.fd.fd, journal)
    self.assertIs(type(definite.exception), RuntimeError)
    self.assertEqual((bytes(first), bytes(second)), (b"left", b"right"))

    first,second = FaultingView(b"left"),FaultingView(b"right", {1}, KeyboardInterrupt)
    with mock.patch.object(self.driver, "resolve_owned", side_effect=(first, second)), self.assertRaises(KeyboardInterrupt):
      self.driver._commit_a630_journal(self.device.fd.fd, journal)
    self.assertEqual((bytes(first), bytes(second)), (b"left", b"right"))

    ambiguous_journal = journal + (KGSLJournalWrite(2, 0, 9, b"THIRD", "third test", False),)
    first,second,third = FaultingView(b"left"),FaultingView(b"right", {2}),FaultingView(b"third", {1})
    with mock.patch.object(self.driver, "resolve_owned", side_effect=(first, second, third)), \
         self.assertRaisesRegex(A630SubmissionAmbiguous, "rollback was incomplete") as ambiguous:
      self.driver._commit_a630_journal(self.device.fd.fd, ambiguous_journal)
    self.assertIs(type(ambiguous.exception), A630SubmissionAmbiguous)
    self.assertEqual((bytes(first), bytes(second), bytes(third)), (b"left", b"RIGHT", b"third"))

    request_code = ioctl_code(kgsl.IOCTL_KGSL_GPU_COMMAND)
    struct_type,_ = self.driver._ioctls[request_code]
    payload = struct_type()
    def fail_ambiguously(_fd, _payload): raise A630SubmissionAmbiguous("ambiguous retirement")
    with mock.patch.dict(self.driver._ioctls, {request_code:(struct_type, fail_ambiguously)}), \
         self.assertRaisesRegex(A630SubmissionAmbiguous, "ambiguous retirement") as ioctl_error:
      kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=payload)
    self.assertIs(type(ioctl_error.exception), A630SubmissionAmbiguous)

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
    with mock.patch.object(a630_module, "decode_a630_ir3", wraps=a630_module.decode_a630_ir3) as decode:
      submission = stage_a630(parse_pm4(words), self._resolve_owned)
    self.assertEqual(len(submission.dispatches), 2)
    self.assertEqual(decode.call_count, 1)
    self.assertIs(submission.dispatches[0].instructions, submission.dispatches[1].instructions)
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
    from test.mockgpu.qcom.qcomdriver import A630SubmissionAmbiguous

    before,before_error = self.device.timeline_value,self.device.error_state
    ambiguous_timeline = later_timeline = None
    try:
      ambiguous = self.device.hw_compute_queue_t()
      with mock.patch.object(ambiguous, "_submit", side_effect=A630SubmissionAmbiguous("ambiguous acceptance")), \
           self.assertRaisesRegex(A630SubmissionAmbiguous, "ambiguous acceptance") as raised:
        self.device.submit_timeline(ambiguous)
      self.assertIs(type(raised.exception), A630SubmissionAmbiguous)
      ambiguous_timeline = self.device.timeline_value
      self.assertIs(self.device.error_state, raised.exception)
      self.device.timeline_value,self.device.error_state = before,None

      def reject_after_later_reservation(_):
        self.device.next_timeline()
        raise HCQSubmissionRejected("definite rejection after a later reservation")
      non_lifo = self.device.hw_compute_queue_t()
      with mock.patch.object(non_lifo, "_submit", side_effect=reject_after_later_reservation), \
           self.assertRaisesRegex(HCQSubmissionRejected, "after a later reservation"):
        self.device.submit_timeline(non_lifo)
      later_timeline = self.device.timeline_value
      self.assertRegex(str(self.device.error_state), "non-latest timeline reservation")
    finally: self.device.timeline_value,self.device.error_state = before,before_error

    self.assertEqual((ambiguous_timeline, later_timeline), (before + 1, before + 2))

  def test_escaped_timeline_retirement_poison_blocks_followup_submission(self):
    from test.mockgpu.qcom.qcomdriver import A630SubmissionAmbiguous

    timeline_view = self.device.timeline_signal.base_buf.cpu_view().mv[:16]
    before = {"timeline_value":self.device.timeline_value, "timeline_image":bytes(timeline_view),
              "context_timestamp":self.driver.context_timestamps[self.device.ctx], "last_cmd":self.device.last_cmd,
              "counter":self.driver.always_on_counter, "error_state":self.device.error_state}
    try:
      def escape_timeline(fd, journal):
        target = next(write for write in journal
                      if write.address == self.device.timeline_signal.value_addr and len(write.data) == 4)
        self.driver.resolve_owned(fd, target.address, len(target.data))[:] = target.data
        raise A630SubmissionAmbiguous("incomplete retirement rollback")

      queue = self.device.hw_compute_queue_t()
      with mock.patch.object(self.driver, "_commit_a630_journal", side_effect=escape_timeline), \
           self.assertRaisesRegex(A630SubmissionAmbiguous, "incomplete retirement rollback") as raised:
        self.device.submit_timeline(queue)
      self.assertEqual((self.device.timeline_value, self.device.timeline_signal.value),
                       (before["timeline_value"] + 1, before["timeline_value"]))
      self.assertIs(self.device.error_state, raised.exception)
      self.assertEqual((self.driver.context_timestamps[self.device.ctx], self.device.last_cmd, self.driver.always_on_counter),
                       (before["context_timestamp"], before["last_cmd"], before["counter"]))

      blocked = self.device.hw_compute_queue_t()
      with mock.patch.object(blocked, "_submit", side_effect=AssertionError("poisoned device submitted follow-up work")) as submit, \
           self.assertRaisesRegex(A630SubmissionAmbiguous, "incomplete retirement rollback") as blocked_error:
        self.device.submit_timeline(blocked)
      submit.assert_not_called()
      self.assertIs(blocked_error.exception, raised.exception)
      self.assertEqual(self.device.timeline_value, before["timeline_value"] + 1)
    finally:
      self.device.timeline_value,timeline_view[:] = before["timeline_value"],before["timeline_image"]
      self.driver.context_timestamps[self.device.ctx],self.device.last_cmd = before["context_timestamp"],before["last_cmd"]
      self.driver.always_on_counter,self.device.error_state = before["counter"],before["error_state"]

  def test_raw_queue_ambiguous_retirement_poison_blocks_graph_style_submission(self):
    from tinygrad import Variable, dtypes
    from test.mockgpu.qcom.qcomdriver import A630SubmissionAmbiguous

    timeline_view = self.device.timeline_signal.base_buf.cpu_view().mv[:16]
    before = {"timeline_value":self.device.timeline_value, "timeline_image":bytes(timeline_view),
              "context_timestamp":self.driver.context_timestamps[self.device.ctx], "last_cmd":self.device.last_cmd,
              "counter":self.driver.always_on_counter, "error_state":self.device.error_state}
    try:
      def escape_timeline(fd, journal):
        target = next(write for write in journal
                      if write.address == self.device.timeline_signal.value_addr and len(write.data) == 4)
        self.driver.resolve_owned(fd, target.address, len(target.data))[:] = target.data
        raise A630SubmissionAmbiguous("raw queue retirement escaped rollback")

      target = Variable("raw_timeline_target", 0, 0xffffffff, dtypes.uint32)
      queue = self.device.hw_compute_queue_t().signal(self.device.timeline_signal, target)
      queue.bind(self.device)
      with mock.patch.object(self.driver, "_commit_a630_journal", side_effect=escape_timeline), \
           self.assertRaisesRegex(A630SubmissionAmbiguous, "raw queue retirement escaped rollback") as raised:
        queue.submit(self.device, {target.expr:self.device.timeline_value})
      poisoned_command = bytes(queue._q)
      self.assertEqual((self.device.timeline_value, self.device.timeline_signal.value),
                       (before["timeline_value"], before["timeline_value"]))
      self.assertIs(self.device.error_state, raised.exception)
      self.assertEqual((self.driver.context_timestamps[self.device.ctx], self.device.last_cmd, self.driver.always_on_counter),
                       (before["context_timestamp"], before["last_cmd"], before["counter"]))

      with self.assertRaisesRegex(A630SubmissionAmbiguous, "raw queue retirement escaped rollback") as blocked_error:
        queue.submit(self.device, {target.expr:self.device.timeline_value + 1})
      self.assertIs(blocked_error.exception, raised.exception)
      self.assertEqual(bytes(queue._q), poisoned_command)
      self.assertEqual(self.device.timeline_value, before["timeline_value"])
    finally:
      self.device.timeline_value,timeline_view[:] = before["timeline_value"],before["timeline_image"]
      self.driver.context_timestamps[self.device.ctx],self.device.last_cmd = before["context_timestamp"],before["last_cmd"]
      self.driver.always_on_counter,self.device.error_state = before["counter"],before["error_state"]

class TestQCOMDriverFailures(_QCOMTestBase):
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

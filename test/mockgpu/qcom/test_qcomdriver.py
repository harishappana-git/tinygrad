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
    kgsl.IOCTL_KGSL_DRAWCTXT_DESTROY(self.device.fd, drawctxt_id=context.drawctxt_id)
    self.assertNotIn(context.drawctxt_id, self.driver.contexts)
    with self.assertRaisesRegex(RuntimeError, "unknown context"):
      kgsl.IOCTL_KGSL_DRAWCTXT_DESTROY(self.device.fd, drawctxt_id=context.drawctxt_id)

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

  def test_gpu_command_validation_does_not_retire(self):
    from tinygrad.runtime.autogen import kgsl, mesa
    from tinygrad.runtime.ops_qcom import pkt7_hdr
    from test.mockgpu.qcom.qcomdriver import ioctl_code
    words = [pkt7_hdr(mesa.CP_WAIT_FOR_IDLE, 0)]
    buffer, command, request = self.gpu_command(words)
    request.timestamp = 0x12345678
    before_bytes = bytes(buffer.cpu_view().mv[:4])
    before_state = (dict(self.driver.contexts), dict(self.driver.allocations), dict(self.driver.user_mappings))
    with self.assertRaisesRegex(RuntimeError, "execution and retirement are not implemented"):
      kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=request)
    self.assertEqual(request.timestamp, 0x12345678)
    self.assertEqual(bytes(buffer.cpu_view().mv[:4]), before_bytes)
    self.assertEqual((self.driver.contexts, self.driver.allocations, self.driver.user_mappings), before_state)
    self.assertEqual(bytes(self.driver.resolve_owned(self.device.fd.fd, int(buffer.va_addr), 4, internal_only=True)), before_bytes)

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

  def test_production_add_queue_decodes_without_retirement(self):
    from tinygrad import Device, Tensor
    from tinygrad.codegen import to_program
    from tinygrad.engine.realize import get_runtime
    from tinygrad.runtime.ops_qcom import QCOMComputeQueue, QCOMProgram
    from tinygrad.runtime.autogen import kgsl, mesa
    from test.mockgpu.qcom.pm4 import PM4Type7Packet, parse_pm4
    last_command = self.device.last_cmd
    source = Tensor([0., 1.], device=Device.DEFAULT).realize()
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
    queue.signal(self.device.timeline_signal, self.device.timeline_value)
    packets = parse_pm4(tuple(queue._q))
    self.assertIs(type(runtime), QCOMProgram)
    self.assertIs(type(queue), QCOMComputeQueue)
    self.assertGreater(runtime.image_size, 0)
    self.assertEqual(runtime.image_size % 128, 0)
    self.assertTrue(any(isinstance(packet, PM4Type7Packet) and packet.opcode == mesa.CP_EXEC_CS for packet in packets))
    command_buffer, _, request = self.gpu_command(tuple(queue._q))
    request.timestamp = 0x87654321
    with self.assertRaisesRegex(RuntimeError, "execution and retirement are not implemented"):
      kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, __payload=request)
    self.assertEqual(request.timestamp, 0x87654321)
    self.device._gpu_free(command_buffer)
    self.assertEqual(self.device.last_cmd, last_command)

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
    with self.assertRaisesRegex(RuntimeError, "unsupported KGSL ioctl"):
      kgsl.IOCTL_KGSL_DEVICE_WAITTIMESTAMP_CTXTID(self.device.fd, context_id=self.device.ctx, timestamp=0, timeout=0)
    with self.assertRaisesRegex(RuntimeError, "invalid command-list pointer"):
      kgsl.IOCTL_KGSL_GPU_COMMAND(self.device.fd, context_id=self.device.ctx)

if __name__ == '__main__':
  unittest.main()

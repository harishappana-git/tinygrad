from __future__ import annotations
import ctypes, functools, mmap, os, struct, time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, cast
from tinygrad.runtime.autogen import kgsl, libc
from tinygrad.helpers import DEV, Target, to_mv
from test.mockgpu.driver import VirtDriver, VirtFile, VirtFileDesc
from test.mockgpu.qcom.a630 import A630Submission, execute_a630, stage_a630
from test.mockgpu.qcom.pm4 import parse_pm4

PAGE_SIZE = 0x1000
A630_CHIP_ID = 0x060300FF
MAP_FAILED = ctypes.c_void_p(-1).value

def ioctl_code(ioctl:functools.partial) -> int:
  direction, base, nr, struct_type = ioctl.args[:4]
  return (direction << 30) | (ctypes.sizeof(struct_type) << 16) | (base << 8) | nr

@dataclass
class KGSLAllocation:
  owner: int
  size: int
  flags: int
  addr: int|None = None

@dataclass(frozen=True)
class KGSLJournalWrite:
  word_offset: int
  ordinal: int
  address: int
  data: bytes
  purpose: str
  from_dispatch: bool

class KGSLFileDesc(VirtFileDesc):
  def __init__(self, fd:int, driver:QCOMDriver):
    super().__init__(fd)
    self.driver = driver

  def _bound_fd(self, fd:int) -> int:
    if fd != self.fd: raise RuntimeError(f"invalid KGSL descriptor {fd}, expected {self.fd}")
    return self.fd

  def ioctl(self, fd, request, argp): return self.driver.ioctl(self._bound_fd(fd), request, argp)
  def mmap(self, start, sz, prot, flags, fd, offset): return self.driver.mmap(self._bound_fd(fd), start, sz, prot, flags, offset)
  def close(self, fd): return self.driver.close(self._bound_fd(fd))

class QCOMDriver(VirtDriver):
  def __init__(self, target:Target|None=None):
    if target is None:
      matches = [x for x in DEV.value if x.interface == "MOCK" and x.device == "QCOM"]
      if len(matches) != 1: raise RuntimeError(f"expected one QCOM mock target, found {len(matches)}")
      target = matches[0]
    if (target.interface, target.device, target.renderer, target.arch) != ("MOCK", "QCOM", "IR3", "a630"):
      raise RuntimeError(f"unsupported QCOM mock target {target!r}; expected MOCK+QCOM:IR3:a630")
    super().__init__()
    self.next_fd, self.next_allocation_id, self.next_context_id = 1 << 28, 1, 1
    self.open_fds:set[int] = set()
    self.allocations:dict[int, KGSLAllocation] = {}
    self.user_mappings:dict[int, tuple[int, int]] = {}
    self.contexts:dict[int, tuple[int, int]] = {}
    self.context_timestamps:dict[int, int] = {}
    self.power_levels:dict[int, int] = {}
    self.always_on_counter = 0
    self.tracked_files.append(VirtFile('/dev/kgsl-3d0', functools.partial(KGSLFileDesc, driver=self)))
    self._ioctls:dict[int, tuple[Any, Callable[[int, Any], int]]] = {
      ioctl_code(kgsl.IOCTL_KGSL_GPUOBJ_ALLOC): (kgsl.struct_kgsl_gpuobj_alloc, self._gpuobj_alloc),
      ioctl_code(kgsl.IOCTL_KGSL_GPUOBJ_FREE): (kgsl.struct_kgsl_gpuobj_free, self._gpuobj_free),
      ioctl_code(kgsl.IOCTL_KGSL_DRAWCTXT_CREATE): (kgsl.struct_kgsl_drawctxt_create, self._drawctxt_create),
      ioctl_code(kgsl.IOCTL_KGSL_DRAWCTXT_DESTROY): (kgsl.struct_kgsl_drawctxt_destroy, self._drawctxt_destroy),
      ioctl_code(kgsl.IOCTL_KGSL_SETPROPERTY): (kgsl.struct_kgsl_device_getproperty, self._setproperty),
      ioctl_code(kgsl.IOCTL_KGSL_DEVICE_GETPROPERTY): (kgsl.struct_kgsl_device_getproperty, self._getproperty),
      ioctl_code(kgsl.IOCTL_KGSL_MAP_USER_MEM): (kgsl.struct_kgsl_map_user_mem, self._map_user_mem),
      ioctl_code(kgsl.IOCTL_KGSL_SHAREDMEM_FREE): (kgsl.struct_kgsl_sharedmem_free, self._sharedmem_free),
      ioctl_code(kgsl.IOCTL_KGSL_GPU_COMMAND): (kgsl.struct_kgsl_gpu_command, self._gpu_command),
    }

  @staticmethod
  def _require(condition:bool, message:str):
    if not condition: raise RuntimeError(f"invalid KGSL request: {message}")

  def _alloc_fd(self) -> int:
    fd, self.next_fd = self.next_fd, self.next_fd + 1
    return fd

  def open(self, name, flags, mode, virtfile):
    self._require(name == '/dev/kgsl-3d0', f"unsupported path {name!r}")
    self._require(flags == os.O_RDWR, f"unsupported open flags {flags:#x}")
    self.open_fds.add(fd:=self._alloc_fd())
    return virtfile.fdcls(fd)

  def close(self, fd:int) -> int:
    self._require(fd in self.open_fds, f"closed descriptor {fd}")
    unmap_failed = False
    for allocation_id in [allocation_id for allocation_id,allocation in self.allocations.items() if allocation.owner == fd]:
      allocation = self.allocations.pop(allocation_id)
      if allocation.addr is not None: unmap_failed |= libc.munmap(allocation.addr, allocation.size) != 0
    for gpuaddr in [gpuaddr for gpuaddr,(owner,_) in self.user_mappings.items() if owner == fd]: self.user_mappings.pop(gpuaddr)
    for context_id in [context_id for context_id,(owner,_) in self.contexts.items() if owner == fd]:
      self.contexts.pop(context_id)
      self.context_timestamps.pop(context_id)
      self.power_levels.pop(context_id, None)
    self.open_fds.remove(fd)
    self._require(not unmap_failed, f"failed to unmap descriptor {fd} allocation")
    return 0

  def ioctl(self, fd:int, request:int, argp:int) -> int:
    self._require(fd in self.open_fds, f"closed descriptor {fd}")
    if (entry:=self._ioctls.get(request)) is None: raise RuntimeError(f"unsupported KGSL ioctl {request:#x}")
    self._require(argp != 0, "null ioctl payload")
    struct_type, handler = entry
    return handler(fd, struct_type.from_address(argp))

  def mmap(self, fd:int, start:int, size:int, prot:int, flags:int, offset:int) -> int:
    self._require(fd in self.open_fds, f"closed descriptor {fd}")
    self._require(start == 0, f"fixed mmap address {start:#x}")
    self._require(prot == mmap.PROT_READ | mmap.PROT_WRITE, f"mmap protection {prot:#x}")
    self._require(flags == mmap.MAP_SHARED, f"mmap flags {flags:#x}")
    self._require(offset % PAGE_SIZE == 0, f"unaligned mmap offset {offset:#x}")
    allocation_id = offset // PAGE_SIZE
    self._require((allocation:=self.allocations.get(allocation_id)) is not None, f"unknown allocation {allocation_id}")
    assert allocation is not None
    self._require(allocation.owner == fd, f"allocation {allocation_id} belongs to another descriptor")
    self._require(size == allocation.size, f"mmap size {size:#x} for {allocation.size:#x}-byte allocation")
    self._require(allocation.addr is None, f"allocation {allocation_id} is already mapped")
    addr = libc.mmap(0, size, prot, flags | mmap.MAP_ANONYMOUS, -1, 0)
    self._require(addr is not None and addr != MAP_FAILED, f"anonymous mmap failed for {size:#x} bytes")
    if self._range_overlaps(int(addr), size):
      libc.munmap(addr, size)
      raise RuntimeError("invalid KGSL request: overlapping internal mapping")
    allocation.addr = int(addr)
    return allocation.addr

  def _mapped_ranges(self) -> Iterator[tuple[int, int]]:
    for allocation in self.allocations.values():
      if allocation.addr is not None: yield allocation.addr, allocation.size
    for addr,(_,size) in self.user_mappings.items(): yield addr, size

  def _range_overlaps(self, addr:int, size:int) -> bool:
    return any(addr + size > start and start + mapped_size > addr for start,mapped_size in self._mapped_ranges())

  def resolve(self, gpuaddr:int, size:int) -> memoryview:
    self._require(gpuaddr > 0 and size > 0 and gpuaddr + size <= 1 << 64, f"invalid GPU range {gpuaddr:#x}+{size:#x}")
    ranges = [(start, mapped_size) for start,mapped_size in self._mapped_ranges()
              if start <= gpuaddr and gpuaddr + size <= start + mapped_size]
    self._require(len(ranges) == 1, f"unmapped or ambiguous GPU range {gpuaddr:#x}+{size:#x}")
    return to_mv(gpuaddr, size)

  def resolve_owned(self, fd:int, gpuaddr:int, size:int, *, internal_only:bool=False) -> memoryview:
    self._require(fd in self.open_fds, f"closed descriptor {fd}")
    self._require(gpuaddr > 0 and size > 0 and gpuaddr + size <= 1 << 64, f"invalid GPU range {gpuaddr:#x}+{size:#x}")
    ranges = [(allocation.addr, allocation.size) for allocation in self.allocations.values()
              if allocation.owner == fd and allocation.addr is not None]
    if not internal_only: ranges += [(addr, mapping_size) for addr,(owner,mapping_size) in self.user_mappings.items() if owner == fd]
    matches = [(start, mapped_size) for start,mapped_size in ranges
               if start is not None and start <= gpuaddr and gpuaddr + size <= start + mapped_size]
    self._require(len(matches) == 1, f"GPU range {gpuaddr:#x}+{size:#x} is not in one owned mapping")
    return to_mv(gpuaddr, size)

  def _gpuobj_alloc(self, fd:int, req:kgsl.struct_kgsl_gpuobj_alloc) -> int:
    self._require(req.size > 0 and req.size == req.mmapsize and req.size % PAGE_SIZE == 0,
                  f"invalid allocation sizes size={req.size:#x} mmapsize={req.mmapsize:#x}")
    self._require(req.va_len == 0 and req.id == 0 and req.metadata_len == 0 and req.metadata == 0, "unsupported allocation metadata")
    allowed_flags = kgsl.KGSL_MEMFLAGS_USE_CPU_MAP | kgsl.KGSL_MEMALIGN_MASK | kgsl.KGSL_CACHEMODE_MASK
    self._require(req.flags & ~allowed_flags == 0, f"unsupported allocation flags {req.flags:#x}")
    self._require(req.flags & kgsl.KGSL_MEMFLAGS_USE_CPU_MAP != 0, "allocation without USE_CPU_MAP")
    alignment = (req.flags & kgsl.KGSL_MEMALIGN_MASK) >> kgsl.KGSL_MEMALIGN_SHIFT
    cache_mode = (req.flags & kgsl.KGSL_CACHEMODE_MASK) >> kgsl.KGSL_CACHEMODE_SHIFT
    self._require(alignment == 12, f"unsupported alignment {alignment}")
    self._require(cache_mode in (kgsl.KGSL_CACHEMODE_WRITECOMBINE, kgsl.KGSL_CACHEMODE_UNCACHED), f"unsupported cache mode {cache_mode}")
    req.id, self.next_allocation_id = self.next_allocation_id, self.next_allocation_id + 1
    self.allocations[req.id] = KGSLAllocation(fd, req.size, req.flags)
    return 0

  def _gpuobj_free(self, fd:int, req:kgsl.struct_kgsl_gpuobj_free) -> int:
    self._require(req.flags == 0 and req.priv == 0 and req.type == 0 and req.len == 0, "asynchronous GPUOBJ_FREE")
    self._require((allocation:=self.allocations.get(req.id)) is not None, f"unknown allocation {req.id}")
    assert allocation is not None
    self._require(allocation.owner == fd, f"allocation {req.id} belongs to another descriptor")
    self.allocations.pop(req.id)
    return 0

  def _drawctxt_create(self, fd:int, req:kgsl.struct_kgsl_drawctxt_create) -> int:
    required = kgsl.KGSL_CONTEXT_PREAMBLE | kgsl.KGSL_CONTEXT_PWR_CONSTRAINT | kgsl.KGSL_CONTEXT_NO_FAULT_TOLERANCE \
      | kgsl.KGSL_CONTEXT_NO_GMEM_ALLOC
    allowed = required | kgsl.KGSL_CONTEXT_PRIORITY_MASK | kgsl.KGSL_CONTEXT_PREEMPT_STYLE_MASK
    preemption = (req.flags & kgsl.KGSL_CONTEXT_PREEMPT_STYLE_MASK) >> kgsl.KGSL_CONTEXT_PREEMPT_STYLE_SHIFT
    self._require(req.drawctxt_id == 0 and req.flags & required == required, "missing required compute-context flags")
    self._require(req.flags & ~allowed == 0, f"unsupported context flags {req.flags:#x}")
    self._require(preemption == kgsl.KGSL_CONTEXT_PREEMPT_STYLE_FINEGRAIN, f"unsupported preemption style {preemption}")
    req.drawctxt_id, self.next_context_id = self.next_context_id, self.next_context_id + 1
    self.contexts[req.drawctxt_id] = (fd, req.flags)
    self.context_timestamps[req.drawctxt_id] = 0
    return 0

  def _drawctxt_destroy(self, fd:int, req:kgsl.struct_kgsl_drawctxt_destroy) -> int:
    self._require((context:=self.contexts.get(req.drawctxt_id)) is not None, f"unknown context {req.drawctxt_id}")
    assert context is not None
    self._require(context[0] == fd, f"context {req.drawctxt_id} belongs to another descriptor")
    self.contexts.pop(req.drawctxt_id)
    self.context_timestamps.pop(req.drawctxt_id)
    self.power_levels.pop(req.drawctxt_id, None)
    return 0

  def _setproperty(self, fd:int, req:kgsl.struct_kgsl_device_getproperty) -> int:
    self._require(req.type == kgsl.KGSL_PROP_PWR_CONSTRAINT, f"unsupported property {req.type:#x}")
    value_ptr = cast(int, req.value or 0)
    self._require(value_ptr != 0 and req.sizebytes == ctypes.sizeof(kgsl.struct_kgsl_device_constraint), "invalid power-constraint payload")
    constraint = kgsl.struct_kgsl_device_constraint.from_address(value_ptr)
    self._require(constraint.type == kgsl.KGSL_CONSTRAINT_PWRLEVEL, f"unsupported power constraint type {constraint.type}")
    self._require((context:=self.contexts.get(constraint.context_id)) is not None, f"unknown context {constraint.context_id}")
    assert context is not None
    self._require(context[0] == fd, f"context {constraint.context_id} belongs to another descriptor")
    data_ptr = cast(int, constraint.data or 0)
    self._require(data_ptr != 0 and constraint.size == ctypes.sizeof(kgsl.struct_kgsl_device_constraint_pwrlevel), "invalid power-level payload")
    level = kgsl.struct_kgsl_device_constraint_pwrlevel.from_address(data_ptr).level
    self._require(level == kgsl.KGSL_CONSTRAINT_PWR_MAX, f"unsupported power level {level}")
    self.power_levels[constraint.context_id] = level
    return 0

  def _getproperty(self, fd:int, req:kgsl.struct_kgsl_device_getproperty) -> int:
    self._require(req.type == kgsl.KGSL_PROP_DEVICE_INFO, f"unsupported property {req.type:#x}")
    value_ptr = cast(int, req.value or 0)
    self._require(value_ptr != 0 and req.sizebytes == ctypes.sizeof(kgsl.struct_kgsl_devinfo), "invalid device-info payload")
    info = kgsl.struct_kgsl_devinfo.from_address(value_ptr)
    # Mesa's pinned freedreno device table derives 0x060300ff and gpu_id 630 from GPUId(630).
    info.device_id, info.chip_id, info.mmu_enabled, info.gpu_id = kgsl.KGSL_DEVICE_3D0, A630_CHIP_ID, 1, 630
    info.gmem_gpubaseaddr, info.gmem_sizebytes = 0, 0
    return 0

  def _map_user_mem(self, fd:int, req:kgsl.struct_kgsl_map_user_mem) -> int:
    self._require(req.fd == 0 and req.gpuaddr == 0 and req.offset == 0 and req.flags == 0, "unsupported user-memory fields")
    self._require(req.memtype == kgsl.KGSL_USER_MEM_TYPE_ADDR, f"unsupported user-memory type {req.memtype}")
    self._require(req.hostptr > 0 and req.hostptr % PAGE_SIZE == 0 and req.len > 0 and req.len % PAGE_SIZE == 0,
                  f"unaligned user-memory range {req.hostptr:#x}+{req.len:#x}")
    self._require(req.hostptr + req.len <= 1 << 64, "overflowing user-memory range")
    self._require(not self._range_overlaps(req.hostptr, req.len), "overlapping user-memory range")
    req.gpuaddr = req.hostptr
    self.user_mappings[req.gpuaddr] = (fd, req.len)
    return 0

  def _sharedmem_free(self, fd:int, req:kgsl.struct_kgsl_sharedmem_free) -> int:
    self._require((mapping:=self.user_mappings.get(req.gpuaddr)) is not None, f"unknown user mapping {req.gpuaddr:#x}")
    assert mapping is not None
    self._require(mapping[0] == fd, f"user mapping {req.gpuaddr:#x} belongs to another descriptor")
    self.user_mappings.pop(req.gpuaddr)
    return 0

  @staticmethod
  def _overlaps(left_address:int, left_size:int, right_address:int, right_size:int) -> bool:
    return left_address + left_size > right_address and right_address + right_size > left_address

  def _plan_a630_retirement(self, fd:int, submission:A630Submission, command_address:int, command_size:int) \
      -> tuple[tuple[KGSLJournalWrite, ...], int]:
    self._require(len(submission.dispatches) <= 1, "A630 retirement supports at most one dispatch")
    effects = [write.word_offset for write in submission.writes] + [dispatch.word_offset for dispatch in submission.dispatches]
    self._require(all(not effects or wait.word_offset < min(effects) for wait in submission.waits),
                  "memory wait after a submission effect")
    for wait in submission.waits:
      current = struct.unpack("<I", bytes(self.resolve_owned(fd, wait.address, 4)))[0]
      # The pinned adreno_pm4.xml WRITE_GE condition compares the two masked values as unsigned words.
      self._require((current & wait.mask) >= (wait.reference & wait.mask),
                    f"unsatisfied memory wait at {wait.address:#x}: {current & wait.mask:#x} < {wait.reference & wait.mask:#x}")

    journal:list[KGSLJournalWrite] = []
    planned_counter = self.always_on_counter
    ordered_effects = [(write.word_offset, False, index) for index,write in enumerate(submission.writes)]
    if submission.dispatches: ordered_effects.append((submission.dispatches[0].word_offset, True, 0))
    for word_offset,is_dispatch,index in sorted(ordered_effects):
      if is_dispatch:
        for ordinal,execution_write in enumerate(execute_a630(submission, lambda address,size: self.resolve_owned(fd, address, size))):
          journal.append(KGSLJournalWrite(word_offset, ordinal, execution_write.address,
                                          execution_write.data, "A630 global store", True))
        continue
      pm4_write = submission.writes[index]
      if pm4_write.value is None:
        # QCOMSignal interprets A6XX_CP_ALWAYS_ON_COUNTER at 19.2 ticks per microsecond.
        planned_counter = max(planned_counter + 1, time.perf_counter_ns() * 12 // 625)
        self._require(planned_counter < 1 << 64, "always-on counter overflow")
        data = struct.pack("<Q", planned_counter)
      else: data = struct.pack("<I", pm4_write.value)
      journal.append(KGSLJournalWrite(word_offset, index, pm4_write.address, data, pm4_write.purpose, False))
    journal.sort(key=lambda journal_write: (journal_write.word_offset, journal_write.ordinal))

    for journal_write in journal:
      self._require(not self._overlaps(journal_write.address, len(journal_write.data), command_address, command_size),
                    f"{journal_write.purpose} aliases the command image")
    for index,left in enumerate(journal):
      for right in journal[index+1:]:
        if not self._overlaps(left.address, len(left.data), right.address, len(right.data)): continue
        repeated_pm4_target = not left.from_dispatch and not right.from_dispatch and \
          (left.address, len(left.data)) == (right.address, len(right.data))
        self._require(repeated_pm4_target, f"overlapping {left.purpose} and {right.purpose}")

    if submission.dispatches:
      dispatch = submission.dispatches[0]
      input_address = struct.unpack_from("<Q", dispatch.constants_image, 8)[0]
      immutable_reads = [(memory_range.address, memory_range.size, memory_range.purpose)
                         for memory_range in submission.memory_ranges if memory_range.read and memory_range.purpose != "wait value"]
      immutable_reads.append((input_address, dispatch.local_size[0] * 4, "A630 global input"))
      for journal_write in journal:
        for address,size,purpose in immutable_reads:
          self._require(not self._overlaps(journal_write.address, len(journal_write.data), address, size),
                        f"{journal_write.purpose} aliases snapshotted {purpose}")

    return tuple(journal), planned_counter

  def _commit_a630_journal(self, fd:int, journal:tuple[KGSLJournalWrite, ...]) -> None:
    targets:list[tuple[KGSLJournalWrite, memoryview]] = []
    originals:dict[tuple[int, int], tuple[memoryview, bytes]] = {}
    for write in journal:
      view = self.resolve_owned(fd, write.address, len(write.data))
      self._require(len(view) == len(write.data), f"short resolved {write.purpose} range")
      self._require(not view.readonly, f"read-only {write.purpose} range")
      targets.append((write, view))
      originals.setdefault((write.address, len(write.data)), (view, bytes(view)))
    try:
      for write,view in targets: view[:] = write.data
    except Exception as error:
      for view,image in reversed(tuple(originals.values())): view[:] = image
      raise RuntimeError(f"invalid KGSL request: failed to commit A630 retirement: {error}") from error

  def _gpu_command(self, fd:int, req:kgsl.struct_kgsl_gpu_command) -> int:
    self._require(req.flags == 0, f"unsupported GPU command flags {req.flags:#x}")
    self._require(req.cmdlist != 0 and req.cmdlist % ctypes.alignment(ctypes.c_uint64) == 0 and
                  req.cmdlist + ctypes.sizeof(kgsl.struct_kgsl_command_object) <= 1 << 64, "invalid command-list pointer")
    self._require(req.cmdsize == ctypes.sizeof(kgsl.struct_kgsl_command_object) and req.numcmds == 1,
                  f"invalid command-list shape size={req.cmdsize} count={req.numcmds}")
    self._require((req.objlist, req.objsize, req.numobjs) == (0, 0, 0), "unsupported GPU object list")
    self._require((req.synclist, req.syncsize, req.numsyncs) == (0, 0, 0), "unsupported GPU sync list")
    self._require((context:=self.contexts.get(req.context_id)) is not None, f"unknown context {req.context_id}")
    assert context is not None
    self._require(context[0] == fd, f"context {req.context_id} belongs to another descriptor")

    # cmdlist is a trusted in-process UAPI pointer; validate its complete scalar shape before the unavoidable ctypes dereference.
    command = kgsl.struct_kgsl_command_object.from_address(req.cmdlist)
    self._require(command.offset == 0 and command.id == 0, "unsupported command-object offset or id")
    self._require(command.flags == kgsl.KGSL_CMDLIST_IB, f"unsupported command-object flags {command.flags:#x}")
    self._require(command.gpuaddr != 0 and command.gpuaddr % 4 == 0, f"unaligned command address {command.gpuaddr:#x}")
    self._require(command.size > 0 and command.size % 4 == 0, f"invalid command size {command.size:#x}")
    command_bytes = bytes(self.resolve_owned(fd, command.gpuaddr, command.size, internal_only=True))
    try:
      packets = parse_pm4(struct.unpack(f"<{command.size // 4}I", command_bytes))
      submission = stage_a630(packets, lambda address,size: self.resolve_owned(fd, address, size))
      journal,planned_counter = self._plan_a630_retirement(fd, submission, command.gpuaddr, command.size)
    except ValueError as error: raise RuntimeError(f"invalid KGSL request: {error}") from error
    timestamp = self.context_timestamps[req.context_id]
    self._require(timestamp < 0xffffffff, f"context {req.context_id} timestamp overflow")
    self._commit_a630_journal(fd, journal)
    self.always_on_counter = planned_counter
    # KGSL assigns a separate per-context command sequence after accepting the complete submission.
    self.context_timestamps[req.context_id] = req.timestamp = timestamp + 1
    return 0

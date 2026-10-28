"""Fail-closed Windows process-tree limits using only the Python standard library.

The memory cap is job-wide committed virtual memory, not resident working set.
See docs/process-monitor.md for guarantees and measurement limitations.
"""
from __future__ import annotations

import contextlib
import ctypes
import math
import os
import shutil
import subprocess
import time
from ctypes import wintypes as w


class ProcessMonitoringError(RuntimeError):
    """No result is valid when process isolation/enforcement cannot be established."""


SIZE_T = ctypes.c_size_t
ULONG_PTR = ctypes.c_size_t


class _BasicLimits(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64), ("LimitFlags", w.DWORD),
                ("MinimumWorkingSetSize", SIZE_T), ("MaximumWorkingSetSize", SIZE_T),
                ("ActiveProcessLimit", w.DWORD), ("Affinity", ULONG_PTR),
                ("PriorityClass", w.DWORD), ("SchedulingClass", w.DWORD)]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in
                ("ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                 "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BasicLimits), ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", SIZE_T), ("JobMemoryLimit", SIZE_T),
                ("PeakProcessMemoryUsed", SIZE_T), ("PeakJobMemoryUsed", SIZE_T)]


class _Accounting(ctypes.Structure):
    _fields_ = [("TotalUserTime", ctypes.c_int64), ("TotalKernelTime", ctypes.c_int64),
                ("ThisPeriodTotalUserTime", ctypes.c_int64),
                ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                ("TotalPageFaultCount", w.DWORD), ("TotalProcesses", w.DWORD),
                ("ActiveProcesses", w.DWORD), ("TotalTerminatedProcesses", w.DWORD)]


class _CompletionPort(ctypes.Structure):
    _fields_ = [("CompletionKey", ctypes.c_void_p), ("CompletionPort", w.HANDLE)]


class _StartupInfo(ctypes.Structure):
    _fields_ = [("cb", w.DWORD), ("lpReserved", w.LPWSTR), ("lpDesktop", w.LPWSTR),
                ("lpTitle", w.LPWSTR), ("dwX", w.DWORD), ("dwY", w.DWORD),
                ("dwXSize", w.DWORD), ("dwYSize", w.DWORD),
                ("dwXCountChars", w.DWORD), ("dwYCountChars", w.DWORD),
                ("dwFillAttribute", w.DWORD), ("dwFlags", w.DWORD),
                ("wShowWindow", w.WORD), ("cbReserved2", w.WORD),
                ("lpReserved2", ctypes.POINTER(w.BYTE)), ("hStdInput", w.HANDLE),
                ("hStdOutput", w.HANDLE), ("hStdError", w.HANDLE)]


class _StartupInfoEx(ctypes.Structure):
    _fields_ = [("StartupInfo", _StartupInfo), ("lpAttributeList", ctypes.c_void_p)]


class _ProcessInformation(ctypes.Structure):
    _fields_ = [("hProcess", w.HANDLE), ("hThread", w.HANDLE),
                ("dwProcessId", w.DWORD), ("dwThreadId", w.DWORD)]


class _MemoryCounters(ctypes.Structure):
    _fields_ = [("cb", w.DWORD), ("PageFaultCount", w.DWORD)] + [
        (name, SIZE_T) for name in ("PeakWorkingSetSize", "WorkingSetSize",
        "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
        "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]


class _Win32:
    """Typed Win32 entry points; no subprocess private attributes or PID kills."""
    def __init__(self):
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.psapi = ctypes.WinDLL("psapi", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([ctypes.c_void_p, w.LPCWSTR], w.HANDLE),
            "SetInformationJobObject": ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD], w.BOOL),
            "QueryInformationJobObject": ([w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.POINTER(w.DWORD)], w.BOOL),
            "AssignProcessToJobObject": ([w.HANDLE, w.HANDLE], w.BOOL),
            "TerminateJobObject": ([w.HANDLE, w.UINT], w.BOOL),
            "CloseHandle": ([w.HANDLE], w.BOOL),
            "GetCurrentProcess": ([], w.HANDLE),
            "DuplicateHandle": ([w.HANDLE, w.HANDLE, w.HANDLE, ctypes.POINTER(w.HANDLE), w.DWORD, w.BOOL, w.DWORD], w.BOOL),
            "InitializeProcThreadAttributeList": ([ctypes.c_void_p, w.DWORD, w.DWORD, ctypes.POINTER(SIZE_T)], w.BOOL),
            "UpdateProcThreadAttribute": ([ctypes.c_void_p, w.DWORD, ULONG_PTR, ctypes.c_void_p, SIZE_T, ctypes.c_void_p, ctypes.c_void_p], w.BOOL),
            "DeleteProcThreadAttributeList": ([ctypes.c_void_p], None),
            "CreateProcessW": ([w.LPCWSTR, w.LPWSTR, ctypes.c_void_p, ctypes.c_void_p, w.BOOL, w.DWORD, ctypes.c_void_p, w.LPCWSTR, ctypes.POINTER(_StartupInfoEx), ctypes.POINTER(_ProcessInformation)], w.BOOL),
            "ResumeThread": ([w.HANDLE], w.DWORD),
            "TerminateProcess": ([w.HANDLE, w.UINT], w.BOOL),
            "WaitForSingleObject": ([w.HANDLE, w.DWORD], w.DWORD),
            "GetExitCodeProcess": ([w.HANDLE, ctypes.POINTER(w.DWORD)], w.BOOL),
            "OpenProcess": ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            "IsProcessInJob": ([w.HANDLE, w.HANDLE, ctypes.POINTER(w.BOOL)], w.BOOL),
            "QueryFullProcessImageNameW": ([w.HANDLE, w.DWORD, w.LPWSTR, ctypes.POINTER(w.DWORD)], w.BOOL),
            "CreateIoCompletionPort": ([w.HANDLE, w.HANDLE, ULONG_PTR, w.DWORD], w.HANDLE),
            "GetQueuedCompletionStatus": ([w.HANDLE, ctypes.POINTER(w.DWORD), ctypes.POINTER(ULONG_PTR), ctypes.POINTER(ctypes.c_void_p), w.DWORD], w.BOOL),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.kernel, name)
            fn.argtypes, fn.restype = args, result
            setattr(self, name, fn)
        self.GetProcessMemoryInfo = self.psapi.GetProcessMemoryInfo
        self.GetProcessMemoryInfo.argtypes = [w.HANDLE, ctypes.POINTER(_MemoryCounters), w.DWORD]
        self.GetProcessMemoryInfo.restype = w.BOOL

    @staticmethod
    def check(ok, operation):
        if not ok:
            error = ctypes.get_last_error()
            raise ProcessMonitoringError(f"{operation}: {ctypes.WinError(error)}")
        return ok

    def query(self, job, kind, result):
        self.check(self.QueryInformationJobObject(job, kind, ctypes.byref(result), ctypes.sizeof(result), None),
                   "QueryInformationJobObject")
        return result

    def process_ids(self, job):
        capacity = 16
        while True:
            class ProcessIds(ctypes.Structure):
                _fields_ = [("assigned", w.DWORD), ("count", w.DWORD),
                            ("ids", ULONG_PTR * capacity)]
            result = ProcessIds()
            ok = self.QueryInformationJobObject(job, 3, ctypes.byref(result), ctypes.sizeof(result), None)
            if ok and result.count == result.assigned:
                return [int(result.ids[i]) for i in range(result.count)]
            if not ok and ctypes.get_last_error() != 234:  # ERROR_MORE_DATA
                self.check(ok, "QueryInformationJobObject(process list)")
            capacity = max(capacity * 2, result.assigned)


def run_limited(argv, stdout, stderr, timeout, memory_limit, cwd=None, *, poll_interval=0.05):
    """Run an argv under a pre-execution Windows Job memory cap and timeout.

    ``stdout``/``stderr`` are file objects, paths, or None (discard); PIPE and
    shell commands are deliberately unsupported. All arguments are passed
    literally, with Windows argv quoting. ``memory_limit`` is bytes;
    ``timeout`` is seconds. Return a JSON-safe resource record. A nonzero child
    returncode is data, while setup/monitor/cleanup failures raise
    ProcessMonitoringError. On non-Windows, fail closed without launching.
    """
    if isinstance(argv, (str, bytes)) or not argv:
        raise ValueError("argv must be a nonempty sequence, not a shell command")
    argv = [os.fspath(x) for x in argv]
    if any(not isinstance(x, str) or "\0" in x for x in argv):
        raise ValueError("argv entries must be NUL-free text")
    if isinstance(memory_limit, bool) or not isinstance(memory_limit, int) or memory_limit <= 0:
        raise ValueError("memory_limit must be a positive integer number of bytes")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    if not math.isfinite(poll_interval) or not 0 < poll_interval <= 1:
        raise ValueError("poll_interval must be in (0, 1] seconds")
    if os.name != "nt":
        raise ProcessMonitoringError("Windows Job enforcement unavailable; no worker was launched")
    if memory_limit >= 1 << (ctypes.sizeof(SIZE_T) * 8):
        raise ValueError("memory_limit does not fit SIZE_T")
    executable = shutil.which(argv[0])
    if not executable:
        raise FileNotFoundError(argv[0])
    # An explicit application path prevents executable-name ambiguity at spaces.
    executable = os.path.abspath(executable)
    return _run_windows(argv, executable, stdout, stderr, float(timeout), memory_limit,
                        os.fspath(cwd) if cwd is not None else None, poll_interval)


def _run_windows(argv, executable, stdout, stderr, timeout, memory_limit, cwd, poll_interval):
    import msvcrt

    api = _Win32()
    start = time.monotonic()
    job = port = None
    process = _ProcessInformation()
    assigned = False
    attribute_initialized = False
    attribute_buffer = None
    inherited = []
    held = {}  # Retaining handles prevents PID reuse while measurements are kept.
    records = {}
    event_counts = {}
    samples = 0
    max_rss = 0
    max_private_commit = 0
    max_active = 0
    memory_event = False
    reason = "exited"
    complete = False

    def remember(pid):
        if pid in records:
            return
        record = {"pid": pid, "image": None, "peak_working_set_bytes": None,
                  "peak_commit_bytes": None, "sample_count": 0,
                  "memory_query_failures": 0, "handle_verified_in_job": False}
        records[pid] = record
        handle = api.OpenProcess(0x100410, False, pid)  # SYNCHRONIZE | QUERY_INFORMATION | VM_READ
        if not handle:
            record["handle_unavailable_winerror"] = ctypes.get_last_error()
            return  # A short-lived descendant may already have exited.
        belongs = w.BOOL()
        if not api.IsProcessInJob(handle, job, ctypes.byref(belongs)) or not belongs.value:
            api.CloseHandle(handle)
            record["handle_unavailable_reason"] = "exited or PID no longer belongs to job"
            return
        held[pid] = handle
        record["handle_verified_in_job"] = True
        name = ctypes.create_unicode_buffer(32768)
        length = w.DWORD(len(name))
        if api.QueryFullProcessImageNameW(handle, 0, name, ctypes.byref(length)):
            record["image"] = name.value

    def drain_events():
        nonlocal memory_event
        while True:
            message, key, value = w.DWORD(), ULONG_PTR(), ctypes.c_void_p()
            ok = api.GetQueuedCompletionStatus(port, ctypes.byref(message), ctypes.byref(key), ctypes.byref(value), 0)
            if not ok:
                if ctypes.get_last_error() == 258:  # WAIT_TIMEOUT
                    return
                api.check(ok, "GetQueuedCompletionStatus")
            event_counts[str(message.value)] = event_counts.get(str(message.value), 0) + 1
            if message.value in (6, 7, 8, 9, 10) and value.value:
                remember(int(value.value))
            if message.value == 10:  # JOB_OBJECT_MSG_JOB_MEMORY_LIMIT
                memory_event = True

    def sample():
        nonlocal samples, max_rss, max_active, max_private_commit
        active = api.process_ids(job)
        total_rss = 0
        total_private_commit = 0
        for pid in active:
            remember(pid)
            handle = held.get(pid)
            if not handle:
                continue
            counters = _MemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if api.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                record = records[pid]
                record["sample_count"] += 1
                record["peak_working_set_bytes"] = max(record["peak_working_set_bytes"] or 0, counters.PeakWorkingSetSize)
                record["peak_commit_bytes"] = max(record["peak_commit_bytes"] or 0, counters.PeakPagefileUsage)
                total_rss += counters.WorkingSetSize
                total_private_commit += counters.PrivateUsage
            else:
                records[pid]["memory_query_failures"] += 1
        samples += 1
        max_rss = max(max_rss, total_rss)
        max_private_commit = max(max_private_commit, total_private_commit)
        max_active = max(max_active, len(active))

    def wait_empty():
        deadline = time.monotonic() + 5
        while api.query(job, 1, _Accounting()).ActiveProcesses:
            if time.monotonic() >= deadline:
                raise ProcessMonitoringError("Job termination did not reach zero active processes within 5 seconds")
            time.sleep(0.01)

    try:
        job = api.check(api.CreateJobObjectW(None, None), "CreateJobObjectW")
        limits = _ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000 | 0x200  # KILL_ON_JOB_CLOSE | JOB_MEMORY
        limits.JobMemoryLimit = memory_limit
        api.check(api.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)), "SetInformationJobObject(limits)")
        actual = api.query(job, 9, _ExtendedLimits())
        if actual.JobMemoryLimit != memory_limit or actual.BasicLimitInformation.LimitFlags != 0x2200:
            raise ProcessMonitoringError("Job limits did not round-trip exactly; refusing to start worker")
        port = api.check(api.CreateIoCompletionPort(w.HANDLE(-1), None, 0, 1), "CreateIoCompletionPort")
        association = _CompletionPort(1, port)
        api.check(api.SetInformationJobObject(job, 7, ctypes.byref(association), ctypes.sizeof(association)), "SetInformationJobObject(completion port)")
        with contextlib.ExitStack() as files:
            def file_handle(value, mode):
                if value is None:
                    value = files.enter_context(open(os.devnull, mode))
                elif isinstance(value, (str, os.PathLike)):
                    value = files.enter_context(open(value, mode))
                else:
                    value.flush()
                raw = msvcrt.get_osfhandle(value.fileno())
                duplicate = w.HANDLE()
                current = api.GetCurrentProcess()
                api.check(api.DuplicateHandle(current, raw, current, ctypes.byref(duplicate), 0, True, 2), "DuplicateHandle(stdio)")
                inherited.append(duplicate.value)
                return duplicate.value

            startup = _StartupInfoEx()
            startup.StartupInfo.cb = ctypes.sizeof(startup)
            startup.StartupInfo.dwFlags = 0x100 | 0x1  # USESTDHANDLES | USESHOWWINDOW
            startup.StartupInfo.wShowWindow = 0  # SW_HIDE
            startup.StartupInfo.hStdInput = file_handle(None, "rb")
            startup.StartupInfo.hStdOutput = file_handle(stdout, "wb")
            startup.StartupInfo.hStdError = file_handle(stderr, "wb")
            size = SIZE_T()
            api.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size))
            if not size.value:
                raise ProcessMonitoringError("InitializeProcThreadAttributeList returned no size")
            attribute_buffer = ctypes.create_string_buffer(size.value)
            api.check(api.InitializeProcThreadAttributeList(attribute_buffer, 1, 0, ctypes.byref(size)), "InitializeProcThreadAttributeList")
            attribute_initialized = True
            startup.lpAttributeList = ctypes.cast(attribute_buffer, ctypes.c_void_p)
            allowed = (w.HANDLE * len(inherited))(*inherited)
            api.check(api.UpdateProcThreadAttribute(attribute_buffer, 0, 0x20002, allowed, ctypes.sizeof(allowed), None, None), "UpdateProcThreadAttribute(handle list)")
            command = ctypes.create_unicode_buffer(subprocess.list2cmdline(argv))
            # No code executes until assignment succeeds, including venv redirectors.
            flags = 0x4 | 0x80000 | 0x08000000  # SUSPENDED | EXTENDED_STARTUPINFO_PRESENT | NO_WINDOW
            api.check(api.CreateProcessW(executable, command, None, None, True, flags, None, cwd,
                                        ctypes.byref(startup), ctypes.byref(process)), "CreateProcessW(suspended)")
            api.check(api.AssignProcessToJobObject(job, process.hProcess), "AssignProcessToJobObject")
            assigned = True
            in_job = w.BOOL()
            api.check(api.IsProcessInJob(process.hProcess, job, ctypes.byref(in_job)), "IsProcessInJob")
            if not in_job.value:
                raise ProcessMonitoringError("Suspended worker was not attached to job")
            remember(int(process.dwProcessId))
            if api.ResumeThread(process.hThread) == 0xFFFFFFFF:
                api.check(False, "ResumeThread")
            # Stdio handles are inherited only by the target; the Job handle is not.
            for handle in inherited:
                api.CloseHandle(handle)
            inherited.clear()
            while True:
                drain_events()
                sample()
                accounting = api.query(job, 1, _Accounting())
                if memory_event:
                    reason = "memory_limit"
                    if accounting.ActiveProcesses:
                        api.check(api.TerminateJobObject(job, 0xE0000002), "TerminateJobObject(memory)")
                    wait_empty()
                    break
                if not accounting.ActiveProcesses:
                    break
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    reason = "timeout"
                    api.check(api.TerminateJobObject(job, 0xE0000001), "TerminateJobObject(timeout)")
                    wait_empty()
                    break
                time.sleep(min(poll_interval, timeout - elapsed))
            drain_events()
            if memory_event:
                reason = "memory_limit"
            accounting = api.query(job, 1, _Accounting())
            limits = api.query(job, 9, _ExtendedLimits())
            # STILL_ACTIVE is also a legal application exit code. The process
            # handle's signaled state, not an exit-code comparison, proves exit.
            # Job ActiveProcesses can reach zero just before the launcher
            # handle is signaled, so allow a bounded completion wait here.
            if accounting.ActiveProcesses or api.WaitForSingleObject(process.hProcess, 5000) != 0:
                raise ProcessMonitoringError("Process tree not fully reaped")
            exitcode = w.DWORD()
            api.check(api.GetExitCodeProcess(process.hProcess, ctypes.byref(exitcode)), "GetExitCodeProcess")
            for pid, handle in held.items():
                child_exit = w.DWORD()
                if api.GetExitCodeProcess(handle, ctypes.byref(child_exit)):
                    records[pid]["returncode"] = int(child_exit.value)
                counters = _MemoryCounters()
                counters.cb = ctypes.sizeof(counters)
                if api.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                    records[pid]["peak_working_set_bytes"] = max(records[pid]["peak_working_set_bytes"] or 0, counters.PeakWorkingSetSize)
                    records[pid]["peak_commit_bytes"] = max(records[pid]["peak_commit_bytes"] or 0, counters.PeakPagefileUsage)
            complete = True
            return {"schema": "css-2d-process-resource-v1", "returncode": int(exitcode.value),
                    "termination_reason": reason, "wall_seconds": time.monotonic() - start,
                    "launcher_pid": int(process.dwProcessId),
                    "observed_process_ids": sorted(records), "processes": [records[k] for k in sorted(records)],
                    "job_total_processes": int(accounting.TotalProcesses),
                    "all_process_ids_observed": len(records) == accounting.TotalProcesses,
                    "peak_job_memory_bytes": int(limits.PeakJobMemoryUsed),
                    "peak_job_process_memory_bytes": int(limits.PeakProcessMemoryUsed),
                    "peak_tree_working_set_bytes_sampled": max_rss,
                    "peak_tree_private_commit_bytes_sampled": max_private_commit,
                    "peak_active_processes_sampled": max_active, "sample_count": samples,
                    "poll_interval_seconds": poll_interval,
                    "job_cpu_user_seconds": accounting.TotalUserTime / 10_000_000,
                    "job_cpu_kernel_seconds": accounting.TotalKernelTime / 10_000_000,
                    "job_events": event_counts, "job_memory_limit_event_observed": memory_event,
                    "memory_limit_event_delivery_guaranteed": False,
                    "memory_limit_bytes": memory_limit, "timeout_seconds": timeout,
                    "enforcement": "windows-job-commit", "tree_cleanup_verified": True,
                    "memory_metric": "job-wide committed virtual memory; raw OS peak may include rejected allocation",
                    "working_set_metric": "sampled sum; shared pages may be counted more than once"}
    finally:
        if process.hProcess and not complete:
            if assigned:
                api.TerminateJobObject(job, 0xE0000003)
                # Also wait for descendants when the monitor itself raises.
                # If querying is unavailable, closing the noninherited job handle
                # below still enforces KILL_ON_JOB_CLOSE; no success is returned.
                try:
                    wait_empty()
                except ProcessMonitoringError:
                    pass
            else:
                # The unassigned worker is still suspended and has no descendants.
                api.TerminateProcess(process.hProcess, 0xE0000003)
            api.WaitForSingleObject(process.hProcess, 5000)
        if job:
            api.CloseHandle(job)  # KILL_ON_JOB_CLOSE also handles monitor exceptions.
        for handle in held.values():
            api.CloseHandle(handle)
        for handle in inherited:
            api.CloseHandle(handle)
        if process.hThread:
            api.CloseHandle(process.hThread)
        if process.hProcess:
            api.CloseHandle(process.hProcess)
        if attribute_initialized:
            api.DeleteProcThreadAttributeList(attribute_buffer)
        if port:
            api.CloseHandle(port)

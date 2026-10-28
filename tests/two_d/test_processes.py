"""Integration tests: real Windows Job caps, venv redirectors and descendants."""
import ctypes
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from css_decoupling.two_d import processes


@unittest.skipUnless(os.name == "nt", "Windows Job integration tests")
class ProcessTests(unittest.TestCase):
    def setUp(self):
        base = os.environ.get("CSS_2D_PROCESS_TEST_OUTPUT")
        if base:
            Path(base).mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix=self._testMethodName + "-", dir=base))
        self.python = os.environ.get("CSS_2D_PROCESS_TEST_PYTHON", sys.executable)

    def script(self, source):
        path = self.directory / "worker.py"
        path.write_text(source, encoding="utf-8", newline="\n")
        return path

    def run_worker(self, script, *args, timeout=8, cap=256 * 1024**2):
        result = processes.run_limited([self.python, str(script), *args],
            self.directory / "stdout.txt", self.directory / "stderr.txt", timeout, cap,
            cwd=self.directory, poll_interval=0.02)
        (self.directory / "resources.json").write_text(json.dumps(result, indent=2) + "\n",
                                                      encoding="utf-8", newline="\n")
        self.assertTrue(result["tree_cleanup_verified"])
        self.assertEqual(result["memory_limit_bytes"], cap)
        return result

    def test_real_worker_memory_and_literal_arguments(self):
        script = self.script("""import json, os, sys, time
data = bytearray(24 * 1024**2)
print(json.dumps({'pid':os.getpid(), 'argv':sys.argv[1:]}), flush=True)
time.sleep(0.2)
""")
        args = ['argument with spaces', 'quote"inside', 'ends' + chr(92), '$()', '中文']
        result = self.run_worker(script, *args)
        actual = json.loads((self.directory / "stdout.txt").read_text(encoding="utf-8"))
        self.assertEqual(actual["argv"], args)
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["termination_reason"], "exited")
        worker = next(x for x in result["processes"] if x["pid"] == actual["pid"])
        self.assertTrue(worker["handle_verified_in_job"])
        self.assertGreater(worker["peak_working_set_bytes"], 24 * 1024**2)
        self.assertGreater(result["peak_job_memory_bytes"], 24 * 1024**2)
        self.assertGreater(result["peak_tree_working_set_bytes_sampled"], 24 * 1024**2)

    def test_timeout_kills_venv_child_and_grandchild_without_late_writes(self):
        script = self.script("""import os, pathlib, subprocess, sys, time
role = int(sys.argv[1])
pathlib.Path(str(role)+'.pid').write_text(str(os.getpid()))
if role < 2:
    subprocess.Popen([sys.executable, __file__, str(role+1)])
time.sleep(2.2)
pathlib.Path(str(role)+'.late').write_text('ORPHAN')
time.sleep(20)
""")
        result = self.run_worker(script, "0", timeout=1.1)
        self.assertEqual(result["termination_reason"], "timeout")
        actual_pids = [int((self.directory / f"{i}.pid").read_text()) for i in range(3)]
        self.assertTrue(set(actual_pids).issubset(result["observed_process_ids"]))
        self.assertGreaterEqual(result["job_total_processes"], 3)
        self.assertGreaterEqual(result["peak_active_processes_sampled"], 3)
        time.sleep(1.4)
        self.assertEqual(list(self.directory.glob("*.late")), [])

    def test_job_cap_covers_aggregate_descendant_commit(self):
        script = self.script("""import os, pathlib, subprocess, sys, time
role = int(sys.argv[1])
chunks = [bytearray(28 * 1024**2)]
pathlib.Path(str(role)+'.pid').write_text(str(os.getpid()))
if role < 2:
    subprocess.Popen([sys.executable, __file__, str(role+1)])
else:
    time.sleep(0.2)
    try:
        for _ in range(20):
            chunks.append(bytearray(4 * 1024**2))
            time.sleep(0.025)
    except MemoryError:
        print('allocation denied by job', flush=True)
time.sleep(3)
pathlib.Path(str(role)+'.late').write_text('ORPHAN')
time.sleep(20)
""")
        result = self.run_worker(script, "0", timeout=6, cap=128 * 1024**2)
        self.assertEqual(result["termination_reason"], "memory_limit")
        self.assertTrue(result["job_memory_limit_event_observed"])
        actual_pids = [int((self.directory / f"{i}.pid").read_text()) for i in range(3)]
        self.assertTrue(set(actual_pids).issubset(result["observed_process_ids"]))
        self.assertGreater(result["peak_job_memory_bytes"], 100 * 1024**2)
        self.assertLess(result["peak_job_process_memory_bytes"], 80 * 1024**2)
        time.sleep(3.2)
        self.assertEqual(list(self.directory.glob("*.late")), [])

    def test_kernel_cap_remains_when_notifications_are_unavailable(self):
        script = self.script("""import time
try:
    allocation = bytearray(256 * 1024**2)
except MemoryError:
    print('kernel denied allocation', flush=True)
else:
    raise RuntimeError('memory cap was not enforced')
time.sleep(0.1)
""")
        api = processes._Win32()
        def no_notifications(*args):
            ctypes.set_last_error(258)
            return False
        with patch.object(api, "GetQueuedCompletionStatus", no_notifications), patch.object(processes, "_Win32", return_value=api):
            result = self.run_worker(script, cap=64 * 1024**2)
        self.assertEqual(result["returncode"], 0)
        self.assertFalse(result["job_memory_limit_event_observed"])
        self.assertIn('kernel denied allocation', (self.directory / 'stdout.txt').read_text())

    def test_root_exit_does_not_abandon_descendants(self):
        script = self.script("""import pathlib, subprocess, sys, time
if len(sys.argv) == 1:
    subprocess.Popen([sys.executable, __file__, 'child'])
else:
    pathlib.Path('child-started').write_text('yes')
    time.sleep(1.3)
    pathlib.Path('child-late').write_text('ORPHAN')
    time.sleep(20)
""")
        result = self.run_worker(script, timeout=0.6)
        self.assertEqual(result["termination_reason"], "timeout")
        self.assertTrue((self.directory / "child-started").exists())
        time.sleep(0.9)
        self.assertFalse((self.directory / "child-late").exists())

    def test_explicit_exit_259_is_not_mistaken_for_running(self):
        script = self.script("import sys\nsys.exit(259)\n")
        result = self.run_worker(script)
        self.assertEqual(result["returncode"], 259)
        self.assertEqual(result["termination_reason"], "exited")

    def test_assignment_failure_never_resumes_worker(self):
        script = self.script("from pathlib import Path\nPath('must-not-run').write_text('FAILED')\n")
        api = processes._Win32()
        def denied(*args):
            ctypes.set_last_error(5)
            return False
        with patch.object(api, "AssignProcessToJobObject", denied), patch.object(processes, "_Win32", return_value=api):
            with self.assertRaisesRegex(processes.ProcessMonitoringError, "AssignProcessToJobObject"):
                self.run_worker(script)
        time.sleep(0.1)
        self.assertFalse((self.directory / "must-not-run").exists())

    def test_no_handle_leak_on_repeated_success_and_failure(self):
        api = processes._Win32()
        query = api.kernel.GetProcessHandleCount
        query.argtypes = [processes.w.HANDLE, ctypes.POINTER(processes.w.DWORD)]
        query.restype = processes.w.BOOL
        def count():
            result = processes.w.DWORD()
            self.assertTrue(query(api.GetCurrentProcess(), ctypes.byref(result)))
            return result.value
        script = self.script("pass\n")
        before = count()
        for _ in range(5):
            self.run_worker(script)
            with self.assertRaises(processes.ProcessMonitoringError):
                self.run_worker(script, timeout=0.01, cap=1)
        self.assertLessEqual(count(), before + 2)

    def test_monitor_exception_terminates_descendants(self):
        script = self.script("""import pathlib, subprocess, sys, time
if len(sys.argv) == 1:
    subprocess.Popen([sys.executable, __file__, 'child'])
else:
    pathlib.Path('child-started').write_text('yes')
    time.sleep(0.8)
    pathlib.Path('child-late').write_text('ORPHAN')
time.sleep(20)
""")
        api = processes._Win32()
        def broken_query(*args):
            time.sleep(0.3)
            raise processes.ProcessMonitoringError("injected process-list failure")
        with patch.object(api, "process_ids", broken_query), patch.object(processes, "_Win32", return_value=api):
            with self.assertRaisesRegex(processes.ProcessMonitoringError, "injected process-list"):
                self.run_worker(script)
        self.assertTrue((self.directory / "child-started").exists())
        time.sleep(0.7)
        self.assertFalse((self.directory / "child-late").exists())


class ProcessContractTests(unittest.TestCase):
    def test_invalid_contract_rejected_before_launch(self):
        with self.assertRaises(ValueError):
            processes.run_limited("echo hello", None, None, 1, 1024)
        for cap in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                processes.run_limited(["ignored"], None, None, 1, cap)
        for timeout in (0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                processes.run_limited(["ignored"], None, None, timeout, 1024)

    def test_other_platform_fails_closed(self):
        with patch.object(processes.os, "name", "posix"):
            with self.assertRaisesRegex(processes.ProcessMonitoringError, "no worker was launched"):
                processes.run_limited(["ignored"], None, None, 1, 1024)


if __name__ == "__main__":
    unittest.main()

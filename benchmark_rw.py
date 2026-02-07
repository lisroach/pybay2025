
"""
This file uses an unbalanced workload and writes to a shared dict.
It demonstrates the benefits of using ft_utils over multiprocessing.

Run with flags to explore different scenarios:
  - --no-locks: Run without locks
  - --use-stdlib-locks: Use standard library locks instead of ft_utils

Change --work-range or --shared-data-size to adjust workload size.
"""
# pyre-strict

import argparse
import math
import multiprocessing as mp
import queue
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing.managers import DictProxy, SyncManager
from multiprocessing.shared_memory import ShareableList

from ft_utils.concurrency import AtomicInt64, ConcurrentDict, ConcurrentQueue


# Get number of CPU cores
CPU_COUNT: int = mp.cpu_count()


def expensive_calculation(n: int) -> float:
    # UNBALANCED WORKLOAD: Some tasks are 500x heavier than others!
    # This creates poor load balance with static partitioning (multiprocessing)
    # but perfect balance with work-stealing (free-threading)
    result = 0

    # Every 10th task is 200x more expensive
    iterations = 20000 if n % 10 == 0 else 100

    for i in range(iterations):
        result += (
            math.sin(n + i)
            * math.cos(n - i)
            * math.sqrt(abs(n - i + 1))
            * math.log(abs(n + i + 1))
        )
    return result


def worker_with_work_stealing(
    work_queue: ConcurrentQueue,
    results_dict: ConcurrentDict,
    progress_counter: AtomicInt64,
    shared_data: list[int],
) -> tuple[float, int, int]:
    """
    Free-threading worker with work stealing and shared mutable state.

    Demonstrates efficient sharing with ConcurrentDict/AtomicInt64:
    - ConcurrentDict: Sharded dict (fine-grained per-object locking)
    - AtomicInt64: True lock-free hardware atomic operations
    - No IPC or serialization overhead
    - Perfect load balancing through work stealing
    """
    total = 0
    lookups = 0
    tasks_completed = 0

    while True:
        try:
            # Pop work from shared queue
            work_item = work_queue.pop(timeout=0.001)
        except Exception:  # Queue empty or shutdown
            break

        # Do computation
        calc_result = expensive_calculation(work_item)
        total += calc_result

        # Read from shared data
        if work_item % 100 < len(shared_data):
            lookups += shared_data[work_item % 100]

        # Write result to shared dict
        results_dict[work_item] = calc_result

        # Update shared atomic counter
        progress_counter.incr()
        tasks_completed += 1

    return total, lookups, tasks_completed


def worker_mp_with_shared_dict(
    chunk_start: int, chunk_end: int, shm_name: str, shared_dict: DictProxy[int, float]
) -> tuple[float, int, int]:
    """
    Multiprocessing worker with Manager.dict() for shared mutable state.
    """
    # Attach to shared memory
    shared_data = ShareableList(name=shm_name)

    total = 0
    lookups = 0
    tasks_completed = 0

    # Process assigned chunk (no work stealing possible)
    for work_item in range(chunk_start, chunk_end):
        # Do computation (same as free-threading)
        calc_result = expensive_calculation(work_item)
        total += calc_result

        # Read from shared data (same as free-threading)
        if work_item % 100 < len(shared_data):
            lookups += shared_data[work_item % 100]

        shared_dict[work_item] = calc_result
        tasks_completed += 1

    shared_data.shm.close()
    return total, lookups, tasks_completed


def benchmarking_mp_with_shared_dict(
    shared_data: list[int], chunk_size: int
) -> tuple[float, list[tuple[float, int, int]]]:
    """
    Multiprocessing with Manager.dict() to share mutable state across processes.

    This demonstrates the massive overhead of sharing mutable state in multiprocessing:
    - Manager spawns a separate server process
    - Every dict operation requires IPC (inter-process communication)
    - Data must be serialized/deserialized with pickle
    - Context switches between processes

    Compare this to free-threading's ConcurrentDict:
    - Sharded dict with fine-grained per-object locking
    - No IPC or serialization
    - Low contention due to sharding
    """
    chunks = [(i * chunk_size, (i + 1) * chunk_size) for i in range(CPU_COUNT)]

    # Create shared memory using ShareableList
    shared_list = ShareableList(shared_data)

    manager: SyncManager = mp.Manager()
    shared_dict: DictProxy[int, float] = manager.dict()

    results = []
    start_time = time.time()
    try:
        with ProcessPoolExecutor(max_workers=CPU_COUNT) as executor:
            futures = [
                executor.submit(
                    worker_mp_with_shared_dict,
                    start,
                    end,
                    shared_list.shm.name,
                    shared_dict,
                )
                for start, end in chunks
            ]
            results = [f.result() for f in futures]
    finally:
        # Cleanup shared memory
        shared_list.shm.close()
        shared_list.shm.unlink()
        manager.shutdown()

    return time.time() - start_time, results


def worker_with_stdlib_locks(
    work_queue: queue.Queue[int | None],
    results_dict: dict[int | str, float],
    dict_lock: threading.Lock,
    progress_counter: list[int],
    counter_lock: threading.Lock,
    shared_data: list[int],
) -> tuple[float, int, int]:
    """
    Free-threading worker using standard library locks (no ft_utils).

    Uses traditional threading primitives:
    - queue.Queue (thread-safe but with locks)
    - Regular dict + Lock for shared state
    - Regular int + Lock for counter

    This demonstrates the overhead of explicit locks compared to ft_utils:
    - Explicit lock acquisition/release for every shared state access
    - High lock contention (single lock per structure)
    - Less efficient than ft_utils' sharded approach
    """
    total = 0
    lookups = 0
    tasks_completed = 0

    while True:
        try:
            # Get work from queue (uses internal locks)
            work_item = work_queue.get(timeout=0.001)
            if work_item is None:  # Sentinel value for shutdown
                break
        except queue.Empty:
            break

        # Do computation
        calc_result = expensive_calculation(work_item)
        total += calc_result

        # Read from shared data (no lock needed for reads)
        if work_item % 100 < len(shared_data):
            lookups += shared_data[work_item % 100]

        # Write result to shared dict (REQUIRES LOCK!)
        # Unlike ConcurrentDict's sharded approach, this requires:
        # - Explicit lock acquisition
        # - High lock contention (single lock for entire dict)
        # - Lock release
        with dict_lock:
            results_dict[work_item] = calc_result

        # Update shared counter (REQUIRES LOCK!)
        # Unlike AtomicInt64's lock-free hardware atomic operation, this requires:
        # - Explicit lock acquisition
        # - Read-modify-write under lock
        # - Lock release
        with counter_lock:
            progress_counter[0] += 1
        tasks_completed += 1

    return total, lookups, tasks_completed


def benchmarking_threads(
    shared_data: list[int], work_range: int
) -> tuple[float, list[tuple[float, int, int]]]:
    """
    Free-threading with work strealing and efficient shared mutable state.

    Uses ConcurrentDict/AtomicInt64 for efficient in-memory sharing:
    - ConcurrentDict: Sharded dict with fine-grained locking (low contention)
    - AtomicInt64: True lock-free hardware atomic operations
    """
    # Create shared data structures
    work_queue = ConcurrentQueue(CPU_COUNT)
    results_dict = ConcurrentDict(CPU_COUNT)
    progress_counter = AtomicInt64(0)

    # Fill work queue with all tasks
    for i in range(work_range):
        work_queue.push(i)

    # Signal queue is complete
    work_queue.shutdown()

    results = []
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=CPU_COUNT) as executor:
        futures = [
            executor.submit(
                worker_with_work_stealing,
                work_queue,
                results_dict,
                progress_counter,
                shared_data,
            )
            for _ in range(CPU_COUNT)
        ]
        results = [f.result() for f in futures]

    return time.time() - start_time, results


def worker_without_locks(
    work_queue: queue.Queue[int | None],
    results_dict: dict[int, float],
    progress_counter: list[int],
    shared_data: list[int],
) -> tuple[float, int, int]:
    """
    Free-threading worker without any locks

    This is INTENTIONALLY BROKEN to show what happens without proper synchronization:
    - Race conditions on dict writes (lost updates)
    - Race conditions on counter (incorrect final count)
    - Data corruption possible

    WARNING: This will produce incorrect results!
    """
    total = 0
    lookups = 0
    tasks_completed = 0

    while True:
        try:
            # Queue has internal locks, so this is safe
            work_item = work_queue.get(timeout=0.001)
            if work_item is None:
                break
        except queue.Empty:
            break

        # Do computation
        calc_result = expensive_calculation(work_item)
        total += calc_result

        # Read from shared data (reads are generally safe)
        if work_item % 100 < len(shared_data):
            lookups += shared_data[work_item % 100]

        # Write to dict without lock
        # Multiple threads can write simultaneously causing:
        # - Lost updates (some writes never happen)
        # - Potentially corrupted dict internal state
        results_dict[work_item] = calc_result

        # Increment counter without lock
        # The += operation is NOT atomic, it's actually:
        # 1. Read current value
        # 2. Add 1
        # 3. Write new value
        # Multiple threads doing this simultaneously will lose updates
        progress_counter[0] += 1
        tasks_completed += 1

    return total, lookups, tasks_completed


def benchmarking_threads_stdlib(
    shared_data: list[int], work_range: int
) -> tuple[float, list[tuple[float, int, int]]]:
    """
    Free-threading with standard library locks (no ft_utils).

    Uses traditional Python threading primitives:
    - queue.Queue with internal locks
    - Regular dict + Lock for shared state
    - Regular int + Lock for counter

    This demonstrates:
    - Explicit lock acquisition/release overhead for every shared access
    - High lock contention (single lock per structure)
    - Much slower than ft_utils' sharded approach
    - But still faster than multiprocessing IPC overhead
    """
    # Create shared data structures using standard library
    work_queue: queue.Queue[int | None] = queue.Queue()
    results_dict: dict[int | str, float] = {}
    dict_lock = threading.Lock()
    progress_counter = [0]  # Use list so it's mutable
    counter_lock = threading.Lock()

    # Fill work queue with all tasks
    for i in range(work_range):
        work_queue.put(i)

    # Add sentinel values for each worker to signal shutdown
    for _ in range(CPU_COUNT):
        work_queue.put(None)

    results = []
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=CPU_COUNT) as executor:
        futures = [
            executor.submit(
                worker_with_stdlib_locks,
                work_queue,
                results_dict,
                dict_lock,
                progress_counter,
                counter_lock,
                shared_data,
            )
            for _ in range(CPU_COUNT)
        ]
        results = [f.result() for f in futures]

    return time.time() - start_time, results


def benchmarking_threads_no_locks(
    shared_data: list[int], work_range: int
) -> tuple[float, list[tuple[float, int, int]], dict[int, float], int]:
    """
    Free-threading without locks
    """
    # Create shared data structures WITHOUT locks
    work_queue: queue.Queue[int | None] = queue.Queue()
    results_dict: dict[int, float] = {}
    progress_counter = [0]

    # Fill work queue with all tasks
    for i in range(work_range):
        work_queue.put(i)

    # Add sentinel values for each worker to signal shutdown
    for _ in range(CPU_COUNT):
        work_queue.put(None)

    results = []
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=CPU_COUNT) as executor:
        futures = [
            executor.submit(
                worker_without_locks,
                work_queue,
                results_dict,
                progress_counter,
                shared_data,
            )
            for _ in range(CPU_COUNT)
        ]
        results = [f.result() for f in futures]

    return time.time() - start_time, results, results_dict, progress_counter[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark free-threading vs multiprocessing with work-stealing"
    )
    parser.add_argument(
        "--work-range",
        type=int,
        default=50000,
        help="Total number of work items to process (default: 50000)",
    )
    parser.add_argument(
        "--shared-data-size",
        type=int,
        default=2000000,
        help="Size of shared data array (default: 2000000)",
    )
    parser.add_argument(
        "--use-stdlib-locks",
        action="store_true",
        help="Use standard library locks for threading instead of ft_utils (demonstrates lock overhead)",
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Run all benchmarks: ft_utils threading, stdlib threading, and multiprocessing",
    )
    parser.add_argument(
        "--no-locks",
        action="store_true",
        help="Run free-threading WITHOUT locks to demonstrate race conditions (WILL PRODUCE INCORRECT RESULTS)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    work_range = args.work_range
    shared_data_size = args.shared_data_size
    chunk_size = work_range // CPU_COUNT

    # Create shared data
    shared_data = list(range(shared_data_size))

    print("CPU-intensive computation with shared data access")
    print("System specs:")
    print(f"  - CPU cores: {CPU_COUNT}")
    print(f"  - Shared data size: {len(shared_data):,} elements")
    print("Running performance comparison...\n")

    # Collect timing results
    times = {}

    # Run threading benchmarks
    if args.no_locks:
        print("Testing threading WITHOUT locks...")
        time_no_locks, results_no_locks, results_dict, final_counter = (
            benchmarking_threads_no_locks(shared_data, work_range)
        )
        print(f"Threading (NO LOCKS):           {time_no_locks:.5f}s")
        times["no_locks"] = time_no_locks

        # Show the data corruption
        expected_dict_size = work_range
        actual_dict_size = len(results_dict)
        lost_dict_entries = expected_dict_size - actual_dict_size

        expected_counter = work_range
        actual_counter = final_counter
        lost_counter_updates = expected_counter - actual_counter

        print("\n" + "=" * 70)
        print("RACE CONDITION RESULTS (DATA CORRUPTION!)")
        print("=" * 70)
        print(f"Expected dict entries:    {expected_dict_size:,}")
        print(f"Actual dict entries:      {actual_dict_size:,}")
        print(
            f"Lost dict entries:        {lost_dict_entries:,} ({lost_dict_entries*100.0/expected_dict_size:.1f}% data loss!)"
        )
        print()
        print(f"Expected counter value:   {expected_counter:,}")
        print(f"Actual counter value:     {actual_counter:,}")
        print(
            f"Lost counter increments:  {lost_counter_updates:,} ({lost_counter_updates*100.0/expected_counter:.1f}% data loss!)"
        )
        print()
        print("WHY THIS HAPPENS:")
        print("  - Dict writes without locks: Multiple threads write simultaneously")
        print("  - Counter += without locks: Read-modify-write is NOT atomic")
        print("  - Results are NON-DETERMINISTIC: Run again for different numbers!")
        print()
        print(
            "SOLUTION: Use locks (stdlib) or efficient concurrent structures (ft_utils)"
        )
        print("=" * 70)
        return  # Exit early to focus on the race condition demonstration

    if args.compare_all or not args.use_stdlib_locks:
        print("Testing threading with ft_utils...")
        time_thread, results_freethread = benchmarking_threads(shared_data, work_range)
        print(f"Threading (ft_utils):           {time_thread:.5f}s")
        times["ft_utils"] = time_thread

    if args.compare_all or args.use_stdlib_locks:
        print("Testing threading with stdlib locks...")
        time_stdlib, results_stdlib = benchmarking_threads_stdlib(
            shared_data, work_range
        )
        print(f"Threading (stdlib locks):       {time_stdlib:.5f}s")
        times["stdlib"] = time_stdlib

    # Run multiprocessing benchmark
    print("Testing multiprocessing...")
    time_mp, results_mp_shared = benchmarking_mp_with_shared_dict(
        shared_data, chunk_size
    )
    print(f"Multiprocessing:                {time_mp:.5f}s")
    times["multiprocessing"] = time_mp

    # Compare results
    if "ft_utils" in times and "stdlib" in times:
        if times["ft_utils"] < times["stdlib"]:
            speedup = times["stdlib"] / times["ft_utils"]
            print(f"\nft_utils is {speedup:.1f}x faster than stdlib locks")
        else:
            slowdown = times["ft_utils"] / times["stdlib"]
            print(f"stdlib locks is {slowdown:.1f}x faster than ft_utils")

    if "ft_utils" in times and "multiprocessing" in times:
        if times["ft_utils"] < times["multiprocessing"]:
            speedup = times["multiprocessing"] / times["ft_utils"]
            print(f"ft_utils is {speedup:.1f}x faster than multiprocessing")
        else:
            slowdown = times["ft_utils"] / times["multiprocessing"]
            print(f"multiprocessing is {slowdown:.1f}x faster than ft_utils")

    if "stdlib" in times and "multiprocessing" in times:
        if times["stdlib"] < times["multiprocessing"]:
            speedup = times["multiprocessing"] / times["stdlib"]
            print(f"stdlib locks is {speedup:.1f}x faster than multiprocessing")
        else:
            slowdown = times["stdlib"] / times["multiprocessing"]
            print(f"multiprocessing is {slowdown:.1f}x faster than stdlib locks")

    print()


if __name__ == "__main__":
    main()

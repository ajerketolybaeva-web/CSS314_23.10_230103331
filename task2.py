
import random
import threading
import time
from pathlib import Path
import math


# ==========================================
# CONFIGURATION
# ==========================================

STUDENT_ID = 230103331  # Replace with your Student ID
TOTAL_POINTS = 40_000_000
THREAD_COUNT = 4

OUTPUT_FILE = str(
    Path(__file__).resolve().parent / "task2_output.txt"
)

shared_generator = random.Random(STUDENT_ID)
prng_lock = threading.Lock()


# ==========================================
# MONTE CARLO WORKER: SHARED PRNG
# ==========================================

def shared_prng_worker(points, results, thread_index):
    """
    All threads share one PRNG instance.
    Access to the generator is protected by a mutex.
    """

    inside = 0

    for _ in range(points):

        # Shared PRNG access protected by mutex
        with prng_lock:
            x = shared_generator.random()
            y = shared_generator.random()

        # Check whether the point is inside the quarter circle
        if x * x + y * y <= 1.0:
            inside += 1

    results[thread_index] = inside


# ==========================================
# MONTE CARLO WORKER: INDEPENDENT PRNG
# ==========================================

def independent_prng_worker(points, thread_index, results):
    """
    Each thread has its own independent PRNG.
    """

    seed = (STUDENT_ID * 10) + thread_index

    generator = random.Random(seed)

    inside = 0

    for _ in range(points):

        x = generator.random()
        y = generator.random()

        if x * x + y * y <= 1.0:
            inside += 1

    results[thread_index] = inside


# ==========================================
# RUN THREADS
# ==========================================

def run_benchmark(worker_function, independent=False):

    results = [0] * THREAD_COUNT

    # Divide points evenly between threads
    points_per_thread = TOTAL_POINTS // THREAD_COUNT
    remainder = TOTAL_POINTS % THREAD_COUNT

    threads = []

    start_time = time.perf_counter()

    for i in range(THREAD_COUNT):

        points = points_per_thread

        # Distribute remaining points if necessary
        if i < remainder:
            points += 1

        if independent:

            thread = threading.Thread(
                target=worker_function,
                args=(points, i, results)
            )

        else:

            thread = threading.Thread(
                target=worker_function,
                args=(points, results, i)
            )

        threads.append(thread)
        thread.start()

    # Wait for all threads
    for thread in threads:
        thread.join()

    end_time = time.perf_counter()

    elapsed_time = (end_time - start_time) * 1000

    total_inside = sum(results)

    pi_estimate = 4.0 * total_inside / TOTAL_POINTS

    absolute_error = abs(pi_estimate - math.pi)

    return pi_estimate, absolute_error, elapsed_time


# ==========================================
# MAIN
# ==========================================

def main():

    output_lines = []

    def log(message=""):
        print(message)
        output_lines.append(str(message))

    log("=" * 55)
    log("TASK 2: PSEUDO-RANDOM GENERATOR STATE BOTTLENECK")
    log("=" * 55)

    log(f"Student ID: {STUDENT_ID}")
    log(f"Threads: {THREAD_COUNT}")
    log(f"Total sample points: {TOTAL_POINTS:,}")

    # ======================================
    # MODE A: SHARED PRNG
    # ======================================

    log("\nRunning Shared PRNG...")

    shared_pi, shared_error, shared_time = run_benchmark(
        shared_prng_worker,
        independent=False
    )

    # ======================================
    # MODE B: INDEPENDENT PRNG
    # ======================================

    log("\nRunning Independent PRNG...")

    independent_pi, independent_error, independent_time = (
        run_benchmark(
            independent_prng_worker,
            independent=True
        )
    )

    # ======================================
    # SPEEDUP
    # ======================================

    speedup = shared_time / independent_time

    # ======================================
    # RESULTS
    # ======================================

    log("\n" + "=" * 55)
    log("BENCHMARK RESULTS")
    log("=" * 55)

    log("\nShared PRNG:")
    log(f"Computed PI: {shared_pi:.10f}")
    log(f"Absolute Error: {shared_error:.10f}")
    log(f"Wall Time: {shared_time:.2f} ms")

    log("\nIndependent PRNG:")
    log(f"Computed PI: {independent_pi:.10f}")
    log(f"Absolute Error: {independent_error:.10f}")
    log(f"Wall Time: {independent_time:.2f} ms")

    log(f"\nSpeedup Factor: {speedup:.4f}x")

    log("=" * 55)

    # ======================================
    # SAVE OUTPUT
    # ======================================

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        file.write("\n".join(output_lines))
        file.write("\n")

    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

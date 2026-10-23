
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np


# ==========================================
# CONFIGURATION
# ==========================================

STUDENT_ID = 2301033331
ARRAY_SIZE = 2_000_000
THREAD_COUNT = 4

OUTPUT_FILE = (
    Path(__file__).resolve().parent / "task3_output.txt"
)


# ==========================================
# GENERATE INPUT ARRAY
# ==========================================

def generate_array():
    """
    Generate integers from 1 to 10 using Student ID seed.
    """

    rng = np.random.default_rng(STUDENT_ID)

    return rng.integers(
        low=1,
        high=11,
        size=ARRAY_SIZE,
        dtype=np.int64
    )


# ==========================================
# SERIAL INCLUSIVE PREFIX SCAN
# ==========================================

def serial_prefix_scan(array):
    """
    output[i] = output[i - 1] + input[i]

    Inclusive prefix sum.
    """

    output = np.empty_like(array)

    output[0] = array[0]

    for i in range(1, len(array)):
        output[i] = output[i - 1] + array[i]

    return output


# ==========================================
# PARALLEL PREFIX SCAN
# ==========================================

def local_prefix_scan(chunk):
    """
    Compute the local inclusive prefix sum
    and return its total sum.
    """

    local_output = np.cumsum(chunk, dtype=np.int64)

    local_sum = int(local_output[-1])

    return local_output, local_sum


def parallel_prefix_scan(array):
    """
    Two-pass parallel prefix scan:

    Pass 1:
        Compute local prefix sums and chunk sums.

    Offset reduction:
        Calculate offsets for each chunk.

    Pass 2:
        Apply offsets to local prefix sums.
    """

    # Divide array into chunks
    chunks = np.array_split(array, THREAD_COUNT)

    # ======================================
    # PASS 1: LOCAL PREFIX SCANS
    # ======================================

    with ThreadPoolExecutor(
        max_workers=THREAD_COUNT
    ) as executor:

        local_results = list(
            executor.map(local_prefix_scan, chunks)
        )

    local_outputs = [
        result[0]
        for result in local_results
    ]

    chunk_sums = np.array(
        [result[1] for result in local_results],
        dtype=np.int64
    )

    # ======================================
    # OFFSET REDUCTION
    # ======================================

    offsets = np.zeros(
        THREAD_COUNT,
        dtype=np.int64
    )

    for i in range(1, THREAD_COUNT):

        offsets[i] = (
            offsets[i - 1] + chunk_sums[i - 1]
        )

    # ======================================
    # PASS 2: APPLY OFFSETS
    # ======================================

    def apply_offset(args):

        local_output, offset = args

        if offset != 0:
            local_output += offset

        return local_output

    with ThreadPoolExecutor(
        max_workers=THREAD_COUNT
    ) as executor:

        parallel_chunks = list(
            executor.map(
                apply_offset,
                zip(local_outputs, offsets)
            )
        )

    # Combine all chunks
    parallel_output = np.concatenate(
        parallel_chunks
    )

    return parallel_output


# ==========================================
# MAIN
# ==========================================

def main():

    output_lines = []

    def log(message=""):

        print(message)

        output_lines.append(str(message))

    log("=" * 55)
    log("TASK 3: PARALLEL PREFIX SCAN")
    log("=" * 55)

    log(f"Student ID: {STUDENT_ID}")
    log(f"Array size: {ARRAY_SIZE:,}")
    log(f"Threads: {THREAD_COUNT}")

    # ======================================
    # GENERATE ARRAY
    # ======================================

    log("\nGenerating input array...")

    array = generate_array()

    # ======================================
    # SERIAL BENCHMARK
    # ======================================

    log("\nRunning Serial Prefix Scan...")

    serial_start = time.perf_counter()

    serial_output = serial_prefix_scan(array)

    serial_time = (
        time.perf_counter() - serial_start
    ) * 1000

    # ======================================
    # PARALLEL BENCHMARK
    # ======================================

    log("Running Parallel Prefix Scan...")

    parallel_start = time.perf_counter()

    parallel_output = parallel_prefix_scan(array)

    parallel_time = (
        time.perf_counter() - parallel_start
    ) * 1000

    # ======================================
    # VERIFICATION
    # ======================================

    try:

        assert (
            parallel_output[-1] == serial_output[-1]
        ), "Parallel and serial final sums do not match"

        verification_status = "PASS"

    except AssertionError as error:

        verification_status = "FAIL"

        log("\nVerification Status: FAIL")
        log(str(error))

        with open(
            OUTPUT_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            file.write("\n".join(output_lines))
            file.write("\n")

        sys.exit(1)

    # ======================================
    # SPEEDUP
    # ======================================

    speedup = serial_time / parallel_time

    # ======================================
    # RESULTS
    # ======================================

    log("\n" + "=" * 55)
    log("BENCHMARK RESULTS")
    log("=" * 55)

    log(f"Serial Time: {serial_time:.2f} ms")
    log(f"Parallel Time: {parallel_time:.2f} ms")
    log(f"Speedup: {speedup:.4f}x")

    log(f"Verification Status: {verification_status}")

    log(f"\nSerial Final Sum: {serial_output[-1]}")
    log(f"Parallel Final Sum: {parallel_output[-1]}")

    log("=" * 55)

    # ======================================
    # SAVE OUTPUT
    # ======================================

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("\n".join(output_lines))
        file.write("\n")

    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

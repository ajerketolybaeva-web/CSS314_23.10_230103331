
import os
import time
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np
import psutil


# ==========================================
# CONFIGURATION
# ==========================================

STUDENT_ID = 230103331  
MATRIX_SIZE = 8000
DTYPE = np.float64

PHYSICAL_CORES = psutil.cpu_count(logical=False)
LOGICAL_CORES = os.cpu_count()

WORKERS = PHYSICAL_CORES or 1


# ==========================================
# SHARED MEMORY WORKER
# ==========================================

def init_worker(shm_name, shape):
    """
    Attach each worker process to the shared matrix.
    """
    global matrix
    global shared_memory_block

    shared_memory_block = shared_memory.SharedMemory(
        name=shm_name
    )

    matrix = np.ndarray(
        shape,
        dtype=DTYPE,
        buffer=shared_memory_block.buf
    )


# ==========================================
# MODE A: ROW-MAJOR
# ==========================================

def sum_rows(row_range):
    """
    Each worker processes a range of rows.
    Elements are accessed linearly across columns.
    """
    start_row, end_row = row_range

    total = 0.0

    for row in range(start_row, end_row):
        total += np.sum(matrix[row, :])

    return total


# ==========================================
# MODE B: COLUMN-MAJOR / STRIDED
# ==========================================

def sum_columns(column_range):
    """
    Each worker processes a range of columns.
    Each column is traversed across rows.

    In a C-contiguous NumPy matrix, this creates
    strided memory access.
    """
    start_column, end_column = column_range

    total = 0.0

    for column in range(start_column, end_column):
        total += np.sum(matrix[:, column])

    return total


# ==========================================
# CREATE MATRIX
# ==========================================

def create_matrix():
    rng = np.random.default_rng(STUDENT_ID)

    return rng.random(
        (MATRIX_SIZE, MATRIX_SIZE),
        dtype=DTYPE
    )


# ==========================================
# CREATE WORKER RANGES
# ==========================================

def create_ranges(total_items, workers):
    """
    Divide rows or columns between workers.
    """
    boundaries = np.linspace(
        0,
        total_items,
        workers + 1,
        dtype=int
    )

    return [
        (boundaries[i], boundaries[i + 1])
        for i in range(workers)
    ]


# ==========================================
# RUN BENCHMARK
# ==========================================

def benchmark(pool, ranges, function):
    start_time = time.perf_counter()

    partial_sums = pool.map(function, ranges)

    elapsed_time = time.perf_counter() - start_time

    total_sum = sum(partial_sums)

    return total_sum, elapsed_time


# ==========================================
# MAIN
# ==========================================

def main():

    print("=" * 50)
    print("TASK 1: ROW-MAJOR VS COLUMN-MAJOR")
    print("=" * 50)

    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical cores: {LOGICAL_CORES}")
    print(f"Physical cores: {PHYSICAL_CORES}")
    print(f"Workers: {WORKERS}")
    print(f"Matrix size: {MATRIX_SIZE} x {MATRIX_SIZE}")
    print(f"Data type: {DTYPE}")

    print("\nCreating matrix...")

    matrix_data = create_matrix()

    # Shared memory allocation
    shm = shared_memory.SharedMemory(
        create=True,
        size=matrix_data.nbytes
    )

    shared_matrix = np.ndarray(
        matrix_data.shape,
        dtype=DTYPE,
        buffer=shm.buf
    )

    shared_matrix[:] = matrix_data

    del matrix_data

    row_ranges = create_ranges(
        MATRIX_SIZE,
        WORKERS
    )

    column_ranges = create_ranges(
        MATRIX_SIZE,
        WORKERS
    )

    try:

        with mp.Pool(
            processes=WORKERS,
            initializer=init_worker,
            initargs=(shm.name, (MATRIX_SIZE, MATRIX_SIZE))
        ) as pool:

            # Warm-up
            pool.map(sum_rows, row_ranges)

            # MODE A
            print("\nRunning Mode A: Row-Major...")

            sum_a, time_a = benchmark(
                pool,
                row_ranges,
                sum_rows
            )

            # MODE B
            print("Running Mode B: Column-Major...")

            sum_b, time_b = benchmark(
                pool,
                column_ranges,
                sum_columns
            )

    finally:
        shm.close()
        shm.unlink()

    # ==========================================
    # RESULTS
    # ==========================================

    time_a_ms = time_a * 1000
    time_b_ms = time_b * 1000

    penalty_factor = time_b / time_a

    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)

    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical cores: {LOGICAL_CORES}")

    print(f"Mode A (Row-Major): {time_a_ms:.2f} ms")
    print(f"Mode B (Column-Major): {time_b_ms:.2f} ms")

    print(
        f"Cache Penalty Factor: "
        f"{penalty_factor:.2f}x"
    )

    print(f"\nMode A sum: {sum_a:.6f}")
    print(f"Mode B sum: {sum_b:.6f}")

    print("=" * 50)


if __name__ == "__main__":
    mp.freeze_support()
    main()

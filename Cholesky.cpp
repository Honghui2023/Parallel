/*
bench_cholesky.cpp

Unified benchmarking/demo program for Cholesky decomposition:
- Serial (in-place, lower-triangular)
- OpenMP-parallelized (column scaling + trailing updates)

Menu-driven console that follows the example layout shown in your images:
- small 4x4 demonstration
- single performance test (custom size)
- scalability analysis (multiple sizes)
- thread-scaling analysis (varying thread counts)
- run all analyses

Build:
  g++ -O3 -std=c++17 -fopenmp bench_cholesky.cpp -o bench_cholesky

Usage:
  ./bench_cholesky
  (then follow on-screen menu)

Notes:
- Uses column-major storage to match LAPACK/cuSOLVER conventions.
- Random SPD generator: A = M * M^T + n * I
- For reproducibility, a fixed seed is used by default.
*/

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include <omp.h>
using namespace std;
using highres_clock_t = chrono::high_resolution_clock;

static inline size_t idx(size_t i, size_t j, size_t n) { return i + j * n; } // column-major

// Generate SPD matrix A = M * M^T + n*I
vector<double> generate_spd(size_t n, unsigned seed = 123456) {
    mt19937_64 rng(seed);
    normal_distribution<double> dist(0.0, 1.0);
    vector<double> M(n * n);
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i)
            M[idx(i, j, n)] = dist(rng);

    vector<double> A(n * n, 0.0);
    for (size_t j = 0; j < n; ++j)
        for (size_t k = 0; k < n; ++k) {
            double mkj = M[idx(k, j, n)];
            for (size_t i = 0; i < n; ++i)
                A[idx(i, j, n)] += M[idx(i, k, n)] * mkj;
        }
    for (size_t i = 0; i < n; ++i) A[idx(i, i, n)] += static_cast<double>(n);
    return A;
}

// Serial in-place Cholesky (lower triangular stored in lower part)
bool cholesky_serial(vector<double>& A, size_t n) {
    for (size_t k = 0; k < n; ++k) {
        double akk = A[idx(k, k, n)];
        if (akk <= 0.0) return false;
        double diag = sqrt(akk);
        A[idx(k, k, n)] = diag;
        for (size_t i = k + 1; i < n; ++i)
            A[idx(i, k, n)] /= diag;
        for (size_t j = k + 1; j < n; ++j) {
            for (size_t i = j; i < n; ++i) {
                A[idx(i, j, n)] -= A[idx(i, k, n)] * A[idx(j, k, n)];
            }
        }
    }
    // zero upper triangle (for neat printing)
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < j; ++i)
            A[idx(i, j, n)] = 0.0;
    return true;
}

// OpenMP-parallel in-place Cholesky
bool cholesky_openmp(vector<double>& A, size_t n) {
    for (size_t k = 0; k < n; ++k) {
        double akk = A[idx(k, k, n)];
        if (akk <= 0.0) return false;
        double diag = sqrt(akk);
        A[idx(k, k, n)] = diag;
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = (ptrdiff_t)k + 1; i < (ptrdiff_t)n; ++i)
            A[idx(i, k, n)] /= diag;
#pragma omp parallel for schedule(dynamic)
        for (ptrdiff_t j = (ptrdiff_t)k + 1; j < (ptrdiff_t)n; ++j) {
            for (ptrdiff_t i = j; i < (ptrdiff_t)n; ++i) {
                A[idx(i, j, (size_t)n)] -= A[idx(i, k, (size_t)n)] * A[idx(j, k, (size_t)n)];
            }
        }
    }
#pragma omp parallel for collapse(2)
    for (ptrdiff_t j = 0; j < (ptrdiff_t)n; ++j)
        for (ptrdiff_t i = 0; i < j; ++i)
            A[idx(i, j, n)] = 0.0;
    return true;
}

// Compute relative Frobenius residual ||A_orig - L*L^T||_F / ||A_orig||_F
double relative_residual(const vector<double>& A_orig, const vector<double>& L, size_t n) {
    vector<double> B(n * n, 0.0);
    for (size_t j = 0; j < n; ++j)
        for (size_t k = 0; k < n; ++k) {
            double lkj = L[idx(k, j, n)];
            if (lkj == 0.0) continue;
            for (size_t i = 0; i < n; ++i)
                B[idx(i, j, n)] += L[idx(i, k, n)] * lkj;
        }
    double norm_diff = 0.0, normA = 0.0;
    for (size_t i = 0; i < n * n; ++i) {
        double d = A_orig[i] - B[i];
        norm_diff += d * d;
        normA += A_orig[i] * A_orig[i];
    }
    return sqrt(norm_diff) / sqrt(normA);
}

// Timing wrapper for function pointer (returns seconds). fn modifies matrix in-place.
template<typename Func>
double time_run(Func fn, vector<double>& A, size_t n, int repeats = 1) {
    double total = 0.0;
    for (int r = 0; r < repeats; ++r) {
        vector<double> tmp = A; // preserve original for repeated runs
        auto t0 = highres_clock_t::now();
        bool ok = fn(tmp, n);
        auto t1 = highres_clock_t::now();
        if (!ok) return -1.0;
        total += chrono::duration<double>(t1 - t0).count();
    }
    return total / repeats;
}

// Pretty-print small matrix (row-major-ish display but data is column-major)
void print_matrix(const vector<double>& A, size_t n, const string& name = "A") {
    cout << name << " (" << n << "x" << n << "):\n";
    cout << fixed << setprecision(6);
    for (size_t i = 0; i < n; ++i) {
        cout << "[ ";
        for (size_t j = 0; j < n; ++j) {
            cout << setw(10) << A[idx(i, j, n)];
            if (j + 1 < n) cout << ", ";
        }
        cout << " ]\n";
    }
}

// Single-run performance test and print result in same style as example
void single_performance_test(size_t n, int omp_threads, int runs = 3) {
    cout << "Processing " << n << "x" << n << " matrix...\n\n";
    auto A = generate_spd(n);
    // serial
    double serial_time = time_run(cholesky_serial, A, n, runs);
    // parallel
    omp_set_num_threads(omp_threads);
    double parallel_time = time_run(cholesky_openmp, A, n, runs);

    double speedup = serial_time / parallel_time;
    double efficiency = speedup / omp_threads * 100.0;

    cout << "Performance Results for " << n << "x" << n << " matrix with " << omp_threads << " threads:\n\n";
    cout << "Serial Execution Time:   " << fixed << setprecision(6) << serial_time << " seconds\n";
    cout << "Parallel Execution Time: " << fixed << setprecision(6) << parallel_time << " seconds\n";
    cout << "Speedup:                 " << fixed << setprecision(4) << speedup << "x\n";
    cout << "Efficiency:              " << fixed << setprecision(2) << efficiency << "%\n\n";
}

// Run scalability analysis over a vector of sizes and print a summary table
void scalability_analysis(const vector<size_t>& sizes, int omp_threads, int runs = 3) {
    cout << "SCALABILITY SUMMARY TABLE\n\n";
    cout << setw(8) << "Size" << setw(15) << "Serial(s)" << setw(15) << "Parallel(s)" << setw(12) << "Speedup" << setw(12) << "Efficiency\n";
    cout << string(70, '-') << "\n";
    for (size_t n : sizes) {
        auto A = generate_spd(n);
        double serial_time = time_run(cholesky_serial, A, n, runs);
        omp_set_num_threads(omp_threads);
        double parallel_time = time_run(cholesky_openmp, A, n, runs);
        double speedup = serial_time / parallel_time;
        double efficiency = speedup / omp_threads * 100.0;
        cout << setw(8) << n
            << setw(15) << fixed << setprecision(6) << serial_time
            << setw(15) << fixed << setprecision(6) << parallel_time
            << setw(12) << fixed << setprecision(3) << speedup << "x"
            << setw(11) << fixed << setprecision(2) << efficiency << "%\n";
    }
    cout << "\n";
}

// Thread-scaling analysis: fix matrix size and vary thread counts (prints table similar to example)
void thread_scaling_analysis(size_t n, const vector<int>& threads_to_test, int runs = 3) {
    cout << "THREAD SCALING ANALYSIS\n\n";
    cout << "Matrix size: " << n << "x" << n << "\n";
    cout << "Maximum available threads: " << omp_get_max_threads() << "\n\n";
    auto A = generate_spd(n);

    cout << "Measuring serial performance...\n";
    double serial_time = time_run(cholesky_serial, A, n, runs);
    cout << "Testing with ";
    for (size_t i = 0; i < threads_to_test.size(); ++i) {
        cout << threads_to_test[i] << " thread(s)" << (i + 1 < threads_to_test.size() ? ", " : "...\n");
    }

    cout << "\n\n";
    cout << setw(8) << "Threads" << setw(15) << "Serial(s)" << setw(15) << "Parallel(s)" << setw(12) << "Speedup" << setw(12) << "Efficiency\n";
    cout << string(70, '-') << "\n";

    for (int t : threads_to_test) {
        omp_set_num_threads(t);
        double parallel_time = time_run(cholesky_openmp, A, n, runs);
        double speedup = serial_time / parallel_time;
        double efficiency = speedup / t * 100.0;
        cout << setw(8) << t
            << setw(15) << fixed << setprecision(6) << serial_time
            << setw(15) << fixed << setprecision(6) << parallel_time
            << setw(12) << fixed << setprecision(3) << speedup << "x"
            << setw(11) << fixed << setprecision(2) << efficiency << "%\n";
    }
    cout << "\n";
}

int main() {
    vector<size_t> default_sizes = { 100, 500, 1000, 2000 };
    vector<int> default_threads = { 1, 2, 4 }; // example; adapt to your CPU

    while (true) {
        cout << "---------------------------------------------------------\n";
        cout << "Select an option:\n";
        cout << "1. Demonstrate on small 4x4 example\n";
        cout << "2. Run single performance test (custom size)\n";
        cout << "3. Run scalability analysis (multiple sizes)\n";
        cout << "4. Run thread scaling analysis (multiple threads)\n";
        cout << "5. Run all analyses\n";
        cout << "0. Exit\n";
        cout << "Enter choice: ";
        int choice;
        if (!(cin >> choice)) break;

        if (choice == 0) break;
        else if (choice == 1) {
            size_t n = 4;
            auto A = generate_spd(n, 42);
            cout << "\nDEMONSTRATION: Small 4x4 System\n\n";
            print_matrix(A, n, "Original A");
            // Serial solve
            auto Acopy = A;
            bool ok = cholesky_serial(Acopy, n);
            if (!ok) { cout << "Serial Cholesky failed.\n"; continue; }
            print_matrix(Acopy, n, "L (serial)");
            double res = relative_residual(A, Acopy, n);
            cout << "Residual norm (serial): " << scientific << setprecision(6) << res << "\n\n";
            // Parallel solve
            Acopy = A;
            omp_set_num_threads(4);
            ok = cholesky_openmp(Acopy, n);
            if (!ok) { cout << "Parallel Cholesky failed.\n"; continue; }
            print_matrix(Acopy, n, "L (parallel)");
            res = relative_residual(A, Acopy, n);
            cout << "Residual norm (parallel): " << scientific << setprecision(6) << res << "\n\n";
        }
        else if (choice == 2) {
            cout << "Enter matrix size n (e.g., 512): ";
            size_t n; cin >> n;
            cout << "Enter number of threads for parallel run: ";
            int t; cin >> t;
            single_performance_test(n, t, 3);
        }
        else if (choice == 3) {
            cout << "Running scalability analysis for sizes:";
            for (auto s : default_sizes) cout << " " << s;
            cout << "\nEnter number of threads for parallel runs (e.g., 4): ";
            int t; cin >> t;
            scalability_analysis(default_sizes, t, 3);
        }
        else if (choice == 4) {
            cout << "Enter matrix size for thread-scaling (e.g., 1000): ";
            size_t n; cin >> n;
            cout << "Enter thread counts to test (space-separated, finish with 0):\n";
            vector<int> threads;
            while (true) {
                int x; cin >> x;
                if (x == 0) break;
                threads.push_back(x);
            }
            if (threads.empty()) threads = default_threads;
            thread_scaling_analysis(n, threads, 3);
        }
        else if (choice == 5) {
            // Run everything with defaults
            cout << "Enter number of threads for parallel runs (e.g., 4): ";
            int t; cin >> t;
            // small demo
            cout << "\n--- Small demo ---\n";
            {
                size_t n = 4;
                auto A = generate_spd(n, 42);
                print_matrix(A, n, "Original A");
                auto Acopy = A;
                cholesky_serial(Acopy, n);
                print_matrix(Acopy, n, "L (serial)");
                cout << "Residual (serial): " << relative_residual(A, Acopy, n) << "\n\n";
            }
            // single tests for default sizes
            for (auto s : default_sizes) single_performance_test(s, t, 3);
            // scalability summary
            scalability_analysis(default_sizes, t, 3);
            // thread scaling for one size (use 1000)
            thread_scaling_analysis(1000, vector<int>{1, 2, 4, 8}, 3);
        }
        else {
            cout << "Unknown choice\n";
        }
    }

    cout << "Exiting.\n";
    return 0;
}
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
    srand(seed);
    vector<double> M(n * n), A(n * n, 0.0);

    for (size_t j = 0; j < n; j++)
        for (size_t i = 0; i < n; i++)
            M[idx(i, j, n)] = (rand() % 5) + 1;

    for (size_t j = 0; j < n; j++)
        for (size_t k = 0; k < n; k++)
            for (size_t i = 0; i < n; ++i)
                A[idx(i, j, n)] += M[idx(i, k, n)] * M[idx(j, k, n)];

    for (size_t i = 0; i < n; i++)
        A[idx(i, i, n)] += n;

    return A;
}

// Serial in-place Cholesky (lower triangular)
bool cholesky_serial(vector<double>& A, size_t n) {
    for (size_t k = 0; k < n; k++) {    // k=step
        double diagonalValue = A[idx(k, k, n)];
        if (diagonalValue <= 0.0) {
            return false;
        }
            
        double diagonalElement = sqrt(diagonalValue);
        A[idx(k, k, n)] = diagonalElement;

        for (size_t i = k + 1; i < n; i++) {
            A[idx(i, k, n)] /= diagonalElement;
        }
           
        for (size_t i = k + 1; i < n; i++) {
            for (size_t j = k + 1; j <= i; j++) {
                A[idx(i, j, n)] -= A[idx(i, k, n)] * A[idx(j, k, n)];
            }
        }
    }

	// Clear upper triangle for clean output
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < i; j++) {
            A[idx(j, i, n)] = 0.0;
		}
    }
   
    return true;
}

// OpenMP-parallel in-place Cholesky
bool cholesky_openmp(vector<double>& A, size_t n) {
    for (size_t k = 0; k < n; k++) {
        double diagonalValue = A[idx(k, k, n)];
        if (diagonalValue <= 0.0) {
            return false;
        } 

        double diag = sqrt(diagonalValue);
        A[idx(k, k, n)] = diag;

#pragma omp parallel for schedule(static)
        for (int i = static_cast<int>(k + 1); i < static_cast<int>(n); i++)
            A[idx(i, k, n)] /= diag;

#pragma omp parallel for schedule(dynamic)
        for (int row_i = static_cast<int>(k) + 1; row_i < static_cast<int>(n); row_i++) {
            size_t i = static_cast<size_t>(row_i);
            for (size_t j = k + 1; j <= i; j++) {
                A[idx(i, j, n)] -= A[idx(i, k, n)] * A[idx(j, k, n)];
            }
        }

#pragma omp parallel for 
        for (int row_i = 0; row_i < static_cast<int>(n); row_i++) {
            for (int col_j = row_i + 1; col_j < static_cast<int>(n); col_j++) {
                A[idx(static_cast<size_t>(row_i), static_cast<size_t>(col_j), n)] = 0.0;
            }
        }
    }
    return true;
}

// Compute relative Frobenius residual ||A_orig - L*L^T||_F / ||A_orig||_F
double relative_residual(const vector<double>& A_orig, const vector<double>& L, size_t n) {
    vector<double> B(n * n, 0.0);

	// Compute B = L * L^T
    for (size_t col = 0; col < n; col++) {
        for (size_t k = 0; k < n; k++) {
            double element = L[idx(k, col, n)];

            if (element == 0.0) 
                continue;

            for (size_t row = 0; row < n; row++) {
                B[idx(row, col, n)] += L[idx(row, k, n)] * element;
            }
        }
    }
        
    double squared_diff = 0.0, matrix_size = 0.0;
    for (size_t i = 0; i < n * n; i++) {
        double d = A_orig[i] - B[i];
        squared_diff += d * d;
        matrix_size += A_orig[i] * A_orig[i];
    }
    return sqrt(squared_diff) / sqrt(matrix_size);
}

// Timing wrapper for function modifies matrix in-place, return average seconds
template<typename Func>
double time_run(Func fn, const vector<double>& A, size_t n, int repeats = 1) {
    double total_time = 0.0;

    for (int r = 0; r < repeats; ++r) {
        vector<double> tmp = A; // Copy original matrix
        auto start = highres_clock_t::now();
        bool success = fn(tmp, n);
        auto end = highres_clock_t::now();

        if (!success) {
            return -1.0; // Function failed
        }
        total_time += chrono::duration<double>(end - start).count();
    }
    return total_time / repeats;
}

// Print small matrix in readable format
// The matrix is stored in column-major order, but printed row by row
void print_matrix(const vector<double>& A, size_t n, const string& name = "A") {
    cout << name << " (" << n << "x" << n << "):\n";
    cout << fixed << setprecision(6);

    for (size_t i = 0; i < n; i++) {
        cout << "[ ";
        for (size_t j = 0; j < n; j++) {
            cout << setw(10) << A[idx(i, j, n)];
            if (j < n - 1) 
                cout << ", ";
        }
        cout << " ]\n";
    }
}

// Single-run performance test for a given matrix size and thread count
// Compare serial and parallel execution times, speedup, efficiency
void single_performance_test(size_t n, int num_threads, int runs = 3) {
    cout << "Processing " << n << "x" << n << " matrix...\n\n";

	// Generate SPD matrix
    vector<double> A = generate_spd(n);

    // serial
    double serial_time = time_run(cholesky_serial, A, n, runs);

    // parallel
    omp_set_num_threads(num_threads);
    double parallel_time = time_run(cholesky_openmp, A, n, runs);

    double speedup = serial_time / parallel_time;
    double efficiency = (speedup / num_threads) * 100.0;

    cout << "Performance Results for " << n << " x " << n << " matrix using " << num_threads << " threads:\n\n";
    cout << "Serial Execution Time:   " << fixed << setprecision(6) << serial_time << " seconds\n";
    cout << "Parallel Execution Time: " << fixed << setprecision(6) << parallel_time << " seconds\n";
    cout << "Speedup:                 " << fixed << setprecision(4) << speedup << "x\n";
    cout << "Efficiency:              " << fixed << setprecision(2) << efficiency << "%\n\n";
}

// Test scalability with different matrix sizes
void scalability_analysis(const vector<size_t>& sizes, int num_threads, int runs = 3) {
    cout << "SCALABILITY SUMMARY TABLE\n\n";
    cout << setw(8) << "Size" << setw(15) << "Serial(s)" << setw(15) << "Parallel(s)" << setw(12) << "Speedup" << setw(12) << "Efficiency\n";
    
    cout << string(70, '-') << "\n";

    for (size_t n : sizes) {
        vector<double> A = generate_spd(n);
        double serial_time = time_run(cholesky_serial, A, n, runs);

        omp_set_num_threads(num_threads);
        double parallel_time = time_run(cholesky_openmp, A, n, runs);
        double speedup = serial_time / parallel_time;
        double efficiency = speedup / num_threads * 100.0;

        cout << setw(8) << n
            << setw(15) << fixed << setprecision(6) << serial_time
            << setw(15) << fixed << setprecision(6) << parallel_time
            << setw(12) << fixed << setprecision(3) << speedup << "x"
            << setw(11) << fixed << setprecision(2) << efficiency << "%\n";
    }
    cout << "\n";
}

// Test scalability with different number of threads
// Matrix size is fixed, only thread count varies
void thread_scaling_analysis(size_t n, const vector<int>& threads_to_test, int runs = 3) {
    cout << "THREAD SCALING ANALYSIS\n\n";
    cout << "Matrix size: " << n << " x " << n << "\n";
    cout << "Maximum available threads: " << omp_get_max_threads() << "\n\n";
    
    vector<double> A = generate_spd(n);

    cout << "Measuring serial performance...\n";
    double serial_time = time_run(cholesky_serial, A, n, runs);
    cout << "Testing with ";
    for (size_t i = 0; i < threads_to_test.size(); i++) {
        cout << threads_to_test[i] << " thread(s)" << (i + 1 < threads_to_test.size() ? ", " : "...\n");
    }

    cout << "\n\n";
    cout << setw(8) << "Threads" << setw(15) << "Serial(s)" << setw(15) << "Parallel(s)" << setw(12) << "Speedup" << setw(12) << "Efficiency\n";
    cout << string(70, '-') << "\n";

    for (int num_threads : threads_to_test) {
        omp_set_num_threads(num_threads);
        double parallel_time = time_run(cholesky_openmp, A, n, runs);
        double speedup = serial_time / parallel_time;
        double efficiency = (speedup / num_threads) * 100.0;
        cout << setw(8) << num_threads
            << setw(15) << fixed << setprecision(6) << serial_time
            << setw(15) << fixed << setprecision(6) << parallel_time
            << setw(12) << fixed << setprecision(3) << speedup << "x"
            << setw(11) << fixed << setprecision(2) << efficiency << "%\n";
    }
    cout << "\n";
}

int main() {
    vector<size_t> default_sizes = { 100, 500, 1000, 2000 };
	vector<int> default_threads = { 1, 2, 4 }; // Default, if no input

    while (true) {
        cout << "+-------------------------------------------------------+\n";
        cout << "| Select an option:                                     |\n";
        cout << "| 1. Demonstrate on small 4x4 example                   |\n";
        cout << "| 2. Run single performance test (custom size)          |\n";
        cout << "| 3. Run scalability analysis (multiple sizes)          |\n";
        cout << "| 4. Run thread scaling analysis (multiple threads)     |\n";
        cout << "| 5. Run all analyses                                   |\n";
        cout << "| 0. Exit                                               |\n";
        cout << "+-------------------------------------------------------+\n";
        cout << "Enter choice: ";
        int choice;
        if (!(cin >> choice)) 
            break;

        if (choice == 0) 
            break;

        else if (choice == 1) {
            size_t n = 4;
            vector<double> A = generate_spd(n, 42);
            cout << "\nDEMONSTRATION: Small 4x4 System\n\n";
            print_matrix(A, n, "Original A");

            // Serial
            vector<double> Acopy = A;
            if (!cholesky_serial(Acopy, n)) { 
                cout << "Serial Cholesky failed.\n"; 
                continue; 
            }
            print_matrix(Acopy, n, "L (serial)");
            double res = relative_residual(A, Acopy, n);
            cout << "Residual norm (serial): " << scientific << setprecision(6) << res << "\n\n";
            
            // Parallel solve
            Acopy = A;
            omp_set_num_threads(4);
            if (!cholesky_openmp(Acopy, n)) { 
                cout << "Parallel Cholesky failed.\n"; 
                continue; 
            }
            print_matrix(Acopy, n, "L (parallel)");
            res = relative_residual(A, Acopy, n);
            cout << "Residual norm (parallel): " << scientific << setprecision(6) << res << "\n\n";
        }
        else if (choice == 2) {
            size_t n;
            int num_threads;

            cout << "Enter matrix size n (Exp - 512): ";
            cin >> n;
            cout << "Enter number of threads for parallel run: ";
            cin >> num_threads;

            single_performance_test(n, num_threads, 3);
        }
        else if (choice == 3) {
            int num_threads;
            cout << "Running scalability analysis for sizes:";

            for (size_t s : default_sizes) {
				cout << " " << s;
            }
            cout << "\nEnter number of threads for parallel runs (Exp - 4): ";
            cin >> num_threads;

            scalability_analysis(default_sizes, num_threads, 3);
        }
        else if (choice == 4) {
            size_t n;
            cout << "Enter matrix size for thread-scaling (Exp - 1000): ";
            cin >> n;
            cout << "Enter thread counts to test (space-separated, finish with 0):\n";
            vector<int> threads;
            while (true) {
                int x;
                cin >> x;

                if (x == 0) 
                    break;

                threads.push_back(x);
            }
            if (threads.empty()) 
                threads = default_threads;

            thread_scaling_analysis(n, threads, 3);
        }
        else if (choice == 5) {
            int num_threads;
            // Run everything with defaults
            cout << "Enter number of threads for parallel runs (Exp - 4): ";
            cin >> num_threads;

            // small 4x4 demo
            cout << "\n===== Small demo =====\n";
            {
                size_t n = 4;
                vector<double> A = generate_spd(n, 42);
                print_matrix(A, n, "Original A");
                vector<double> Acopy = A;
                cholesky_serial(Acopy, n);
                print_matrix(Acopy, n, "L (serial)");
                cout << "Residual (serial): " << relative_residual(A, Acopy, n) << "\n\n";
            }

            // single tests for default sizes
            for (size_t s : default_sizes) {
                single_performance_test(s, num_threads, 3);
            }

            // scalability summary
            scalability_analysis(default_sizes, num_threads, 3);

            // thread scaling for one size ( 1000)
			vector<int> threads_to_test = { 1, 2, 4, 8 };
            thread_scaling_analysis(1000, threads_to_test, 3);
        }
        else {
            cout << "Invalid choice\n";
        }
    }

    cout << "Exiting.\n";
    return 0;
}
1. Project Description
This program implements how Cholesky Decomposition solve symmetric positive definite (SPD) matrices.
This program include both a serial version and a parallel version using OpenMP and is menu-based run in terminal.
Allow users to:
- Measure execution time
- Compare serial and parallel performanc
- Check correctness by using small example
- Study speedup and efficiency

2. Prerequisites
To execute this program, you need A C++ compiler that supports OpenMP, in this right click your program file, go to properties --> Configuration Properties --> C/C++ --> Language --> Open MP Support --> Choose Yes(/openmp) and click Apply.
Also need an operating system like Windows, macOS, or Linux

3. Compile and Run the Program
Open terminal in the folder that contains the source file and run.
After run, a menu will appear with several options. 

4. Menu Options
- Demonstrate on small 4x4 example
- Run single performance test
- Run scalability analysis
- Run thread scaling analysis
- Run all analyses
- Exit
    Close the program.

  5. Set Number of threads (optional)
     The number of threads can be set in two methods:
     - Input the number when program ask
     - Using environment variable like 4

6. Output Information
   Display:
   1. Serial execution time
   2. Parallel execution time
   3. Speedup
   4. Efficiency
   5. Residual error
   Each test run multiple times and average time is reported

 7. Extra information
- All matrices generate automatically and guarenteed to be SPD
- For large matrix size, parallel performance is more obvious

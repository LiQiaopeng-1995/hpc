/**
 * 矩阵乘法优化学习程序 (C++17版本)
 * 
 * 本程序实现了多种矩阵乘法算法，用于学习和比较不同优化技术的性能。
 * 包含基础算法、循环重排、分块算法和转置优化等多种实现。
 * 支持从内存块直接构造矩阵，提高内存效率。
 * author: liqiaopeng
 */

#include <iostream>      // 输入输出流
#include <vector>        // 动态数组
#include <iomanip>       // 输出格式控制
#include <chrono>        // 高精度时间测量
#include <random>        // 随机数生成
#include <cstring>       // 字符串操作
#include <algorithm>     // 算法库
#include <functional>    // 函数对象
#include <fstream>       // 文件操作
#include <cmath>         // 数学函数

/**
 * CPU频率检测类
 * 用于自动获取当前CPU的运行频率，用于计算理论峰值性能
 */
class CPUFrequency {
private:
    double frequency_ghz;  // CPU频率，单位GHz

public:
    /**
     * 构造函数：自动检测CPU频率
     */
    CPUFrequency() {
        frequency_ghz = get_cpu_frequency();
    }

    /**
     * 获取CPU频率
     * @return CPU频率（GHz）
     */
    double get_frequency_ghz() const {
        return frequency_ghz;
    }

private:
    /**
     * 从系统文件读取CPU频率
     * @return CPU频率（GHz）
     */
    double get_cpu_frequency() {
        // 尝试从/proc/cpuinfo读取CPU频率（Linux系统）
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line;
        
        while (std::getline(cpuinfo, line)) {
            if (line.find("cpu MHz") != std::string::npos) {
                size_t pos = line.find(':');
                if (pos != std::string::npos) {
                    std::string freq_str = line.substr(pos + 1);
                    // 移除前导空格和制表符
                    freq_str.erase(0, freq_str.find_first_not_of(" \t"));
                    try {
                        double freq_mhz = std::stod(freq_str);
                        return freq_mhz / 1000.0; // 转换为GHz
                    } catch (...) {
                        break;
                    }
                }
            }
        }
        
        // 如果无法读取，使用默认值
        std::cout << "警告：无法读取CPU频率，使用默认值2.0 GHz" << std::endl;
        return 2.0;
    }
};

/**
 * 矩阵类
 */
/**
 * 矩阵类
 */
template<typename T>
class Matrix {
private:
    std::vector<T> data;
    size_t rows, cols;

public:
    Matrix(int r, int c) : rows(r), cols(c) {
        data.resize(rows * cols);
    }
    ~Matrix() = default;
    int getRows() const { return rows; }
    int getCols() const { return cols; }

    // 可写的operator()方法
    T& operator()(int i, int j) {
        return data[i * cols + j];
    }
    
    // 常量版本的operator()方法
    const T& operator()(int i, int j) const {
        return data[i * cols + j];
    }
    
    // 打印矩阵
    void print() const {
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(2) 
                          << data[i * cols + j] << " ";
            }
            std::cout << std::endl;
        }
    }
    
    /**
     * 用指定数据填充矩阵
     * @param values 要填充的数据
     */
    void fill(const std::vector<std::vector<T>>& values) {
        if (values.size() != rows || values[0].size() != cols) {
            throw std::invalid_argument("输入数据维度与矩阵不匹配");
        }
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                data[i * cols + j] = values[i][j];
            }
        }
    }

    void random_fill(T min_val = T(0), T max_val = T(1)) {
        std::random_device rd;  // 随机数种子
        std::mt19937 gen(rd()); // Mersenne Twister随机数生成器
        std::uniform_real_distribution<double> dis(static_cast<double>(min_val), static_cast<double>(max_val));
        
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                data[i * cols + j] = static_cast<T>(dis(gen));
            }
        }
    }

    /**
     * 检查两个矩阵是否相等
     * @param other 要比较的矩阵
     * @param tolerance 容差
     * @return 是否相等
     */
    bool equals(const Matrix& other, T tolerance = T(1e-10)) const {
        if (rows != other.rows || cols != other.cols) {
            return false;
        }
        
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                if (std::abs(data[i * cols + j] - other.data[i * other.cols + j]) > tolerance) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * 将矩阵所有元素清零
     */
    void zero() {
        std::fill(data.begin(), data.end(), T(0));
    }
};
/**
 * 性能评测类
 * 用于比较不同矩阵乘法算法的性能
 */
template<typename T>
class PerformanceBenchmark {
private:
    std::vector<std::string> method_names;  // 方法名称列表
    std::vector<std::function<Matrix<T>(const Matrix<T>&, const Matrix<T>&)>> methods;  // 方法函数列表
    CPUFrequency cpu_freq;  // CPU频率检测器

public:
    /**
     * 添加要评测的方法
     * @param name 方法名称
     * @param method 方法函数
     */
    void add_method(const std::string& name, 
                   std::function<Matrix<T>(const Matrix<T>&, const Matrix<T>&)> method) {
        method_names.push_back(name);
        methods.push_back(method);
    }

    /**
     * 执行性能评测
     * @param A 第一个矩阵
     * @param B 第二个矩阵
     * @param iterations 迭代次数
     */
    void benchmark(const Matrix<T>& A, const Matrix<T>& B, int iterations = 5) {
        // 打印评测信息
        std::cout << "\n=== 性能评测 (矩阵大小: " << A.getRows() << "x" << A.getCols() 
                  << " * " << B.getRows() << "x" << B.getCols() << ") ===" << std::endl;
        std::cout << "CPU频率: " << std::fixed << std::setprecision(2) 
                  << cpu_freq.get_frequency_ghz() << " GHz" << std::endl;
        std::cout << "迭代次数: " << iterations << std::endl;
        std::cout << std::string(100, '-') << std::endl;
        
        // 打印表头
        std::cout << std::setw(20) << "方法" << std::setw(15) << "时间(ms)" 
                  << std::setw(15) << "平均(ms)" << std::setw(15) << "GFLOPS" 
                  << std::setw(15) << "理论峰值%" << std::endl;
        std::cout << std::string(100, '-') << std::endl;

        // 计算理论FLOPS (每个矩阵元素需要2个浮点运算：乘法和加法)
        double flops = 2.0 * A.getRows() * A.getCols() * B.getCols();
        double theoretical_peak_gflops = cpu_freq.get_frequency_ghz() * 4.0; // 假设4个浮点运算单元
        
        // 对每个方法进行评测
        for (size_t i = 0; i < methods.size(); i++) {
            std::vector<double> times;
            
            // 预热：避免首次运行的冷启动开销
            methods[i](A, B);
            
            // 多次测试取平均值
            for (int iter = 0; iter < iterations; iter++) {
                auto start = std::chrono::high_resolution_clock::now();
                Matrix result = methods[i](A, B);
                auto end = std::chrono::high_resolution_clock::now();
                
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                times.push_back(duration.count() / 1000.0); // 转换为毫秒
            }
            
            // 计算统计信息
            double avg_time = 0.0;
            for (double t : times) avg_time += t;
            avg_time /= times.size();
            
            // 计算GFLOPS：时间单位是毫秒，需要转换为秒
            double gflops = (flops / (avg_time * 1e-3)) / 1e9; // 转换为GFLOPS
            double peak_percentage = (gflops / theoretical_peak_gflops) * 100.0;
            
            // 打印结果
            std::cout << std::setw(20) << method_names[i] 
                      << std::setw(15) << std::fixed << std::setprecision(2) << times[0]
                      << std::setw(15) << std::fixed << std::setprecision(2) << avg_time
                      << std::setw(15) << std::fixed << std::setprecision(2) << gflops
                      << std::setw(15) << std::fixed << std::setprecision(1) << peak_percentage << "%" << std::endl;
        }
        
        // 打印总结信息
        std::cout << std::string(100, '-') << std::endl;
        std::cout << "理论峰值: " << std::fixed << std::setprecision(2) 
                  << theoretical_peak_gflops << " GFLOPS" << std::endl;
        std::cout << "总浮点运算数: " << std::scientific << std::setprecision(2) << flops << std::endl;
    }
};

/**
 * 基础矩阵乘法（三重循环，i-j-k顺序）
 * 
 * 这是最经典的矩阵乘法实现，但内存访问模式不够优化。
 * 时间复杂度：O(n³)
 * 空间复杂度：O(n²)
 * 
 * @param other 要相乘的矩阵
 * @return 乘法结果矩阵
 */
template<typename T>
Matrix<T> simple_gemm(const Matrix<T>& A, const Matrix<T>& B)  {
    // 检查矩阵维度是否匹配
    if (A.getCols() != B.getRows()) {
        throw std::invalid_argument("矩阵维度不匹配，无法相乘");
    }
    int A_rows = A.getRows();
    int A_cols = A.getCols();
    int B_cols = B.getCols();
    Matrix<T> result(A_rows, B_cols);
    
    // 经典的三重循环实现
    // 外层循环：遍历结果矩阵的每个元素
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            // 内层循环：计算C[i][j] = Σ(A[i][k] * B[k][j])
            for (int k = 0; k < A.getCols(); k++) {
                result(i, j) += A(i, k) * B(k, j);
            }
        }
    }
    
    return result;
}

/**
 * 优化1：循环重排（i-k-j顺序）
 * 
 * 通过改变循环顺序来提高缓存局部性。
 * 优势：内层循环访问连续内存位置，减少缓存未命中。
 * 
 * @param other 要相乘的矩阵
 * @return 乘法结果矩阵
 */
template<typename T>
Matrix<T> multiply_optimized1(const Matrix<T>& A, const Matrix<T>& B) {
    if (A.getCols() != B.getRows()) {
        throw std::invalid_argument("矩阵维度不匹配，无法相乘");
    }

    Matrix<T> result(A.getRows(), B.getCols());
    
    // i-k-j顺序：更好的缓存局部性
    for (int i = 0; i < A.getRows(); i++) {
        for (int k = 0; k < A.getCols(); k++) {
            for (int j = 0; j < B.getCols(); j++) {
                result(i, j) += A(i, k) * B(k, j);
            }
        }
    }
    
    return result;
}

/**
 * 优化2：分块矩阵乘法
 * 
 * 将大矩阵分解为小块进行计算，充分利用CPU缓存层次结构。
 * 每个块的大小应该能够完全装入CPU的L1缓存。
 * 
 * @param other 要相乘的矩阵
 * @param block_size 分块大小（默认32）
 * @return 乘法结果矩阵
 */
template<typename T>
Matrix<T> multiply_blocked(const Matrix<T>& A, const Matrix<T>& B, int block_size = 32){
    if (A.getCols() != B.getRows()) {
        throw std::invalid_argument("矩阵维度不匹配，无法相乘");
    }

    Matrix<T> result(A.getRows(), B.getCols());
    
    // 外层循环：遍历所有块
    for (int i = 0; i < A.getRows(); i += block_size) {
        for (int j = 0; j < B.getCols(); j += block_size) {
            for (int k = 0; k < A.getCols(); k += block_size) {
                // 计算当前块的边界
                int i_end = std::min(i + block_size, A.getRows());
                int j_end = std::min(j + block_size, B.getCols());
                int k_end = std::min(k + block_size, A.getCols());
                
                // 处理当前块内的所有元素
                for (int ii = i; ii < i_end; ii++) {
                    for (int kk = k; kk < k_end; kk++) {
                        for (int jj = j; jj < j_end; jj++) {
                            result(ii, jj) += A(ii, kk) * B(kk, jj);
                        }
                    }
                }
            }
        }
    }
    
    return result;
}

/**
 * 优化3：转置优化
 * 
 * 通过转置第二个矩阵来提高内存访问的连续性。
 * 转置后，内存访问模式更加友好，但需要额外的转置开销。
 * 
 * @param A 第一个矩阵
 * @param B 第二个矩阵
 * @return 乘法结果矩阵
 */
template<typename T>
Matrix<T> multiply_transpose(const Matrix<T>& A, const Matrix<T>& B) {
    if (A.getCols() != B.getRows()) {
        throw std::invalid_argument("矩阵维度不匹配，无法相乘");
    }

    int A_rows = A.getRows();
    int A_cols = A.getCols();
    int B_rows = B.getRows();
    int B_cols = B.getCols();

    // 转置第二个矩阵以提高缓存命中率
    Matrix<T> B_transpose(B_cols, B_rows);
    for (int i = 0; i < B_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            B_transpose(j, i) = B(i, j);
        }
    }

    Matrix<T> result(A_rows, B_cols);
    
    // 使用转置后的矩阵进行乘法
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            for (int k = 0; k < A_cols; k++) {
                result(i, j) += A(i, k) * B_transpose(j, k);
            }
        }
    }
    
    return result;
}

/**
 * 主函数：演示矩阵乘法的各种优化算法
 */
int main() {
    try {
        std::cout << "=== 矩阵乘法优化学习程序 (C++17版本) ===" << std::endl;
        std::cout << "本程序演示了多种矩阵乘法优化技术及其性能对比" << std::endl;
        
        // 创建性能评测器
        PerformanceBenchmark<double> benchmark;
        
        // 添加不同的矩阵乘法方法
        benchmark.add_method("基础算法", [](const Matrix<double>& A, const Matrix<double>& B) { return simple_gemm(A, B); });
        benchmark.add_method("循环重排", [](const Matrix<double>& A, const Matrix<double>& B) { return multiply_optimized1(A, B); });
        benchmark.add_method("分块算法", [](const Matrix<double>& A, const Matrix<double>& B) { return multiply_blocked(A, B, 32); });
        benchmark.add_method("转置优化", [](const Matrix<double>& A, const Matrix<double>& B) { return multiply_transpose(A, B); });

        // 测试不同大小的矩阵
        std::vector<int> sizes = {64, 128, 256, 512};
        
        std::cout << "\n开始性能评测..." << std::endl;
        for (int size : sizes) {
            std::cout << "\n正在测试 " << size << "x" << size << " 矩阵..." << std::endl;
            
            // 创建测试矩阵
            Matrix<double> A(size, size);
            Matrix<double> B(size, size);
            
            // 随机填充矩阵
            A.random_fill();
            B.random_fill();
            
            // 运行性能评测
            benchmark.benchmark(A, B, 5);
            
            // 验证结果正确性
            Matrix<double> result1 = simple_gemm(A, B);
            Matrix<double> result2 = multiply_optimized1(A, B);
            Matrix<double> result3 = multiply_blocked(A, B, 32);
            Matrix<double> result4 = multiply_transpose(A, B);
            
            if (!result1.equals(result2) || !result1.equals(result3) || !result1.equals(result4)) {
                std::cout << "警告：优化算法结果与基础算法不一致！" << std::endl;
            } else {
                std::cout << "✓ 所有算法结果一致" << std::endl;
            }
        }

        std::cout << "\n=== 程序执行完成 ===" << std::endl;
        std::cout << "感谢使用矩阵乘法优化学习程序！" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

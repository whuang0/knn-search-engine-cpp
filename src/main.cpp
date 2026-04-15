#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <future>
#include <sys/time.h>
#include <sys/resource.h>
#include <sched.h>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <fcntl.h>
#include <sys/stat.h>
#include <cstring>
#include <iomanip>
#include "kdtree.hpp"
#include "file_io.hpp"
#include "point.hpp"

void printUsage(const char* program) {
    std::cerr << "Usage: " << program << " n_cores training_file [query_file result_file | pipe_name]\n"
              << "  n_cores: Number of threads to use\n"
              << "  training_file: Binary file containing training data\n"
              << "  query_file: Binary file containing query points (optional)\n"
              << "  result_file: Output file for k-NN results (optional)\n"
              << "  pipe_name: Named pipe for single queries (optional)\n";
}

// Function to process a single query from the pipe
void processPipeQuery(const KDTree& tree, const std::string& pipe_name) {
    std::string in_pipe = pipe_name + "_in";
    std::string out_pipe = pipe_name + "_out";
    
    // Create named pipes if they don't exist
    if (mkfifo(in_pipe.c_str(), 0666) == -1 && errno != EEXIST) {
        std::cerr << "Error creating input pipe: " << strerror(errno) << std::endl;
        return;
    }
    if (mkfifo(out_pipe.c_str(), 0666) == -1 && errno != EEXIST) {
        std::cerr << "Error creating output pipe: " << strerror(errno) << std::endl;
        return;
    }
    
    std::cout << "Waiting for queries on pipe: " << pipe_name << std::endl;
    
    while (true) {
        // Open input pipe for reading FIRST
        int in_fd = open(in_pipe.c_str(), O_RDONLY | O_NONBLOCK);
        if (in_fd == -1) {
            if (errno == EAGAIN) {
                // No writer yet, wait a bit and try again
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }
            std::cerr << "Error opening input pipe: " << strerror(errno) << std::endl;
            return;
        }
        
        // Successfully opened input pipe, now set it back to blocking mode
        int flags = fcntl(in_fd, F_GETFL);
        fcntl(in_fd, F_SETFL, flags & ~O_NONBLOCK);
        
        // Read query dimensions
        uint64_t n_dimensions;
        ssize_t bytes_read = read(in_fd, &n_dimensions, sizeof(n_dimensions));
        if (bytes_read != sizeof(n_dimensions)) {
            if (bytes_read == 0) {
                // End of pipe, client disconnected
                close(in_fd);
                continue;
            }
            std::cerr << "Error reading query dimensions" << std::endl;
            close(in_fd);
            return;
        }
        
        // Read query point
        std::vector<float> coordinates(n_dimensions);
        if (read(in_fd, coordinates.data(), n_dimensions * sizeof(float)) != (ssize_t)(n_dimensions * sizeof(float))) {
            std::cerr << "Error reading query coordinates" << std::endl;
            close(in_fd);
            return;
        }
        close(in_fd);
        
        Point query(coordinates);
        
        // Find k nearest neighbors
        auto start = std::chrono::high_resolution_clock::now();
        
        // Read k from query or use default value of 3
        uint64_t n_neighbors = 3;  // Default to 3 neighbors
        std::vector<Point> neighbors = tree.findKNearest(query, n_neighbors);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Open output pipe for writing
        int out_fd = open(out_pipe.c_str(), O_WRONLY);
        if (out_fd == -1) {
            std::cerr << "Error opening output pipe: " << strerror(errno) << std::endl;
            return;
        }
        
        // Write results
        uint64_t n_neighbors_out = neighbors.size();
        if (write(out_fd, &n_neighbors_out, sizeof(n_neighbors_out)) != sizeof(n_neighbors_out)) {
            std::cerr << "Error writing number of neighbors" << std::endl;
            close(out_fd);
            return;
        }
        
        for (const auto& neighbor : neighbors) {
            if (write(out_fd, neighbor.coordinates.data(), n_dimensions * sizeof(float)) != (ssize_t)(n_dimensions * sizeof(float))) {
                std::cerr << "Error writing neighbor coordinates" << std::endl;
                close(out_fd);
                return;
            }
        }
        
        uint64_t duration_count = duration.count();
        if (write(out_fd, &duration_count, sizeof(duration_count)) != sizeof(duration_count)) {
            std::cerr << "Error writing processing time" << std::endl;
            close(out_fd);
            return;
        }
        
        close(out_fd);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4 && argc != 5) {
        printUsage(argv[0]);
        return 1;
    }
    
    try {
        // Parse command line arguments
        int n_cores = std::stoi(argv[1]);
        std::string training_file = argv[2];
        std::string query_file = argv[3];
        std::string result_file = (argc == 5) ? argv[4] : "";
        
        // Check if we're using pipe mode
        bool pipe_mode = result_file.empty();
        std::string pipe_name = pipe_mode ? query_file : "";
        
        // Verify core count
        cpu_set_t cpu_set;
        CPU_ZERO(&cpu_set);
        if (sched_getaffinity(0, sizeof(cpu_set), &cpu_set) == 0) {
            int available_cores = CPU_COUNT(&cpu_set);
            if (n_cores > available_cores) {
                std::cerr << "Warning: Requested " << n_cores << " cores but only "
                         << available_cores << " are available" << std::endl;
                n_cores = available_cores;
            }
        }
        
        // Start total timing
        auto total_start = std::chrono::high_resolution_clock::now();
        
        // Read training data
        auto read_start = std::chrono::high_resolution_clock::now();
        FileIO training_io;
        if (!training_io.openInputFile(training_file)) {
            std::cerr << "Error: Failed to open training file" << std::endl;
            return 1;
        }
        std::vector<Point> training_points;
        if (!training_io.readPoints(training_points)) {
            std::cerr << "Error: Failed to read training points" << std::endl;
            return 1;
        }
        auto read_end = std::chrono::high_resolution_clock::now();
        auto read_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            read_end - read_start).count();
        std::cout << "Read " << training_points.size() << " training points in "
                  << read_time << "ms" << std::endl;
        
        // Build k-d tree
        auto build_start = std::chrono::high_resolution_clock::now();
        KDTree tree(training_points, n_cores);
        auto build_end = std::chrono::high_resolution_clock::now();
        auto build_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            build_end - build_start).count();
        std::cout << "Built k-d tree in " << build_time << "ms" << std::endl;
        
        if (pipe_mode) {
            // Process queries from named pipe
            std::cout << "Waiting for queries on pipe: " << pipe_name << std::endl;
            processPipeQuery(tree, pipe_name);
        } else {
            // Process queries from file
            auto query_start = std::chrono::high_resolution_clock::now();
            
            // Read query file and get n_neighbors
            uint64_t query_file_id;
            uint64_t n_neighbors;
            std::vector<Point> query_points = FileIO::readQueryFile(query_file, query_file_id, n_neighbors);
            std::cout << "Number of neighbors to find: " << n_neighbors << std::endl;
            
            if (query_points.empty()) {
                std::cerr << "Error: Failed to read query points" << std::endl;
                return 1;
            }
            
            std::cout << "Query points dimensions: " << query_points[0].coordinates.size() << std::endl;
            std::cout << "Training points dimensions: " << training_points[0].coordinates.size() << std::endl;
            
            if (query_points[0].coordinates.size() != training_points[0].coordinates.size()) {
                std::cerr << "Error: Query and training points have different dimensions" << std::endl;
                return 1;
            }
            
            auto query_end = std::chrono::high_resolution_clock::now();
            auto query_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                query_end - query_start).count();
            std::cout << "Read " << query_points.size() << " query points in "
                      << query_time << "ms" << std::endl;
            
            // Process queries in parallel
            auto search_start = std::chrono::high_resolution_clock::now();
            std::vector<std::vector<Point>> results(query_points.size());
            std::atomic<size_t> processed_queries(0);
            std::mutex cout_mutex;
            
            // Use larger chunks for better performance
            size_t chunk_size = std::max(size_t(1), query_points.size() / n_cores);
            size_t total_chunks = (query_points.size() + chunk_size - 1) / chunk_size;
            
            std::cout << "Processing " << total_chunks << " chunks of size " << chunk_size << std::endl;
            
            try {
                // Process chunks in parallel with a fixed thread pool
                std::vector<std::thread> threads;
                threads.reserve(n_cores);
                
                for (size_t chunk = 0; chunk < total_chunks; ++chunk) {
                    size_t chunk_start = chunk * chunk_size;
                    size_t chunk_end = std::min(chunk_start + chunk_size, query_points.size());
                    
                    // Wait for a thread to finish if we have too many running
                    while (threads.size() >= static_cast<size_t>(n_cores)) {
                        for (auto it = threads.begin(); it != threads.end();) {
                            if (it->joinable()) {
                                it->join();
                                it = threads.erase(it);
                            } else {
                                ++it;
                            }
                        }
                    }
                    
                    threads.emplace_back([&tree, &query_points, &results, chunk_start, chunk_end, 
                                        &processed_queries, &cout_mutex, n_neighbors]() {
                        try {
                            for (size_t idx = chunk_start; idx < chunk_end; ++idx) {
                                results[idx] = tree.findKNearest(query_points[idx], n_neighbors);  // Use n_neighbors from query file
                                
                                // Sort results by distance from query point
                                std::sort(results[idx].begin(), results[idx].end(),
                                    [&query_points, idx](const Point& a, const Point& b) {
                                        float dist_a = 0, dist_b = 0;
                                        for (size_t j = 0; j < a.coordinates.size(); ++j) {
                                            float diff_a = a.coordinates[j] - query_points[idx].coordinates[j];
                                            float diff_b = b.coordinates[j] - query_points[idx].coordinates[j];
                                            dist_a += diff_a * diff_a;
                                            dist_b += diff_b * diff_b;
                                        }
                                        return dist_a < dist_b;
                                    });
                                
                                size_t processed = ++processed_queries;
                                if (processed % 1000000 == 0) {  // Print progress every 1M queries
                                    std::lock_guard<std::mutex> lock(cout_mutex);
                                    std::cout << "Processed " << processed << " queries..." << std::endl;
                                }
                            }
                        } catch (const std::exception& e) {
                            std::lock_guard<std::mutex> lock(cout_mutex);
                            std::cerr << "Error in query processing: " << e.what() << std::endl;
                        }
                    });
                }
                
                // Wait for remaining threads to complete
                for (auto& thread : threads) {
                    if (thread.joinable()) {
                        thread.join();
                    }
                }
            } catch (const std::exception& e) {
                std::cerr << "Error during query processing: " << e.what() << std::endl;
                return 1;
            }
            
            auto search_end = std::chrono::high_resolution_clock::now();
            auto search_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                search_end - search_start).count();
            std::cout << "Found k-NN for all queries in " << search_time << "ms" << std::endl;
            
            // Write results
            auto write_start = std::chrono::high_resolution_clock::now();
            
            // Get file IDs from training and query files before closing them
            uint64_t training_file_id = training_io.getTrainingFileId();
            
            // Close input files before writing results
            training_io.closeInputFile();
            
            // Write results with proper file IDs
            FileIO::writeResultFile(result_file, training_file_id, query_file_id, results);
            
            auto write_end = std::chrono::high_resolution_clock::now();
            auto write_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                write_end - write_start).count();
            std::cout << "Wrote results in " << write_time << "ms" << std::endl;
        }
        
        // Print total time and resource usage
        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            total_end - total_start).count();
        
        struct rusage usage;
        if (getrusage(RUSAGE_SELF, &usage) == 0) {
            std::cout << "\nResource Usage:\n"
                      << "User CPU time: " << usage.ru_utime.tv_sec << "." 
                      << std::setfill('0') << std::setw(6) << usage.ru_utime.tv_usec << "s\n"
                      << "System CPU time: " << usage.ru_stime.tv_sec << "."
                      << std::setfill('0') << std::setw(6) << usage.ru_stime.tv_usec << "s\n"
                      << "Maximum resident set size: " << usage.ru_maxrss << "KB\n"
                      << "Page faults: " << usage.ru_majflt << " major, " 
                      << usage.ru_minflt << " minor\n"
                      << "Total time: " << total_time << "ms" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 
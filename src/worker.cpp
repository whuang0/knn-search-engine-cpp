#include <iostream>
#include <string>
#include <unordered_map>
#include <unistd.h>
#include "kdtree.hpp"
#include "file_io.hpp"
#include "named_pipe.hpp"

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " training_file input_pipe output_pipe\n";
        return 1;
    }

    try {
        std::string training_file = argv[1];
        std::string input_pipe    = argv[2];
        std::string output_pipe   = argv[3];

        // Load training data and build k-d tree
        uint64_t file_id;
        std::vector<Point> training_points = FileIO::readTrainingFile(training_file, file_id);
        KDTree tree(training_points);

        // Open named pipes
        int input_fd  = NamedPipe::openReadPipe(input_pipe);
        int output_fd = NamedPipe::openWritePipe(output_pipe);

        // Cache results by query ID to avoid redundant searches
        std::unordered_map<int, std::vector<Point>> result_cache;

        while (true) {
            try {
                if (!NamedPipe::waitForData(input_fd))
                    continue;

                // Read batch header: size and k
                size_t batch_size;
                if (read(input_fd, &batch_size, sizeof(size_t)) == 0)
                    break; // pipe closed, exit cleanly

                size_t k;
                if (read(input_fd, &k, sizeof(size_t)) == -1)
                    throw std::runtime_error("Failed to read k");

                size_t n_dims;
                if (read(input_fd, &n_dims, sizeof(size_t)) == -1)
                    throw std::runtime_error("Failed to read dimensions");

                std::vector<Point> queries = NamedPipe::readBatch(input_fd, n_dims, batch_size);

                std::vector<std::vector<Point>> results;
                results.reserve(queries.size());

                for (const auto& query : queries) {
                    auto it = result_cache.find(query.id);
                    if (it != result_cache.end()) {
                        results.push_back(it->second);
                        continue;
                    }

                    std::vector<Point> result = tree.findKNearest(query, k);
                    result_cache[query.id] = result;
                    results.push_back(std::move(result));
                }

                NamedPipe::writeBatchResults(output_fd, results);

            } catch (const std::exception& e) {
                std::cerr << "Error processing batch: " << e.what() << std::endl;
                break;
            }
        }

        NamedPipe::closePipe(input_fd);
        NamedPipe::closePipe(output_fd);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
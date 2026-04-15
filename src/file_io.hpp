#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include "point.hpp"
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdint>

struct FileHeader {
    char file_type[8];  // "TRAINING" or "QUERY"
    uint64_t file_id;
    uint64_t n_points;
    uint64_t n_dimensions;
};

struct QueryHeader {
    char file_type[8];
    uint64_t file_id;
    uint64_t n_queries;
    uint64_t n_dimensions;
    uint64_t n_neighbors;
};

struct ResultHeader {
    char file_type[8];
    uint64_t training_file_id;
    uint64_t query_file_id;
    uint64_t result_file_id;
    uint64_t n_queries;
    uint64_t n_dimensions;
    uint64_t n_neighbors;
};

class FileIO {
public:
    FileIO() : input_fd(-1), output_fd(-1), input_mmap(nullptr), output_mmap(nullptr),
               input_size(0), output_size(0) {}
    ~FileIO();

    bool openInputFile(const std::string& filename);
    bool openOutputFile(const std::string& filename, bool append = false);
    bool readPoints(std::vector<Point>& points);
    void closeInputFile();
    void closeOutputFile();

    static std::vector<Point> readTrainingFile(const std::string& filename, uint64_t& file_id);
    static std::vector<Point> readQueryFile(const std::string& filename, uint64_t& file_id, uint64_t& n_neighbors);
    static void writeResultFile(const std::string& filename,
                                uint64_t training_file_id,
                                uint64_t query_file_id,
                                const std::vector<std::vector<Point>>& results);

    uint64_t getTrainingFileId() const;
    uint64_t getQueryFileId() const;

private:
    void cleanup();
    void validateHeader(const FileHeader& header);
    static uint64_t generateFileId();

    int input_fd;
    int output_fd;
    void* input_mmap;
    void* output_mmap;
    size_t input_size;
    size_t output_size;
};
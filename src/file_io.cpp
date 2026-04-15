#include "file_io.hpp"
#include <string.h>
#include <stdexcept>
#include <algorithm>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <thread>

FileIO::~FileIO() {
    cleanup();
}

void FileIO::cleanup() {
    if (input_mmap) {
        munmap(input_mmap, input_size);
        input_mmap = nullptr;
    }
    if (output_mmap) {
        munmap(output_mmap, output_size);
        output_mmap = nullptr;
    }
    if (input_fd != -1) {
        close(input_fd);
        input_fd = -1;
    }
    if (output_fd != -1) {
        close(output_fd);
        output_fd = -1;
    }
}

bool FileIO::openInputFile(const std::string& filename) {
    cleanup();

    input_fd = open(filename.c_str(), O_RDONLY);
    if (input_fd == -1) return false;

    struct stat st;
    if (fstat(input_fd, &st) == -1) {
        close(input_fd);
        input_fd = -1;
        return false;
    }

    input_size = st.st_size;
    input_mmap = mmap(nullptr, input_size, PROT_READ, MAP_PRIVATE, input_fd, 0);

    if (input_mmap == MAP_FAILED) {
        close(input_fd);
        input_fd = -1;
        return false;
    }

    return true;
}

bool FileIO::openOutputFile(const std::string& filename, bool append) {
    cleanup();

    int flags = O_RDWR | O_CREAT;
    if (!append) flags |= O_TRUNC;

    output_fd = open(filename.c_str(), flags, 0644);
    if (output_fd == -1) return false;

    struct stat st;
    if (fstat(output_fd, &st) == -1) {
        close(output_fd);
        output_fd = -1;
        return false;
    }

    output_size = st.st_size;
    if (output_size == 0) output_size = 4096;

    if (ftruncate(output_fd, output_size) == -1) {
        close(output_fd);
        output_fd = -1;
        return false;
    }

    output_mmap = mmap(nullptr, output_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                       output_fd, 0);

    if (output_mmap == MAP_FAILED) {
        close(output_fd);
        output_fd = -1;
        return false;
    }

    return true;
}

void FileIO::validateHeader(const FileHeader& header) {
    if (strncmp(header.file_type, "TRAINING", 8) != 0 &&
        strncmp(header.file_type, "QUERY", 8) != 0) {
        throw std::runtime_error("Invalid file type");
    }
}

bool FileIO::readPoints(std::vector<Point>& points) {
    if (!input_mmap) return false;

    const char* data = static_cast<const char*>(input_mmap);
    const FileHeader* header = reinterpret_cast<const FileHeader*>(data);
    validateHeader(*header);

    data += sizeof(FileHeader);
    uint64_t n_points = header->n_points;
    uint64_t n_dimensions = header->n_dimensions;

    points.resize(n_points);
    for (uint64_t i = 0; i < n_points; ++i) {
        points[i].coordinates.resize(n_dimensions);
        memcpy(points[i].coordinates.data(), data, n_dimensions * sizeof(float));
        data += n_dimensions * sizeof(float);
    }

    return true;
}

void FileIO::closeInputFile() {
    if (input_mmap) {
        munmap(input_mmap, input_size);
        input_mmap = nullptr;
    }
    if (input_fd != -1) {
        close(input_fd);
        input_fd = -1;
    }
}

void FileIO::closeOutputFile() {
    if (output_mmap) {
        munmap(output_mmap, output_size);
        output_mmap = nullptr;
    }
    if (output_fd != -1) {
        close(output_fd);
        output_fd = -1;
    }
}

std::vector<Point> FileIO::readTrainingFile(const std::string& filename, uint64_t& file_id) {
    FileIO io;
    if (!io.openInputFile(filename))
        throw std::runtime_error("Failed to open training file");

    std::vector<Point> points;
    if (!io.readPoints(points))
        throw std::runtime_error("Failed to read training points");

    const FileHeader* header = reinterpret_cast<const FileHeader*>(io.input_mmap);
    file_id = header->file_id;

    return points;
}

std::vector<Point> FileIO::readQueryFile(const std::string& filename, uint64_t& file_id, uint64_t& n_neighbors) {
    FileIO io;
    if (!io.openInputFile(filename))
        throw std::runtime_error("Failed to open query file");

    const char* data = static_cast<const char*>(io.input_mmap);
    const QueryHeader* header = reinterpret_cast<const QueryHeader*>(data);

    if (strncmp(header->file_type, "QUERY", 8) != 0)
        throw std::runtime_error("Invalid file type: expected QUERY");

    file_id = header->file_id;
    n_neighbors = header->n_neighbors;

    data += sizeof(QueryHeader);
    uint64_t n_queries = header->n_queries;
    uint64_t n_dimensions = header->n_dimensions;

    std::vector<Point> points(n_queries);
    for (uint64_t i = 0; i < n_queries; ++i) {
        points[i].coordinates.resize(n_dimensions);
        memcpy(points[i].coordinates.data(), data, n_dimensions * sizeof(float));
        data += n_dimensions * sizeof(float);
    }

    return points;
}

void FileIO::writeResultFile(const std::string& filename,
                             uint64_t training_file_id,
                             uint64_t query_file_id,
                             const std::vector<std::vector<Point>>& results) {
    FileIO io;
    if (!io.openOutputFile(filename))
        throw std::runtime_error("Failed to open result file");

    size_t header_size = sizeof(ResultHeader);
    size_t total_points = results.size() * results[0].size();
    size_t n_dims = results[0][0].coordinates.size();
    size_t points_size = total_points * n_dims * sizeof(float);
    size_t total_size = header_size + points_size;

    if (total_size > io.output_size) {
        if (munmap(io.output_mmap, io.output_size) == -1)
            throw std::runtime_error("Failed to unmap output file");
        if (ftruncate(io.output_fd, total_size) == -1)
            throw std::runtime_error("Failed to resize output file");
        io.output_size = total_size;
        io.output_mmap = mmap(nullptr, total_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                              io.output_fd, 0);
        if (io.output_mmap == MAP_FAILED)
            throw std::runtime_error("Failed to remap output file");
    }

    char* data = static_cast<char*>(io.output_mmap);

    ResultHeader* header = reinterpret_cast<ResultHeader*>(data);
    strncpy(header->file_type, "RESULT\0\0", 8);
    header->training_file_id = training_file_id;
    header->query_file_id = query_file_id;
    header->result_file_id = generateFileId();
    header->n_queries = results.size();
    header->n_dimensions = n_dims;
    header->n_neighbors = results[0].size();

    data += sizeof(ResultHeader);
    size_t chunk_size = 1000000;
    size_t total_chunks = (total_points + chunk_size - 1) / chunk_size;

    std::vector<std::thread> threads;
    threads.reserve(4);

    for (size_t chunk = 0; chunk < total_chunks; ++chunk) {
        size_t chunk_start = chunk * chunk_size;
        size_t chunk_end = std::min(chunk_start + chunk_size, total_points);

        while (threads.size() >= 4) {
            for (auto it = threads.begin(); it != threads.end();) {
                if (it->joinable()) {
                    it->join();
                    it = threads.erase(it);
                } else {
                    ++it;
                }
            }
        }

        threads.emplace_back([&results, data, chunk_start, chunk_end, n_dims]() {
            char* chunk_data = data + chunk_start * n_dims * sizeof(float);
            size_t query_idx = chunk_start / results[0].size();
            size_t point_idx = chunk_start % results[0].size();

            for (size_t idx = chunk_start; idx < chunk_end; ++idx) {
                memcpy(chunk_data, results[query_idx][point_idx].coordinates.data(),
                       n_dims * sizeof(float));
                chunk_data += n_dims * sizeof(float);
                if (++point_idx >= results[0].size()) {
                    point_idx = 0;
                    query_idx++;
                }
            }
        });
    }

    for (auto& thread : threads) {
        if (thread.joinable()) thread.join();
    }

    if (msync(io.output_mmap, total_size, MS_SYNC) == -1)
        throw std::runtime_error("Failed to sync output file");
}

uint64_t FileIO::generateFileId() {
    uint64_t id;
    int fd = open("/dev/urandom", O_RDONLY);
    (void)read(fd, &id, sizeof(id));
    close(fd);
    return id;
}

uint64_t FileIO::getTrainingFileId() const {
    if (!input_mmap)
        throw std::runtime_error("No input file mapped");
    const FileHeader* header = reinterpret_cast<const FileHeader*>(input_mmap);
    return header->file_id;
}

uint64_t FileIO::getQueryFileId() const {
    if (!input_mmap)
        throw std::runtime_error("No input file mapped");
    const QueryHeader* header = reinterpret_cast<const QueryHeader*>(input_mmap);
    return header->file_id;
}
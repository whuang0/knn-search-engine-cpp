#pragma once
#include <string>
#include <vector>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <poll.h>
#include "point.hpp"

// Utility class for binary point serialization over POSIX named pipes.
// All operations are static — no instance state needed.
class NamedPipe {
public:
    static constexpr size_t BUFFER_SIZE = 4096;
    static constexpr size_t BATCH_SIZE  = 32;

    static void createPipe(const std::string& pipe_name);
    static void removePipe(const std::string& pipe_name);

    static int  openReadPipe(const std::string& pipe_name);
    static int  openWritePipe(const std::string& pipe_name);
    static void closePipe(int fd);

    // Returns true if data is available within timeout_ms milliseconds
    static bool waitForData(int fd, int timeout_ms = 100);

    static void writeBatch(int fd, const std::vector<Point>& points);
    static std::vector<Point> readBatch(int fd, size_t n_dimensions, size_t batch_size);

    static void writeBatchResults(int fd, const std::vector<std::vector<Point>>& results);
    static std::vector<std::vector<Point>> readBatchResults(int fd);

private:
    static void  writePointData(int fd, const Point& point);
    static Point readPointData(int fd, size_t n_dimensions);
};
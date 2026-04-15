#include "named_pipe.hpp"
#include <stdexcept>
#include <cstring>

void NamedPipe::createPipe(const std::string& pipe_name) {
    if (mkfifo(pipe_name.c_str(), 0666) == -1) {
        if (errno != EEXIST) {  // Ignore error if pipe already exists
            throw std::runtime_error("Failed to create named pipe: " + pipe_name);
        }
    }
}

void NamedPipe::removePipe(const std::string& pipe_name) {
    unlink(pipe_name.c_str());
}

int NamedPipe::openReadPipe(const std::string& pipe_name) {
    int fd = open(pipe_name.c_str(), O_RDONLY | O_NONBLOCK);
    if (fd == -1) {
        throw std::runtime_error("Failed to open pipe for reading: " + pipe_name);
    }
    return fd;
}

int NamedPipe::openWritePipe(const std::string& pipe_name) {
    int fd = open(pipe_name.c_str(), O_WRONLY | O_NONBLOCK);
    if (fd == -1) {
        throw std::runtime_error("Failed to open pipe for writing: " + pipe_name);
    }
    return fd;
}

void NamedPipe::closePipe(int fd) {
    if (close(fd) == -1) {
        throw std::runtime_error("Failed to close pipe");
    }
}

bool NamedPipe::waitForData(int fd, int timeout_ms) {
    struct pollfd pfd;
    pfd.fd = fd;
    pfd.events = POLLIN;
    
    int ret = poll(&pfd, 1, timeout_ms);
    return ret > 0 && (pfd.revents & POLLIN);
}

void NamedPipe::writePointData(int fd, const Point& point) {
    // Write coordinates
    if (write(fd, point.coordinates.data(), point.coordinates.size() * sizeof(float)) == -1) {
        throw std::runtime_error("Failed to write point coordinates");
    }
    
    // Write point ID
    if (write(fd, &point.id, sizeof(int)) == -1) {
        throw std::runtime_error("Failed to write point ID");
    }
}

Point NamedPipe::readPointData(int fd, size_t n_dimensions) {
    std::vector<float> coords(n_dimensions);
    
    // Read coordinates
    if (read(fd, coords.data(), n_dimensions * sizeof(float)) == -1) {
        throw std::runtime_error("Failed to read point coordinates");
    }
    
    // Read point ID
    int id;
    if (read(fd, &id, sizeof(int)) == -1) {
        throw std::runtime_error("Failed to read point ID");
    }
    
    return Point(coords, id);
}

void NamedPipe::writeBatch(int fd, const std::vector<Point>& points) {
    // Write batch size
    size_t batch_size = points.size();
    if (write(fd, &batch_size, sizeof(size_t)) == -1) {
        throw std::runtime_error("Failed to write batch size");
    }
    
    // Write number of dimensions (same for all points)
    if (!points.empty()) {
        size_t n_dims = points[0].coordinates.size();
        if (write(fd, &n_dims, sizeof(size_t)) == -1) {
            throw std::runtime_error("Failed to write dimensions");
        }
    }
    
    // Write all points
    for (const auto& point : points) {
        writePointData(fd, point);
    }
}

std::vector<Point> NamedPipe::readBatch(int fd, size_t n_dimensions, size_t batch_size) {
    std::vector<Point> points;
    points.reserve(batch_size);
    
    for (size_t i = 0; i < batch_size; ++i) {
        points.push_back(readPointData(fd, n_dimensions));
    }
    
    return points;
}

void NamedPipe::writeBatchResults(int fd, const std::vector<std::vector<Point>>& results) {
    // Write number of result sets
    size_t n_sets = results.size();
    if (write(fd, &n_sets, sizeof(size_t)) == -1) {
        throw std::runtime_error("Failed to write number of result sets");
    }
    
    // Write each result set
    for (const auto& result_set : results) {
        // Write number of points in this set
        size_t n_points = result_set.size();
        if (write(fd, &n_points, sizeof(size_t)) == -1) {
            throw std::runtime_error("Failed to write result set size");
        }
        
        // Write points
        for (const auto& point : result_set) {
            writePointData(fd, point);
        }
    }
}

std::vector<std::vector<Point>> NamedPipe::readBatchResults(int fd) {
    // Read number of result sets
    size_t n_sets;
    if (read(fd, &n_sets, sizeof(size_t)) == -1) {
        throw std::runtime_error("Failed to read number of result sets");
    }
    
    std::vector<std::vector<Point>> results;
    results.reserve(n_sets);
    
    // Read each result set
    for (size_t i = 0; i < n_sets; ++i) {
        // Read number of points in this set
        size_t n_points;
        if (read(fd, &n_points, sizeof(size_t)) == -1) {
            throw std::runtime_error("Failed to read result set size");
        }
        
        // Read points
        std::vector<Point> result_set;
        result_set.reserve(n_points);
        
        for (size_t j = 0; j < n_points; ++j) {
            // Read number of dimensions
            size_t n_dims;
            if (read(fd, &n_dims, sizeof(size_t)) == -1) {
                throw std::runtime_error("Failed to read point dimensions");
            }
            
            result_set.push_back(readPointData(fd, n_dims));
        }
        
        results.push_back(std::move(result_set));
    }
    
    return results;
} 
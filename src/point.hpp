#pragma once
#include <vector>
#include <cmath>

struct Point {
    std::vector<float> coordinates;
    int id;

    Point() : id(-1) {}
    Point(const std::vector<float>& coords, int id) : coordinates(coords), id(id) {}
    Point(const std::vector<float>& coords) : coordinates(coords), id(-1) {}

    // Squared Euclidean distance avoids costly sqrt during search comparisons.
    // Final distances are only rooted when results are returned to the caller.
    float squaredDistanceTo(const Point& other) const {
        float dist = 0.0f;
        for (size_t i = 0; i < coordinates.size(); ++i) {
            float diff = coordinates[i] - other.coordinates[i];
            dist += diff * diff;
        }
        return dist;
    }

    float distanceTo(const Point& other) const {
        return std::sqrt(squaredDistanceTo(other));
    }
};
#include "kdtree.hpp"
#include <algorithm>
#include <cmath>
#include <thread>
#include <future>
#include <queue>
#include <mutex>
#include <iostream>

KDTree::KDTree(const std::vector<Point>& points, int n_threads) : n_threads(n_threads) {
    if (points.empty()) {
        std::cerr << "Warning: Empty points vector" << std::endl;
        return;
    }

    try {
        if (n_threads > 1) {
            root = buildTreeParallel(points, 0, n_threads);
        } else {
            root = buildTree(points, 0);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error during k-d tree construction: " << e.what() << std::endl;
        throw;
    }
}

std::unique_ptr<KDTree::Node> KDTree::buildTree(const std::vector<Point>& points, int depth) {
    if (points.empty()) return nullptr;

    // Cycle through dimensions round-robin at each depth level
    int dimension = depth % points[0].coordinates.size();

    // nth_element gives O(n) median finding vs O(n log n) full sort
    std::vector<Point> working = points;
    size_t median_idx = working.size() / 2;

    std::nth_element(working.begin(), working.begin() + median_idx, working.end(),
                     [dimension](const Point& a, const Point& b) {
                         return a.coordinates[dimension] < b.coordinates[dimension];
                     });

    auto node = std::make_unique<Node>(working[median_idx], dimension);

    node->left  = buildTree({working.begin(), working.begin() + median_idx}, depth + 1);
    node->right = buildTree({working.begin() + median_idx + 1, working.end()}, depth + 1);

    return node;
}

std::unique_ptr<KDTree::Node> KDTree::buildTreeParallel(const std::vector<Point>& points, int depth, int threads_available) {
    if (points.empty()) return nullptr;

    // Fall back to sequential build for small subsets or when threads are exhausted
    if (threads_available <= 1 || points.size() < 100000) {
        return buildTree(points, depth);
    }

    int dimension = depth % points[0].coordinates.size();

    std::vector<Point> working = points;
    size_t median_idx = working.size() / 2;

    std::nth_element(working.begin(), working.begin() + median_idx, working.end(),
                     [dimension](const Point& a, const Point& b) {
                         return a.coordinates[dimension] < b.coordinates[dimension];
                     });

    auto node = std::make_unique<Node>(working[median_idx], dimension);

    std::vector<Point> left_points(working.begin(), working.begin() + median_idx);
    std::vector<Point> right_points(working.begin() + median_idx + 1, working.end());

    int left_threads  = threads_available / 2;
    int right_threads = threads_available - left_threads;

    // Build left subtree asynchronously while right is built on this thread
    auto left_future = std::async(std::launch::async,
        [this, &left_points, depth, left_threads]() {
            return buildTreeParallel(left_points, depth + 1, left_threads);
        });

    node->right = buildTreeParallel(right_points, depth + 1, right_threads);
    node->left  = left_future.get();

    return node;
}

// Free function: squared distance avoids sqrt during search hot path
static float calculateDistanceSquared(const Point& a, const Point& b) {
    float dist = 0.0f;
    for (size_t i = 0; i < a.coordinates.size(); ++i) {
        float diff = a.coordinates[i] - b.coordinates[i];
        dist += diff * diff;
    }
    return dist;
}

// Tie-break equal distances lexicographically for deterministic results
static bool compareDistances(const std::pair<float, Point>& a, const std::pair<float, Point>& b) {
    if (std::abs(a.first - b.first) > 1e-6f)
        return a.first < b.first;

    for (size_t i = 0; i < a.second.coordinates.size(); ++i) {
        if (std::abs(a.second.coordinates[i] - b.second.coordinates[i]) > 1e-6f)
            return a.second.coordinates[i] < b.second.coordinates[i];
    }
    return false;
}

std::vector<Point> KDTree::findKNearest(const Point& query, int k) const {
    std::vector<std::pair<float, Point>> distanced_neighbors;
    distanced_neighbors.reserve(k);

    searchKNN(root.get(), query, k, distanced_neighbors);

    std::sort(distanced_neighbors.begin(), distanced_neighbors.end(), compareDistances);

    std::vector<Point> neighbors;
    neighbors.reserve(distanced_neighbors.size());
    for (const auto& pair : distanced_neighbors)
        neighbors.push_back(pair.second);

    return neighbors;
}

void KDTree::searchKNN(const Node* node, const Point& query, int k,
                       std::vector<std::pair<float, Point>>& neighbors) const {
    if (!node) return;

    float dist_sq = calculateDistanceSquared(node->point, query);

    // Maintain a max-heap of size k — heap front is always the furthest neighbor
    if (neighbors.size() < static_cast<size_t>(k)) {
        neighbors.emplace_back(dist_sq, node->point);
        std::push_heap(neighbors.begin(), neighbors.end(),
                       [](const auto& a, const auto& b) { return a.first < b.first; });
    } else if (dist_sq < neighbors.front().first) {
        std::pop_heap(neighbors.begin(), neighbors.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });
        neighbors.back() = {dist_sq, node->point};
        std::push_heap(neighbors.begin(), neighbors.end(),
                       [](const auto& a, const auto& b) { return a.first < b.first; });
    }

    // Signed distance to splitting hyperplane
    float diff = query.coordinates[node->dimension] - node->point.coordinates[node->dimension];
    float dist_to_plane_sq = diff * diff;

    // Search the side the query falls on first, then check if we need to cross the hyperplane
    const Node* near_side = (diff < 0) ? node->left.get()  : node->right.get();
    const Node* far_side  = (diff < 0) ? node->right.get() : node->left.get();

    searchKNN(near_side, query, k, neighbors);

    // Only cross hyperplane if it could contain a closer neighbor
    if (neighbors.size() < static_cast<size_t>(k) || dist_to_plane_sq < neighbors.front().first)
        searchKNN(far_side, query, k, neighbors);
}

std::vector<std::vector<Point>> KDTree::findKNearestBatch(const std::vector<Point>& queries, int k) const {
    std::vector<std::vector<Point>> results(queries.size());

    std::vector<std::future<void>> futures;
    size_t chunk_size = queries.size() / n_threads + 1;

    for (size_t i = 0; i < queries.size(); i += chunk_size) {
        size_t end = std::min(i + chunk_size, queries.size());
        futures.push_back(std::async(std::launch::async,
            [this, &queries, &results, k, i, end]() {
                for (size_t j = i; j < end; ++j)
                    results[j] = findKNearest(queries[j], k);
            }));
    }

    for (auto& future : futures)
        future.wait();

    return results;
}
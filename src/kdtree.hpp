#pragma once

#include "point.hpp"
#include <memory>
#include <vector>
#include <utility>

class KDTree {
public:
    // Constructor with parallel median finding
    KDTree(const std::vector<Point>& points, int n_threads = 1);
    
    // Find k nearest neighbors of a query point
    std::vector<Point> findKNearest(const Point& query, int k) const;
    
    // Find k nearest neighbors of multiple query points
    std::vector<std::vector<Point>> findKNearestBatch(const std::vector<Point>& queries, int k) const;
    
private:
    struct Node {
        Point point;
        int dimension;
        std::unique_ptr<Node> left;
        std::unique_ptr<Node> right;
        
        Node(const Point& p, int dim) : point(p), dimension(dim) {}
    };
    
    std::unique_ptr<Node> root;
    int n_threads;
    
    // Helper functions
    std::unique_ptr<Node> buildTree(const std::vector<Point>& points, int depth);
    std::unique_ptr<Node> buildTreeParallel(const std::vector<Point>& points, int depth, int threads_available);
    void searchKNN(const Node* node, const Point& query, int k,
                   std::vector<std::pair<float, Point>>& neighbors) const;
}; 
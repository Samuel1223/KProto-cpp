#pragma once

#include <string>
#include <vector>

namespace kproto {

// Squared Euclidean distance between two numeric vectors:
//   sum_i (a[i] - b[i])^2
// (Squared, to match the k-prototypes cost function — the numeric term is the
// squared Euclidean distance, not its square root.)
// Throws std::invalid_argument if the lengths differ or both are empty.
double numeric_distance(const std::vector<double>& a, const std::vector<double>& b);

// Simple matching dissimilarity between two categorical vectors: the number of
// positions at which the two differ. Empty vectors are allowed and yield 0.
// Throws std::invalid_argument if the lengths differ.
int categorical_distance(const std::vector<std::string>& a,
                         const std::vector<std::string>& b);

// Combined k-prototypes dissimilarity between a point (an, ac) and a prototype
// (bn, bc):
//   numeric_distance(an, bn) + gamma * categorical_distance(ac, bc)
// Throws std::invalid_argument if gamma is negative or either part's lengths
// are inconsistent (propagated from the two helpers above).
double mixed_distance(const std::vector<double>& an, const std::vector<std::string>& ac,
                      const std::vector<double>& bn, const std::vector<std::string>& bc,
                      double gamma);

}  // namespace kproto

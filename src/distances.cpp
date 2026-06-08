#include "kproto/distances.hpp"

#include <stdexcept>

namespace kproto {

double numeric_distance(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("numeric_distance: length mismatch");
  }
  if (a.empty()) {
    throw std::invalid_argument("numeric_distance: empty vector");
  }
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = a[i] - b[i];
    sum += d * d;
  }
  return sum;
}

int categorical_distance(const std::vector<std::string>& a,
                         const std::vector<std::string>& b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument("categorical_distance: length mismatch");
  }
  int mismatches = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      ++mismatches;
    }
  }
  return mismatches;
}

double mixed_distance(const std::vector<double>& an, const std::vector<std::string>& ac,
                      const std::vector<double>& bn, const std::vector<std::string>& bc,
                      double gamma) {
  if (gamma < 0.0) {
    throw std::invalid_argument("mixed_distance: gamma must be non-negative");
  }
  double total = 0.0;
  if (!an.empty() || !bn.empty()) {
    total += numeric_distance(an, bn);
  } else if (an.size() != bn.size()) {
    throw std::invalid_argument("mixed_distance: numeric length mismatch");
  }
  total += gamma * static_cast<double>(categorical_distance(ac, bc));
  return total;
}

}  // namespace kproto

#include "kproto/kprototypes.hpp"

#include <stdexcept>

namespace kproto {

namespace {
[[noreturn]] void not_implemented() { throw std::logic_error("not implemented"); }
}  // namespace

KPrototypes::KPrototypes(int n_clusters, double gamma, int max_iter)
    : n_clusters_(n_clusters), gamma_(gamma), max_iter_(max_iter) {
  if (n_clusters <= 0) {
    throw std::invalid_argument("n_clusters must be positive");
  }
  if (gamma < 0.0) {
    throw std::invalid_argument("gamma must be non-negative");
  }
  if (max_iter <= 0) {
    throw std::invalid_argument("max_iter must be positive");
  }
}

void KPrototypes::fit(const std::vector<Point>&, const std::vector<int>&) { not_implemented(); }
void KPrototypes::fit(const std::vector<Point>&) { not_implemented(); }
int KPrototypes::predict(const Point&) const { not_implemented(); }
const std::vector<int>& KPrototypes::labels() const { not_implemented(); }
const std::vector<Centroid>& KPrototypes::centroids() const { not_implemented(); }
double KPrototypes::cost() const { not_implemented(); }
int KPrototypes::n_iter() const { not_implemented(); }
bool KPrototypes::converged() const { not_implemented(); }

}  // namespace kproto

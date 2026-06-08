#pragma once

#include <vector>

#include "kproto/types.hpp"

namespace kproto {

// K-prototypes clustering for mixed numeric + categorical data (Huang, 1998).
//
// The model partitions a set of Points into `n_clusters` groups by alternating
// assignment and update steps (Lloyd's algorithm), using the combined
// dissimilarity from distances.hpp:
//
//     D(point, centroid) = numeric_distance(point.numeric, centroid.numeric)
//                          + gamma * categorical_distance(point.categorical,
//                                                         centroid.categorical)
//
// Determinism: there is no randomness. Initial centroids are taken from
// caller-supplied row indices (or the first `n_clusters` rows by default), and
// every tie is broken deterministically (see fit()).
class KPrototypes {
 public:
  // n_clusters: number of clusters (> 0).
  // gamma:      weight on the categorical dissimilarity (>= 0).
  // max_iter:   maximum number of assignment passes (> 0).
  // Throws std::invalid_argument if any argument is out of range.
  KPrototypes(int n_clusters, double gamma, int max_iter = 100);

  // Cluster `data` using `init_indices` as the rows for the initial centroids.
  //
  // Validation (all std::invalid_argument):
  //   - data must be non-empty;
  //   - every point must share data[0]'s numeric arity and categorical arity;
  //   - init_indices.size() must equal n_clusters;
  //   - every index must be in [0, data.size()) and the indices must be distinct.
  //
  // Algorithm:
  //   centroid[c] starts as a copy of data[init_indices[c]];
  //   repeat up to max_iter times:
  //     1. assignment: label each point with argmin_c D(point, centroid[c]);
  //        ties are broken by the smallest cluster index;
  //     2. if the labels are identical to the previous pass, mark converged and
  //        stop;
  //     3. update: for each cluster with at least one member, set its numeric
  //        centroid to the per-column mean and its categorical centroid to the
  //        per-column mode (most frequent value; ties broken by the
  //        lexicographically smallest string). A cluster with no members keeps
  //        its centroid from the previous pass unchanged.
  //
  // After fit(), labels(), centroids(), cost(), n_iter(), converged() and
  // is_fitted() reflect the result. Calling fit() again fully replaces prior
  // state.
  void fit(const std::vector<Point>& data, const std::vector<int>& init_indices);

  // Convenience overload: initialize from the first `n_clusters` rows, i.e.
  // init_indices = {0, 1, ..., n_clusters - 1}.
  void fit(const std::vector<Point>& data);

  // Return the cluster index a single point would be assigned to under the
  // fitted centroids (argmin D, ties broken by smallest index).
  // Throws std::logic_error if the model is not fitted, and
  // std::invalid_argument if the point's arities do not match the fitted data.
  int predict(const Point& p) const;

  // Accessors (each throws std::logic_error if called before a successful fit,
  // except is_fitted()/n_clusters()/gamma()/max_iter()).
  const std::vector<int>& labels() const;          // cluster id per input row
  const std::vector<Centroid>& centroids() const;  // size == n_clusters
  double cost() const;  // sum over points of D(point, its assigned centroid)
  int n_iter() const;   // assignment passes performed (>= 1 after fit)
  bool converged() const;

  bool is_fitted() const { return fitted_; }
  int n_clusters() const { return n_clusters_; }
  double gamma() const { return gamma_; }
  int max_iter() const { return max_iter_; }

 private:
  int n_clusters_;
  double gamma_;
  int max_iter_;

  bool fitted_ = false;
  bool converged_ = false;
  int n_iter_ = 0;
  double cost_ = 0.0;
  int n_numeric_ = 0;
  int n_categorical_ = 0;

  std::vector<int> labels_;
  std::vector<Centroid> centroids_;
};

}  // namespace kproto

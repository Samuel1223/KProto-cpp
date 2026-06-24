#pragma once

#include <functional>
#include <string>
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
// Numeric features are standardized (z-scored) from the fit data before
// clustering; the per-column mean/std learned at fit are stored so that
// predict() can standardize its query the same way. All clustering state
// (centroids, cost) lives in the standardized numeric space.
//
// Determinism: there is no randomness. Initial centroids are taken from
// caller-supplied row indices (or the first `n_clusters` rows by default), and
// every tie is broken deterministically (see fit()).
class KPrototypes {
 public:
  // Strict-weak ordering over categorical values, used to break ties when
  // selecting the per-column mode: among the values that share the highest
  // frequency in a column, the centroid takes the one that compares "less" than
  // every other tied value under this predicate. The default,
  // std::less<std::string>, therefore picks the lexicographically smallest
  // value; passing std::greater<std::string> would pick the largest.
  using TieBreaker = std::function<bool(const std::string&, const std::string&)>;

  // Strategy for choosing the initial centroids in fit():
  //   Provided      -- use the caller's init_indices directly (init_indices.size()
  //                    must equal n_clusters).
  //   FarthestFirst -- deterministic farthest-first (maxmin) seeding from a single
  //                    given row (init_indices.size() must equal 1); see fit().
  enum class Init { Provided, FarthestFirst };

  // n_clusters:       number of clusters (> 0).
  // gamma:            weight on the categorical dissimilarity (>= 0).
  // max_iter:         maximum number of assignment passes (> 0).
  // mode_tie_breaker: ordering used to resolve categorical mode ties (see the
  //                   TieBreaker note above); defaults to lexicographically
  //                   smallest.
  // init:             initial-centroid strategy (see the Init enum and fit());
  //                   defaults to Provided.
  // Throws std::invalid_argument if any of the numeric arguments are out of range.
  KPrototypes(int n_clusters, double gamma, int max_iter = 100,
              TieBreaker mode_tie_breaker = std::less<std::string>(),
              Init init = Init::Provided);

  // Cluster `data` using `init_indices` as the rows for the initial centroids.
  //
  // sample_weight optionally assigns a positive weight to each row (think of it
  // as a fractional multiplicity). It must either be empty -- meaning every row
  // has weight 1 -- or have exactly data.size() entries, each strictly positive.
  // Every weighted quantity below uses these weights; with the default (all 1s)
  // the result is identical to the unweighted algorithm.
  //
  // Validation (all std::invalid_argument):
  //   - data must be non-empty;
  //   - every point must share data[0]'s numeric arity and categorical arity;
  //   - with Init::Provided, init_indices.size() must equal n_clusters; with
  //     Init::FarthestFirst, init_indices.size() must equal 1 (the first seed);
  //   - every index must be in [0, data.size()) and the indices must be distinct;
  //   - sample_weight, if non-empty, must have size data.size() and every entry
  //     must be > 0.
  //
  // Algorithm:
  //   standardize each numeric column using the fit data's WEIGHTED mean and
  //   weighted population std -- mean_j = (sum_i w_i x_ij) / (sum_i w_i) and
  //   var_j = (sum_i w_i (x_ij - mean_j)^2) / (sum_i w_i); a zero-std column is
  //   treated as std 1. Cluster in this standardized space. The initial centroids
  //   come from `init`:
  //     - Provided: centroid[c] is the standardized row data[init_indices[c]].
  //     - FarthestFirst: centroid[0] is the standardized row data[init_indices[0]];
  //       then each subsequent centroid is the data row whose MINIMUM unweighted
  //       mixed_distance (in standardized space, with this gamma) to the
  //       already-chosen centroids is the LARGEST, ties broken by the lowest row
  //       index, until there are n_clusters centroids.
  //   Then repeat up to max_iter times:
  //     1. assignment: label each point with argmin_c D(point, centroid[c]);
  //        ties are broken by the smallest cluster index (the weights do not
  //        affect which centroid is nearest);
  //     2. if the labels are identical to the previous pass, mark converged and
  //        stop;
  //     3. update: for each cluster with at least one member, set its numeric
  //        centroid to the WEIGHTED per-column mean of its members, and its
  //        categorical centroid to the WEIGHTED per-column mode -- the value with
  //        the greatest total member weight (NOT the raw count), ties broken with
  //        mode_tie_breaker (lexicographically smallest by default). A cluster
  //        with no members keeps its centroid from the previous pass unchanged.
  //
  // cost() is the weighted sum over points of w_i * D(point_i, its centroid).
  //
  // Missing values: a numeric feature that is NaN, or a categorical feature equal
  // to the empty string "", is treated as MISSING, and missing entries are handled
  // pairwise/columnwise rather than propagating:
  //   - distance D skips any dimension where EITHER operand is missing (no
  //     normalization for the number of present dimensions);
  //   - standardization computes each column's weighted mean/std over only its
  //     present (non-NaN) values (a column with no present value uses mean 0,
  //     std 1); a present value is standardized as usual, a missing value stays
  //     missing (NaN) in the standardized space;
  //   - a cluster's numeric centroid coordinate is the weighted mean over the
  //     members that are present in that dimension (NaN if none are present), and
  //     its categorical centroid coordinate is the weighted mode over the members
  //     that are present in that column (the empty string "" if none are present);
  //   - predict() and cost() use this same missing-aware distance.
  // With no missing values every rule above reduces to the dense behavior.
  //
  // After fit(), labels(), centroids(), cost(), n_iter(), converged() and
  // is_fitted() reflect the result. Calling fit() again fully replaces prior
  // state.
  void fit(const std::vector<Point>& data, const std::vector<int>& init_indices,
           const std::vector<double>& sample_weight = {});

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
  double cost() const;  // weighted sum over points of w_i * D(point, its centroid)
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
  TieBreaker mode_tie_breaker_;
  Init init_;

  bool fitted_ = false;
  bool converged_ = false;
  int n_iter_ = 0;
  double cost_ = 0.0;
  int n_numeric_ = 0;
  int n_categorical_ = 0;

  std::vector<double> feature_mean_;  // per-numeric-column mean from fit (standardization)
  std::vector<double> feature_std_;   // per-numeric-column population std from fit (0 -> treated as 1)

  std::vector<int> labels_;
  std::vector<Centroid> centroids_;
};

}  // namespace kproto

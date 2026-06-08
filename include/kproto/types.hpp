#pragma once

#include <string>
#include <vector>

namespace kproto {

// A single observation with a numeric part and a categorical part. Every point
// fed to a KPrototypes model must share the same numeric arity and the same
// categorical arity (the model validates this on fit).
struct Point {
  std::vector<double> numeric;
  std::vector<std::string> categorical;
};

// A cluster prototype: the per-feature mean of the numeric columns and the
// per-feature mode of the categorical columns of its assigned members.
struct Centroid {
  std::vector<double> numeric;
  std::vector<std::string> categorical;

  bool operator==(const Centroid& o) const {
    return numeric == o.numeric && categorical == o.categorical;
  }
  bool operator!=(const Centroid& o) const { return !(*this == o); }
};

}  // namespace kproto

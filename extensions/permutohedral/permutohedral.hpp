// Copyright 2016 Max Planck Society
// Distributed under the BSD-3 Software license,
// (See accompanying file ../../../../LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)

#ifndef PERMUTOHEDRAL_HPP
#define PERMUTOHEDRAL_HPP

#include <boost/shared_ptr.hpp>
#include <vector>

#include "math_utils.hpp"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>

#include <boost/array.hpp>
#include <boost/cstdint.hpp>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

/************************************************/
/***         Some Utility Functions           ***/
/************************************************/

int ipow(int base, int exp) {
  int result = 1;
  while (exp) {
    if (exp & 1) result *= base;
    exp >>= 1;
    base *= base;
  }
  return result;
}

int get_filter_size(int neighborhood_size, int feature_size) {
  return ipow(neighborhood_size + 1, feature_size + 1) -
         ipow(neighborhood_size, feature_size + 1);
}

inline void advance_in_dimension(const std::size_t dimension,
                                 const int increment,
                                 std::vector<boost::int16_t>* key) {
  const int d_ = key->size() - 1;
  for (int k = 0; k <= d_; ++k) {
    (*key)[k] -= increment;
  }
  (*key)[dimension] += increment * (1 + d_);
}

/************************************************/
/***                Hash Table                ***/
/************************************************/

/*! \brief Hash Table for keeping track of lattice occupancy. Taken from
 * Krähenbühl's original DenseCRF lattice code.
 *         (http://www.philkr.net/home/densecrf)
 */

class HashTable {
 private:
  // Don't copy!
  HashTable(const HashTable& o)
    : key_size_(o.key_size_), filled_(0), capacity_(o.capacity_) {
    table_ = new int[capacity_];
    keys_  = new boost::int16_t[(capacity_ / 2 + 10) * key_size_];
    memset(table_, -1, capacity_ * sizeof(int));
  }

  size_t key_size_, filled_, capacity_;
  boost::int16_t* keys_;
  int* table_;
  void grow() {
    // Swap out the old memory
    boost::int16_t* old_keys = keys_;
    int* old_table           = table_;
    int old_capacity         = capacity_;
    capacity_ *= 2;
    // Allocate the new memory
    keys_  = new boost::int16_t[(old_capacity + 10) * key_size_];
    table_ = new int[capacity_];
    memset(table_, -1, capacity_ * sizeof(int));
    memcpy(keys_, old_keys, filled_ * key_size_ * sizeof(boost::int16_t));

    // Reinsert each element
    for (int i = 0; i < old_capacity; i++)
      if (old_table[i] >= 0) {
        int e    = old_table[i];
        size_t h = hash(old_keys + (getKey(e) - keys_)) % capacity_;
        for (; table_[h] >= 0; h = h < capacity_ - 1 ? h + 1 : 0)
          ;
        table_[h] = e;
      }

    delete[] old_keys;
    delete[] old_table;
  }

  size_t hash(const boost::int16_t* k) {
    size_t r = 0;
    for (size_t i = 0; i < key_size_; i++) {
      r += k[i];
      r *= 1664525;
    }
    return r;
  }

 public:
  HashTable(int key_size, int n_elements)
    : key_size_(key_size), filled_(0), capacity_(2 * n_elements) {
    table_ = new int[capacity_];
    keys_  = new boost::int16_t[(capacity_ / 2 + 10) * key_size_];
    memset(table_, -1, capacity_ * sizeof(int));
  }

  ~HashTable() {
    delete[] keys_;
    delete[] table_;
  }

  int size() const { return filled_; }

  void reset() {
    filled_ = 0;
    memset(table_, -1, capacity_ * sizeof(int));
  }

  int find(const boost::int16_t* k, bool create = false) {
    if (2 * filled_ >= capacity_) grow();
    // Get the hash value
    size_t h = hash(k) % capacity_;
    // Find the element with he right key, using linear probing
    while (1) {
      int e = table_[h];
      if (e == -1) {
        if (create) {
          // Insert a new key and return the new id
          for (size_t i = 0; i < key_size_; i++)
            keys_[filled_ * key_size_ + i] = k[i];
          return table_[h] = filled_++;
        } else {
          return -1;
        }
      }
      // Check if the current key is The One
      bool good = true;
      for (size_t i = 0; i < key_size_ && good; i++)
        if (keys_[e * key_size_ + i] != k[i]) good = false;
      if (good) return e;
      // Continue searching
      h++;
      if (h == capacity_) h = 0;
    }
  }

  const boost::int16_t* getKey(int i) const {
    assert(static_cast<std::size_t>(i) < filled_);
    return keys_ + i * key_size_;
  }
};

/************************************************/
/***     Permutohedral Lattice Traversal      ***/
/************************************************/

/*! \brief Class functions for traversing the lattice to build neighborhood
 * stucture for convolutions.
 */

class LatticeTraversal {
 public:
  typedef std::vector<boost::int16_t> key_type;

 public:
  explicit LatticeTraversal(int neighborhood_size, int d)
    : neighborhood_size_(neighborhood_size), d_(d) {}

  template <typename TFun>
  void go(const key_type& start_key, TFun yield) const {
    assert(start_key.size() == d_ + 1);

    std::vector<key_type> walking_keys(d_ + 1);
    for (int i = 0; static_cast<std::size_t>(i) < walking_keys.size(); ++i) {
      walking_keys[i].resize(start_key.size());
    }

    walk_cuboid(start_key, 0, false, walking_keys, yield);
  }

 private:
  template <typename TFun>
  void walk_cuboid(const key_type& start_key,
                   const int d,
                   const bool has_zero,
                   std::vector<key_type>& walking_keys,
                   TFun yield) const {
    if (d <= d_) {
      key_type& walking_key = walking_keys[d];
      walking_key           = start_key;

      const int range_end = (d < d_ || has_zero) ? neighborhood_size_ + 1 : 1;
      for (int i = 0; i < range_end; ++i) {
        walk_cuboid(walking_key, d + 1, has_zero || i == 0, walking_keys,
                    yield);
        advance_in_dimension(d, 1, &walking_key);
      }
    } else {
      yield(start_key);
    }
  }

  int neighborhood_size_;
  int d_;
};

/************************************************/
/***         Neighborhood Callback            ***/
/************************************************/

/*! \brief Used for approximate lattice traversal.
 */

class NeighborhoodCallback {
 public:
  NeighborhoodCallback(const int step, int* const neighbors, int* n)
    : step_(step), neighbors_(neighbors), n_(*n) {}

  void operator()(const int indx) {
    if (n_ >= 0) neighbors_[n_ * step_] = indx;
    ++n_;
  }

 private:
  const int step_;
  int* const neighbors_;
  int& n_;
};

/************************************************/
/***     Approximate Lattice Traversal        ***/
/************************************************/

/*! \brief Class functions for faster and Approximately
 *         traversing the lattice to build neighborhood stucture for
 * convolutions.
 */

class LatticeApproximateTraversal {
 public:
  LatticeApproximateTraversal(int neighborhood_size,
                              int d,
                              const std::vector<int>& immediate_neighbors,
                              int M)
    : neighborhood_size_(neighborhood_size),
      d_(d),
      immediate_neighbors_(immediate_neighbors),
      M_(M) {}

  template <typename TFun>
  void go(const int start, TFun yield) const {
    walk_approximate(start, 0, false, yield);
  }

 private:
  template <typename TFun>
  void walk_approximate(const int start,
                        const int d,
                        const bool has_zero,
                        TFun yield) const {
    if (d <= d_) {
      int walking = start;

      const int range_end = (d < d_ || has_zero) ? neighborhood_size_ + 1 : 1;
      for (int i = 0; i < range_end; ++i) {
        walk_approximate(walking, d + 1, has_zero || i == 0, yield);
        if (walking >= 0) walking = immediate_neighbors_[walking + M_ * d];
      }
    } else {
      yield(start);
    }
  }

  int neighborhood_size_;
  int d_;
  const std::vector<int>& immediate_neighbors_;
  int M_;
};

/************************************************/
/*** Gaussian Filter for Permutohedral Lattice **/
/************************************************/

/*! \brief Class for high-dimensional Gaussian filter construction.
 *         Useful as 'offset' for the learnable filter.
 */

class GaussianFilter {
 public:
  GaussianFilter(int neighborhood_size, int feature_size)
    : neighborhood_size_(neighborhood_size), feature_size_(feature_size) {
    build_filter();
  }

  const float* filter() { return filter_.data(); }

 private:
  class TraversalCallback {
   public:
    TraversalCallback(HashTable& hash_table) : hash_table_(hash_table) {}

    void operator()(const std::vector<boost::int16_t>& key) {
      hash_table_.find(key.data(), true);
    }

   private:
    HashTable& hash_table_;
  };

  void build_filter() {
    boost::array<float, 2> gauss = {{1, 0.5}};

    const int size = get_filter_size(neighborhood_size_, feature_size_);

    HashTable hash_table(feature_size_, size * (feature_size_ + 1));

    std::vector<float> lattice(size + 1);

    // Insert center of lattice into hash table.
    std::vector<boost::int16_t> center(feature_size_ + 1);
    const int center_index = hash_table.find(center.data(), true) + 1;
    assert(center_index == 1);

    // Insert all other lattice points into the hash table.
    LatticeTraversal traversal(neighborhood_size_, feature_size_);
    TraversalCallback yield(hash_table);
    traversal.go(center, yield);

    // Initialize the center of the lattice.
    lattice[center_index] = 1;

    std::vector<float> tmp_lattice(size + 1);
    std::vector<boost::int16_t> walking_key_up(feature_size_ + 1);
    std::vector<boost::int16_t> walking_key_down(feature_size_ + 1);
    for (int d = 0; d <= feature_size_; ++d) {
      std::fill(tmp_lattice.begin(), tmp_lattice.end(), 0);

      for (int i = 0; i < size; i++) {
        const boost::int16_t* key = hash_table.getKey(i);
        std::copy(key, key + feature_size_ + 1, walking_key_up.begin());
        std::copy(key, key + feature_size_ + 1, walking_key_down.begin());

        float& v = tmp_lattice[i + 1];
        v        = lattice[i + 1] * gauss[0];

        for (int n = 1; n < neighborhood_size_ + 1; ++n) {
          advance_in_dimension(d, 1, &walking_key_up);
          advance_in_dimension(d, -1, &walking_key_down);

          v += (lattice[hash_table.find(walking_key_up.data()) + 1] +
                lattice[hash_table.find(walking_key_down.data()) + 1]) *
               (n < gauss.size() ? gauss[n] : 0);
        }
      }

      lattice.swap(tmp_lattice);
      lattice[0] = 0;
    }

    filter_.resize(size);
    // Normalize the filter according to the center lattice point. Like that we
    // are not creating additional energy for it.
    const float alpha = lattice[1];
    for (int i = 0; i < size; ++i) {
      filter_[i] = lattice[i + 1] / alpha;
    }
  }

  int neighborhood_size_;
  int feature_size_;
  std::vector<float> filter_;
};

/************************************************/
/***          Permutohedral Lattice           ***/
/************************************************/

/*! \brief This is the main class for lattice construction and forward
 * operations in 'learnable' sparse high dimensional filtering. This class
 * defines 'forward' functionatlity. See CPU/GPU specific
 *         'PermutohedralReverseCpu' and 'PermutohedralReverseGpu' for the
 * definition of 'splat', 'blur', 'max' and 'slice' forward functions and the
 * respective backward functions.
 *
 *  'Filter' weights are generic non-seperable high-dimensional permutohedral
 * filters. This class has both CPU and GPU functionality except for the lattice
 * construction, At present, there is no dedicated GPU functions for
 * permutohedral lattice construction.
 *
 *  Some parts of the code are adapted and heavily modified from the separable
 * filter code from Adams et al. 2010
 * (http://graphics.stanford.edu/papers/permutohedral/).
 */

class PermutohedralReverse;

class Permutohedral {
 public:
  typedef GaussianFilter gauss_type;
  struct Lattice {
    int N_, d_;
    int neighborhood_size_;
    int M_;
    std::vector<float> barycentric_;
    std::vector<int> offset_;
    std::vector<int> blur_neighbors_;
  };

 private:
  Permutohedral(const Permutohedral& rhs);
  boost::shared_ptr<const Lattice> lattice_;
  bool check_unique_neighbors(const int* neighbors);
  static void map_back(const std::vector<boost::int16_t>& key, float* const x);

 public:
  Permutohedral();
  static int get_filter_size(int neighborhood_size, int feature_size);
  void init(const float* feature,
            int data_count,
            int feature_size,
            int neighborhood_size,
            bool do_visualization);
  boost::shared_ptr<PermutohedralReverse> compute(const float* filter,
                                                  const float* in,
                                                  int num_output,
                                                  int group,
                                                  int value_size,
                                                  bool do_skip_blur,
                                                  int in_offset,
                                                  int out_offset,
                                                  int in_size,
                                                  int out_size,
                                                  float* out) const;
  boost::shared_ptr<PermutohedralReverse> max_compute(const float* filter,
                                                      const float* in,
                                                      int value_size,
                                                      int in_offset,
                                                      int out_offset,
                                                      int in_size,
                                                      int out_size,
                                                      float* out);
};

/************************************************/
/***          Permutohedral Reverse           ***/
/************************************************/

/*! \brief This is the main class for reverse operations in
 *         'learnable' sparse high dimensional filtering.
 *         This class defines 'backward' functionatlity. See CPU/GPU specific
 *         'PermutohedralReverseCpu' and 'PermutohedralReverseGpu' for the
 * definition of 'splat_tick', 'blur_tick', 'max_tick' and 'slice_tick' backward
 * functions and the respective foward functions.
 *
 */

class PermutohedralReverse {
 public:
  void reverse(const float* diff_in,
               float* diff_out_filter,
               float* diff_out_in);
  void max_reverse(const float* diff_in, float* diff_out_in);

 private:
  PermutohedralReverse(const PermutohedralReverse& rhs);
  PermutohedralReverse();

  void init(
    const float* filter,
    int num_output,
    int group,
    int value_size,
    bool do_skip_blur,
    int in_offset,
    int out_offset,
    int in_size,
    int out_size,
    const boost::shared_ptr<const typename Permutohedral::Lattice> lattice);
  void compute(const float* in, float* out);
  void max_compute(const float* in, float* out);
  void slice(const std::vector<float>& data, float* sliced) const;
  void blur(const std::vector<float>& splatted,
            const float* filter,
            std::vector<float>* blurred) const;
  void max(const std::vector<float>& splatted, std::vector<float>* maxxed);
  void splat(const float* in, std::vector<float>* splatted) const;
  static void im2col(const float* im,
                     const std::size_t value_size,
                     const std::size_t filter_size,
                     const std::size_t M,
                     const std::size_t start,
                     const std::size_t end,
                     const std::vector<int>& blur_neighbors,
                     float* col);
  static void col2im(const float* col,
                     const std::size_t value_size,
                     const std::size_t filter_size,
                     const std::size_t M,
                     const std::size_t start,
                     const std::size_t end,
                     const std::vector<int>& blur_neighbors,
                     float* im);
  void slice_tick(const float* sliced_tick,
                  std::vector<float>* sliced_out) const;
  void blur_tick(const std::vector<float>& blurred_tick,
                 std::vector<float>* blurred_out,
                 float* filter_out);
  void max_tick(const std::vector<float>& maxxed_tick,
                std::vector<float>* maxxed_out);
  void splat_tick(const std::vector<float>& splatted_tick, float* splatted_out);

  std::vector<float> filter_;
  std::vector<float> splatted_;

  int d_, N_;
  int neighborhood_size_;
  int M_;
  std::vector<int> max_idx_;
  boost::shared_ptr<const typename Permutohedral::Lattice> lattice_;

  int in_offset_, out_offset_, in_size_, out_size_;
  int num_output_, group_;
  int value_size_;
  bool do_skip_blur_;

  friend class Permutohedral;
};

/************************************************/
/***          Permutohedral Lattice           ***/
/************************************************/

void Permutohedral::map_back(const std::vector<boost::int16_t>& key,
                             float* const x) {
  const int d_      = key.size() - 1;
  float inv_std_dev = std::sqrt(2.0 / 3.0) * (d_ + 1);

  std::vector<float> scale_factor(d_);
  for (int i = 0; i < d_; ++i) {
    scale_factor[i] = 1.0 / std::sqrt((i + 2) * (i + 1)) * inv_std_dev;
  }

  float sum = 0;
  for (int j = d_; j > 0; --j) {
    float cf = (sum - key[j]) / j;
    x[j - 1] = cf / scale_factor[j - 1];
    sum += cf;
  }
  assert(std::abs(sum - key[0]) < 1e-3);
}

Permutohedral::Permutohedral() {}

int Permutohedral::get_filter_size(int neighborhood_size, int feature_size) {
  return ::get_filter_size(neighborhood_size, feature_size);
}

void Permutohedral::init(const float* feature,
                         int data_count,
                         int feature_size,
                         int neighborhood_size,
                         bool do_visualization) {
  const int N = data_count;
  const int d = feature_size;

  boost::shared_ptr<Lattice> lattice = boost::make_shared<Lattice>();
  // Set the read only shared lattice data
  lattice_ = lattice;

  lattice->N_                 = N;
  lattice->d_                 = d;
  lattice->neighborhood_size_ = neighborhood_size;

  // Allocate enough storage
  lattice->barycentric_.resize(static_cast<std::size_t>((d + 1) * N));
  std::vector<boost::int16_t> ranks;
  ranks.resize((d + 1) * N);
  lattice->offset_.resize((d + 1) * N);

  // Compute the lattice coordinates for each feature [there is going to be
  // a lot of magic here
  HashTable hash_table(d, N * (d + 1));

  // Allocate the local memory
  std::vector<float> scale_factor(d);
  std::vector<float> elevated(d + 1);
  std::vector<float> rem0(d + 1);
  std::vector<float> barycentric(d + 2);
  std::vector<boost::int16_t> canonical((d + 1) * (d + 1));
  std::vector<boost::int16_t> key(d + 1);

  // Compute the canonical simplex
  for (int i = 0; i <= d; i++) {
    for (int j = 0; j <= d - i; j++) canonical[i * (d + 1) + j] = i;
    for (int j = d - i + 1; j <= d; j++)
      canonical[i * (d + 1) + j] = i - (d + 1);
  }

  // Expected standard deviation of our filter (p.6 in [Adams etal 2010])
  float inv_std_dev = sqrt(2.0 / 3.0) * (d + 1);
  // Compute the diagonal part of E (p.5 in [Adams etal 2010])
  for (int i = 0; i < d; i++)
    scale_factor[i] = 1.0 / sqrt((i + 2) * (i + 1)) * inv_std_dev;

  const float* f = feature;

  std::vector<boost::int16_t> min_key(d + 1);
  std::vector<boost::int16_t> max_key(d + 1);

  // Compute the simplex each feature lies in
  for (int k = 0; k < N; k++) {
    // Elevate the feature ( y = Ep, see p.5 in [Adams etal 2010])
    // const float * f = feature + k*feature_size;

    // sm contains the sum of 1..n of our faeture vector
    float sm(0);
    for (int j = d; j > 0; j--) {
      const int fIndex = (j - 1) * N + k;
      float cf         = f[fIndex] * scale_factor[j - 1];
      elevated[j]      = sm - j * cf;
      sm += cf;
    }
    elevated[0] = sm;

    // Find the closest 0-colored simplex through rounding
    float down_factor = 1.0f / (d + 1);
    float up_factor   = (d + 1);
    int sum           = 0;
    for (int i = 0; i <= d; i++) {
      int rd  = round(down_factor * static_cast<float>(elevated[i]));
      rem0[i] = rd * up_factor;
      sum += rd;
    }

    // Find the simplex we are in and store it in rank (where rank
    // describes what position coorinate i has in the sorted order of the
    // features values)
    boost::int16_t* rank = ranks.data() + (d + 1) * k;
    for (int i = 0; i < d; i++) {
      double di = static_cast<float>(elevated[i]) - rem0[i];
      for (int j = i + 1; j <= d; j++)
        if (di < static_cast<float>(elevated[j]) - rem0[j])
          rank[i]++;
        else
          rank[j]++;
    }

    // If the point doesn't lie on the plane (sum != 0) bring it back
    for (int i = 0; i <= d; i++) {
      rank[i] += sum;
      if (rank[i] < 0) {
        rank[i] += d + 1;
        rem0[i] += d + 1;
      } else if (rank[i] > d) {
        rank[i] -= d + 1;
        rem0[i] -= d + 1;
      }
    }

    // If do_visualization is true, fill barycentric weights with 1.0
    // Otherwise, comptue the barycentric coordinates (p.10 in [Adams et al.
    // 2010])
    if (do_visualization) {
      for (int i = 0; i <= d + 1; i++) {
        barycentric[i] = 1.0;
      }
    } else {
      for (int i = 0; i <= d + 1; i++) barycentric[i] = 0;
      for (int i = 0; i <= d; i++) {
        float v = (elevated[i] - rem0[i]) * down_factor;

        if (d - rank[i] < 0 || d - rank[i] + 1 >= d + 2)
          throw std::runtime_error("Permutohedral: rank access error");

        // assert(d_-rank[i]   >= 0);
        // assert(d_-rank[i]+1 <  d_+2);
        barycentric[d - rank[i]] += v;
        barycentric[d - rank[i] + 1] -= v;
      }
      // Wrap around
      barycentric[0] += 1.0 + barycentric[d + 1];
    }

    // Compute all vertices and their offset
    std::vector<boost::int16_t> neighborKeyUp(d + 1);
    std::vector<boost::int16_t> neighborKeyDown(d + 1);
    for (int remainder = 0; remainder <= d; remainder++) {
      for (int i = 0; i < d; i++)
        key[i] = rem0[i] + canonical[remainder * (d + 1) + rank[i]];
      assert(k * (d + 1) + remainder < (d + 1) * N);
      lattice->offset_[k * (d + 1) + remainder] =
        hash_table.find(key.data(), true);
      lattice->barycentric_[k * (d + 1) + remainder] = barycentric[remainder];

      // Gather the extent statistics of the lattice.
      for (int j = 0; j < d; ++j) {
        min_key[j] = (std::min)(key[j], min_key[j]);
        max_key[j] = (std::max)(key[j], max_key[j]);
      }
    }
  }

  // Find the Neighbors of each lattice point
  // Get the number of vertices in the lattice
  const int M = hash_table.size();
  lattice->M_ = M;

  // Gather some debug information.
  std::ostringstream extent_string;
  for (int i = 0; i < d; ++i) {
    extent_string << (max_key[i] - min_key[i]) << ", ";
  }

  // LOG(INFO) << "lattice size: " << M
  //           << ", samples: " << N
  //           << ", mean occupancy: " << static_cast<float>(N * (d+1)) /
  //           M
  //           << ", extent: " << extent_string.str();

  // Create the neighborhood structure
  // blur_neighbors (filter_size-1) x M_ row-major
  const int size = get_filter_size(lattice->neighborhood_size_, d);
  lattice->blur_neighbors_.resize((size - 1) * M);

  std::vector<boost::int16_t> start_key(d + 1);
  std::vector<boost::int16_t> walking_key(d + 1);

  //  extract (d+1) x M matrix of immediate neighbour indices row-major
  std::vector<int> immediate_neighbors((d + 1) * M);
  for (int i = 0; i < M; ++i) {
    const boost::int16_t* key = hash_table.getKey(i);
    for (int dim = 0; dim <= d; ++dim) {
      std::copy(key, key + d + 1, walking_key.begin());
      advance_in_dimension(dim, 1, &walking_key);
      immediate_neighbors[i + M * dim] =
        hash_table.find(walking_key.data(), false);
    }
  }
  assert(immediate_neighbors.size() == (M * (d + 1)));

  // Lattice traversal using immediate neighbour indices.
  LatticeApproximateTraversal traverse(lattice->neighborhood_size_, d,
                                       immediate_neighbors, M);
  for (int i = 0; i < M; ++i) {
    int* neighbors = &lattice->blur_neighbors_[i];
    int n          = -1;
    NeighborhoodCallback yield(M, neighbors, &n);
    traverse.go(i, yield);
    assert(n + 1 == size);
  }
}

boost::shared_ptr<PermutohedralReverse> Permutohedral::compute(
  const float* filter,
  const float* in,
  int num_output,
  int group,
  int value_size,
  bool do_skip_blur,
  int in_offset,
  int out_offset,
  int in_size,
  int out_size,
  float* out) const {
  // Setup blur operation. This op will be returned to be able to compute the
  // gradient later.
  // TODO(mkiefel): probably move to some kind of constructor or init.
  boost::shared_ptr<PermutohedralReverse> reverse_operation(
    new PermutohedralReverse());

  reverse_operation->init(filter, num_output, group, value_size, do_skip_blur,
                          in_offset, out_offset, in_size, out_size, lattice_);
  reverse_operation->compute(in, out);

  return reverse_operation;
}

boost::shared_ptr<PermutohedralReverse> Permutohedral::max_compute(
  const float* filter,
  const float* in,
  int value_size,
  int in_offset,
  int out_offset,
  int in_size,
  int out_size,
  float* out) {
  // Setup max operation. This op will be returned to be able to compute the
  // gradient later.
  // TODO(mkiefel): probably move to some kind of constructor or init.
  boost::shared_ptr<PermutohedralReverse> reverse_operation(
    new PermutohedralReverse());

  reverse_operation->init(filter, value_size, value_size, value_size, false,
                          in_offset, out_offset, in_size, out_size, lattice_);
  reverse_operation->max_compute(in, out);

  return reverse_operation;
}

/************************************************/
/***           PermutohedralReverse           ***/
/************************************************/

/*! \brief This class has functionality for 'forward' and 'reverse'
 *         permutohedral operations for computations specific to CPU.
 *
 */

void PermutohedralReverse::reverse(const float* diff_in,
                                   float* diff_out_filter,
                                   float* diff_out_in) {
  std::vector<float> sliced_out;

  slice_tick(diff_in, &sliced_out);

  std::vector<float> blurred_out;
  if (do_skip_blur_) {
    blurred_out.resize(value_size_ * (M_ + 1), 0);
    std::fill(blurred_out.begin(), blurred_out.end(), 0);
    for (int t = 0; t < M_; ++t) {
      for (int k = 0; k < value_size_; ++k) {
        blurred_out[k * (M_ + 1) + t + 1] += sliced_out[k * M_ + t];
      }
    }
  } else {
    blur_tick(sliced_out, &blurred_out, diff_out_filter);
  }

  splat_tick(blurred_out, diff_out_in);
}

void PermutohedralReverse::max_reverse(const float* diff_in,
                                       float* diff_out_in) {
  std::vector<float> sliced_out;

  slice_tick(diff_in, &sliced_out);

  std::vector<float> blurred_out;
  max_tick(sliced_out, &blurred_out);

  splat_tick(blurred_out, diff_out_in);
}

// Only Permutohedral initializes this.
PermutohedralReverse::PermutohedralReverse() {}

void PermutohedralReverse::init(
  const float* filter,
  int num_output,
  int group,
  int value_size,
  bool do_skip_blur,
  int in_offset,
  int out_offset,
  int in_size,
  int out_size,
  const boost::shared_ptr<const typename Permutohedral::Lattice> lattice) {
  lattice_           = lattice;
  d_                 = lattice->d_;
  N_                 = lattice->N_;
  neighborhood_size_ = lattice->neighborhood_size_;
  M_                 = lattice->M_;

  in_offset_  = in_offset;
  out_offset_ = out_offset;
  in_size_    = in_size;
  out_size_   = out_size;

  const int size = get_filter_size(neighborhood_size_, d_);

  filter_.resize(((num_output * value_size) / group) * size);
  std::copy(filter, filter + filter_.size(), filter_.begin());

  num_output_ = num_output;
  group_      = group;
  value_size_ = value_size;

  do_skip_blur_ = do_skip_blur;

  splatted_.resize((M_ + 1) * value_size, 0);

  max_idx_.resize(M_ * value_size, 0);
}

void PermutohedralReverse::compute(const float* in, float* out) {
  splat(in, &splatted_);

  std::vector<float> blurred_(M_ * num_output_);

  if (do_skip_blur_) {
    for (int t = 0; t < M_; ++t) {
      for (int k = 0; k < num_output_; ++k) {
        blurred_[k * M_ + t] = splatted_[k * (M_ + 1) + t + 1];
      }
    }
  } else {
    blur(splatted_, filter_.data(), &blurred_);
  }

  slice(blurred_, out);
}

void PermutohedralReverse::max_compute(const float* in, float* out) {
  std::vector<float> blurred_(M_ * num_output_);
  splat(in, &splatted_);
  max(splatted_, &blurred_);
  slice(blurred_, out);
}

void PermutohedralReverse::slice(const std::vector<float>& data,
                                 float* sliced) const {
  // data           num_output x M_                               row-major
  // sliced         num_output x out_size                         row-major

  for (int i = 0; i < static_cast<int>(out_size_); i++) {
    for (int k = 0; k < static_cast<int>(num_output_); k++) {
      sliced[k * out_size_ + i] = 0;
    }

    for (int j = 0; j <= d_; j++) {
      int o   = lattice_->offset_[(out_offset_ + i) * (d_ + 1) + j];
      float w = lattice_->barycentric_[(out_offset_ + i) * (d_ + 1) + j];

      for (int k = 0; k < static_cast<int>(num_output_); k++) {
        sliced[k * out_size_ + i] += w * data[k * M_ + o];
      }
    }
  }
}

void PermutohedralReverse::blur(const std::vector<float>& splatted,
                                const float* filter,
                                std::vector<float>* blurred) const {
  // filter         num_output x value_size / group x filter_size row-major
  // splatted       value_size x (M_+1)                           row-major
  // blur_neighbors filter_size x M_                              row-major
  // blurred        num_output x M_                               row-major

  const int size = get_filter_size(neighborhood_size_, d_);

  const std::size_t M = num_output_ / group_;
  const std::size_t K = value_size_ / group_ * size;
  const std::size_t N = M_;

  const std::size_t max_size = 1024 * 1024 * 200;
  const std::size_t chunk_size =
    std::max<std::size_t>(1, std::min<std::size_t>(max_size / K, N));
  const std::size_t chunks = std::ceil(static_cast<double>(N) / chunk_size);

  std::vector<float> col_data(K * chunk_size);

  // number of filter parameters in a group
  const std::size_t filter_offset = M * K;
  // number of values in an output region / column
  const std::size_t top_offset = M * N;

  for (std::size_t g = 0; g < group_; ++g) {
    for (std::size_t c = 0; c < chunks; ++c) {
      const std::size_t start = c * chunk_size;
      const std::size_t end   = std::min<std::size_t>(N, start + chunk_size);

      im2col(splatted.data() + (value_size_ / group_) * g * (N + 1),
             value_size_ / group_, size, M_, start, end,
             lattice_->blur_neighbors_, col_data.data());

      ::cpu_gemm_ex(CblasNoTrans, CblasNoTrans, M, end - start, K,
                    static_cast<float>(1), filter + filter_offset * g, K,
                    col_data.data(), end - start, static_cast<float>(0),
                    blurred->data() + top_offset * g + chunk_size * c, N);
    }
  }
}

void PermutohedralReverse::max(const std::vector<float>& splatted,
                               std::vector<float>* maxxed) {
  // splatted       value_size x (M_+1)                           row-major
  // blur_neighbors filter_size x M_                              row-major
  // maxxed         num_output x M_                               row-major

  const int filter_size = get_filter_size(neighborhood_size_, d_);

  const float* data = splatted.data();

  for (std::size_t i = 0; i < value_size_; ++i) {
    for (std::size_t j = 0; j < M_; ++j) {
      int idx         = j;
      float max_value = data[i * (M_ + 1) + idx + 1];
      for (std::size_t k = 1; k < filter_size; ++k) {
        const int* neighbors = &lattice_->blur_neighbors_[(k - 1) * M_];
        float value          = data[i * (M_ + 1) + neighbors[j] + 1];
        if (value > max_value) {
          max_value = value;
          idx       = neighbors[j];
        }
      }
      (*maxxed)[i * M_ + j] = max_value;
      max_idx_[i * M_ + j]  = i * (M_ + 1) + idx + 1;
    }
  }
}

void PermutohedralReverse::splat(const float* in,
                                 std::vector<float>* splatted) const {
  // in             value_size x in_size                          row-major
  // splatted       value_size x (M_+1)                           row-major
  std::fill(splatted->begin(), splatted->end(), 0);
  for (int i = 0; i < in_size_; i++) {
    for (int j = 0; j <= d_; j++) {
      int o          = lattice_->offset_[(in_offset_ + i) * (d_ + 1) + j] + 1;
      const float& w = lattice_->barycentric_[(in_offset_ + i) * (d_ + 1) + j];

      for (int k = 0; k < static_cast<int>(value_size_); k++) {
        (*splatted)[k * (M_ + 1) + o] += w * in[k * in_size_ + i];
      }
    }
  }
}

void PermutohedralReverse::im2col(const float* im,
                                  const std::size_t value_size,
                                  const std::size_t filter_size,
                                  const std::size_t M,
                                  const std::size_t start,
                                  const std::size_t end,
                                  const std::vector<int>& blur_neighbors,
                                  float* col) {
  // im             value_size      x (M_+1) row-major blur_neighbors
  // (filter_size-1) x M_                              row-major col
  // value_size x filter_size x (end - start)          row-major

  const std::size_t output_size = end - start;

  for (std::size_t i = 0; i < output_size; ++i) {
    for (std::size_t k = 0; k < value_size; ++k) {
      col[(k * filter_size + 0) * output_size + i] =
        im[k * (M + 1) + (i + start + 1)];

      for (std::size_t f = 1; f < filter_size; ++f) {
        const int* neighbors = &blur_neighbors[(f - 1) * M + 0];

        col[(k * filter_size + f) * output_size + i] =
          im[k * (M + 1) + (neighbors[i + start] + 1)];
      }
    }
  }
}

void PermutohedralReverse::col2im(const float* col,
                                  const std::size_t value_size,
                                  const std::size_t filter_size,
                                  const std::size_t M,
                                  const std::size_t start,
                                  const std::size_t end,
                                  const std::vector<int>& blur_neighbors,
                                  float* im) {
  // col            value_size x filter_size x (end - start) row-major
  // blur_neighbors (filter_size-1) x M_ row-major im             value_size
  // x (M_+1)                          row-major

  const std::size_t output_size = end - start;

  for (std::size_t i = 0; i < output_size; ++i) {
    for (std::size_t k = 0; k < value_size; ++k) {
      im[k * (M + 1) + (i + start + 1)] +=
        col[(k * filter_size + 0) * output_size + i];

      for (std::size_t f = 1; f < filter_size; ++f) {
        const int* neighbors = &blur_neighbors[(f - 1) * M + 0];

        im[k * (M + 1) + (neighbors[i + start] + 1)] +=
          col[(k * filter_size + f) * output_size + i];
      }
    }
  }
}

void PermutohedralReverse::slice_tick(const float* sliced_tick,
                                      std::vector<float>* sliced_out) const {
  // sliced_tick        num_output x out_size row-major sliced_out num_output
  // x M_                               row-major
  sliced_out->resize(num_output_ * M_);
  std::fill(sliced_out->begin(), sliced_out->end(), 0);

  for (int i = 0; i < static_cast<int>(out_size_); ++i) {
    for (int j = 0; j <= d_; ++j) {
      int o   = lattice_->offset_[(out_offset_ + i) * (d_ + 1) + j];
      float w = lattice_->barycentric_[(out_offset_ + i) * (d_ + 1) + j];

      for (int k = 0; k < static_cast<int>(num_output_); ++k) {
        (*sliced_out)[k * M_ + o] += w * sliced_tick[k * out_size_ + i];
      }
    }
  }
}

void PermutohedralReverse::blur_tick(const std::vector<float>& blurred_tick,
                                     std::vector<float>* blurred_out,
                                     float* filter_out) {
  // filter_        num_output x value_size / group x filter_size row-major
  // blurred_out    value_size x (M_+1)                           row-major
  // blur_neighbors filter_size x M_                              row-major
  // blurred_tick   num_output x M_                               row-major
  blurred_out->resize(value_size_ * (M_ + 1));
  std::fill(blurred_out->begin(), blurred_out->end(), 0);

  const int size = get_filter_size(neighborhood_size_, d_);

  const std::size_t M = num_output_ / group_;
  const std::size_t K = value_size_ / group_ * size;
  const std::size_t N = M_;

  const std::size_t max_size = 1024 * 1024 * 200 / 2;
  const std::size_t chunk_size =
    std::max<std::size_t>(1, std::min<std::size_t>(max_size / K, N));
  const std::size_t chunks = std::ceil(static_cast<double>(N) / chunk_size);

  std::vector<float> col_data(K * chunk_size);

  // number of filter parameters in a group
  const std::size_t filter_offset = M * K;
  // number of values in an output region / column
  const std::size_t top_offset = M * N;

  std::vector<float> col_diff(K * chunk_size);

  for (std::size_t g = 0; g < group_; ++g) {
    for (std::size_t c = 0; c < chunks; ++c) {
      const std::size_t start = c * chunk_size;
      const std::size_t end   = std::min<std::size_t>(N, start + chunk_size);

      im2col(splatted_.data() + (value_size_ / group_) * g * (N + 1),
             value_size_ / group_, size, M_, start, end,
             lattice_->blur_neighbors_, col_data.data());

      // Gradient w.r.t. filter.
      ::cpu_gemm_ex(CblasNoTrans, CblasTrans, M, K, end - start,
                    static_cast<float>(1),
                    blurred_tick.data() + top_offset * g + chunk_size * c, N,
                    col_data.data(), end - start, static_cast<float>(1),
                    filter_out + filter_offset * g, K);

      // Gradient w.r.t. data.
      ::cpu_gemm_ex(CblasTrans, CblasNoTrans, K, end - start, M,
                    static_cast<float>(1), filter_.data() + filter_offset * g,
                    K, blurred_tick.data() + top_offset * g + chunk_size * c, N,
                    static_cast<float>(0), col_diff.data(), end - start);

      col2im(col_diff.data(), value_size_ / group_, size, M_, start, end,
             lattice_->blur_neighbors_,
             blurred_out->data() + (value_size_ / group_) * g * (N + 1));
    }
  }
}

void PermutohedralReverse::max_tick(const std::vector<float>& maxxed_tick,
                                    std::vector<float>* maxxed_out) {
  // filter_        num_output x value_size / group x filter_size row-major
  // maxxed_out    value_size x (M_+1)                           row-major
  // blur_neighbors filter_size x M_                              row-major
  // maxxed_tick   value_size x M_                               row-major
  maxxed_out->resize(value_size_ * (M_ + 1));
  std::fill(maxxed_out->begin(), maxxed_out->end(), 0);

  const int filter_size = get_filter_size(neighborhood_size_, d_);

  const float* tick_data = maxxed_tick.data();

  for (std::size_t i = 0; i < value_size_; ++i) {
    // Looping over variables
    for (std::size_t j = 0; j < M_; ++j) {
      // Looping only over the neighbors
      if (max_idx_[i * M_ + j] == i * (M_ + 1) + j + 1) {
        (*maxxed_out)[i * (M_ + 1) + j + 1] += tick_data[i * M_ + j];
      }
      for (std::size_t k = 1; k < filter_size; ++k) {
        const int* neighbors = &lattice_->blur_neighbors_[(k - 1) * M_];
        if (max_idx_[i * M_ + j] == i * (M_ + 1) + neighbors[j] + 1) {
          (*maxxed_out)[i * (M_ + 1) + neighbors[j] + 1] +=
            tick_data[i * M_ + j];
        }
      }
    }
  }
}

void PermutohedralReverse::splat_tick(const std::vector<float>& splatted_tick,
                                      float* splatted_out) {
  // splatted_tick  value_size x (M_+1)                           row-major
  // splatted_out   value_size x in_size                          row-major
  for (int i = 0; i < in_size_; ++i) {
    for (int j = 0; j <= d_; ++j) {
      int o          = lattice_->offset_[(in_offset_ + i) * (d_ + 1) + j] + 1;
      const float& w = lattice_->barycentric_[(in_offset_ + i) * (d_ + 1) + j];

      for (int k = 0; k < value_size_; k++) {
        splatted_out[k * in_size_ + i] += w * splatted_tick[k * (M_ + 1) + o];
      }
    }
  }
}

#endif /* PERMUTOHEDRAL_HPP */

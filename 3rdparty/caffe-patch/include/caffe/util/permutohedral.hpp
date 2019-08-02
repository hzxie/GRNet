// Copyright 2016 Max Planck Society
// Distributed under the BSD-3 Software license,
// (See accompanying file ../../../../LICENSE.txt or copy at
// https://opensource.org/licenses/BSD-3-Clause)

#ifndef CAFFE_UTIL_PERMUTOHEDRAL_H_
#define CAFFE_UTIL_PERMUTOHEDRAL_H_

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "boost/array.hpp"
#include "boost/cstdint.hpp"
#include "boost/shared_ptr.hpp"

#include "caffe/blob.hpp"

namespace caffe {

namespace permutohedral {

/************************************************/
/***         Some Utility Functions           ***/
/************************************************/

int get_filter_size(int neighborhood_size, int feature_size);

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
  void go(const key_type& start_key, TFun yield) const;

 private:
  template <typename TFun>
  void walk_cuboid(const key_type& start_key,
                   const int d,
                   const bool has_zero,
                   std::vector<key_type>& walking_keys,
                   TFun yield) const;

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
  void go(const int start, TFun yield) const;

 private:
  template <typename TFun>
  void walk_approximate(const int start,
                        const int d,
                        const bool has_zero,
                        TFun yield) const;

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

template <typename T>
class GaussianFilter {
 public:
  typedef T value_type;

  GaussianFilter(int neighborhood_size, int feature_size)
    : neighborhood_size_(neighborhood_size), feature_size_(feature_size) {
    build_filter();
  }

  const value_type* filter() { return filter_.data(); }

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

  void build_filter();

  int neighborhood_size_;
  int feature_size_;
  std::vector<value_type> filter_;
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

template <typename T>
class Permutohedral;

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

template <typename T>
class PermutohedralReverse {
 public:
  typedef T value_type;

  virtual ~PermutohedralReverse() {}

  virtual void reverse(const value_type* diff_in,
                       value_type* diff_out_filter,
                       value_type* diff_out_in) = 0;

  virtual void max_reverse(const value_type* diff_in,
                           value_type* diff_out_in) = 0;
};

template <typename T>
class Permutohedral {
 public:
  typedef T value_type;
  typedef GaussianFilter<value_type> gauss_type;

  struct Lattice {
    int N_, d_;
    int neighborhood_size_;
    int M_;
    std::vector<value_type> barycentric_;
    std::vector<int> offset_;
    std::vector<int> blur_neighbors_;
  };

 private:
  // don't copy
  Permutohedral(const Permutohedral& rhs);

  boost::shared_ptr<const Lattice> lattice_;

  bool check_unique_neighbors(const int* neighbors);

  static void map_back(const std::vector<boost::int16_t>& key,
                       value_type* const x);

 public:
  Permutohedral() {}

  static int get_filter_size(int neighborhood_size, int feature_size);

  void init(const value_type* feature,
            int data_count,
            int feature_size,
            int neighborhood_size,
            bool do_visualization);

  boost::shared_ptr<PermutohedralReverse<value_type> > compute(
    const value_type* filter,
    const value_type* in,
    int num_output,
    int group,
    int value_size,
    bool do_skip_blur,
    int in_offset,
    int out_offset,
    int in_size,
    int out_size,
    value_type* out) const;

  boost::shared_ptr<PermutohedralReverse<value_type> > max_compute(
    const value_type* filter,
    const value_type* in,
    int value_size,
    int in_offset,
    int out_offset,
    int in_size,
    int out_size,
    value_type* out);

  boost::shared_ptr<PermutohedralReverse<value_type> > compute_gpu(
    const value_type* filter,
    const value_type* in,
    int num_output,
    int group,
    int value_size,
    bool do_skip_blur,
    int in_offset,
    int out_offset,
    int in_size,
    int out_size,
    value_type* out) const;

  boost::shared_ptr<PermutohedralReverse<value_type> > max_compute_gpu(
    const value_type* filter,
    const value_type* in,
    int value_size,
    int in_offset,
    int out_offset,
    int in_size,
    int out_size,
    value_type* out);
};

/************************************************/
/***         PermutohedralReverseCpu          ***/
/************************************************/

/*! \brief This class has functionality for 'forward' and 'reverse'
 *         permutohedral operations for computations specific to CPU.
 *
 */

template <typename T>
class PermutohedralReverseCpu : public PermutohedralReverse<T> {
 public:
  typedef T value_type;

  void reverse(const value_type* diff_in,
               value_type* diff_out_filter,
               value_type* diff_out_in);

  void max_reverse(const value_type* diff_in, value_type* diff_out_in);

 private:
  // don't copy
  PermutohedralReverseCpu(const PermutohedralReverseCpu& rhs);

  // Only Permutohedral initializes this.
  PermutohedralReverseCpu() {}

  void init(
    const value_type* filter,
    int num_output,
    int group,
    int value_size,
    bool do_skip_blur,
    int in_offset,
    int out_offset,
    int in_size,
    int out_size,
    const boost::shared_ptr<const typename Permutohedral<value_type>::Lattice>
      lattice);

  void compute(const value_type* in, value_type* out);

  void max_compute(const value_type* in, value_type* out);

  void slice(const std::vector<value_type>& data, value_type* sliced) const;

  void blur(const std::vector<value_type>& splatted,
            const value_type* filter,
            std::vector<value_type>* blurred) const;

  void max(const std::vector<value_type>& splatted,
           std::vector<value_type>* maxxed);

  void splat(const value_type* in, std::vector<value_type>* splatted) const;

  static void im2col(const value_type* im,
                     const std::size_t value_size,
                     const std::size_t filter_size,
                     const std::size_t M,
                     const std::size_t start,
                     const std::size_t end,
                     const std::vector<int>& blur_neighbors,
                     value_type* col);

  static void col2im(const value_type* col,
                     const std::size_t value_size,
                     const std::size_t filter_size,
                     const std::size_t M,
                     const std::size_t start,
                     const std::size_t end,
                     const std::vector<int>& blur_neighbors,
                     value_type* im);

  void slice_tick(const value_type* sliced_tick,
                  std::vector<value_type>* sliced_out) const;

  void blur_tick(const std::vector<value_type>& blurred_tick,
                 std::vector<value_type>* blurred_out,
                 value_type* filter_out);

  void max_tick(const std::vector<value_type>& maxxed_tick,
                std::vector<value_type>* maxxed_out);

  void splat_tick(const std::vector<value_type>& splatted_tick,
                  value_type* splatted_out);

  std::vector<value_type> filter_;

  std::vector<value_type> splatted_;

  int d_, N_;
  int neighborhood_size_;
  int M_;
  std::vector<int> max_idx_;
  boost::shared_ptr<const typename Permutohedral<value_type>::Lattice> lattice_;

  int in_offset_, out_offset_, in_size_, out_size_;
  int num_output_, group_;
  int value_size_;

  bool do_skip_blur_;

  friend class Permutohedral<value_type>;
};

/************************************************/
/***         PermutohedralReverseGpu          ***/
/************************************************/

/*! \brief This class has functionality for 'forward' and 'reverse'
 *         permutohedral operations for computations specific to GPU.
 *
 */

template <typename T>
class PermutohedralReverseGpu : public PermutohedralReverse<T> {
 public:
  typedef T value_type;

  void reverse(const value_type* diff_in,
               value_type* diff_out_filter,
               value_type* diff_out_in);

  void max_reverse(const value_type* diff_in, value_type* diff_out_in);

 private:
  // don't copy
  PermutohedralReverseGpu(const PermutohedralReverseGpu& rhs);

  // Only Permutohedral initializes this.
  PermutohedralReverseGpu() {}

  void init(
    const value_type* filter,
    int num_output,
    int group,
    int value_size,
    bool do_skip_blur,
    int in_offset,
    int out_offset,
    int in_size,
    int out_size,
    const boost::shared_ptr<const typename Permutohedral<value_type>::Lattice>
      lattice);

  void compute(const value_type* in, value_type* out);
  void max_compute(const value_type* in, value_type* out);

  void slice(const Blob<value_type>& data, value_type* sliced) const;

  void blur(const Blob<value_type>& splatted,
            const Blob<value_type>& filter,
            Blob<value_type>* blurred) const;

  void max(const Blob<value_type>& splatted, Blob<value_type>* maxxed);

  void splat(const value_type* in, Blob<value_type>* splatted) const;

  static void im2col(const value_type* im,
                     const std::size_t value_size,
                     const std::size_t filter_size,
                     const std::size_t M,
                     const std::size_t start,
                     const std::size_t end,
                     const int* blur_neighbors,
                     value_type* col);

  static void col2im(const value_type* col,
                     const std::size_t value_size,
                     const std::size_t filter_size,
                     const std::size_t M,
                     const std::size_t start,
                     const std::size_t end,
                     const int* blur_neighbors,
                     value_type* im);

  void slice_tick(const value_type* sliced_tick,
                  Blob<value_type>* sliced_out) const;

  void blur_tick(const Blob<value_type>& blurred_tick,
                 Blob<value_type>* blurred_out,
                 value_type* filter_out);

  void max_tick(const Blob<value_type>& maxxed_tick,
                Blob<value_type>* maxxed_out);

  void splat_tick(const Blob<value_type>& splatted_tick,
                  value_type* splatted_out);

  Blob<value_type> filter_;

  Blob<value_type> splatted_;

  int d_, N_;
  int neighborhood_size_;
  int M_;
  Blob<int> max_idx_;

  Blob<value_type> barycentric_;
  Blob<int> offset_;
  Blob<int> blur_neighbors_;

  int in_offset_, out_offset_, in_size_, out_size_;
  int num_output_, group_;
  int value_size_;

  bool do_skip_blur_;

  friend class Permutohedral<value_type>;
};

}  // namespace permutohedral

}  // namespace caffe

#endif /* CAFFE_UTIL_PERMUTOHEDRAL_H */

#ifndef KMEANS_H
#define KMEANS_H

#include <stdint.h>
#include <stdlib.h>
#include <float.h>

typedef double value_t;
#define VALUE_MAX DBL_MAX

static inline void *
alloc_rm2(size_t x, size_t y) {
  value_t **array =
    (value_t **)malloc(sizeof(value_t *) * x + sizeof(value_t) * x * y);
  for (size_t i = 0; i < x; ++i)
    array[i] = (value_t *)array + x + i * y;

  return array;
}

static inline void
free_rm2(void *array) {
  value_t **arr = (value_t **) array;
  free(arr);
}

inline value_t euclid_dist_2(value_t *pt1, value_t *pt2, size_t dimension) {
  value_t ans = 0;
  for (size_t i = 0; i < dimension; ++i) {
    value_t diff = pt1[i] - pt2[i];
    ans += diff * diff;
  }
  return ans;
}

inline int find_nearest_point(value_t *pt,
                              size_t nfeatures,
                              value_t **pts,
                              size_t npts) {
  size_t index = 0;
  value_t max_dist = VALUE_MAX;

  // Find the cluster center id with min distance to pt.
  for (int i = 0; i < npts; ++i) {
    value_t dist = euclid_dist_2(pt, pts[i], nfeatures);
    if (dist < max_dist) {
      max_dist = dist;
      index = i;
    }
  }

  return index;
}

#endif


#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include <omp.h>

#include <kmeans.h>

#define min(x, y) (((x) < (y)) ? (x) : (y))

void kmeans(size_t nclusters,
            value_t *attributes,
            size_t nattributes,
            size_t nobjects,
            int is_random,
            size_t niterations,
            size_t nthreads) {
  omp_set_num_threads(nthreads);

  value_t **clusters = alloc_rm2(nclusters, nattributes);

  // Randomly pick the cluster centers.
  for (size_t i = 0; i < nclusters; ++i) {
    size_t index = rand() % nobjects;
    for (size_t j = 0; j < nattributes; ++j) {
      clusters[i][j] = attributes[index * nattributes + j];
    }
  }

  value_t ***partial_new_centers = malloc(sizeof(value_t **) * nthreads);
  size_t **partial_new_center_len = malloc(sizeof(size_t *) * nthreads);

  #pragma omp parallel for
  for (size_t i = 0; i < nthreads; ++i) {
    partial_new_centers[i] = alloc_rm2(nclusters, nattributes);
    partial_new_center_len[i] = calloc(nclusters, sizeof(size_t));
  }

  printf("[Info] Entering kernel...\n");
  struct timeval start_time;
  gettimeofday(&start_time, NULL);

  #pragma omp parallel
  {
    const size_t tid = omp_get_thread_num();

    value_t **local_centers = partial_new_centers[tid];
    size_t *local_center_len = partial_new_center_len[tid];

    for (size_t iter = 0; iter < niterations; ++iter) {
      // Initialize the local storage.
      for (size_t i = 0; i < nclusters; ++i) {
        local_center_len[i] = 0;
        for (size_t j = 0; j < nattributes; ++j) {
          local_centers[i][j] = 0;
        }
      }

      #pragma omp for
      for (size_t i = 0; i < nobjects; ++i) {
        // Find the index of the nearest cluster centers.
        size_t nearest = find_nearest_point(
            &attributes[i * nattributes], nattributes, clusters, nclusters);

        // Update new cluster centers: sum of all objects located within.
        local_center_len[nearest]++;
        for (size_t j = 0; j < nattributes; ++j) {
          local_centers[nearest][j] += attributes[i * nattributes + j];
        }
      }

      // Perform reduction at the master core.
      #pragma omp single
      for (size_t i = 0; i < nclusters; ++i) {
        for (size_t j = 0; j < nattributes; ++j) {
          value_t sum = 0;
          size_t length = 0;

          for (size_t k = 0; k < nthreads; ++k) {
            sum += partial_new_centers[k][i][j];
            length += partial_new_center_len[k][i];
          }

          if (length > 0) {
            clusters[i][j] = sum / length;
          } else {
            clusters[i][j] = 0;
          }
        }
      }
    }
  }

  struct timeval end_time;
  gettimeofday(&end_time, NULL);
  printf("[Info] Exiting kernel...\n");

  struct timeval diff_time;
  timersub(&end_time, &start_time, &diff_time);
  printf("[Info] Runtime = %lf s\n", diff_time.tv_sec + diff_time.tv_usec * 1e-6);

  #pragma omp parallel for
  for (size_t i = 0; i < nthreads; ++i) {
    free_rm2(partial_new_centers[i]);
    free(partial_new_center_len[i]);
  }

  free_rm2(clusters);

  free(partial_new_centers);
  free(partial_new_center_len);
}

int main(int argc, char *argv[]) {
  const char *filename = NULL;
  size_t nclusters = 8;
  size_t nattributes = 4;
  size_t nobjects = 256;
  size_t niterations = 10;
  size_t nthreads = 1;
  size_t nchunks = 1;

  int opt;
  while ((opt = getopt(argc, argv, "f:n:k:d:c:i:p:t:h")) != -1) {
    switch (opt) {
      case 'f':
        filename = optarg;
        break;
      case 'n':
        nchunks = atoi(optarg);
        break;
      case 'k':
        nclusters = atoi(optarg);
        break;
      case 'd':
        nattributes = atoi(optarg);
        break;
      case 'c':
        nobjects = atoi(optarg);
        break;
      case 'i':
        niterations = atoi(optarg);
        break;
      case 'p':
        fprintf(stderr, "error: kmeans++ is not supported\n");
        return 1;
      case 't':
        nthreads = atoi(optarg);
        break;

      case 'h':
      default:
        fprintf(stderr, "usage: %s [-f filename] [-k clusters] "
                "[-d dimensionality] [-c cardinality] [-i iterations] "
                "[-t threads]\n", argv[0]);
        return 1;
    }
  }

  fprintf(stderr, "Determining %lu clusters on %lu rows with %lu attributes..\n"
          "Running %lu iterations with %lu threads..\n",
      nclusters, nobjects, nattributes, niterations, nthreads);

  value_t *attributes = alloc_rm2(nobjects, nattributes);

  // Simple NUMA optimization based on the first-touch policy.
  #pragma omp parallel for
  for (size_t i = 0; i < nobjects; ++i) {
    for (size_t j = 0; j < nattributes; ++j) {
      attributes[i * nattributes + j] = 0;
    }
  }

  if (filename) {
    fprintf(stderr, "Loading from file not implemented!\n");
    exit(-1);
  } else {
    // Deterministically initialize the attributes for validation.
    for (size_t i = 0; i < nobjects; i++) {
      for (size_t j = 0; j < nattributes; j++) {
        attributes[i * nattributes + j] = i % 1000;
      }
    }
  }

  kmeans(nclusters, attributes, nattributes, nobjects, filename == NULL,
         niterations, nthreads);

  free_rm2(attributes);

  return 0;
}


//
// Created by pravi on 21.10.2024.
//
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <pthread.h>
#include <string.h>
#include <sys/types.h>
#include <complex.h>

#include "mandelbrot.h"
#include "timer.h"

/* command line arguments:
 * [1] number of threads
 * [2] approximate number of points to collect
 */
#define MB_NUM_ARGS      3
#define MB_ITERATIONS    4000
#define MB_OUTPUT_FILE   "./mandelbrot_output.csv"

static uint64_t       g_requested_points;
static uint64_t       g_collected_points = 0;
static complex double *g_points_buffer = NULL;
pthread_mutex_t       g_mutex;

/* Returns 1 if the complex number c belongs to the Mandelbrot set,
 * 0 otherwise.
 */
static int mandelbrot_in_set(const complex double c) {
    complex double z = 0.0;

    for (uint16_t i = 0; i < MB_ITERATIONS; i++) {
        z = z * z + c;
        if (cabs(z) >= 2.0) {
            return 0;
        }
    }
    return 1;
}

/* Worker routine for a single thread.
 * Each thread scans its own segment on the real axis.
 */
static void *thread_worker(void *vargs) {
    const pthread_args_t *args = (pthread_args_t *)vargs;
    const double_t x_start = args->x_start;
    const double_t x_end   = args->x_end;

    const double_t step = 0.00015;

    for (double_t x = x_start; x < x_end; x += step) {
        for (double_t y = -1.0; y < 1.0; y += step) {
            const complex double c = x + y * I;

            if (!mandelbrot_in_set(c)) {
                continue;
            }

            pthread_mutex_lock(&g_mutex);

            if (g_collected_points < g_requested_points) {
                g_points_buffer[g_collected_points++] = c;
                pthread_mutex_unlock(&g_mutex);
            } else {
                pthread_mutex_unlock(&g_mutex);
                return NULL;
            }
        }

        if (g_collected_points >= g_requested_points) {
            break;
        }
    }

    return NULL;
}


int mandelbrot(const int argc, char *argv[]) {
    double start = 0.0;
    double finish = 0.0;

    if (argc != MB_NUM_ARGS) {
        fprintf(stderr, "Usage:\n%s [nthreads] [ntrials]\n", argv[0]);
        return -1;
    }

    const uint64_t thread_count = strtoull(argv[1], NULL, 10);
    g_requested_points = strtoull(argv[2], NULL, 10);

    const double_t x_segment_width = 4.0 / (double_t)thread_count;

    pthread_t *threads = malloc(thread_count * sizeof(pthread_t));
    pthread_args_t *thread_args_arr = malloc(thread_count * sizeof(pthread_args_t));
    g_points_buffer = malloc(g_requested_points * sizeof(complex double));

    if (threads == NULL || thread_args_arr == NULL || g_points_buffer == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        free(threads);
        free(thread_args_arr);
        free(g_points_buffer);
        return -1;
    }

    pthread_mutex_init(&g_mutex, NULL);

    /* Prepare arguments for each worker thread */
    for (uint64_t i = 0; i < thread_count; i++) {
        pthread_args_t *args = &thread_args_arr[i];
        args->tid = i;
        args->x_start = -2.0 + x_segment_width * (double_t)i;
        args->x_end   = args->x_start + x_segment_width;
    }

    GET_TIME(start);

    /* Launch worker threads */
    for (uint64_t i = 0; i < thread_count; i++) {
        const int err = pthread_create(&threads[i], NULL, thread_worker, &thread_args_arr[i]);
        if (err != 0) {
            fprintf(stderr, "Creating pthread #%llu failed (err=%d)\n", (unsigned long long)i, err);
        }
    }

    /* Wait for all threads to finish */
    for (uint64_t i = 0; i < thread_count; i++) {
        pthread_join(threads[i], NULL);
    }

    GET_TIME(finish);

    /* Save points into CSV file */
    FILE *file = fopen(MB_OUTPUT_FILE, "w");
    if (file != NULL) {
        for (uint64_t i = 0; i < g_collected_points; i++) {
            fprintf(file, "(%lf, %lf)\n", creal(g_points_buffer[i]), cimag(g_points_buffer[i]));
        }
        fclose(file);
    } else {
        fprintf(stderr, "Cannot open file %s\n", MB_OUTPUT_FILE);
    }

    printf("Check result in the %s\n", MB_OUTPUT_FILE);
    printf("Done in %lfs ( %llu threads, %llu requested points, %llu collected points )\n",
           finish - start,
           (unsigned long long)thread_count,
           (unsigned long long)g_requested_points,
           (unsigned long long)g_collected_points);

    pthread_mutex_destroy(&g_mutex);
    free(thread_args_arr);
    free(g_points_buffer);
    free(threads);

    return 0;
}

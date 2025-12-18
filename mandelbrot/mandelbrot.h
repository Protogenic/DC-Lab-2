//
// Created by pravi on 21.10.2024.
//

#ifndef MAND_SET_H
#define MAND_SET_H

#include <math.h>
#include <stdint.h>

typedef struct {
    uint64_t tid;

    double_t x_start;
    double_t x_end;
} pthread_args_t;

int mandelbrot(int argc, char *argv[]);
#endif //MAND_SET_H

#ifndef RWLOCK_H
#define RWLOCK_H

#include <pthread.h>
#include <stdint.h>


typedef struct
{
  pthread_mutex_t mutex;

  pthread_cond_t r_cv;
  pthread_cond_t w_cv;

  uint64_t rlock_wait_c;
  uint64_t wlock_wait_c;

  uint64_t r_locked_c;
  int8_t w_locked;
} my_rwlock_t;

void rdlock(my_rwlock_t* p);

void wrlock(my_rwlock_t* p);

void unlock(my_rwlock_t* p);

#endif
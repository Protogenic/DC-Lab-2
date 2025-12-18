#include <pthread.h>

#include "rwlock.h"

void rdlock(my_rwlock_t *p) {
    pthread_mutex_lock(&p->mutex);
    p->rlock_wait_c++;
    /*
     * Writer-preference to avoid starving writers:
     * - block readers if a writer holds the lock OR at least one writer is waiting
     */
    while (p->w_locked != 0 || p->wlock_wait_c > 0) {
        pthread_cond_wait(&p->r_cv, &p->mutex);
    }
    --p->rlock_wait_c;
    ++p->r_locked_c;
    pthread_mutex_unlock(&p->mutex);
}

void wrlock(my_rwlock_t *p) {
    pthread_mutex_lock(&p->mutex);
    ++p->wlock_wait_c;
    /*
     * Wait until there are no active readers and no active writer.
     * (Important: must always check BOTH conditions to avoid writer entering
     * while readers slip in between wake-up and lock acquisition.)
     */
    while (p->w_locked != 0 || p->r_locked_c > 0) {
        pthread_cond_wait(&p->w_cv, &p->mutex);
    }
    --p->wlock_wait_c;
    p->w_locked = 1;
    pthread_mutex_unlock(&p->mutex);
}

void unlock(my_rwlock_t *p) {
    pthread_mutex_lock(&p->mutex);
    if (p->w_locked) {
        p->w_locked = 0;
    } else if (p->r_locked_c) {
        --p->r_locked_c;
    }
    /*
     * Wake-up policy:
     * - If writers are waiting, let one writer proceed when possible.
     * - Otherwise, wake all waiting readers.
     */
    if (p->wlock_wait_c > 0) {
        if (p->w_locked == 0 && p->r_locked_c == 0) {
            pthread_cond_signal(&p->w_cv);
        }
    } else if (p->rlock_wait_c > 0) {
        pthread_cond_broadcast(&p->r_cv);
    }
    pthread_mutex_unlock(&p->mutex);
}

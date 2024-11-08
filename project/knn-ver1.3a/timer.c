#include "timer.h"


#if _WIN32
static double get_time_in_ms() {
    LARGE_INTEGER t, f;
    QueryPerformanceCounter(&t);
    QueryPerformanceFrequency(&f);
    return (double)t.QuadPart/(double)f.QuadPart * 1000;
}
#endif

#if __linux__
static double calc_delta_ms(struct timespec *this_instant, struct timespec *start_instant);
#endif

/*
 * Returns a new timer. Call this before anything else.
 * */
Timer *timer_init() {

    Timer *timer_ptr = malloc(sizeof(Timer));
    if (timer_ptr == NULL) {
        return NULL;
    }

#if __linux__
    timer_ptr->start_instant = malloc(sizeof(struct timespec));
    if (timer_ptr->start_instant == NULL) {
        return NULL;
    }
#endif

    timer_ptr->state = STOP;

    return timer_ptr;
}

/*
 * Destroys the timer, deallocating the memory and NULLing the pointer.
 * */
Timer *timer_destroy(Timer *timer) {

    if (timer != NULL) {

#if __linux__
        if (timer->start_instant != NULL) {
            free(timer->start_instant);
        }
#endif

        free(timer);
    }

    return NULL;
}

/*
 * This starts the timer. Used at the beginning and after pause. If used after stop, resets the timer.
 * */
void timer_start(Timer *timer) {

    switch (timer->state) {

        case STOP:
            timer->total_time = 0.0;
            timer->state = START;
#if _WIN32
            timer->start_instant = get_time_in_ms();
#endif
#if __linux__
            clock_gettime(CLOCK_MONOTONIC, timer->start_instant);
#endif
            break;
        case PAUSE:

            timer->state = START;
#if _WIN32
            timer->start_instant = get_time_in_ms();
#endif
#if __linux__
            clock_gettime(CLOCK_MONOTONIC, timer->start_instant);
#endif

            break;
        default:
            break;
    }
}


/*
 * This pauses the timer. The timer can be continued by calling start.
 * */
void timer_pause(Timer *timer) {

#if _WIN32
    double this_instant;
#endif
#if __linux__
    struct timespec this_instant;
#endif

    double delta;

    switch (timer->state) {

        case START:

#if _WIN32
            this_instant = get_time_in_ms();
            delta = this_instant - timer->start_instant;
#endif
#if __linux__
            clock_gettime(CLOCK_MONOTONIC, &this_instant);
            delta = calc_delta_ms(&this_instant, timer->start_instant);
#endif
            timer->total_time += delta;

            timer->state = PAUSE;
            break;
        default:
            break;
    }
}

/*
 * Stops the timer. If start is called after this, the timer will be restarted.
 * */
void timer_stop(Timer *timer) {

#if _WIN32
    double this_instant;
#endif
#if __linux__
    struct timespec this_instant;
#endif

    double delta;

    switch (timer->state) {
        case START:

#if _WIN32
            this_instant = get_time_in_ms();
            delta = this_instant - timer->start_instant;
#endif
#if __linux__
            clock_gettime(CLOCK_MONOTONIC, &this_instant);
            delta = calc_delta_ms(&this_instant, timer->start_instant);
#endif
            timer->total_time += delta;

            timer->state = STOP;
            break;
        case PAUSE:
            timer->state = STOP;
            break;
        default:
            break;
    }
}


/*
 * Returns the time elapsed in milliseconds.
 * */
double timer_get_ms(Timer *timer) {

#if _WIN32
    double this_instant;
#endif
#if __linux__
    struct timespec this_instant;
#endif
    double delta;

    switch (timer->state) {
        case START:

#if _WIN32
            this_instant = get_time_in_ms();
            delta = this_instant - timer->start_instant;
#endif
#if __linux__
            clock_gettime(CLOCK_MONOTONIC, &this_instant);
            delta = calc_delta_ms(&this_instant, timer->start_instant);
#endif

            return timer->total_time + delta;
        case STOP:
            return timer->total_time;
        case PAUSE:
            return timer->total_time;
        default:
            return 0;
    }
}

double timer_get_s(Timer *timer) {
    return timer_get_ms(timer) * 1e-3;
}

double timer_get_us(Timer *timer) {
    return timer_get_ms(timer) * 1e3;
}

#if __linux__

/*
 * Calculates the timer difference between two instants in milliseconds.
 * */
static double calc_delta_ms(struct timespec *this_instant, struct timespec *start_instant) {

    return (this_instant->tv_sec - start_instant->tv_sec) * 1e3 +
           (double) (this_instant->tv_nsec - start_instant->tv_nsec) * 1e-6;
}

#endif
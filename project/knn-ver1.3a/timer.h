#ifndef TIMER_H_
#define TIMER_H_

#include <stdlib.h>

#if _WIN32
#include <windows.h>
#endif

#if __linux__
#include <time.h>
#include <sys/time.h>
#endif


typedef enum {
    START, STOP, PAUSE
} timerstate_t;

typedef struct {

    /*
     * The timer state.
     * */
    timerstate_t state;

    /*
     * The start instant and the total time elapsed.
     * */
#if _WIN32
    double start_instant; // in ms
#endif
#if __linux__
    struct timespec *start_instant;
#endif

    double total_time;
} Timer;

Timer *timer_init(void);

Timer *timer_destroy(Timer *timer);

void timer_start(Timer *timer);

void timer_stop(Timer *timer);

void timer_pause(Timer *timer);

double timer_get_ms(Timer *timer);

double timer_get_s(Timer *timer);

double timer_get_us(Timer *timer);

#endif

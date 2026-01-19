/*
 * C++ library for Coupled Simulated Annealing.
 * (Patched for Windows/MinGW and OpenMP correctness)
 */

#ifndef CSA_H
#define CSA_H

#ifndef CSA_ITER_MONO
#define CSA_ITER_MONO 0
#endif

#include <cmath>
#include <omp.h>
#include <vector>
#include <cstdlib> // Added for rand()

namespace CSA {

template<typename Scalar_x, typename Scalar_fx>
class State
{
public:
    std::vector<Scalar_x> x;
    std::vector<Scalar_x> best_x;
    Scalar_fx cost;
    Scalar_fx best_cost;

    State(int n, const Scalar_x* x0, Scalar_fx fx0)
    {
        this->x = std::vector<Scalar_x>(x0, x0 + n);
        this->best_x = std::vector<Scalar_x>(x0, x0 + n);
        this->cost = fx0;
        this->best_cost = fx0;
    }

    inline void step(std::vector<Scalar_x> &y, Scalar_fx y_cost)
    {
        this->cost = y_cost;
        this->x.swap(y);
    }
};

template<typename Scalar_x, typename Scalar_fx>
class SharedStates
{
public:
    const int m;
    const int n;
    std::vector<State<Scalar_x, Scalar_fx>> states;

    State<Scalar_x, Scalar_fx>& operator[](int i) { return this->states[i]; }
    const State<Scalar_x, Scalar_fx>& operator[](int i) const { return this->states[i]; }

    SharedStates(int m, int n, const Scalar_x* x0, Scalar_fx fx0)
        : m(m), n(n), states(m, State<Scalar_x, Scalar_fx>(n, x0, fx0))
    {
    }
};

template<typename Scalar_x, typename Scalar_fx>
class Solver
{
public:
    int m = 4;
    int max_iterations = 1000000;
    float tgen_initial = 0.01;
    float tgen_schedule = 0.99999;
    float tacc_initial = 0.9;
    float tacc_schedule = 0.01;
    float desired_variance = 0.99;

    Solver() {  };

    inline int minimize(
        int n,
        Scalar_x* x,
        Scalar_fx (*fx)(void*, Scalar_x*),
        void (*step)(void*, Scalar_x* y, const Scalar_x*, float tgen),
        void (*progress)(void*, Scalar_fx cost, float tgen, float tacc, int opt_id, int iter),
        void* instance)
    {
        Scalar_fx fx0 = fx(instance, x);

        SharedStates<Scalar_x, Scalar_fx> shared_states(this->m, n, x, fx0);
        float tacc = this->tacc_initial;
        float tgen = this->tgen_initial;
        float tmp = 0.0, sum_a = 0.0, prob_var = 0.0, gamma = m;

        omp_lock_t lock;
        omp_init_lock(&lock);

        // FIXED: Added missing variables to shared() list
        #pragma omp parallel shared(n, shared_states, tacc, tgen, gamma, lock, step, instance, fx, progress, sum_a, prob_var, tmp) num_threads(this->m) default(none)
        {
            int k, opt_id = omp_get_thread_num();

            Scalar_fx max_cost = shared_states[0].cost;
            Scalar_fx cost;
            std::vector<Scalar_x> y(n, Scalar_x(0));
            float unif, prob;

#if CSA_ITER_MONO == 0
            #pragma omp for
#else
            #pragma omp for schedule(monotonic:static)
#endif
            for (int iter = 0; iter < this->max_iterations; ++iter) {
                step(instance, y.data(), shared_states[opt_id].x.data(), tgen);
                cost = fx(instance, y.data());

                if (cost < shared_states[opt_id].cost) {
                    omp_set_lock(&lock);
                    if (cost < shared_states[opt_id].best_cost) {
                        shared_states[opt_id].best_cost = cost;
                        shared_states[opt_id].best_x = y;
                        if (progress != nullptr)
                            progress(instance, cost, tgen, tacc, opt_id, iter);
                    }
                    shared_states[opt_id].step(y, cost);
                    omp_unset_lock(&lock);
                } else {
                    // FIXED: Replaced drand48() with standard rand()
                    unif = (double)rand() / (double)RAND_MAX;
                    prob = std::exp((shared_states[opt_id].cost - max_cost) / tacc) / gamma;
                    if (prob > unif) {
                        omp_set_lock(&lock);
                        shared_states[opt_id].step(y, cost);
                        omp_unset_lock(&lock);
                    }
                }

                if (omp_test_lock(&lock)) {
                    max_cost = shared_states[0].cost;
                    for (k = 0; k < this->m; ++k)
                        if (shared_states[k].cost > max_cost)
                            max_cost = shared_states[k].cost;

                    gamma = sum_a = 0.;
                    for (k = 0; k < this->m; ++k) {
                        tmp = (shared_states[k].cost - max_cost) / tacc;
                        gamma += std::exp(tmp);
                        sum_a += std::exp(2.0 * tmp);
                    }
                    prob_var = (this->m * (sum_a / (gamma * gamma)) - 1.) /
                               (this->m * this->m);

                    if (prob_var > this->desired_variance)
                        tacc += this->tacc_schedule * tacc;
                    else
                        tacc -= this->tacc_schedule * tacc;

                    tgen = this->tgen_schedule * tgen;
                    omp_unset_lock(&lock);
                }
            } 
        } 

        int best_ind = 0;
        Scalar_fx best_cost = shared_states[0].best_cost;
        for (int k = 0; k < this->m; ++k) {
            if (shared_states[k].best_cost < best_cost) {
                best_cost = shared_states[k].best_cost;
                best_ind = k;
            }
        }
        State<Scalar_x,Scalar_fx> best_state = shared_states[best_ind];
        for (int i = 0; i < n; ++i)
            x[i] = best_state.best_x[i];

        omp_destroy_lock(&lock);
        return 0;
    }
};

}

#endif
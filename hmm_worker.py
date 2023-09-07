import queue
from braingeneers.iot.messaging import MessageBroker
from hmmsupport import get_fitted_hmm

if __name__ == "__main__":
    q = MessageBroker().get_queue("atspaeth/hmm-job-queue")

    try:
        while True:
            # Keep popping queue items and fitting HMMs with those parameters.
            try:
                job = q.get()
                get_fitted_hmm(verbose=True, **job["params"])
                try:
                    q.task_done()
                except ValueError as e:
                    print("Queue misaligned:", e)

            # If one of the fits fails, put it back on the queue with one less
            # retry allowed.
            except ZeroDivisionError:
                s = job["params"]["source"]
                e = job["params"]["exp"]
                r = job["params"]["surrogate"]
                T = job["params"]["bin_size_ms"]
                K = job["params"]["n_states"]
                if job["retries_allowed"] > 0:
                    print(f"Retrying {s}/{e}[{r}] with {T=}ms, {K=}.")
                    job["retries_allowed"] -= 1
                    q.put(job)
                else:
                    print(f"Failed {s}/{e}[{r}] with {T=}ms, {K=}!")

    # If there are no more jobs, let the worker quit.
    except queue.Empty:
        print("No more jobs in queue.")

    # Any other exception is a problem with the worker, so put the job
    # back in the queue unaltered and quit. Also issue task_done because we
    # are not going to process the original job.
    except BaseException as e:
        print(f"Worker terminated with exception {e}.")
        q.put(job)
        try:
            q.task_done()
        except ValueError:
            pass
        print("Job requeued.")

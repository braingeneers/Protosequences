import os
import queue
from braingeneers.iot.messaging import MessageBroker

class Job:
    def __init__(self, q, item):
        self._q = q
        self._item = item
        self.params = item["params"]
        self.retries_allowed = item.get("retries_allowed", 3)

    def requeue(self):
        if self.retries_allowed > 0:
            self._item["retries_allowed"] = self.retries_allowed - 1
            self.q.put(self._item)
            return True
        else:
            return False

def become_worker(what, how):
    q = MessageBroker().get_queue(f"{os.environ['S3_USER']}/{what}-job-queue")

    try:
        while True:
            # Keep popping queue items and fitting HMMs with those parameters.
            job = Job(q, q.get())

            try:
                how(job)
            finally:
                # Always issue task_done, even if the worker failed. If the
                # task counts are misaligned, log it but continue.
                try:
                    q.task_done()
                except ValueError as e:
                    print("Queue misaligned:", e)

    # If there are no more jobs, let the worker quit.
    except queue.Empty:
        print("No more jobs in queue.")

    # Any other exception is a problem with the worker, so put the job
    # back in the queue unaltered and quit. Also issue task_done because we
    # are not going to process the original job.
    except BaseException as e:
        print(f"Worker terminated with exception {e}.")
        q.put(job._item)
        print("Job requeued.")

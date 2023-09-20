from hmmsupport import cv_scores
from worker import become_worker

if __name__ == "__main__":

    def do(job):
        try:
            cv_scores(**job.params)
        except ZeroDivisionError:
            if job.requeue():
                print("Optimization failed, retrying.")
            else:
                print("Optimization failed.")

    become_worker("cv", do)

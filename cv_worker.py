import os
import pickle
from braingeneers.utils.smart_open_braingeneers import open
from hmmsupport import get_raster, Model
from worker import become_worker

BASEPATH = f"s3://braingeneers/personal/{os.environ['S3_USER']}/cache/cv_scores/"

def cross_validate(job):
    "Run cross-validation for a given parameter set."
    model = Model(**job.params, verbose=True)
    raster = get_raster(model.source, model.exp, model.bin_size_ms)
    try:
        validation, training = model.cross_validate(raster, 5, verbose=True)
        with open(f'{BASEPATH}/{model.tag}.pickle', "wb") as f:
            pickle.dump(dict(validation=validation, training=training), f)
    except ZeroDivisionError:
        if job.requeue():
            print("Optimization failed, retrying.")
        else:
            print("Optimization failed.")

become_worker("cv", cross_validate)

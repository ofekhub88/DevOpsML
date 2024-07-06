
import os
import timeit
import numpy as np

def ingestion_timing():
    os.system("python ingestion.py")

def training_timing():
    os.system("python  training.py")

def measure_and_save_timings():
    time_stat = {"ingestion":[],
                "training": [] }
    for i in range(20):
        starttime = timeit.default_timer()
        ingestion_timing()
        time_stat["ingestion"].append(timeit.default_timer()-starttime)
        starttime = timeit.default_timer()
        training_timing()
        time_stat["training"].append(timeit.default_timer()-starttime)
    return time_stat

  
time_stat = measure_and_save_timings()

print("the mean of 20 ingestion_timing() outputs: ",np.mean(time_stat["ingestion"]))
print("the standard deviation of 20 ingestion_timing() outputs:",np.std(time_stat["ingestion"]))
print("the minimum of 20 ingestion_timing() outputs :",min((time_stat["ingestion"])))
print("the maximum of 20 ingestion_timing() outputs :",max((time_stat["ingestion"])))
print("the mean of 20 training_timing() outputs :",np.mean(time_stat["training"]))
print("the standard deviation of 20 training_timing() outputs :",np.std((time_stat["training"])))
print("the minimum of 20 training_timing() outputs :",min((time_stat["training"])))
print("the maximum of 20 training_timing() outputs :",max((time_stat["training"])))

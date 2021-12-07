from parse import read_input_file, write_output_file
import os
import random
from pathlib import Path
import numpy as np
from itertools import permutations
from tqdm import tqdm
from Task import Task
import math
import threading

def solve(tasks):
    """
    Args:
        tasks: list[Task], list of igloos to polish
    Returns:
        output: list of igloos in order of polishing
    """
    return dpSolverMemoized(tasks)

def dpSolverMemoized(tasks):
    mapping = dict()
    for task in tasks:
        mapping[task] = task
        precision = 100
        for decay in [i/precision for i in range(1, precision, 1)]:
            decayed_benefit = round(decay*task.perfect_benefit)
            delay = round(abs(math.log(decay)) / 0.0170)
            new_task = Task(task.task_id, task.deadline + delay, task.duration, decayed_benefit)
            mapping[new_task] = task

    tasks = np.concatenate((tasks, list(mapping.keys())))
    tasks = np.array(sorted(tasks, key=lambda x: x.deadline))
    
    dp_array = [  [ [0, []] for __ in range(1441) ]  for _ in range(2) ]

    for task in tasks:
        for t in range(1441):
            #want the task to end by time t so if duration > t then can't do task i
            t_prime = t - task.duration
            if t_prime < 0:
                #we can't do task i so best order does not change
                dp_array[1][t][0] = dp_array[0][t][0]
                dp_array[1][t][1] = dp_array[0][t][1]
            else:
                if task.deadline < t:
                    profit = Task.get_late_benefit(mapping[task], t - mapping[task].deadline)
                else:
                    profit = task.perfect_benefit
                seen = set(dp_array[0][t_prime][1])
                if dp_array[0][t][0] > profit + dp_array[0][t_prime][0] or task.task_id in seen:
                    dp_array[1][t][0] = dp_array[0][t][0]
                    dp_array[1][t][1] = dp_array[0][t][1]
                else:
                    dp_array[1][t][0] = profit + dp_array[0][t_prime][0]
                    dp_array[1][t][1] = [*dp_array[0][t_prime][1], task.task_id]
        dp_array[0] = dp_array[1]
        dp_array[1] = [ [0, []] for __ in range(1441) ]

    optimal = [0, 0]
    for deez in dp_array:
        cur = max(deez, key=lambda x: x[0])
        optimal = max(optimal, cur, key=lambda x: x[0])
    maxProfit = optimal[0]
    optimalSchedule = optimal[1]

    print(maxProfit)
    return optimalSchedule

if __name__ == '__main__':
    for size in os.listdir('inputs/'):
        if size not in ['small', 'medium', 'large']:
            continue
        for input_file in os.listdir('inputs/{}/'.format(size)):
            if size not in input_file:
                continue
            input_path = 'inputs/{}/{}'.format(size, input_file)
            output_path = 'outputs/{}/{}.out'.format(size, input_file[:-3])
            print(input_path, output_path)
            tasks = read_input_file(input_path)
            output = solve(tasks)
            write_output_file(output_path, output)


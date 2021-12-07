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
    # ans = dpSolutionTaskCopies(tasks)
    return dpSolutionTaskCopiesMemoized(tasks)

def dpSolutionTaskCopies(tasks):
    #perhaps remove this line. Why do you need to sort it twice?
    tasks = sorted(tasks, key=lambda x: x.deadline)
    #end remove
    
    root = dict()
    for task in tasks:
        root[task] = task
        granularity = 100
        for alpha in [i/granularity for i in range(1, granularity, 1)]:
            new_benefit = round(alpha*task.perfect_benefit)
            time_delay = round(abs(math.log(alpha)) / 0.0170)
            new_task = Task(task.task_id, task.deadline + time_delay, task.duration, new_benefit)
            root[new_task] = task

    #for this, why not just say
    # tasks = np.concatenate((tasks, list(root.keys())))
    for task in root:
        tasks.append(task)
    #end mod

    tasks = sorted(tasks, key=lambda x: x.deadline)
    
    #bro what the fuck. consider memoizing this since we only use i + 1 and i
    # dp = [  [ [0, []] for __ in range(1441) ]  for _ in range(len(tasks) + 1) ]
    dp = []
    for i in range(len(tasks)+1):
        temp = []
        for j in range(1441):
            temp.append([0, []])
        dp.append(temp)
    #end mod

    for i, task in enumerate(tasks):
        for t in range(1441):
            #want the task to end by time t so if duration > t then can't do task i
            t_prime = t - task.duration
            if t_prime < 0:
                #order of which to do tasks up to i + 1 (exclusive) is the same as order of doing tasks up to i (exclusive)
                #when we want all tasks completed by time t
                dp[i+1][t][0] = dp[i][t][0]
                dp[i+1][t][1] = dp[i][t][1]
            else:
                if task.deadline < t:
                    actual_profit = Task.get_late_benefit(root[task], t - root[task].deadline)
                else:
                    actual_profit = task.perfect_benefit
                set_seen = set(dp[i][t_prime][1])
                if dp[i][t][0] > actual_profit + dp[i][t_prime][0] or task.task_id in set_seen:
                    dp[i+1][t][0] = dp[i][t][0]
                    dp[i+1][t][1] = dp[i][t][1]
                else:
                    dp[i+1][t][0] = actual_profit + dp[i][t_prime][0]
                    dp[i+1][t][1] = [*dp[i][t_prime][1], task.task_id]

    bestAnswer = 0
    res = []
    for d in dp:
        curr = max(d, key=lambda x: x[0])
        if curr[0] > bestAnswer:
            bestAnswer = curr[0]
            res = curr[1]

    print(bestAnswer)
    return res

def dpSolutionTaskCopiesMemoized(tasks):
    root = dict()
    for task in tasks:
        root[task] = task
        granularity = 100
        for alpha in [i/granularity for i in range(1, granularity, 1)]:
            new_benefit = round(alpha*task.perfect_benefit)
            time_delay = round(abs(math.log(alpha)) / 0.0170)
            new_task = Task(task.task_id, task.deadline + time_delay, task.duration, new_benefit)
            root[new_task] = task

    tasks = np.concatenate((tasks, list(root.keys())))
    tasks = np.array(sorted(tasks, key=lambda x: x.deadline))
    
    dp = [  [ [0, []] for __ in range(1441) ]  for _ in range(2) ]

    for task in tasks:
        for t in range(1441):
            #want the task to end by time t so if duration > t then can't do task i
            t_prime = t - task.duration
            if t_prime < 0:
                #order of which to do tasks up to i + 1 (exclusive) is the same as order of doing tasks up to i (exclusive)
                #when we want all tasks completed by time t
                dp[1][t][0] = dp[0][t][0]
                dp[1][t][1] = dp[0][t][1]
            else:
                if task.deadline < t:
                    actual_profit = Task.get_late_benefit(root[task], t - root[task].deadline)
                else:
                    actual_profit = task.perfect_benefit
                set_seen = set(dp[0][t_prime][1])
                if dp[0][t][0] > actual_profit + dp[0][t_prime][0] or task.task_id in set_seen:
                    dp[1][t][0] = dp[0][t][0]
                    dp[1][t][1] = dp[0][t][1]
                else:
                    dp[1][t][0] = actual_profit + dp[0][t_prime][0]
                    dp[1][t][1] = [*dp[0][t_prime][1], task.task_id]
        dp[0] = dp[1]
        dp[1] = [ [0, []] for __ in range(1441) ]

    bestAnswer = 0
    res = []
    for d in dp:
        curr = max(d, key=lambda x: x[0])
        if curr[0] > bestAnswer:
            bestAnswer = curr[0]
            res = curr[1]

    print(bestAnswer)
    return res

count = 0

test_tasks = [
    Task(1, 20, 11, 5),
    Task(2, 21, 15, 10)
]

# print(dpSolution3(test_tasks))
# print(dpSolutionTaskCopies(test_tasks))


# Here's an example of how to run your solver.
if __name__ == '__main__':
    
    # test_tasks = [
    #     Task(1, 20, 11, 5),
    #     Task(2, 21, 15, 10)
    # ]

    # print(dpSolutionTaskCopies(test_tasks))
    # print(dpSolutionTaskCopiesMemoized(test_tasks))
    # [10.732052, 11.367660400000002, 10.7269991, 10.626864099999999, 10.798250499999995]
    # [10.777539599999997, 10.766100800000004, 10.801782599999996, 10.604425699999993, 10.750248900000003]
    # import timeit
    # print(timeit.repeat("dpSolutionTaskCopies(test_tasks)", "from __main__ import dpSolutionTaskCopies, test_tasks",
    #               number=10))
    # print(timeit.repeat("dpSolutionTaskCopies(test_tasks)", "from __main__ import dpSolutionTaskCopies, test_tasks",
    #               number=10))
    def solve_prob_thread(input_path):
        if not (input_path).is_file() or ".in" not in input_path.name:
            return
        output_path2 = Path('outputs') / (input_path.name[:-3] + '.out')
        tasks = read_input_file(input_path)
        print(input_path)
        output = solve(tasks)
        if "large" in output_path2.name:
            output_path = Path('outputs/large') / (input_path.name[:-3] + '.out')
        if "medium" in output_path2.name:
            output_path = Path('outputs/medium') / (input_path.name[:-3] + '.out')
        if "small" in output_path2.name:
            output_path = Path('outputs/small') / (input_path.name[:-3] + '.out')
        write_output_file(output_path, output)
    
    for input_path in tqdm(list(Path('inputs/').rglob("*"))):
        threading.Thread(target=lambda: solve_prob_thread(input_path)).start()


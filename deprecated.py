def constructLPSolver(n, approximation='tangent'):
    
    if approximation == 'tangent':
        f = lambda p, lateness: p + (-0.0170 * lateness)
    elif approximation == 'full penalty':
        f = lambda p, lateness: 0

    end = cp.Variable(n) #end times
    t = cp.Variable(n)
    q, r = cp.Variable(n * (n - 1) // 2), cp.Variable(n * (n - 1) // 2)
    start = cp.Variable(n)
    deadlines = cp.Parameter(n)
    durations = cp.Parameter(n)
    profits = cp.Parameter(n)

    constraints = [start >= 0]
    z = 0
    objective = 0
    for i in tqdm(range(n)):
        objective += t[i]
        constraints.append(t[i] <= profits[i])
        constraints.append(t[i] <= f(profits[i], end[i] - deadlines[i]))
        constraints.append(end[i] - start[i] == durations[i])
        for j in range(i + 1, n):
            c = [q[z] + r[z] <= - durations[i] - durations[j],
                q[z] <= -end[i],
                q[z] <= -end[j],
                r[z] <= start[i],
                r[z] <= start[j]]
            z += 1
            constraints.extend(c)

    prob = cp.Problem(cp.Maximize(objective), constraints)

    def solver(tasks):
        deadlines.value = np.array([task.get_deadline() for task in tasks])
        durations.value = np.array([task.get_duration() for task in tasks])
        profits.value = np.array([task.get_max_benefit() for task in tasks])

        result = prob.solve(verbose=True)

        task_ids = [tasks[i] for i in range(n)]
        paired_x_ids = zip(task_ids, list(end.value))
        sorted_pairs = sorted(paired_x_ids, key=lambda x: x[1])

        if True:
            profit = 0
            for i, (task, when) in enumerate(sorted_pairs):
                if math.ceil(when) > 1440:
                    sorted_pairs.pop(i)
                else:
                    profit += task.get_late_benefit(math.ceil(when) - task.get_deadline())

            print('estimated profit', result, 'actual profit', profit)

        return [pair[0].get_task_id() for pair in sorted_pairs]

    return solver


def getMaxAndMinRatios(tasks):
    for task in tasks:
        task.ratio = task.perfect_benefit / task.duration

    tasks = sorted(tasks, key=lambda x: x.ratio, reverse=True)
    return tasks[0].ratio, tasks[-(int(len(tasks)*.2))].ratio

def getBestResult(tasks):
    best_res = []
    best_profit = 0
    results = [solveGreedyRatio(tasks), solveGreedyRatio2(tasks), getBestFromRandomSubsets(tasks, 50), getBestFromProfitOnly(tasks), getBestFromProfitOnly2(tasks)]
    results.extend([getBetterResultWithSplitRandomFromRatio(tasks, i, 50) for i in range(5, 20)])
    results.append(dpSolution6(tasks))
    for i, result in enumerate(results):
        curr_profit, curr_res = result
        if curr_profit > best_profit:
            best_res = curr_res
            best_profit = curr_profit
    return best_res

def getBetterResultWithSplitRandomFromRatio(tasks, num_splits, num_random):
    for task in tasks:
        task.ratio = task.perfect_benefit / task.duration
    best_res = []
    best_profit = 0
    for _ in range(num_random):
        tasks = sorted(tasks, key=lambda x: x.deadline)
        tasks = sorted(tasks, key=lambda x: x.ratio, reverse=True)
        split_tasks = np.array_split(tasks, num_splits)
        res_tasks = []
        for split_task in split_tasks:
            np.random.shuffle(split_task)
            res_tasks.extend(split_task)

        curr_profit, curr_res = measureProfitAndGetBackTaskOrder(res_tasks)
        if curr_profit > best_profit:
            best_profit = curr_profit
            best_res = curr_res

        tasks = sorted(tasks, key=lambda x: x.ratio, reverse=True)
        tasks = sorted(tasks, key=lambda x: x.deadline)
        split_tasks = np.array_split(tasks, num_splits)
        res_tasks = []
        for split_task in split_tasks:
            np.random.shuffle(split_task)
            res_tasks.extend(split_task)

        curr_profit, curr_res = measureProfitAndGetBackTaskOrder(res_tasks)
        if curr_profit > best_profit:
            best_profit = curr_profit
            best_res = curr_res

    return best_profit, best_res

def getBestFromProfitOnly(tasks):
    tasks = sorted(tasks, key=lambda x: x.perfect_benefit, reverse=True)
    tasks = sorted(tasks, key=lambda x: x.deadline)
    return measureProfitAndGetBackTaskOrder(tasks)

def getBestFromProfitOnly2(tasks):
    tasks = sorted(tasks, key=lambda x: x.deadline)
    tasks = sorted(tasks, key=lambda x: x.perfect_benefit, reverse=True)
    return measureProfitAndGetBackTaskOrder(tasks)

def solveGreedyRatio(tasks):
    for task in tasks:
        task.ratio = task.perfect_benefit / task.duration

    tasks = sorted(tasks, key=lambda x: x.deadline)
    tasks = sorted(tasks, key=lambda x: x.ratio, reverse=True)
    return measureProfitAndGetBackTaskOrder(tasks)

def solveGreedyRatio2(tasks):
    for task in tasks:
        task.ratio = task.perfect_benefit / task.duration

    tasks = sorted(tasks, key=lambda x: x.ratio, reverse=True)
    tasks = sorted(tasks, key=lambda x: x.deadline)
    return measureProfitAndGetBackTaskOrder(tasks)


def getBestFromRandomSubsets(tasks, num_random):
    best_res = []
    best_profit = 0
    for _ in range(num_random):
        np.random.shuffle(tasks)
        curr_profit, curr_res = measureProfitAndGetBackTaskOrder(tasks)
        if curr_profit > best_profit:
            best_profit = curr_profit
            best_res = curr_res
    return best_profit, best_res

def measureProfitAndGetBackTaskOrder(tasks):
    res = []
    time = 0
    profit = 0
    for task in tasks:
        curr_profit, time = getProfit(task, time)
        if curr_profit is None:
            return profit, res
        profit += curr_profit
        res.append(task.task_id)
    return profit, res

def penaltyEquation(profit, time_late):
    if time_late:
        return profit * np.e**(-0.0170*time_late)
    return profit


def getProfit(task, start):
    end = start + task.duration
    if end > 1440:
        return None, None
    time_penalty = max(end - task.deadline, 0)
    return penaltyEquation(task.perfect_benefit, time_penalty), end

def dpSolution(tasks):
    tasks = sorted(tasks, key=lambda x: x.deadline)
    dp = []
    for i in range(len(tasks)+1):
        temp = []
        for j in range(1441):
            temp.append([0, []])
        dp.append(temp)

    for i, task in enumerate(tasks):
        for t in range(1441):
            t_prime = min(t, task.deadline) - task.duration
            if t_prime < 0: # not enough time to do the task
                dp[i+1][t][0] = dp[i][t][0]
                dp[i+1][t][1] = dp[i][t][1]
            else:
                # have enough time to do the task: so we will deide between (a) not inserting (b) inserting such that we finish the task at time t
                if dp[i][t][0] > task.perfect_benefit + dp[i][t_prime][0]:
                    dp[i+1][t][0] = dp[i][t][0]
                    dp[i+1][t][1] = dp[i][t][1]
                else:
                    dp[i+1][t][0] = task.perfect_benefit + dp[i][t_prime][0]
                    dp[i+1][t][1] = dp[i][t_prime][1] + [task.task_id]

    return dp[-1][-1][0], dp[-1][-1][1]

def dpSolution3(tasks):
    tasks = sorted(tasks, key=lambda x: x.deadline)
    dp = []
    for i in range(len(tasks)+1):
        temp = []
        for j in range(1441):
            temp.append([0, []])
        dp.append(temp)

    for i, task in enumerate(tasks):
        for t in range(1441):
            t_prime = t - task.duration
            if t_prime < 0:
                dp[i+1][t][0] = dp[i][t][0]
                dp[i+1][t][1] = dp[i][t][1]
            else:
                if task.deadline < t:
                    actual_profit = task.get_late_benefit(t - task.deadline)
                else:
                    actual_profit = task.perfect_benefit
                if dp[i][t][0] > actual_profit + dp[i][t_prime][0]:
                    dp[i+1][t][0] = dp[i][t][0]
                    dp[i+1][t][1] = dp[i][t][1]
                else:
                    dp[i+1][t][0] = actual_profit + dp[i][t_prime][0]
                    dp[i+1][t][1] = dp[i][t_prime][1] + [task.task_id]
    return max(dp[-1], key=lambda x: x[0])
    # return dp[-1][-1][0], dp[-1][-1][1]

def dpSolution4(tasks):
    tasks = sorted(tasks, key=lambda x: x.duration)
    tasks = sorted(tasks, key=lambda x: x.deadline)
    dp = []
    for i in range(len(tasks)+1):
        temp = []
        for j in range(1441):
            temp.append([0, []])
        dp.append(temp)

    for i, task in enumerate(tasks):
        for t in range(1441):
            t_prime = t - task.duration
            if t_prime < 0:
                dp[i+1][t][0] = dp[i][t][0]
                dp[i+1][t][1] = dp[i][t][1]
            else:
                if task.deadline < t:
                    actual_profit = task.get_late_benefit(t - task.deadline)
                else:
                    actual_profit = task.perfect_benefit
                if dp[i][t][0] > actual_profit + dp[i][t_prime][0]:
                    dp[i+1][t][0] = dp[i][t][0]
                    dp[i+1][t][1] = dp[i][t][1]
                else:
                    dp[i+1][t][0] = actual_profit + dp[i][t_prime][0]
                    dp[i+1][t][1] = dp[i][t_prime][1] + [task.task_id]

    return dp[-1][-1][1]

def dpSolution5(tasks):
    tasks = sorted(tasks, key=lambda x: x.duration)
    tasks = sorted(tasks, key=lambda x: x.perfect_benefit / x.duration)
    tasks = sorted(tasks, key=lambda x: x.deadline)
    dp = []
    for i in range(len(tasks)+1):
        temp = []
        for j in range(1441):
            temp.append([0, []])
        dp.append(temp)

    for i, task in enumerate(tasks):
        for t in range(1441):
            t_prime = t - task.duration
            if t_prime < 0:
                dp[i+1][t][0] = dp[i][t][0]
                dp[i+1][t][1] = dp[i][t][1]
            else:
                if task.deadline < t:
                    actual_profit = task.get_late_benefit(t - task.deadline)
                else:
                    actual_profit = task.perfect_benefit
                if dp[i][t][0] > actual_profit + dp[i][t_prime][0]:
                    dp[i+1][t][0] = dp[i][t][0]
                    dp[i+1][t][1] = dp[i][t][1]
                else:
                    dp[i+1][t][0] = actual_profit + dp[i][t_prime][0]
                    dp[i+1][t][1] = dp[i][t_prime][1] + [task.task_id]

    return dp[-1][-1][1]

def dpSolution6(tasks):
    max_profit = 0
    max_res = []
    options = list(permutations([lambda x: x.duration, lambda x: x.perfect_benefit, lambda x: x.perfect_benefit / x.duration, lambda x: x.deadline]))
    options.append([lambda x: x.deadline])
    for fs in options:
        for f in fs:
            tasks = sorted(tasks, key=f)
        dp = []
        for i in range(len(tasks)+1):
            temp = []
            for j in range(1441):
                temp.append([0, []])
            dp.append(temp)

        for i, task in enumerate(tasks):
            for t in range(1441):
                t_prime = t - task.duration
                if t_prime < 0:
                    dp[i+1][t][0] = dp[i][t][0]
                    dp[i+1][t][1] = dp[i][t][1]
                else:
                    if task.deadline < t:
                        actual_profit = task.get_late_benefit(t - task.deadline)
                    else:
                        actual_profit = task.perfect_benefit
                    if dp[i][t][0] > actual_profit + dp[i][t_prime][0]:
                        dp[i+1][t][0] = dp[i][t][0]
                        dp[i+1][t][1] = dp[i][t][1]
                    else:
                        dp[i+1][t][0] = actual_profit + dp[i][t_prime][0]
                        dp[i+1][t][1] = dp[i][t_prime][1] + [task.task_id]

        if dp[-1][-1][0] > max_profit:
            max_profit = dp[-1][-1][0]
            max_res = dp[-1][-1][1]
    return max_profit, max_res
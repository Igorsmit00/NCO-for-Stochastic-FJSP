import collections
import re

from ortools.sat.python import cp_model


def update_envs(jobShopEnvs, vars, solver, status, solution_count, time_limit):
    """Update the job shop scheduling environment with the solution found by the solver."""
    # NOTE: We just pick the first one to process and ignore all realizations
    results_list = []
    for i, jobShopEnv in enumerate(jobShopEnvs):
        # Map job operations to their processing times and machines (according to used OR-tools format)
        jobs_operations = [
            [
                [(value, key) for key, value in operation.processing_times.items()]
                for operation in job.operations
            ]
            for job in jobShopEnv.jobs
        ]
        starts, presences = vars["starts"], vars["presences"][i]

        # Check if a solution has been found
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            print("Solution:")

            schedule = []
            # Iterate through all jobs and tasks to construct the schedule
            for job_id, job_operations in enumerate(jobs_operations):
                job_info = {"job": job_id, "tasks": []}

                for task_id, alternatives in enumerate(job_operations):
                    start_time = solver.Value(starts[(i, job_id, task_id)])
                    machine_id, processing_time = -1, -1  # Initialize as not found

                    # Identify the chosen machine and processing time for the task
                    for alt_id, (alt_time, alt_machine_id) in enumerate(alternatives):
                        if solver.Value(presences[(job_id, task_id, alt_id)]):
                            processing_time, machine_id = alt_time, alt_machine_id
                            break  # Exit the loop once the selected alternative is found

                    # Append task information to the job schedule
                    task_info = {
                        "task": task_id,
                        "start": start_time,
                        "machine": machine_id,
                        "duration": processing_time,
                    }
                    job_info["tasks"].append(task_info)

                    # Update the environment with the task's scheduling information
                    job = jobShopEnv.get_job(job_id)
                    machine = jobShopEnv.get_machine(machine_id)
                    operation = job.operations[task_id]
                    setup_time = 0  # No setup time required for FJSP
                    machine.add_operation_to_schedule_at_time(
                        operation, start_time, processing_time, setup_time
                    )

                schedule.append(job_info)

            # Compile results
            results = {
                "time_limit": str(time_limit),
                "status": status,
                "statusString": solver.StatusName(status),
                "objValue": solver.ObjectiveValue(),
                "runtime": solver.WallTime(),
                "numBranches": solver.NumBranches(),
                "conflicts": solver.NumConflicts(),
                "solution_methods": solution_count,
                "Schedule": schedule,
            }
            results_list.append(results)
            print(f"Optimal Schedule Length: {solver.ObjectiveValue()}")
        else:
            print("No solution found.")
        jobShopEnvs[i] = jobShopEnv
    return jobShopEnvs, results_list


def build_product_var(
    model: cp_model.CpModel, b: cp_model.IntVar, x: cp_model.IntVar, name: str
) -> cp_model.IntVar:
    """Builds the product of a Boolean variable and an integer variable."""
    p = model.NewIntVarFromDomain(
        cp_model.Domain.FromFlatIntervals(x.Proto().domain).UnionWith(
            cp_model.Domain(0, 0)
        ),
        name,
    )
    model.Add(p == x).OnlyEnforceIf(b)
    model.Add(p == 0).OnlyEnforceIf(b.Not())
    return p


def fjsp_stoch_cp_sat_model(
    jobShopEnvs: list, obj, alpha
) -> tuple[cp_model.CpModel, dict]:
    """
    Creates a flexible job shop scheduling model using the OR-Tools library.
    """
    nr_scenarios = len(jobShopEnvs)
    # Map job operations to their processing times and machines (according to used OR-tools format)
    jobs_operations = [
        [
            [
                [(value, key) for key, value in operation.processing_times.items()]
                for operation in job.operations
            ]
            for job in jobShopEnv.jobs
        ]
        for jobShopEnv in jobShopEnvs
    ]

    # Computes horizon dynamically as the sum of all durations
    horizon = max(
        sum(
            max(alternative[0] for alternative in task)
            for job in jobs_operations[i]
            for task in job
        )
        for i in range(nr_scenarios)
    )
    print(f"Horizon = {horizon}")

    # Create the model
    model = cp_model.CpModel()

    # Global storage of variables
    intervals_per_resources = [
        collections.defaultdict(list) for _ in range(nr_scenarios)
    ]
    starts = {}  # indexed by (scenario_id,job_id, task_id).
    presences = [
        {} for _ in range(nr_scenarios)
    ]  # scenario_id indexed by (job_id, task_id, alt_id).
    job_ends = [[] for _ in range(nr_scenarios)]  # (nr_scenarios, nr_jobs)
    makespans = []  # nr_scenarios

    # Scan the jobs and create the relevant variables and intervals
    for scenario_id in range(nr_scenarios):
        jobShopEnv = jobShopEnvs[scenario_id]
        for job_id in range(jobShopEnv.nr_of_jobs):
            job = jobs_operations[scenario_id][job_id]
            num_tasks = len(job)
            previous_end = None
            for task_id in range(num_tasks):
                task = job[task_id]

                min_duration = task[0][0]
                max_duration = task[0][0]

                num_alternatives = len(task)
                all_alternatives = range(num_alternatives)

                for alt_id in range(1, num_alternatives):
                    alt_duration = task[alt_id][0]
                    min_duration = min(min_duration, alt_duration)
                    max_duration = max(max_duration, alt_duration)

                # Create main interval for the task
                suffix_name = "_s%i_j%i_t%i" % (scenario_id, job_id, task_id)
                start = model.NewIntVar(0, horizon, "start" + suffix_name)
                duration = model.NewIntVar(
                    min_duration, max_duration, "duration" + suffix_name
                )
                end = model.NewIntVar(0, horizon, "end" + suffix_name)
                interval = model.NewIntervalVar(
                    start, duration, end, "interval" + suffix_name
                )

                # Store the start for the solution
                starts[(scenario_id, job_id, task_id)] = start

                # Add precedence with previous task in the same job
                if previous_end is not None:
                    model.Add(start >= previous_end)
                previous_end = end

                # Create alternative intervals
                if num_alternatives > 1:
                    l_presences = []
                    for alt_id in all_alternatives:
                        alt_suffix = "_s%i_j%i_t%i_a%i" % (
                            scenario_id,
                            job_id,
                            task_id,
                            alt_id,
                        )
                        l_presence = model.NewBoolVar("presence" + alt_suffix)
                        l_start = model.NewIntVar(0, horizon, "start" + alt_suffix)
                        l_duration = task[alt_id][0]
                        l_end = model.NewIntVar(0, horizon, "end" + alt_suffix)
                        l_interval = model.NewOptionalIntervalVar(
                            l_start,
                            l_duration,
                            l_end,
                            l_presence,
                            "interval" + alt_suffix,
                        )
                        l_presences.append(l_presence)

                        # Link the primary/global variables with the local ones
                        model.Add(start == l_start).OnlyEnforceIf(l_presence)
                        model.Add(duration == l_duration).OnlyEnforceIf(l_presence)
                        model.Add(end == l_end).OnlyEnforceIf(l_presence)

                        # Add the local interval to the right machine
                        intervals_per_resources[scenario_id][task[alt_id][1]].append(
                            l_interval
                        )

                        # Store the presences for the solution
                        presences[scenario_id][(job_id, task_id, alt_id)] = l_presence

                    # Select exactly one presence variable
                    model.AddExactlyOne(l_presences)
                else:
                    intervals_per_resources[scenario_id][task[0][1]].append(interval)
                    presences[scenario_id][(job_id, task_id, 0)] = model.NewConstant(1)

            job_ends[scenario_id].append(previous_end)

        # Makespan objective
        makespan = model.NewIntVar(0, horizon, "makespan_s%i" % scenario_id)
        model.AddMaxEquality(makespan, job_ends[scenario_id])
        makespans.append(makespan)

    # Create machines constraints
    for scenario_id in range(nr_scenarios):
        for machine_id in range(jobShopEnv.nr_of_machines):
            intervals = intervals_per_resources[scenario_id][machine_id]
            if len(intervals) > 1:
                model.AddNoOverlap(intervals)

    # Create boolean variables for whether operation A is scheduled on same machine as operation B and starts before operation B
    a_before_b_vars = [
        {} for _ in range(nr_scenarios)
    ]  # key is (machine_id, interval_a_id, interval_b_id) for each scenario
    for scenario_id in range(nr_scenarios):
        # add boolean var whether operation A is scheduled on same machine as operation B and starts before operation B
        for machine_id in range(jobShopEnv.nr_of_machines):
            intervals = intervals_per_resources[scenario_id][machine_id]
            for interval_id, interval in enumerate(intervals):
                split_name = interval.Name().split("_")
                split_name = (
                    split_name[-2:] if len(split_name) == 4 else split_name[-3:]
                )
                interval_info = list(
                    map(lambda x: int(re.split("(\d+)", x)[1]), split_name)
                )
                presence_interval = presences[scenario_id][
                    (
                        (interval_info[0], interval_info[1], 0)
                        if len(interval_info) == 2
                        else (interval_info[0], interval_info[1], interval_info[2])
                    )
                ]
                for other_interval_id, other_interval in enumerate(
                    intervals[interval_id + 1 :], start=interval_id + 1
                ):
                    other_split_name = other_interval.Name().split("_")
                    other_split_name = (
                        other_split_name[-2:]
                        if len(other_split_name) == 4
                        else other_split_name[-3:]
                    )
                    other_interval_info = list(
                        map(lambda x: int(re.split("(\d+)", x)[1]), other_split_name)
                    )
                    presence_other_interval = presences[scenario_id][
                        (
                            (other_interval_info[0], other_interval_info[1], 0)
                            if len(other_interval_info) == 2
                            else (
                                other_interval_info[0],
                                other_interval_info[1],
                                other_interval_info[2],
                            )
                        )
                    ]

                    a_before_b_var = model.NewBoolVar(
                        f"{interval.Name()}_before_{other_interval.Name()}"
                    )
                    a_before_b_vars[scenario_id][
                        (machine_id, interval_id, other_interval_id)
                    ] = a_before_b_var
                    model.Add(
                        interval.EndExpr() <= other_interval.StartExpr()
                    ).OnlyEnforceIf(
                        a_before_b_var, presence_interval, presence_other_interval
                    )

                    b_before_a_var = model.NewBoolVar(
                        f"{other_interval.Name()}_before_{interval.Name()}"
                    )
                    a_before_b_vars[scenario_id][
                        (machine_id, other_interval_id, interval_id)
                    ] = b_before_a_var
                    model.Add(
                        other_interval.EndExpr() <= interval.StartExpr()
                    ).OnlyEnforceIf(
                        b_before_a_var, presence_interval, presence_other_interval
                    )

                    model.AddBoolXOr([a_before_b_var, b_before_a_var])

    for key in a_before_b_vars[0].keys():
        a_before_b_vars_all_scenarios_per_key = [
            a_before_b_vars[scenario_id][key] for scenario_id in range(nr_scenarios)
        ]
        model.AddAllowedAssignments(
            a_before_b_vars_all_scenarios_per_key,
            [(1,) * nr_scenarios, (0,) * nr_scenarios],
        )

    for key in presences[0].keys():
        presences_all_scenarios_per_key = [
            presences[scenario_id][key] for scenario_id in range(nr_scenarios)
        ]
        model.AddAllowedAssignments(
            presences_all_scenarios_per_key, [(1,) * nr_scenarios, (0,) * nr_scenarios]
        )

    if obj == "mean":
        obj_makespan = cp_model.LinearExpr.WeightedSum(
            makespans, [1 / nr_scenarios] * nr_scenarios
        )
    elif obj == "VaR":
        # Create constraints to capture the ordering of the makespan values
        makespan_order_vars = collections.defaultdict(list)
        for scenario_id, makespan in enumerate(makespans):
            for scenario_id_other, other_makespan in enumerate(
                makespans[scenario_id + 1 :], start=scenario_id + 1
            ):
                make_less_then_other_bool = model.NewBoolVar(
                    f"{makespan.Name()} < {makespan.Name()}"
                )
                model.Add(makespan <= other_makespan).OnlyEnforceIf(
                    make_less_then_other_bool
                )
                makespan_order_vars[scenario_id].append(make_less_then_other_bool)
                other_less_then_make_bool = model.NewBoolVar(
                    f"{other_makespan.Name()} < {makespan.Name()}"
                )
                # We use < below and <= above to ensure when multiple scenarios have the same makespan, the one with the lower index is chosen
                model.Add(other_makespan < makespan).OnlyEnforceIf(
                    other_less_then_make_bool
                )
                makespan_order_vars[scenario_id_other].append(other_less_then_make_bool)
                model.AddBoolXOr([make_less_then_other_bool, other_less_then_make_bool])

        target_rank_bools = []
        obj_makespan_additions = []
        target_rank = round((1 - alpha) * nr_scenarios)
        for scenario_id in range(nr_scenarios):
            target_bool = model.NewBoolVar(f"target_rank_{scenario_id}")
            makespan_rank = cp_model.LinearExpr.Sum(makespan_order_vars[scenario_id])
            model.Add(makespan_rank == target_rank).OnlyEnforceIf(target_bool)

            target_rank_bools.append(target_bool)
            obj_makespan_addition = build_product_var(
                model,
                target_bool,
                makespans[scenario_id],
                f"makespan_{scenario_id}_addition",
            )
            obj_makespan_additions.append(obj_makespan_addition)
        model.AddBoolXOr(target_rank_bools)
        obj_makespan = cp_model.LinearExpr.Sum(obj_makespan_additions)
    model.Minimize(obj_makespan)
    return model, {
        "starts": starts,
        "makespans": makespans,
        "presences": presences,
        "intervals": intervals_per_resources,
        "a_before_b_vars": a_before_b_vars,
    }

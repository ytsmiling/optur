import optur


def test_optimize() -> None:
    sampler = optur.samplers.create_random_sampler()
    storage = optur.storages.create_inmemory_storage()
    study = optur.create_study(storage=storage, sampler=sampler)

    def _objective(trial: optur.Trial) -> float:
        return sum((trial.suggest_float(f"f{i}", 0, 1) for i in range(10)), 0.0)

    study.optimize(objective=_objective, n_trials=100)


def test_multithread_parallel_optimize() -> None:
    sampler = optur.samplers.create_random_sampler()
    storage = optur.storages.create_inmemory_storage()
    study = optur.create_study(storage=storage, sampler=sampler)

    def _objective(trial: optur.Trial) -> float:
        return sum((trial.suggest_float(f"f{i}", 0, 1) for i in range(10)), 0.0)

    study.optimize(objective=_objective, n_trials=100, n_jobs=4)


# The objective function must be picklable.
def _multiprocess_objective(trial: optur.Trial) -> float:
    return sum((trial.suggest_float(f"f{i}", 0, 1) for i in range(10)), 0.0)


def test_multiprocess_parallel_optimize() -> None:
    sampler = optur.samplers.create_random_sampler()
    storage = optur.storages.create_inmemory_storage()
    study = optur.create_study(storage=storage, sampler=sampler)

    study.optimize(objective=_multiprocess_objective, n_trials=100, n_jobs=4)


def test_optimize_tpe() -> None:
    sampler = optur.samplers.create_tpe_sampler()
    storage = optur.storages.create_inmemory_storage()
    study = optur.create_study(storage=storage, sampler=sampler)

    def _objective(trial: optur.Trial) -> float:
        return sum((trial.suggest_float(f"f{i}", 0, 1) for i in range(10)), 0.0)

    study.optimize(objective=_objective, n_trials=100)

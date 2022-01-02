import optur


def test_optimize() -> None:
    sampler = optur.samplers.create_random_sampler()
    storage = optur.storages.create_inmemory_storage()
    study = optur.create_study(storage=storage, sampler=sampler)

    def _objective(trial: optur.Trial) -> float:
        return sum((trial.suggest_float(f"f{i}", 0, 1) for i in range(10)), 0.0)

    study.optimize(objective=_objective, n_trials=100)


# def test_optimize_tpe() -> None:
#     sampler = optur.samplers.create_tpe_sampler()
#     storage = optur.storages.create_inmemory_storage()
#     study = optur.create_study(storage=storage, sampler=sampler)
#
#     def _objective(trial: optur.Trial) -> float:
#         return sum((trial.suggest_float(f"f{i}", 0, 1) for i in range(10)), 0.0)
#
#     study.optimize(objective=_objective, n_trials=100)

import logging
import multiprocessing
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.core.singleton import Singleton
from hydra.core.utils import JobReturn, configure_log, filter_overrides, run_job, setup_globals
from hydra.plugins.launcher import Launcher
from hydra.types import HydraContext, TaskFunction
from omegaconf import DictConfig, open_dict

log = logging.getLogger(__name__)


def _worker(
    idx: int,
    overrides: Sequence[str],
    hydra_context: HydraContext,
    config: DictConfig,
    task_function: TaskFunction,
    singleton_state: Dict[Any, Any],
    ret_queue: "multiprocessing.Queue[JobReturn]",
) -> None:
    setup_globals()
    Singleton.set_state(singleton_state)

    sweep_config = hydra_context.config_loader.load_sweep_config(config, list(overrides))
    with open_dict(sweep_config):
        sweep_config.hydra.job.id = "{}_{}".format(sweep_config.hydra.job.name, idx)
        sweep_config.hydra.job.num = idx
    HydraConfig.instance().set_config(sweep_config)

    ret = run_job(
        hydra_context=hydra_context,
        config=sweep_config,
        task_function=task_function,
        job_dir_key="hydra.sweep.dir",
        job_subdir_key="hydra.sweep.subdir",
    )
    ret_queue.put(ret)


class ProcessLauncher(Launcher):
    def __init__(self, max_workers: int = 1) -> None:
        self.max_workers = max_workers
        self.config: Optional[DictConfig] = None
        self.hydra_context: Optional[HydraContext] = None
        self.task_function: Optional[TaskFunction] = None

    def setup(
        self,
        *,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.hydra_context = hydra_context
        self.task_function = task_function

    def launch(
        self, job_overrides: Sequence[Sequence[str]], initial_job_idx: int
    ) -> Sequence[JobReturn]:
        configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)

        log.info(
            f"ProcessLauncher launching {len(job_overrides)} job(s) "
            f"(max_workers={self.max_workers}, fresh process per job)"
        )
        for idx, overrides in enumerate(job_overrides):
            log.info(f"\t#{initial_job_idx + idx} : {' '.join(filter_overrides(overrides))}")

        singleton_state = Singleton.get_state()

        # Each entry: (global_idx, process, ret_queue)
        active: List[tuple] = []
        results: Dict[int, JobReturn] = {}
        pending = list(enumerate(job_overrides))
        pending_iter = iter(pending)
        exhausted = False

        def _spawn_next() -> bool:
            nonlocal exhausted
            try:
                local_idx, overrides = next(pending_iter)
            except StopIteration:
                exhausted = True
                return False
            global_idx = initial_job_idx + local_idx
            ret_queue: multiprocessing.Queue = multiprocessing.Queue()
            p = multiprocessing.Process(
                target=_worker,
                args=(global_idx, overrides, self.hydra_context, self.config,
                      self.task_function, singleton_state, ret_queue),
                daemon=False,
            )
            p.start()
            active.append((global_idx, p, ret_queue))
            return True

        # Fill initial slots
        for _ in range(self.max_workers):
            if not _spawn_next():
                break

        while active:
            for entry in list(active):
                global_idx, p, ret_queue = entry
                if not p.is_alive():
                    p.join()
                    if p.exitcode != 0:
                        log.warning(f"Job #{global_idx} exited with code {p.exitcode}")
                    results[global_idx] = ret_queue.get()
                    active.remove(entry)
                    if not exhausted:
                        _spawn_next()
            time.sleep(0.05)

        return [results[initial_job_idx + i] for i in range(len(job_overrides))]


@dataclass
class ProcessLauncherConf:
    _target_: str = "hydra_plugins.process_launcher.process_launcher.ProcessLauncher"
    max_workers: int = 1


ConfigStore.instance().store(
    group="hydra/launcher",
    name="process",
    node=ProcessLauncherConf,
    provider="process_launcher",
)

import comet_ml
import os
import sys


def init_comet_experiment(comet_api_key, project_name, task_full_name, memo, tags: list = []):
    comet_ml.init()
    experiment = comet_ml.Experiment(api_key=comet_api_key, project_name=project_name)
    experiment_config = sys.argv[-1].replace("+experiments=", "")
    
    tmux_session = "未使用 Tmux"
    for arg in sys.argv:
        if "tmux_session" in arg:
            tmux_session = arg.replace("+tmux_session=", "")
            
    experiment.log_other("备注", memo)
    experiment.log_other("tmux_session", tmux_session)
    experiment.log_other("实验标识", task_full_name)
    experiment.log_other("进程ID", str(os.getpid()))
    experiment.log_other("实验配置", experiment_config)
    for tag in tags:
        experiment.add_tag(tag)
        
    return experiment

def get_experiment(run_id):
    experiment_id = hashlib.sha1(run_id.encode("utf-8")).hexdigest()
    os.environ["COMET_EXPERIMENT_KEY"] = experiment_id

    api = comet_ml.API()  # Assumes API key is set in config/env
    api_experiment = api.get_experiment_by_id(experiment_id)

    if api_experiment is None:
        return comet_ml.Experiment(project_name=PROJECT_NAME)

    else:
        return comet_ml.ExistingExperiment(project_name=PROJECT_NAME)
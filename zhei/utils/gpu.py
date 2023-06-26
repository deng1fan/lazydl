from nvitop import Device, GpuProcess, NA, colored, select_devices
from zhei.utils.log import get_logger
import time
from zhei.utils.catch_error import print_error_info
import json
import os
import datetime
from redis import Redis
from zhei.utils.notice import notice

log = get_logger(__name__)


def set_config_gpus(config):
    redis_client = RedisClient()

    self_occupied_gpus = redis_client.get_self_occupied_gpus()

    if (
        config.use_gpu
        and isinstance(config.visible_cuda, str)
        and "auto_select_" in config.visible_cuda
    ):
        # 如果是自动选择GPU
        min_count = int(config.visible_cuda.split("auto_select_")[-1])
        gpus = select_devices(
            format="index",
            min_count=min_count,
            min_free_memory=config.cuda_min_free_memory,
            max_memory_utilization=config.cuda_max_memory_utilization,
        )
        available_gpus = list(set(gpus) - self_occupied_gpus)
        if len(available_gpus) > 0 and len(available_gpus) >= min_count and len(self_occupied_gpus) < config.limit_the_amount_of_gpu_you_can_use:
            # 有足够可用GPU
            config.wait_gpus = False
            config.visible_cuda = available_gpus[:min_count]
            config.want_gpu_num = len(config.visible_cuda)
            config.default_device = f"cuda:{config.visible_cuda[0]}"
            config.task_id = redis_client.register_gpus(config)
            log.info(f"自动选择GPU：{str(config.visible_cuda)}")
        else:
            # 可用GPU不足
            if config.wait_gpus:
                # 排队
                config.task_id, wait_num = redis_client.join_wait_queue(config)
                # 发送钉钉通知
                try:
                    notice(
                        f"{config.comet_name} 加入排队队列！前方还有{wait_num}个任务🚶🏻‍🧑🏻‍🦼🚶👩‍🦯👨🏻‍🦯", config)
                except Exception as e:
                    print_error_info(e)
                    log.info(f"发送钉钉通知失败: {e}")
            else:
                # 不排队
                raise Exception("可用GPU数量不足，建议使用排队功能！")
    elif config.use_gpu:
        # 如果指定了GPU
        # 转换成相对索引
        reserve_gpus = [i for i, _ in enumerate(config.visible_cuda)]
        # reserve_gpus = config.visible_cuda
        min_count = len(reserve_gpus)
        gpu_all_free = True
        for gpu in reserve_gpus:
            if Device.cuda.from_cuda_indices(gpu)[0].physical_index in self_occupied_gpus:
                gpu_all_free = False
        if len(self_occupied_gpus) >= config.limit_the_amount_of_gpu_you_can_use:
            gpu_all_free = False
        if not config.wait_gpus and not gpu_all_free:
            raise Exception("指定GPU并未全部空闲，建议使用排队功能！")
        elif gpu_all_free:
            available_gpus = reserve_gpus
            config.wait_gpus = False
            config.visible_cuda = available_gpus[:min_count]
            config.want_gpu_num = len(config.visible_cuda)
            config.default_device = f"cuda:{config.visible_cuda[0]}"
            config.task_id = redis_client.register_gpus(config)
        else:
            # 排队
            config.task_id, wait_num = redis_client.join_wait_queue(config)
            # 发送钉钉通知
            try:
                notice(
                    f"{config.comet_name} 加入排队队列！前方还有{wait_num}个任务🚶🏻‍🧑🏻‍🦼🚶👩‍🦯👨🏻‍🦯", config)
            except Exception as e:
                print_error_info(e)
                log.info(f"发送钉钉通知失败: {e}")
    else:
        # 使用CPU
        pass

    ###############################################
    # 检查是否需要等待Gpu
    ###############################################
    while config.use_gpu and config.wait_gpus:
        curr_time = str(time.strftime('%m月%d日 %H:%M:%S', time.localtime()))
        # 判断当前是否轮到自己
        if redis_client.is_my_turn(config):
            # 循环获取当前可用Gpu
            try:
                min_count = config.want_gpu_num
                gpus = select_devices(
                    format="index",
                    min_count=min_count,
                    min_free_memory=config.cuda_min_free_memory,
                    max_memory_utilization=config.cuda_max_memory_utilization,
                )
                self_occupied_gpus = redis_client.get_self_occupied_gpus()

                # 在不超出 GPU 使用数量限制下进行判断
                if len(self_occupied_gpus) < config.limit_the_amount_of_gpu_you_can_use:
                    if not isinstance(config.visible_cuda, str):
                        # 如果指定了GPU
                        reserve_gpus = [
                            i for i, _ in enumerate(config.visible_cuda)]
                        gpu_all_free = True
                        for gpu in reserve_gpus:
                            if Device.cuda.from_cuda_indices(gpu)[0].physical_index in self_occupied_gpus:
                                gpu_all_free = False
                        if gpu_all_free:
                            available_gpus = reserve_gpus
                        else:
                            available_gpus = []
                        min_count = len(reserve_gpus)
                    else:
                        # 自动选择
                        available_gpus = list(set(gpus) - self_occupied_gpus)

                    if len(available_gpus) > 0 and len(available_gpus) >= min_count:
                        # 自动选择，确认等待
                        if (
                            config.confirm_gpu_free
                            and config.last_confirm_gpus == available_gpus[:min_count]
                        ):
                            # 如果满足条件退出循环
                            log.info("发现足够可用GPU并二次确认成功！")
                            config.wait_gpus = False
                            config.visible_cuda = available_gpus[:min_count]
                            config.want_gpu_num = len(config.visible_cuda)
                            config.default_device = f"cuda:{config.visible_cuda[0]}"
                            redis_client.pop_wait_queue(config)
                            config.task_id = redis_client.register_gpus(config)
                            break
                        else:
                            # 设置单次确认空闲
                            log.info("\n发现足够可用GPU！即将进行二次确认！")
                            config.confirm_gpu_free = True
                            config.last_confirm_gpus = available_gpus[:min_count]
                            redis_client.update_queue(config)
                            time.sleep(30)
                            continue
                # 重置确认信息
                print(f"\r{curr_time}: 当前无足够可用GPU，继续等待......",
                      end='',  flush=True)
                if config.confirm_gpu_free:
                    log.info("二次确认失败，继续等待......")
                config.confirm_gpu_free = False
                config.last_confirm_gpus = []
                redis_client.update_queue(config)
                time.sleep(30)
            except Exception as e:
                print_error_info(e)
                raise e
        else:
            # 排队ing......
            wait_num = len(redis_client.client.lrange("wait_queue", 0, -1)) - 1
            print(f"\r{curr_time}: 正在排队中！ 前方还有 {wait_num} 个训练任务！",
                  end='',  flush=True)
            time.sleep(60)

    if config.use_gpu:
        log.info("实验标识： " + config.task_full_name)
        log.info("实验备注： " + config.memo)
        log.info("正在搜集可用GPU信息")
        print_gpu_info(config.visible_cuda)

    return config


class RedisClient:
    def __init__(self):
        self.client = Redis(
            host="127.0.0.1",
            port=6379,
            decode_responses=True,
            charset="UTF-8",
            encoding="UTF-8",
        )

    def get_self_occupied_gpus(self, only_gpus=True):
        """
        获取自己已经占用的Gpu序号
        """
        self_occupied_gpus = self.client.hgetall("self_occupied_gpus")
        if only_gpus:
            all_gpus = []
            for task in self_occupied_gpus.values():
                gpus = [
                    int(device) for device in str(json.loads(task)["cuda_devices"]).split(",")
                ]
                all_gpus.extend(gpus)
            return set(all_gpus)
        return [json.loads(g) for g in self_occupied_gpus.values()]

    def join_wait_queue(self, config):
        """
        加入等待队列
        """
        curr_time = datetime.datetime.now()
        creat_time = datetime.datetime.strftime(curr_time, "%Y-%m-%d %H:%M:%S")
        task_id = (
            str(os.getpid())
            + "*"
            + str(int(time.mktime(time.strptime(creat_time, "%Y-%m-%d %H:%M:%S"))))
        )
        cuda_devices = ','.join([str(cuda) for cuda in config.visible_cuda]) if isinstance(
            config.visible_cuda, list) else "auto"
        content = {
            "want_gpus": config.want_gpu_num,
            "cuda_devices": cuda_devices,
            "create_time": creat_time,
            "update_time": creat_time,
            "system_pid": os.getpid(),
            "task_id": task_id,
            "run_name": config.comet_name,
            "comet_name": config.comet_name,
            "logger_project": config.logger_project,
            "memo": config.memo,
        }
        wait_num = len(self.client.lrange("wait_queue", 0, -1))
        self.client.rpush("wait_queue", json.dumps(content))
        if wait_num == 0:
            log.info(f"正在排队中！ 目前排第一位哦！")
        else:
            log.info(f"正在排队中！ 前方还有 {wait_num} 个训练任务！")
        log.info(
            f"tips: 如果想要对任务进行调整可以移步Redis客户端进行数据修改，只建议进行修改 want_gpus 参数以及删除训练任务操作，其他操作可能会影响Redis读取的稳定性"
        )
        return task_id, wait_num

    def is_my_turn(self, config):
        """
        排队这么长时间，是否轮到我了？
        """
        curr_task = json.loads(self.client.lrange("wait_queue", 0, -1)[0])
        return curr_task["task_id"] == config.task_id

    def update_queue(self, config):
        """
        更新等待队列
        """
        task = json.loads(self.client.lrange("wait_queue", 0, -1)[0])
        if task["task_id"] != config.task_id:
            # 登记异常信息
            log.info("当前训练任务并不排在队列第一位，请检查Redis数据正确性！")
        curr_time = datetime.datetime.now()
        update_time = datetime.datetime.strftime(
            curr_time, "%Y-%m-%d %H:%M:%S")
        task["update_time"] = update_time
        self.client.lset("wait_queue", 0, json.dumps(task))
        # log.info("更新训练任务时间戳成功！")

    def pop_wait_queue(self, config):
        """
        弹出当前排位第一的训练任务
        """
        task = json.loads(self.client.lrange("wait_queue", 0, -1)[0])
        if task["task_id"] != config.task_id:
            # 登记异常信息
            log.info("当前训练任务并不排在队列第一位，请检查Redis数据正确性！")
        next_task = self.client.lpop("wait_queue")
        return next_task

    def register_gpus(self, config):
        """
        将当前训练任务登记到GPU占用信息中
        """
        curr_time = datetime.datetime.now()
        creat_time = datetime.datetime.strftime(curr_time, "%Y-%m-%d %H:%M:%S")
        if not config.task_id:
            task_id = (
                str(os.getpid())
                + "*"
                + str(int(time.mktime(time.strptime(creat_time, "%Y-%m-%d %H:%M:%S"))))
            )
        else:
            task_id = config.task_id

        content = {
            "use_gpus": config.want_gpu_num,
            "cuda_devices": ",".join([str(Device.cuda.from_cuda_indices(gpu)[0].physical_index) for gpu in list(config.visible_cuda)]),
            "register_time": datetime.datetime.strftime(curr_time, "%Y-%m-%d %H:%M:%S"),
            "system_pid": os.getpid(),
            "task_id": task_id,
            "run_name": config.comet_name,
            "comet_name": config.comet_name,
            "logger_project": config.logger_project,
            "memo": config.memo,
        }
        self.client.hset("self_occupied_gpus", task_id, json.dumps(content))
        log.info("成功登记Gpu使用信息到Redis服务器！")
        return task_id

    def deregister_gpus(self, config):
        """
        删除当前训练任务的占用信息
        """
        task = self.client.hget("self_occupied_gpus", config.task_id)
        if task:
            self.client.hdel("self_occupied_gpus", config.task_id)
            log.info("成功删除Redis服务器上的Gpu使用信息！")
        else:
            log.info("无法找到当前训练任务在Redis服务器上的Gpu使用信息！或许可以考虑检查一下Redis的数据 🤔")

    def register_process(self, config):
        """
        将当前训练任务登记到进程信息中
        """
        curr_time = datetime.datetime.now()
        creat_time = datetime.datetime.strftime(curr_time, "%Y-%m-%d %H:%M:%S")
        if not config.task_id:
            task_id = (
                str(os.getpid())
                + "*"
                + str(int(time.mktime(time.strptime(creat_time, "%Y-%m-%d %H:%M:%S"))))
            )
        else:
            task_id = config.task_id

        content = {
            "use_gpus": config.want_gpu_num,
            "memo": config.memo,
            "see_log": "tail -f " + config.get("see_log", "当前进程未使用 nohup 命令启动，无法查看日志"),
            "register_time": datetime.datetime.strftime(curr_time, "%Y-%m-%d %H:%M:%S"),
            "system_pid": os.getpid(),
            "task_id": task_id,
            "run_name": config.comet_name,
            "comet_name": config.comet_name,
            "logger_project": config.logger_project,
        }
        self.client.hset("running_processes", task_id, json.dumps(content))
        log.info("成功登记进程使用信息到Redis服务器！")
        return task_id

    def deregister_process(self, config):
        """
        删除当前训练任务的信息
        """
        task = self.client.hget("running_processes", config.task_id)
        if task:
            self.client.hdel("running_processes", config.task_id)
            log.info("成功删除Redis服务器上的进程使用信息！")
        else:
            log.info("无法找到当前训练任务在Redis服务器上的进程使用信息！或许可以考虑检查一下Redis的数据 🤔")



def print_gpu_info(gpus):
    # or `Device.all()` to use NVML ordinal instead
    devices = Device.cuda.from_cuda_indices(gpus)
    separator = False
    for device in devices:
        processes = device.processes()
        print(colored(str(device), color="green", attrs=("bold",)))
        print(
            colored("  - GPU physical index: ", color="blue", attrs=("bold",))
            + f"{device.physical_index}"
        )
        print(
            colored("  - GPU utilization: ", color="blue", attrs=("bold",))
            + f"{device.gpu_utilization()}%"
        )
        print(
            colored("  - Total memory:    ", color="blue", attrs=("bold",))
            + f"{device.memory_total_human()}"
        )
        print(
            colored("  - Used memory:     ", color="blue", attrs=("bold",))
            + f"{device.memory_used_human()}"
        )
        print(
            colored("  - Free memory:     ", color="blue", attrs=("bold",))
            + f"{device.memory_free_human()}"
        )

        if len(processes) > 0:
            processes = GpuProcess.take_snapshots(
                processes.values(), failsafe=True)
            processes.sort(key=lambda process: (process.username, process.pid))

            print(
                colored(
                    f"  - Processes ({len(processes)}):", color="blue", attrs=("bold",)
                )
            )
            fmt = "    {pid:<5}  {username:<8} {cpu:>5}  {host_memory:>8} {time:>8}  {gpu_memory:>8}  {sm:>3}  {command:<}".format
            print(
                colored(
                    fmt(
                        pid="PID",
                        username="USERNAME",
                        cpu="CPU%",
                        host_memory="HOST-MEM",
                        time="TIME",
                        gpu_memory="GPU-MEM",
                        sm="SM%",
                        command="COMMAND",
                    ),
                    attrs=("bold",),
                )
            )
            for snapshot in processes:
                print(
                    fmt(
                        pid=snapshot.pid,
                        username=snapshot.username[:7]
                        + (
                            "+"
                            if len(snapshot.username) > 8
                            else snapshot.username[7:8]
                        ),
                        cpu=snapshot.cpu_percent,
                        host_memory=snapshot.host_memory_human,
                        time=snapshot.running_time_human,
                        gpu_memory=(
                            snapshot.gpu_memory_human
                            if snapshot.gpu_memory_human is not NA
                            else "WDDM:N/A"
                        ),
                        sm=snapshot.gpu_sm_utilization,
                        command=snapshot.command,
                    )
                )
        else:
            print(colored("  - No Running Processes", attrs=("bold",)))
        if separator:
            print("-" * 120)
        separator = True

import json
import os
import datetime
import time
from redis import Redis
from zhei.utils.log import Logger

log = Logger(__name__) 

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

    def join_wait_queue(self, processing_unit_type, processing_unit, units_count):
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
        content = {
            "processing_unit_type": processing_unit_type,
            "processing_unit": processing_unit,
            "units_count": units_count,
            "create_time": creat_time,
            "update_time": creat_time,
            "system_pid": os.getpid(),
            "task_id": task_id,
            "task_desc": os.environ.get("TASK_DESC", "No task description."),
            "task_id": os.environ.get("TASK_ID", "No task identifier."),
            "comet_project": os.environ.get("COMET_PROJECT_NAME", "No comet project."),
        }
        wait_num = len(self.client.lrange("wait_queue", 0, -1))
        self.client.rpush("wait_queue", json.dumps(content))
        if wait_num == 0:
            log.info(f"正在排队中！ 目前排第一位哦！")
        else:
            log.info(f"正在排队中！ 前方还有 {wait_num} 个训练任务！")
        return task_id, wait_num

    def is_my_turn(self, task_id):
        """
        排队这么长时间，是否轮到我了？
        """
        curr_task = json.loads(self.client.lrange("wait_queue", 0, -1)[0])
        return curr_task["task_id"] == task_id

    def update_queue(self, task_id):
        """
        更新等待队列
        """
        task = json.loads(self.client.lrange("wait_queue", 0, -1)[0])
        if task["task_id"] != task_id:
            # 登记异常信息
            log.info("当前训练任务并不排在队列第一位，请检查Redis数据正确性！")
        curr_time = datetime.datetime.now()
        update_time = datetime.datetime.strftime(
            curr_time, "%Y-%m-%d %H:%M:%S")
        task["update_time"] = update_time
        self.client.lset("wait_queue", 0, json.dumps(task))

    def pop_wait_queue(self, task_id):
        """
        弹出当前排位第一的训练任务
        """
        task = json.loads(self.client.lrange("wait_queue", 0, -1)[0])
        if task["task_id"] != task_id:
            # 登记异常信息
            log.info("当前训练任务并不排在队列第一位，请检查Redis数据正确性！")
        next_task = self.client.lpop("wait_queue")
        return next_task

    def register_gpus(self, task_id, processing_unit_type, processing_unit, units_count):
        """
        将当前训练任务登记到GPU占用信息中
        """
        curr_time = datetime.datetime.now()
        creat_time = datetime.datetime.strftime(curr_time, "%Y-%m-%d %H:%M:%S")
        if not task_id:
            task_id = (
                str(os.getpid())
                + "*"
                + str(int(time.mktime(time.strptime(creat_time, "%Y-%m-%d %H:%M:%S"))))
            )
        else:
            task_id = task_id

        content = {
            "processing_unit_type": processing_unit_type,
            "processing_unit": processing_unit,
            "units_count": units_count,
            "create_time": creat_time,
            "update_time": creat_time,
            "system_pid": os.getpid(),
            "task_id": task_id,
            "task_desc": os.environ.get("TASK_DESC", "No task description."),
            "task_id": os.environ.get("TASK_ID", "No task identifier."),
            "comet_project": os.environ.get("COMET_PROJECT_NAME", "No comet project."),
        }
        self.client.hset("self_occupied_gpus", task_id, json.dumps(content))
        log.info("成功登记Gpu使用信息到Redis服务器！")
        return task_id

    def deregister_gpus(self, task_id):
        """
        删除当前训练任务的占用信息
        """
        task = self.client.hget("self_occupied_gpus", task_id)
        if task:
            self.client.hdel("self_occupied_gpus", task_id)
            log.info("成功删除Redis服务器上的Gpu使用信息！")
        else:
            log.info("无法找到当前训练任务在Redis服务器上的Gpu使用信息！或许可以考虑检查一下Redis的数据 🤔")

    def register_process(self, task_id, processing_unit_type, processing_unit, units_count):
        """
        将当前训练任务登记到进程信息中
        """
        curr_time = datetime.datetime.now()
        creat_time = datetime.datetime.strftime(curr_time, "%Y-%m-%d %H:%M:%S")
        if not task_id:
            task_id = (
                str(os.getpid())
                + "*"
                + str(int(time.mktime(time.strptime(creat_time, "%Y-%m-%d %H:%M:%S"))))
            )
        else:
            task_id = task_id

        content = {
            "processing_unit_type": processing_unit_type,
            "processing_unit": processing_unit,
            "units_count": units_count,
            "create_time": creat_time,
            "update_time": creat_time,
            "system_pid": os.getpid(),
            "task_id": task_id,
            "task_desc": os.environ.get("TASK_DESC", "No task description."),
            "task_id": os.environ.get("TASK_ID", "No task identifier."),
            "comet_project": os.environ.get("COMET_PROJECT_NAME", "No comet project."),
        }
        self.client.hset("running_processes", task_id, json.dumps(content))
        log.info("成功登记进程使用信息到Redis服务器！")
        return task_id

    def deregister_process(self, task_id):
        """
        删除当前训练任务的信息
        """
        task = self.client.hget("running_processes", task_id)
        if task:
            self.client.hdel("running_processes", task_id)
            log.info("成功删除Redis服务器上的进程使用信息！")
        else:
            log.info("无法找到当前训练任务在Redis服务器上的进程使用信息！或许可以考虑检查一下Redis的数据 🤔")

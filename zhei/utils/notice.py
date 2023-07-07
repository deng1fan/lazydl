from dingtalkchatbot.chatbot import DingtalkChatbot
import os
from zhei.utils.log import Logger
import psutil

log = Logger(__name__)

def notice(msg: str = "", warning=False, access_token="", secret=""):
    """钉钉消息通知
    
    """
    access_token = os.environ.get('DINGDING_ACCESS_TOKEN', "") if access_token == "" else access_token
    secret = os.environ.get('DINGDING_SECRET', "") if secret == "" else secret
    if access_token == "" or secret == "":
        log.warning("未设置钉钉Token，无法发送消息: " + msg)
        return
    
    pid = os.getpid()
    proctitle = psutil.Process(pid).name()
    if warning:
        msg = f"⚠️\n{msg}\n\n👾进程ID: {pid}\n👾进程名: {proctitle}"
    else:
        msg = f"🪼\n{msg}\n\n👾进程ID: {pid}\n👾进程名: {proctitle}"
    
    # WebHook地址
    webhook = f'https://oapi.dingtalk.com/robot/send?access_token={access_token}'
    xiaoding = DingtalkChatbot(webhook, secret=secret, pc_slide=True)
    # Text消息@所有人
    xiaoding.send_text(msg=msg)

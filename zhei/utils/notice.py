from dingtalkchatbot.chatbot import DingtalkChatbot
import os

def notice(msg):
    access_token = os.environ.get('dingding_access_token')
    secret = os.environ.get('dingding_secret')
    # WebHook地址
    webhook = f'https://oapi.dingtalk.com/robot/send?access_token={access_token}'
    xiaoding = DingtalkChatbot(webhook, secret=secret, pc_slide=True)
    # Text消息@所有人
    xiaoding.send_text(msg=msg)


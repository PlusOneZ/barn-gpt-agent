import requests
import logging
import re
import os

from dotenv import load_dotenv
load_dotenv()


def call_hook_with_result(hook: str, results: [], status="done", api_response=None):
    # use pattern matching to check the hook style
    if hook == "skip":
        logging.debug("Skipping hook calling without raising error")
        return 402

#    # regex = r"http:\/\/[a-zA-Z0-9.]+:[0-9]+\/task\/[a-zA-Z0-9\-]+\/hook"
#    # corresponding to the backend URL
#    url = os.getenv('NODE_SERVER_URL')
#    ip = os.getenv('NODE_SERVER_IP')
#    regex = (r"^((https?:\/\/)?((" + url +
#             r")|(" + ip +
#             r")|(localhost)))(:[0-9]+)?\/task\/[a-zA-Z0-9\-]+\/hook")

    #new regex after migrate to nginx
    # 获取环境变量
    url = os.getenv('NODE_SERVER_URL', '127.0.0.1')  # 默认值为 127.0.0.1
    ip = os.getenv('NODE_SERVER_IP', '127.0.0.1')
    api_prefix = os.getenv('NODE_SERVER_API_PREFIX', '/barngpt')

    # 定义改进的正则表达式
    regex = (r"^(https?://)?"  # 可选的 http 或 https
             r"(localhost|" + re.escape(url) + "|" + re.escape(ip) + r")"  # URL 或 IP
             r"(:[0-9]+)?"  # 可选端口
             + re.escape(api_prefix) +               # 动态的 API 前缀
             r"/task/[a-zA-Z0-9\-]+/hook$")  # 任务路径和钩子结尾

    if not re.match(regex, hook):
        logging.error(f"Hook '{hook}' is not legal")
        return 400

    res = requests.put(hook, json={
        "status": status,
        "results": results,
        "apiResponse": api_response.json() if api_response else {}
    })
    logging.debug(f"Hook calling with status {res.status_code} and content '{res.text}'", )
    return res.status_code

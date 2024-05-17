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

    # regex = r"http:\/\/[a-zA-Z0-9.]+:[0-9]+\/task\/[a-zA-Z0-9\-]+\/hook"
    # corresponding to the backend URL
    url = os.getenv('NODE_SERVER_URL')
    ip = os.getenv('NODE_SERVER_IP')
    regex = (r"^((http:\/\/" + url + r")" +
             r"|(https:\/\/" + url + r")"
             r"|(" + ip + r"))(:[0-9]+)?\/task\/[a-zA-Z0-9\-]+\/hook")
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

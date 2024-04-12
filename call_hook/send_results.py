import requests
import logging
import re


def call_hook_with_result(hook: str, results: [], status="done", api_response=None):
    # Hook style: "http://<localhost or IP>:<port>/task/<id>/hook"
    # use pattern matching to check the hook style
    regex = r"http:\/\/[a-zA-Z0-9.]+:[0-9]+\/task\/[a-zA-Z0-9]+\/hook"
    if not re.match(regex, hook):
        logging.error(f"Hook '{hook}' is not in the correct format")
        return 400

    res = requests.put(hook, json={
        "status": status,
        "results": results,
        "api_response": api_response if api_response else {}
    })
    logging.debug(f"Hook calling with status {res.status_code} and content '{res.text}'", )
    return res.status_code

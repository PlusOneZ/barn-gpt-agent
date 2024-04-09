import requests
import logging


def call_hook_with_result(hook: str, results: []):
    # todo check hook style
    res = requests.put(hook, json={"status": "done", "results": results})
    logging.debug(f"Hook calling with status {res.status_code} and content '{res.text}'", )
    return res.status_code

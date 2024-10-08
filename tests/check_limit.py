import unittest
from gpt_tasks import DoTask
import requests

class MyTestCase(unittest.TestCase):
    def test_limit_exceed(self):
        task_manager = DoTask()
        self.assertFalse(all(
            task_manager.check_limit("audio-generation") for i in range(10)
        ))

    def test_within_limit(self):
        task_manager = DoTask()
        limit_times = 2
        self.assertTrue(
            all(task_manager.check_limit("audio-generation") for i in range(limit_times))
        )

    def test_with_api(self):
        url = "http://127.0.0.1:5000/api/task/audio/generation"
        limit_times = 2
        status_codes = []
        for i in range(limit_times * 2):
            response = requests.post(url)
            print(response.text, response.status_code)
            status_codes.append(response.status_code)
        self.assertFalse(all(code == 200 for code in status_codes))


if __name__ == '__main__':
    unittest.main()

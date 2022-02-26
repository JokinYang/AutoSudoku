import time

import cv2 as cv
import numpy as np
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from seleniumbase.core.browser_launcher import get_driver

from cv_util import recognize_puzzle
from sudoku import solve_sudoku


class WebSudoku:
    _canvas_xpath = '//*[@id="game"]/canvas'
    _base_url = "https://sudoku.com/"

    def __init__(self, browser_name='firefox', maxsize=True, timeout=10):
        self.driver = get_driver(browser_name, headless=False)
        if maxsize:
            self.driver.maximize_window()
        self.timeout = timeout
        self._is_banner_close = False

    def open(self, url):
        self.driver.get(url)

    def close_cookies_banner(self):
        wait = WebDriverWait(self.driver, self.timeout)
        btn = wait.until(EC.element_to_be_clickable((By.ID, 'banner-close')))
        btn.click()

    def dismiss_tips(self):
        wait = WebDriverWait(self.driver, self.timeout)
        wait.until(EC.invisibility_of_element((By.XPATH, '//*[@id="game-wrapper"]/div[1]')))
        element = wait.until(EC.element_to_be_clickable((By.XPATH, self._canvas_xpath)))
        element.click()

    def click_by_position(self, x, y, keys):
        actions = ActionChains(self.driver)
        actions.move_by_offset(x, y).click()
        actions.send_keys(keys)
        actions.perform()

    def screenshot(self, wait=1) -> np.ndarray:
        if wait:
            time.sleep(wait)
        img_bytes = self.driver.get_screenshot_as_png()
        img_buffer_numpy = np.frombuffer(img_bytes, dtype=np.uint8)
        img_numpy = cv.imdecode(img_buffer_numpy, 1)
        return img_numpy

    def launch_game(self, difficulty):
        self.driver.delete_all_cookies()
        url = self._base_url + difficulty
        self.open(url)
        self.close_cookies_banner()
        self.dismiss_tips()

    def solve_puzzle(self):
        screenshot = self.screenshot(wait=2)
        img_fields = recognize_puzzle(screenshot)
        print('Puzzle:\n', [i.number for i in img_fields])
        solved_puzzle = solve_sudoku([i.number for i in img_fields])
        print('Solution:\n', solved_puzzle)
        last_x, last_y = 0, 0
        for i in img_fields:
            if i.number:
                continue
            print(i)
            self.click_by_position(i.x_center - last_x, i.y_center - last_y, solved_puzzle[i.index])
            last_x = i.x_center
            last_y = i.y_center


sudoku = WebSudoku(browser_name='firefox')
while True:
    sudoku.launch_game('easy')
    try:
        sudoku.solve_puzzle()
    except ValueError:
        continue
    input("Press enter to continue...")

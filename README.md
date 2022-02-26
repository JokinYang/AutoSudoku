# Auto Sudoku

A tool to solve [Sudoku](https://sudoku.com) automatically    
[demo video](https://www.bilibili.com/video/BV1ka411C7kw)

# How to run

Make sure [firefox](https://www.mozilla.org/firefox/new/ ) was installed in your system.(Other browsers are not tested)

```shell
pip3 install -r requirements.txt
python3 ./src/main.py
```

# Known issue

- [ddddocr](https://github.com/sml2h3/ddddocr) (ocr lib used by this tool) may make wrong classification of the number
  in Sudoku board.
- The Chrome driver of [selenium](https://www.selenium.dev/ ) can not work properly
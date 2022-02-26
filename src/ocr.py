import ddddocr

ocr = ddddocr.DdddOcr(show_ad=False)


def get_num(img_bytes) -> int:
    ret = ocr.classification(img_bytes)
    ret = ret or '0'
    if 'l' in ret:
        ret = '1'
    return int(ret)

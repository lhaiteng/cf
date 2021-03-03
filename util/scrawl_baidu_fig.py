# -*- coding: utf-8 -*-
import numpy as np
import time, os, random
import matplotlib.pyplot as plt
import cv2
import requests, json, re
from lxml import etree
from bs4 import BeautifulSoup
import multiprocessing
from pypinyin import lazy_pinyin


class ScrawlFig:
    def __init__(self, src_web):
        self.src_web = src_web
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36",
            "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36",
            "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:30.0) Gecko/20100101 Firefox/30.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/537.75.14",
            "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; Win64; x64; Trident/6.0)",
            'Mozilla/5.0 (Windows; U; Windows NT 5.1; it; rv:1.8.1.11) Gecko/20071127 Firefox/2.0.0.11',
            'Opera/9.25 (Windows NT 5.1; U; en)',
            'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; .NET CLR 1.1.4322; .NET CLR 2.0.50727)',
            'Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.5 (like Gecko) (Kubuntu)',
            'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.8.0.12) Gecko/20070731 Ubuntu/dapper-security Firefox/1.5.0.12',
            'Lynx/2.8.5rel.1 libwww-FM/2.14 SSL-MM/1.4.1 GNUTLS/1.2.9',
            "Mozilla/5.0 (X11; Linux i686) AppleWebKit/535.7 (KHTML, like Gecko) Ubuntu/11.04 Chromium/16.0.912.77 Chrome/16.0.912.77 Safari/535.7",
            "Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:10.0) Gecko/20100101 Firefox/10.0 "]

    def save_names_fig(self, kws, fig_num, save_dirs, pre=''):
        for ind, kw in enumerate(kws):
            print(f'\rGetting No.{ind} {kw}...', end='')
            self.save_name_fig(kw, fig_num, save_dirs[ind], pre)
            time.sleep(0.01)

    def save_name_fig(self, kw, fig_num, save_dir, pre=''):
        if not os.path.isdir(save_dir):
            try:
                os.makedirs(save_dir)
            except:
                pass
        stop = False
        start_num = -30
        has_fig = 0
        while not stop and start_num < fig_num:
            start_num += 30
            web = self.src_web.format(keyword=kw, start_num=start_num)
            try:
                response = requests.get(web, headers=self.headers(kw), timeout=10).content.decode('utf8', 'ignore')
                js = json.loads(response)
            except:
                continue
            for data in js['x']:
                if 'hoverURL' in data:
                    try:
                        if self.download_img(data['hoverURL'], kw, save_dir, has_fig, pre):
                            has_fig += 1
                            if has_fig == fig_num:
                                stop = True
                                break
                    except:
                        continue

    def download_img(self, img_url, kw, save_dir, has_fig, pre=''):
        res = requests.get(img_url, headers=self.headers(kw), timeout=10).status_code
        if res != 200:
            return False
        pic = requests.get(img_url, headers=self.headers(kw), stream=True, timeout=10)
        with open(save_dir + pre + f'{has_fig}.png', 'wb') as file:
            for j in pic.iter_content(2048):
                file.write(j)
        return True

    def headers(self, kw):
        headers = {
            "User-Agent": random.choice(self.user_agents),
            "Referer": f'http://image.baidu.com/search/index?tn=baiduimage&ps=1&ct=201326592&ie=utf-8&word={kw.encode("utf8")}'
        }
        return headers


if __name__ == '__main__':
    src_web = 'http://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord={keyword}&cl=&lm=&ie=utf-8&oe=utf-8&adpicid=&st=&z=&ic=&hd=&latest=&copyright=&word={keyword}&s=&se=&tab=&width=&height=&face=&istype=&qc=&nc=&fr=&expermode=&force=&pn={start_num}&rn=30&gsm=1e&1583653409630='
    scrawl_fig = ScrawlFig(src_web)

    fig_num = 100
    root_dir = 'E:/TEST/AI/datasets/cnface/'
    names_path = root_dir+'cn_names.txt'
    # with open(names_path, 'r', encoding='utf8') as f:
    #     cn_names = f.readlines()
    # cn_names = [nf.strip() for nf in cn_names]
    cn_names = ['孟佳']
    names = [''.join(lazy_pinyin(n)).replace('\n', '') for n in cn_names]
    for name, cn_name in zip(names, cn_names):
        print(f'START scrawling {cn_name}...')
        save_dir = root_dir+f'{name}/'
        kws = [f'{cn_name}壁纸', f'{cn_name}侧脸', f'{cn_name}闭眼', f'{cn_name}张嘴']
        pres = list('abcdefghijk')[:len(kws)]
        _fig_num = round(fig_num / (len(kws)))
        ps = []
        for kw, pre in zip(kws, pres):
            ps.append(multiprocessing.Process(target=scrawl_fig.save_name_fig,
                                              args=(kw, _fig_num, save_dir, pre)))
        for p in ps:
            p.start()
        for p in ps:
            p.join()

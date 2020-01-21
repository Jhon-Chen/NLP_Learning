# -*- coding: utf-8 -*-
import scrapy


class GranbluefanstySpider(scrapy.Spider):
    name = 'granbluefansty'
    allowed_domains = ['https://tieba.baidu.com/f?kw=%E7%A2%A7%E8%93%9D%E5%B9%BB%E6%83%B3&fr=index&fp=0&ie=utf-8']
    start_urls = ['http://https://tieba.baidu.com/f?kw=%E7%A2%A7%E8%93%9D%E5%B9%BB%E6%83%B3&fr=index&fp=0&ie=utf-8/']

    def parse(self, response):
        pass

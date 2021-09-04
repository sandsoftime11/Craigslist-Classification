import scrapy
from scrapy import Request

class GeneralSpider(scrapy.Spider):
    name = 'generalv2'
    allowed_domains = ['craigslist.org']
    start_urls = ['https://newyork.craigslist.org/d/general-for-sale/search/foa']

    def parse(self, response):
        # titles = response.xpath('//a[@class="result-title hdrlnk"]/text()').getall()
        # print(titles)
        # for i in titles :
        #     yield {'Title': i}

        deals = response.xpath('//div[@class="result-info"]')
        for deal in deals:
            # extract_first, if not, may mess up the results
            title = deal.xpath('h3[@class ="result-heading"]/a[@class="result-title hdrlnk"]/text()').extract_first()
            city = deal.xpath('span[@class ="result-meta"]/span[@class="result-hood"]/text()').extract_first("")[2:-1]  #[2:-1] get rid of the bracets
            price = deal.xpath('span[@class ="result-meta"]/span[@class="result-price"]/text()').extract_first("")

            lower_rel_url = deal.xpath('*/a[@class="result-title hdrlnk"]/@href').extract_first()
            lower_url = response.urljoin(lower_rel_url)

            yield Request(lower_url, callback=self.parse_lower, meta={'Title': title, 'City': city, 'Price': price})

        next_rel_url = response.xpath('//a[@class="button next"]/@href').extract_first()
        next_url = response.urljoin(next_rel_url)
        yield Request(next_url, callback=self.parse)

    def parse_lower(self, response):
        text = "".join(line for line in response.xpath('//*[@id="postingbody"]/text()').extract())
        # because there may be multiple texts
        response.meta['Text'] = text
        yield response.meta

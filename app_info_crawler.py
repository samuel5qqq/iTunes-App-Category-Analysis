import urllib.request
from bs4 import BeautifulSoup
import csv
import sys

nav_site = "http://itunes.apple.com/us/genre/ios-books/id6018?mt=8"
alphabet = ['#', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z']


def site_open(site):

    try:

        req = urllib.request.Request(site)

        req.add_header("User-Agent", "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36")

        website = urllib.request.urlopen(req)

        return website
    except urllib.request.URLError:
        pass


def soup_site(site):

    return BeautifulSoup(site_open(site))


def title_get(soup):
    return soup.find(id="title").find("h1").text


def dev_get(soup):
    return soup.find(id="title").find("h2").text[3:]

def des_get(soup):
    return soup.find('div', {'class': 'center-stack'}).find('p', itemprop="description").text

def price_get(soup):
    return soup.find(id="left-stack").find("ul").find("li").text


def category_get(soup):
    return soup.find(id="left-stack").find("ul").find("li", "genre").find("a").text


def size_get(soup):
    return soup.find(id="left-stack").find("ul").find_all("li")[4].text[6:]


def seller_get(soup):
    return soup.find(id="left-stack").find("ul").find_all("li")[6].text[8:]


def rating_get(soup):

    tag = soup.find("div", "customer-ratings").find("div", "rating")

    stars, rating = tag['aria-label'].split(',')

    return (stars.strip(), rating.strip())


def app_info(soup, site):

    try:
        title = title_get(soup)
        des = des_get(soup)
        developer = dev_get(soup)
        price = price_get(soup)
        category = category_get(soup)
        size = size_get(soup)
        seller = seller_get(soup)
        stars, rating = rating_get(soup)


        return (title, des, developer, price, category, size, seller, stars, rating, site)
    except:
        print("I Dunno, There Was Some Error!")
        pass

def read_in(file):

    f = open(file, 'r')
    data = f.readlines()

    return data


def split_data(data, splits):
    n = round(len(data) / splits)
    print(len(data))
    new_data = []
    for i in range(0, splits):
        j = data[(i - 1) * n:i * n]
        new_data.append(j)
    return new_data


def app_info_crawl(source, output, totalnumber):

    data = read_in(source)
    f = open(output, 'w', newline='')
    writer = csv.writer(f, dialect='excel', quoting=csv.QUOTE_NONNUMERIC)
    app_crawl_main_loop(data, writer, totalnumber)
    return


def app_crawl_main_loop(data, writer, totalnumber):

    count = 0
    for i, link in enumerate(data):

        if(count == totalnumber):
            break
        try:
            info = app_info(soup_site(link.strip('"')), link.strip('"'))

            writer.writerow(info)
            count = count+1
            print(count, info)
        except:
            continue

    print('Completed Scrapping Data')
    return


def main():

    links_file = ["ios-games.csv", "ios-education.csv", "ios-business.csv", "ios-music.csv"
                  , "ios-sports.csv", "ios-weather.csv", "ios-photo-video.csv", "ios-shopping.csv"
                  , "ios-news.csv", "ios-travel.csv"]

    # second argument is the number of tuple wanted to get from csv
    totalnumber = int(sys.argv[1])

    for i in range(len(links_file)):
        line = str(links_file[i]).split(".")
        app_info_file = line[0]+"tmp.csv"
        app_info_crawl(links_file[i], app_info_file, totalnumber)
    return


if __name__ == '__main__':
    main()


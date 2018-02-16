import urllib.request
from bs4 import BeautifulSoup
import time
import csv


sleep = 0


sample = ''

threads = 40


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



def genre_link_list(site):

    soup = soup_site(site)

    table = soup.find(id="genre-nav").find_all("a")

    for link in table:
        # returns next link when called
        yield link.get('href')


def app_link_list(site):

    soup = soup_site(site)

    table = soup.find(id="selectedcontent").find_all("a")
    for link in table:
        yield (link.get('href'),)


def general_app_store_crawl(sleep_time=float):

    for i, link in enumerate(genre_link_list(nav_site)):
        line = str(link).split("/")
        category = line[5]
        print('Scraping genre ' + category + '.')
        file = str(category) + '.csv'
        f = open(file, 'a', newline='')
        writer = csv.writer(f, dialect='excel', quoting=csv.QUOTE_NONNUMERIC)

        for letter in alphabet:

            new_site = link + "&" + letter
            print('Scraping from ' + new_site + '.')

            time.sleep(sleep_time)

            for app in app_link_list(new_site):
                writer.writerow(app)

        f.close()
        print('Completed Scrapping')

    return




def main():
    general_app_store_crawl(sleep)
    return


if __name__ == '__main__':
    main()


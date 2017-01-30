import math
import os
import re
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, NoSuchElementException



def get_VC_info(driver, url):
    driver.get(url)
    infos = {}
    try:
        infos['name'] = driver.find_element_by_class_name('slides').text
        #infos['Early-stage investments'] = driver.find_element_by_class_name('title').text
        #infos['Early-stage deal count'] = driver.find_element_by_id(
         #   'overview-summary-current').find_element_by_tag_name(
          #  'td').find_element_by_tag_name('a').text
        #infos['Industries'] = driver.find_element_by_id(
           # 'overview-summary-past').find_element_by_tag_name(
            #'td').find_element_by_tag_name('a').text
        #infos['Regions'] = driver.find_element_by_id(
         #   'background-education').find_elements_by_class_name(
          #  'editable-item')

    except NoSuchElementException:
        print("profile incomplete")


    return(infos)

if __name__=='__main__':
    profiles_infos = None
    driver = webdriver.Firefox()
    url1 = 'https://www.entrepreneur.com/article/242702'
    VC_infos = get_VC_info(driver,url1)
    print(VC)
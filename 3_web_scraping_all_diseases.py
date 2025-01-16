import csv
import requests
from bs4 import BeautifulSoup
import time
import random

"""
Scraper Tool
"""

def scrape_primary_structure(csv_file_path, output_file):
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)

        # Open the output txt file in append mode
        with open(output_file, 'a', encoding='utf-8') as file:
            for row in reader:
                disease_name, url = row['Disease Name'], row['Link']
                print(f"Scraping {disease_name} from {url}...")

                response = requests.get(url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, 'html.parser')

                    # Extract disease name
                    disease_name = soup.find('h1').text.strip()

                    # Extract symptoms
                    symptoms_header = soup.find('h2', id='symptoms')
                    symptoms_paragraph = ""
                    symptoms_list = []

                    if symptoms_header:
                        paragraph = symptoms_header.find_next_sibling('p')
                        if paragraph:
                            symptoms_paragraph = paragraph.text.strip()

                        symptoms_list_element = symptoms_header.find_next_sibling('ul')
                        if symptoms_list_element:
                            symptoms_list = [li.text.strip() for li in symptoms_list_element.find_all('li')]

                    # Save data to file
                    file.write(f"Disease Name: {disease_name}\n")
                    file.write(f"Symptoms Description: {symptoms_paragraph}\n")
                    file.write(f"Symptoms List: {', '.join(symptoms_list)}\n")
                    file.write("---\n")

                    print(f"Finished scraping {disease_name}")
                    # Random delay to avoid getting blocked
                    time.sleep(random.uniform(1, 5))
                else:
                    print(f"Failed to retrieve {url}. Status code: {response.status_code}")


csv_file_path = 'diseases.csv'
scrape_primary_structure(csv_file_path, 'primary_structure.txt')
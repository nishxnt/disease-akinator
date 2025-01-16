
import requests
from bs4 import BeautifulSoup
import csv

"""##Scraper Tool"""

# Send a GET request to the NHS conditions page
url = 'https://www.nhs.uk/conditions/'
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    html_content = response.text

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all the disease links under the relevant <a> tag structure
    base_url = 'https://www.nhs.uk'  # Base URL for relative links
    diseases = []

    # Select all <a> tags that are part of the conditions list
    for link in soup.select('ul.nhsuk-list > li > a[href^="/conditions/"]'):
        disease_name = link.text.strip()  # Extract the text (disease name)
        disease_link = base_url + link['href']  # Append base URL to form full link

        # Append the result to the diseases list
        diseases.append([disease_name, disease_link])

    # Save the data to a CSV file
    with open('diseases.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Disease Name', 'Link'])  # Header row
        writer.writerows(diseases)

    print(f"Scraped {len(diseases)} diseases and their links.")
else:
    print(f"Failed to retrieve the webpage. Status code: {response.status_code}")

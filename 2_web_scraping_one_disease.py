
import requests
from bs4 import BeautifulSoup

"""##Scraper Tool"""

# Fetch the disease-specific page (Abdominal Aortic Aneurysm Screening)
url = 'https://www.nhs.uk/conditions/abdominal-aortic-aneurysm/'  # Example URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    html_content = response.text

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract the disease name from the <h1> tag
    disease_name = soup.find('h1').text.strip()
    print(f"Disease Name: {disease_name}")

    # Extract the symptoms section by finding the <h2> tag with id="symptoms"
    symptoms_header = soup.find('h2', id='symptoms')

    # Check if the symptoms section exists
    if symptoms_header:
        # Get the next sibling paragraphs and lists
        symptoms_paragraph = symptoms_header.find_next_sibling('p').text.strip()
        symptoms_list = symptoms_header.find_next_sibling('ul')

        # Extract each symptom from the list items <li>
        symptoms = [li.text.strip() for li in symptoms_list.find_all('li')]

        # Combine symptoms into a single string for easier display or storage
        symptoms_str = ', '.join(symptoms)

        # Output the results
        print(f"Symptoms: {symptoms_paragraph}")
        print(f"Symptoms List: {symptoms_str}")
    else:
        print("No symptoms section found.")
else:
    print(f"Failed to retrieve the webpage. Status code: {response.status_code}")

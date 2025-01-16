#Disease Akinator: Automated Disease Prediction

This repository contains the source code and resources for the Disease Akinator, a project that combines web-scraped medical data, large language models (LLMs), and an ID3 Decision Tree algorithm to create an automated disease prediction tool. This project was completed as part of the thesis for the Bachelor of Science degree in International Business Information Systems at Hochschule Furtwangen University.

Overview
The Disease Akinator leverages artificial intelligence to assist in preliminary medical diagnostics. It integrates:

  Web scraping: Collecting detailed disease and symptom data from the NHS website.
  
  Data structuring: Utilizing LLMs (Llama 3.2-1B) to clean, simplify, and normalize symptom descriptions.
  
  ID3 Decision Tree: Dynamically guiding users through symptom-based questions to narrow down potential diagnoses.

1_web_scraping_links.py: Script to scrape disease names and links from the NHS conditions page.
2_web_scraping_one_disease.py: Script to test and extract symptoms for a single disease.
3_web_scraping_all_diseases.py: Script to scrape symptoms for all diseases from the NHS.
data_structuring_using_llama.py: Code for cleaning, structuring, and normalizing symptoms using LLMs.
disease_akinator_v2.py: The main tool integrating the decision tree model and GUI.

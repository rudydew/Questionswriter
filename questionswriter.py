import mysql.connector
import openai
import requests
import json
import re
import random
import base64
import math
import hashlib
import html
import argparse
import sys
import time
from bs4 import BeautifulSoup
from io import BytesIO
from PIL import Image, ImageEnhance, ImageOps
from openai.error import APIError, ServiceUnavailableError

# MySQL database credentials
db_config = {
    'host': '***REMOVED***',
    'user': '***REMOVED***',
    'password': '***REMOVED***',
    'database': '***REMOVED***'
}

# OpenAI Api key
openai.api_key = '***REMOVED***'

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Post articles to WordPress with a specified category.')
parser.add_argument('--category', type=str, help='Category name for the WordPress post.')
parser.add_argument('--fetch', type=str, help='Fetch a keyword with a specific subcategory.')
parser.add_argument('--limit', type=int, default=1, help='Limit the number of main keywords to process.')
parser.add_argument('--site', type=str, help='Site to post the article on.', required=True)
args = parser.parse_args()

# Use the category argument value
category_name = args.category
# Use the fetch argument value if provided
subcategory = args.fetch



site_configs = {
    '***REMOVED***': {
        'wp_endpoint': 'https://www.***REMOVED***.com/wp-json/wp/v2/posts',
        'wp_media_endpoint': 'https://www.***REMOVED***.com/wp-json/wp/v2/media',
        'wp_tags_endpoint' : 'https://***REMOVED***.com/wp-json/wp/v2/tags',
        'wp_api_url' : 'https://www.***REMOVED***.com/wp-json',
        'wp_username': '***REMOVED***',
        'wp_password': '***REMOVED***'
    },
    '***REMOVED***': {
        'wp_endpoint': 'https://***REMOVED***.fr/wp-json/wp/v2/posts',
        'wp_media_endpoint': 'https://***REMOVED***.fr/wp-json/wp/v2/media',
        'wp_tags_endpoint' : 'https://***REMOVED***.com/wp-json/wp/v2/tags',
        'wp_api_url' : 'https://www.***REMOVED***.com/wp-json',
        'wp_username': 'rudy@***REMOVED***.fr',
        'wp_password': '***REMOVED***'
    }
}


# Retrieve the site configuration based on the --site argument
site_config = site_configs.get(args.site)
if not site_config:
    print(f"Configuration for site '{args.site}' not found.")
    sys.exit(1)

# Use the site-specific configuration
wp_endpoint = site_config['wp_endpoint']
wp_media_endpoint = site_config['wp_media_endpoint']
wp_api_url = site_config['wp_api_url']
wp_tags_endpoint = site_config['wp_tags_endpoint']
wp_username = site_config['wp_username']
wp_password = site_config['wp_password']
auth = (wp_username, wp_password)
wp_auth = f"{wp_username}:{wp_password}"
wp_auth_header = "Basic " + base64.b64encode(wp_auth.encode()).decode()
main_article_question_answer = None



def generate_wordpress_tag(keyword):
    prompt = f"Generate a concise, relevant tag for an article about '{keyword}'. The tag should reflect the core topic and be suitable for SEO purposes."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You're an AI trained to assist with digital content creation, including SEO optimization."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=60
    )
    tag = response.choices[0].message['content'].strip()
    return tag


def create_wordpress_tag(tag_name):
    data = {
        'name': tag_name
    }
    response = requests.post(wp_tags_endpoint, json=data, auth=auth)
    if response.status_code in [200, 201]:  # HTTP status codes for success and created
        tag_id = response.json()['id']
        print(f"Tag created successfully. Tag ID: {tag_id}")
        return tag_id
    else:
        print(f"Failed to create tag. Response: {response.text}")
        return None

def fetch_or_create_wordpress_tag(tag_name):
    # Check if the tag already exists
    response = requests.get(wp_tags_endpoint, params={'search': tag_name}, auth=auth)
    tags = response.json()
    
    # If the tag exists, return its ID
    for tag in tags:
        if tag['name'].lower() == tag_name.lower():
            print(f"Tag '{tag_name}' already exists. Tag ID: {tag['id']}")
            return tag['id']
    
    # If the tag does not exist, create it
    data = {'name': tag_name}
    response = requests.post(wp_tags_endpoint, json=data, auth=auth)
    if response.status_code in [200, 201]:
        tag_id = response.json()['id']
        print(f"Tag '{tag_name}' created successfully. Tag ID: {tag_id}")
        return tag_id
    else:
        print(f"Failed to create tag '{tag_name}'. Response: {response.text}")
        return None





def generate_image_metadata(keyword, photo_id):
    model = "gpt-3.5-turbo"
    
    # Generate an alt tag using OpenAI based on the keyword
    prompt = f"Générer un alt tag descriptif mais court pour une image sur le sujet: {keyword}."
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "Tu es un assistant spécialiste du sujet."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=60
    )
    alt_tag = response.choices[0].message['content'].strip()

    # Sanitize the alt tag to create a file name
    sanitized_alt_tag = re.sub(r'\W+', '_', alt_tag.lower())
    
    # Remove any trailing underscores from the sanitized alt tag
    sanitized_alt_tag = sanitized_alt_tag.rstrip('_')

    # Construct the file name directly from the sanitized alt tag
    file_name = f"{sanitized_alt_tag}.jpeg"
    
    return file_name, alt_tag


def adjust_image(img):
    # Initial rotation
    rotation_degrees = 90 * 0.03  # 3% of 90 degrees
    rotated_img = img.rotate(rotation_degrees, expand=True, fillcolor='white')

    # Apply skew to the rotated image
    skew_degrees_x = 0.05  # Horizontal skew
    skew_degrees_y = 0.05  # Vertical skew
    width, height = rotated_img.size
    skew_x = math.tan(math.radians(skew_degrees_x))
    skew_y = math.tan(math.radians(skew_degrees_y))
    skew_matrix = (
        1, skew_x, 0,  # Horizontal skew
        skew_y, 1, 0,   # Vertical skew
        0, 0, 1
    )
    skewed_img = rotated_img.transform((width, height), Image.AFFINE, skew_matrix[:6], Image.Resampling.BICUBIC, fillcolor='white')

    # Flip the image horizontally
    flipped_img = ImageOps.mirror(skewed_img)

    # Zoom and crop adjustments with zoom factor
    target_aspect = 1080 / 720
    zoom_factor = 1.3  # Adjust this factor to zoom in more or less aggressively

    # Adjusting based on zoom factor to calculate new dimensions for cropping
    flipped_width, flipped_height = flipped_img.size
    # Apply zoom factor directly to the flipped image's dimensions
    crop_width = flipped_width / zoom_factor
    crop_height = flipped_height / zoom_factor

    # Ensure cropped area maintains the target aspect ratio
    if crop_width / crop_height > target_aspect:
        # Adjust crop_height to maintain aspect ratio
        crop_height = crop_width / target_aspect
    else:
        # Adjust crop_width to maintain aspect ratio
        crop_width = crop_height * target_aspect

    # Calculate the crop area centered on the flipped image
    crop_x_center = flipped_width / 2
    crop_y_center = flipped_height / 2
    crop_area = (
        crop_x_center - crop_width / 2,
        crop_y_center - crop_height / 2,
        crop_x_center + crop_width / 2,
        crop_y_center + crop_height / 2
    )
    cropped_img = flipped_img.crop(crop_area)
    
    # Resize the cropped image to fill the canvas
    final_img = cropped_img.resize((1080, 720), Image.Resampling.LANCZOS)

    # Color enhancement
    final_img = ImageEnhance.Color(final_img).enhance(1.2)  # Increase color saturation
    final_img = ImageEnhance.Contrast(final_img).enhance(1.1)  # Increase contrast
    final_img = ImageEnhance.Brightness(final_img).enhance(1.05)  # Increase brightness

    return final_img






def search_and_upload_photos(keyword, subheadings_count, wp_upload_endpoint, wp_auth):
    # First, generate a query for Pexels using the keyword
    prompt = f"in a single word, in english, what is this topic about?: '{keyword}'"
    pexels_query = generate_with_model_retry(prompt, max_tokens=60).strip()
    print(f"Pexel query generated: \"{pexels_query}\"")
    
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": "***REMOVED***"}
    params = {
        "query": pexels_query,
        "orientation": "landscape",
        "per_page": subheadings_count
    }

    # Search for photos on Pexels
    print("Searching photos on Pexels...")
    response = requests.get(url, headers=headers, params=params)
    print("Pexels Response Code:", response.status_code)
    photos = response.json().get('photos', [])
    print(f"Found {len(photos)} photos")

    uploaded_media_urls = []

    for photo in photos:
        photo_url = photo['src']['original']
        photo_id = photo['id']

        # Generate unique file name and alt tag for this photo
        file_name, alt_tag = generate_image_metadata(keyword, photo_id)
        
        # Download photo
        img_response = requests.get(photo_url)
        if img_response.status_code == 200:
            img = Image.open(BytesIO(img_response.content))

            # Adjust the image (rotate, increase contrast and brightness, adjust warmth, and resize)
            img = adjust_image(img)

            # Save the processed image to a buffer
            processed_img_buffer = BytesIO()
            img.save(processed_img_buffer, format="JPEG")
            processed_img_buffer.seek(0)

            # Upload to WordPress
            files = {'file': (file_name, processed_img_buffer.getvalue())}
            headers = {'Authorization': wp_auth_header}
            print(f"Uploading {file_name} to WordPress...")
            upload_response = requests.post(wp_upload_endpoint, headers=headers, files=files)
            print("Upload Response Code:", upload_response.status_code)
            if upload_response.status_code == 201:  # HTTP 201 Created
                uploaded_media = upload_response.json()
                uploaded_media_url = uploaded_media['source_url']
                uploaded_media_urls.append((uploaded_media_url, alt_tag))
                print("Uploaded Media URL:", uploaded_media_url)
            else:
                print("Failed to upload photo. Response:", upload_response.text)

    return uploaded_media_urls

def fetch_keyword(subcategory=None):
    print("Fetching keyword...")
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor(dictionary=True)
    if subcategory:
        query = "SELECT ID, question FROM questions WHERE is_published IS NULL AND subcategory = %s ORDER BY ID ASC LIMIT 1"
        cursor.execute(query, (subcategory,))
    else:
        query = "SELECT ID, question FROM questions WHERE is_published IS NULL ORDER BY ID ASC LIMIT 1"
        cursor.execute(query)
    result = cursor.fetchone()
    cursor.close()
    connection.close()
    if result:
        print(f"Keyword fetched: {result['question']}")
    else:
        print("No keyword fetched.")
    return result

def update_is_published(keyword_id):
    print(f"Updating is_published for keyword ID: {keyword_id}")
    connection = mysql.connector.connect(**db_config)
    cursor = connection.cursor()
    query = "UPDATE questions SET is_published = TRUE WHERE ID = %s"
    cursor.execute(query, (keyword_id,))
    connection.commit()
    cursor.close()
    connection.close()
    print("is_published updated successfully.")

def generate_with_model(model, prompt, max_tokens=4096, temperature=0.7, **kwargs):
    attempt = 0
    max_attempts = 3
    retry_delay = 10  # seconds
    
    while attempt < max_attempts:
        try:
            messages = [
                {"role": "system", "content": "Tu es un spécialiste de cette question."},
                {"role": "user", "content": prompt}
            ]
            
            if 'role' in kwargs:
                # Adjust 'messages' or handle 'role' as needed
                pass  # Example placeholder for custom logic
                
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                # Note: OpenAI's Chat API may not accept a 'stop' parameter directly
                # stop=kwargs.get('stop', None)  # Example of handling an optional parameter
            )
            return response.choices[0].message['content'].strip()
        except (ServiceUnavailableError, APIError) as e:
            if "server shutdown" in str(e) or isinstance(e, ServiceUnavailableError):
                if attempt == max_attempts - 1:
                    raise  # Re-raise the last exception if all retries fail
                print(f"Attempt {attempt + 1} of {max_attempts} failed due to server issue: {str(e)}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                attempt += 1
            else:
                raise  # If the error is not related to server issues, raise immediately


def generate_with_model_retry(prompt, model="gpt-4-0125-preview", max_retries=3, **kwargs):
    retries = 0
    while retries < max_retries:
        try:
            response = generate_with_model(model, prompt, **kwargs)
            return response
        except Exception as e:  # You can specify the exact exception types if you know them
            print(f"OpenAI APIError: {e} - Retrying ({retries+1}/{max_retries})")
            retries += 1
            time.sleep(1)  # Simple exponential backoff or fixed delay can be implemented here
    raise Exception(f"Failed to get response from OpenAI API after {max_retries} retries.")




def generate_meta_description(keyword):
    # Initial prompt with length instructions
    prompt = f"écrire une meta description concise pour le mot clef: '{keyword}'. La meta description doit donner envie de cliquer et intriguer le lecteur, elle doit inclure le mot clef et être entre 120 et 158 caractères."
    
    # Attempt to generate a meta description that fits the length requirements
    retry_count = 0
    while retry_count < 10:  # Limit the number of retries to avoid infinite loops
        meta_description = generate_with_model_retry(prompt, model="gpt-3.5-turbo-0125", max_tokens=200).strip()
        # Remove double quotes at the beginning and at the end, if any
        meta_description = meta_description.strip('"')
        
        # Check if the meta description is within the desired length range
        if 120 <= len(meta_description) <= 158:
            break  # The meta description meets the requirements
        else:
            # Adjust the prompt to refine the length of the meta description in the next attempt
            prompt = f"Réessayer: une meta description entre 120 et 158 caractères pour '{keyword}'. Doit inclure le mot clef et intriguer le lecteur."
            retry_count += 1

    # Fallback in case all retries fail (use the closest attempt)
    if not (120 <= len(meta_description) <= 158):
        print("Warning: Unable to generate a meta description within the desired length range after retries.")

    return meta_description

def fetch_additional_information(keyword):
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": keyword,
        "gl": "fr",
        "hl": "fr",
        "autocorrect": False
    })
    headers = {
        'X-API-KEY': '***REMOVED***',  # Use your actual API key
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    if response.status_code == 200:
        results = response.json()
        people_also_ask = [item['question'] for item in results.get('peopleAlsoAsk', [])]
        top_search_results = results.get('organic', [])  # Get the search results
        related_searches = [item['query'] for item in results.get('relatedSearches', [])]
        return people_also_ask, top_search_results, related_searches
    else:
        print("Failed to fetch additional information.")
        return [], []


def split_into_paragraphs(text, min_sentences=1, max_sentences=2):
    # Use a regex to split the text into sentences more reliably.
    # This pattern matches periods, question marks, and exclamation points.
    # It's simplified and might need adjustments for edge cases.
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    paragraphs = []
    temp_paragraph = []
    
    for sentence in sentences:
        temp_paragraph.append(sentence)
        # Randomly decide if this sentence should end the current paragraph.
        if len(temp_paragraph) >= random.randint(min_sentences, max_sentences):
            paragraphs.append(' '.join(temp_paragraph))
            temp_paragraph = []
    
    # Add any remaining sentences as a paragraph.
    if temp_paragraph:
        paragraphs.append(' '.join(temp_paragraph))
    
    return paragraphs



def generate_opinion_section_with_paragraphs(keyword, related_searches):
    if not related_searches:
        print("No related searches found for generating the opinion section.")
        return ""
    
    queries_str = ", ".join(related_searches[:-1]) + " et " + related_searches[-1] if len(related_searches) > 1 else related_searches[0]
    prompt = f"Ecrire à la première personne un texte d'opinion sur la question {keyword}, inclure une anecdote et utiliser les expressions suivantes: {queries_str}. Instructions de style: Assurez-vous que les paragraphes et les longueurs de phrases sont hétérogènes, en vous en tenant principalement à des phrases courtes et directes. N'inclure aucun remplissage. Chaque phrase doit apporter de la valeur. N'utilisez pas toujours les mots les plus naturels. Soyez conversationnel, empathique, occasionnellement humoristique, et utilisez des idiomes, métaphores, anecdotes, et un dialogue naturel."
    
    opinion_text = generate_with_model_retry(prompt, model="gpt-4-0125-preview", max_tokens=4096)
    
    # Use the updated splitting logic.
    paragraphs = split_into_paragraphs(opinion_text)
    
    # Format paragraphs for WordPress
    opinion_section = "<!-- wp:heading -->\n<h2>Mon avis (opinion)</h2>\n<!-- /wp:heading -->\n"
    for paragraph in paragraphs:
        opinion_section += f"<!-- wp:paragraph -->\n<p>{paragraph}</p>\n<!-- /wp:paragraph -->\n"
    
    return opinion_section




def clean_html_content(html_content):
    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove unnecessary whitespaces and newlines around and inside <b> tags
    for b_tag in soup.find_all('b'):
        if b_tag.string:
            b_tag.string.replace_with(b_tag.get_text(strip=True))
    
    # Convert the soup object back to string
    cleaned_html = str(soup)
    
    # Ensure proper spacing around <b> tags without introducing new lines
    cleaned_html = re.sub(r'(\w)(<b>)', r'\1 \2', cleaned_html)
    cleaned_html = re.sub(r'(</b>)(\w)', r'\1 \2', cleaned_html)

    # Optionally, extend this cleanup to other tags as needed
    cleaned_html = re.sub(r'\s*<p>\s*', '<p>', cleaned_html)
    cleaned_html = re.sub(r'\s*</p>\s*', '</p>', cleaned_html)
    cleaned_html = re.sub(r'\s*<i>\s*', '<i>', cleaned_html)
    cleaned_html = re.sub(r'\s*</i>\s*', '</i>', cleaned_html)

    # Normalize space usage throughout
    cleaned_html = re.sub(r'\s+', ' ', cleaned_html)  # Collapse multiple spaces into one
    cleaned_html = re.sub(r' >', '>', cleaned_html)   # Remove space before closing tag character
    cleaned_html = re.sub(r'< ', '<', cleaned_html)   # Remove space after opening tag character

    return cleaned_html

# Function to normalize text
def normalize_text(html):
    """Strip HTML tags, then trim and lower the case of the text."""
    return BeautifulSoup(html, "html.parser").get_text().strip().lower()


def generate_content(keyword, is_supporting_article=False, main_article_url=None, main_article_keyword=None, main_article_question=None, main_article_answer=None):
    print(f"Generating content for keyword: {keyword}")

     # Fetch "People Also Ask" questions and top search results
    people_also_ask_questions, top_search_results, related_searches = fetch_additional_information(keyword)

    # Initialize list for scraped contents and a counter for successful scrapes
    scraped_contents = []
    successful_scrapes = 0

    for url in top_search_results[:10]:  # Assuming top_search_results can have up to 10 items
        if successful_scrapes >= 3:  # Target number of successful scrapes
            break  # Stop if we've reached the target number of articles
        
        content = scrape_webpage_content(url['link'])
        if content and len(content) <= 20000:  # Check if content is not excessively long
            scraped_contents.append(content)
            successful_scrapes += 1
        else:
            print(f"Content from {url['link']} is too long or empty, skipping...")

    if successful_scrapes < 3:
        print("Warning: Failed to scrape sufficient content. Proceeding with what we have.")

    # Generate a combined summary of the scraped contents
    combined_summary = generate_combined_summary(scraped_contents)
    table_summary = generate_table_summary(combined_summary)  # Pass the combined_summary as an argument

    
    # Initial title generation attempt
    title_prompt = f"Utiliser les informations contenues dans ce résumé: {combined_summary} : Générer un titre concis (moins de 60 caractères) pour un article sur '{keyword}'. Le titre doit inclure le mot clef exact au début, et des parenthèses. ne pas inclure l'année."
    title = generate_with_model_retry(title_prompt, model="gpt-4-0125-preview")

    # Remove double quotes from the title and initialize shortest_title
    title = title.replace('"', '')
    shortest_title = title

    # Retry logic if title exceeds 60 characters, but stop as soon as a valid title is found
    retry_count = 0
    while retry_count < 15:
        print(f"Title too long ({len(shortest_title)} characters), trying again.")
        title = generate_with_model_retry(title_prompt, model="gpt-4-0125-preview")
        
        # Remove double quotes from the generated title before further processing
        title = title.replace('"', '')

        if len(title) <= 60:
            shortest_title = title
            break
        elif len(title) < len(shortest_title):
            shortest_title = title
        retry_count += 1

    title = shortest_title

    # Generate introduction
    intro_prompt = f"Utiliser les informations contenues dans ce résumé pour écrire une introduction de deux phrases pour un article ciblant le mot clef: {keyword}: {combined_summary}. Instructions de style: Assurez-vous que les paragraphes et les longueurs de phrases sont hétérogènes, en vous en tenant principalement à des phrases courtes et directes. N'inclure aucun remplissage. Chaque phrase doit apporter de la valeur. N'utilisez pas toujours les mots les plus naturels. Soyez conversationnel, empathique, occasionnellement humoristique, et utilisez des idiomes, métaphores, anecdotes, et un dialogue naturel."
    intro = generate_with_model_retry(intro_prompt, model="gpt-4-0125-preview", max_tokens=1000)
    intro_block = f"<!-- wp:paragraph -->\n<p>{intro}</p>\n<!-- /wp:paragraph -->"

    # Keyword paragraph block
    keyword_block = f"<!-- wp:paragraph -->\n<p><em>{keyword}</em></p>\n<!-- /wp:paragraph -->"

    # Generate direct answer
    direct_answer_prompt = f"Utiliser les informations contenues dans ce résumé: - {combined_summary} - écrire une réponse directe, d'une seule phrase, et qui peut se lire en dehors de tout autre contexte, comme une réponse d'encyclopédie, à la question {keyword}."
    direct_answer = generate_with_model_retry(direct_answer_prompt, model="gpt-3.5-turbo-0125", max_tokens=1000)
    global main_article_question_answer
    main_article_question_answer = direct_answer
    direct_answer_block = f"<!-- wp:paragraph -->\n<p><b>{direct_answer}</b></p>\n<!-- /wp:paragraph -->"

    # Shortcode block for [ez-toc]
    shortcode_block = f"<!-- wp:shortcode -->\n[ez-toc]\n<!-- /wp:shortcode -->"

    # Compile the article content with introduction, keyword, direct answer, and shortcode
    body = f"{intro_block}\n\n{keyword_block}\n\n{direct_answer_block}\n\n{shortcode_block}"

    # Generate subheadings and corresponding paragraphs
    subheading_prompt = f"Utiliser les informations contenues dans ce résumé: -- {combined_summary} -- Générer 2 à 4 sous-titres pour un article sur le sujet: '{keyword}' permettant de couvrir le sujet totalement; Ne pas utiliser de numéros, ne pas générer de sous-titre d'introduction ou de conclusion."
    subheadings = generate_with_model_retry(subheading_prompt, model="gpt-4-0125-preview", max_tokens=1000).split('\n')

    # Print and clean the generated subheadings
    cleaned_subheadings = []
    for subheading in subheadings:
        # Remove numeric prefix (e.g., "1. ") using regular expressions
        cleaned_subheading = re.sub(r'^(\d+\.\s*|- )', '', subheading.strip())
        # Remove double quotes
        cleaned_subheading = cleaned_subheading.replace('"', '')

        if cleaned_subheading:  # Ensure the subheading is not empty after cleaning
            cleaned_subheadings.append(cleaned_subheading)

    print("Generated Subheadings:", cleaned_subheadings)

    # Searching for Images on Pexels

    # Call the function to search for photos, download them, and upload to WordPress

    # Define WordPress media upload endpoint and authentication credentials before using them
    wp_media_endpoint = 'https://wordpress-460823-4353340.cloudwaysapps.com/wp-json/wp/v2/media'
    wp_auth = f"{wp_username}:{wp_password}"  # This might need encoding or specific formatting depending on your auth method
    
    # Generate Basic Auth Header (if required by your WordPress setup)
    uploaded_media_urls = search_and_upload_photos(
        keyword,  # Pass the keyword argument here
        len(cleaned_subheadings), 
        wp_media_endpoint, 
        wp_auth_header  # Ensure this is passed correctly
    )



    # Assume uploaded_media_urls contains URLs of uploaded images
    uploaded_media_index = 0  # Initialize an index to keep track of which image to insert next
    
    # Iterate through each cleaned subheading
    for subheading in cleaned_subheadings:
        subheading_block = f"<!-- wp:heading -->\n<h2>{subheading}</h2>\n<!-- /wp:heading -->\n"
        paragraph_prompt = f"Dans un article intitulé {keyword}, écrire les paragraphes pour le sous-titre: {subheading}, en utilisant la balise <b> pour les mots importants. Ne pas écrire le sous-titre ni faire d'introduction. Instructions de style: Assurez-vous que les paragraphes et les longueurs de phrases sont hétérogènes, en vous en tenant principalement à des phrases courtes et directes. N'inclure aucun remplissage. Chaque phrase doit apporter de la valeur. N'utilisez pas toujours les mots les plus naturels. Soyez conversationnel, empathique, occasionnellement humoristique, et utilisez des idiomes, métaphores, anecdotes, et un dialogue naturel."
        paragraph = generate_with_model_retry(paragraph_prompt, model="gpt-4-0125-preview", max_tokens=4096)
        
        paragraphs = paragraph.split('\n')
        paragraph_blocks = ""
        
        # Normalize the subheading for comparison
        normalized_subheading = normalize_text(subheading)
        
        # Iterate over paragraphs to add them to the body content
        for p in paragraphs:
            if p.strip():
                normalized_paragraph = normalize_text(p)
                if normalized_paragraph != normalized_subheading:
                    paragraph_blocks += f"<!-- wp:paragraph -->\n<p>{p}</p>\n<!-- /wp:paragraph -->\n\n"
                
        # After handling all paragraphs for a subheading, handle image insertion
        if uploaded_media_index < len(uploaded_media_urls):
            media_url, alt_tag = uploaded_media_urls[uploaded_media_index]
            # Escape the alt tag to ensure it's safe for HTML insertion
            escaped_alt_tag = html.escape(alt_tag)
            # Strip the double quotes that sometimes appear
            stripped_alt_tag = escaped_alt_tag.replace('&quot;', '')
            image_html = f"""<!-- wp:image {{"sizeSlug":"full"}} -->
        <figure class="wp-block-image"><img src="{media_url}" alt="{stripped_alt_tag}" /></figure>
        <!-- /wp:image -->
        """
            uploaded_media_index += 1
        else:
            image_html = ""
        

        
        # Append the subheading, image, and paragraph blocks to your article content
        body += f"\n{subheading_block}\n{image_html}\n{paragraph_blocks}"

    # Generate "Récapitulatif" heading and table only once, correctly placed before the FAQ section
    recap_header = "<!-- wp:heading -->\n<h2>Récapitulatif</h2>\n<!-- /wp:heading -->\n"
    table_summary = generate_table_summary(body)  # Generate table based on content generated so far
    formatted_table = format_table_for_wordpress(table_summary)  # Format the table for WordPress
    body += recap_header + formatted_table  # Append the "Récapitulatif" section and table to the body

    # Generate the opinion section content:
    opinion_section = generate_opinion_section_with_paragraphs(keyword, related_searches)
    body += f"\n{opinion_section}\n"

    # Start of the FAQ section with a header
    faq_header = "<!-- wp:heading -->\n<h2>Questions fréquentes</h2>\n<!-- /wp:heading -->\n"
    body += faq_header


    if is_supporting_article and main_article_question and main_article_answer and main_article_url:
        # Include the main article's question, answer, and a link to the main article
        main_question_block = f"<!-- wp:heading {{\"level\":3}} -->\n<h3>{main_article_question}</h3>\n<!-- /wp:heading -->\n"
        main_answer_block = f"<!-- wp:paragraph -->\n<p>{main_article_answer}</p><p>Pour plus d'informations, voir mon article sur: <a href='{main_article_url}'>{main_article_keyword}</a>.</p>\n<!-- /wp:paragraph -->\n"
        body += main_question_block + main_answer_block


    for question in people_also_ask_questions:
        # Wrap each question in an H3 Gutenberg block
        question_block = f"<!-- wp:heading {{\"level\":3}} -->\n<h3>{question}</h3>\n<!-- /wp:heading -->\n"
    
        # Generate a direct answer for the question
        direct_answer_prompt = f"écrire une réponse directe à la question: '{question}'. Répondre comme une encyclopédie, la réponse doit pouvoir être lue seule en dehors de tout contexte (sans avoir vu la question)."
        direct_answer = generate_with_model_retry(direct_answer_prompt, model="gpt-4-0125-preview", max_tokens=500)
    
        # Wrap the answer in a paragraph Gutenberg block
        answer_block = f"<!-- wp:paragraph -->\n<p>{direct_answer}</p>\n<!-- /wp:paragraph -->\n"

        # Append the question and its answer to the body
        body += question_block + answer_block

    # Now, append the top search results as links formatted as a bullet point list at the end of the article
    search_results_header = "<!-- wp:heading -->\n<h2>Liens Utiles</h2>\n<!-- /wp:heading -->\n"
    search_results_list_start = "<!-- wp:list -->\n<ul>\n"  # Start of list block
    search_results_list_items = ""

    for result in top_search_results[:3]:  # Limit to the first 3 results
        list_item = f"<li><a href='{result['link']}' target='_blank'>{result['title']}</a></li>\n"
        search_results_list_items += list_item

    search_results_list_end = "</ul>\n<!-- /wp:list -->\n"  # End of list block

    # Combine all parts to form the complete list block
    search_results_list = search_results_list_start + search_results_list_items + search_results_list_end

    # Append this new section to the article body
    body += search_results_header + search_results_list

    # Clean the HTML content of the body
    body = clean_html_content(body)

    return title, body



def generate_table_summary(combined_summary):
    prompt = (
        "A partir de ces informations, créer un tableau utile pour le lecteur pour comprendre rapidement le sujet."
        "Le tableau doit être facile a lire pour un utilisateur comme pour un moteur de recherche."
        "Utiliser les informations de cet article:\n\n"
        f"{combined_summary}"
    )
    # Assuming generate_with_model_retry can handle parameters for ChatCompletion
    table_summary = generate_with_model_retry(
        prompt=prompt,
        model="gpt-4-0125-preview",
        temperature=0.5,
        max_tokens=1024,  # Adjust based on the complexity of your content
        role="system"  # This parameter and how you handle it might need to be adjusted based on your implementation of generate_with_model_retry
    )
    # Remove any occurrences of "**"
    table_summary = table_summary.replace("**", "")
    return table_summary





def is_markdown_table(table_summary):
    # Simple check for Markdown table indicators
    return "|" in table_summary and "---" in table_summary

def convert_markdown_table_to_html(markdown_table):
    lines = markdown_table.strip().split('\n')
    # Assuming the first line is the header
    headers = lines[0].split('|')[1:-1]  # Exclude the outer empty cells
    header_html = ''.join(f'<th>{header.strip()}</th>' for header in headers)
    
    body_html = ''
    for line in lines[2:]:  # Skip the markdown header separator
        cells = line.split('|')[1:-1]  # Exclude the outer empty cells
        row_html = ''.join(f'<td>{cell.strip()}</td>' for cell in cells)
        body_html += f'<tr>{row_html}</tr>'
    
    html_table = f'<table><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>'
    return html_table

def format_table_for_wordpress(table_summary):
    if is_markdown_table(table_summary):
        html_table = convert_markdown_table_to_html(table_summary)
    else:
        html_table = table_summary  # Assume it's already in HTML or other non-Markdown format
    return "<!-- wp:html -->\n" + html_table + "\n<!-- /wp:html -->"



def integrate_table_summary(article_content):
    table_summary = generate_table_summary(article_content)
    formatted_table = format_table_for_wordpress(table_summary)
    recap_section = "<!-- wp:heading -->\n<h2>Récapitulatif</h2>\n<!-- /wp:heading -->\n" + formatted_table
    # Append the Récapitulatif section to the article content
    return article_content + "\n\n" + recap_section


def fetch_or_create_wordpress_category(category_name):
    wp_categories_endpoint = 'https://wordpress-460823-4353340.cloudwaysapps.com/wp-json/wp/v2/categories'
    response = requests.get(wp_categories_endpoint, params={'search': category_name}, auth=auth)
    categories = response.json()
    
    # If the category exists, return its ID
    for category in categories:
        if category['name'].lower() == category_name.lower():
            print(f"Category '{category_name}' already exists. Category ID: {category['id']}")
            return category['id']
    
    # If the category does not exist, create it
    data = {'name': category_name}
    response = requests.post(wp_categories_endpoint, json=data, auth=auth)
    if response.status_code in [200, 201]:
        category_id = response.json()['id']
        print(f"Category '{category_name}' created successfully. Category ID: {category_id}")
        return category_id
    else:
        print(f"Failed to create category '{category_name}'. Response: {response.text}")
        return None







def post_to_wordpress(title, content, meta_description, featured_media_id, tag_id, category_id=None, status='pending'):
    print("Posting to WordPress...")
    data = {
        'title': title,
        'content': content,
        'status': status,
        'featured_media': featured_media_id,  # Set the featured image
        'meta': {
            '_yoast_wpseo_metadesc': meta_description 
        },
        'tags': [tag_id]
    }

    if category_id:
        data['categories'] = [category_id]  # Assign the category ID to the post


    response = requests.post(wp_endpoint, auth=auth, json=data)

    if response.status_code in [200, 201]:  # Successful creation
        post_data = response.json()
        post_id = post_data.get('id')
        post_url = post_data.get('link')  # Extracting the URL
        print(f"Post successfully created with status '{status}'. URL: {post_url}")
        return post_id, post_url  # Returning both ID and URL
    else:
        print(f"Failed to create post. Status Code: {response.status_code}")
    
    # Inspect the HTTP response
    print("Status Code:", response.status_code)  # Print the status code
    try:
        response_json = response.json()
        #print("JSON Response:", json.dumps(response_json, indent=4))  # Print formatted JSON response
        # Check for and print WordPress-specific errors if present
        if 'code' in response_json:
            print("Error Code:", response_json['code'])
        if 'message' in response_json:
            print("Error Message:", response_json['message'])
    except ValueError:
        # If response is not in JSON format, print the raw text
        print("Response is not in JSON format.")
        print("Response Body:", response.text)

    print("Post response received.")
    return response.json()





def publish_and_revert_post(post_id, wp_api_url, auth):
    # Headers for Authorization
    headers = {"Authorization": auth}
    
    # Publish the post
    publish_response = requests.post(
        f"{wp_api_url}/wp/v2/posts/{post_id}",
        headers=headers,
        json={"status": "publish"}
    )

    # Fetch the definitive URL immediately
    definitive_url = publish_response.json().get("link")

    # Check if the post was published successfully before attempting to revert
    if publish_response.status_code == 200 or publish_response.status_code == 201:
        # Revert the post to pending
        requests.post(
            f"{wp_api_url}/wp/v2/posts/{post_id}",
            headers=headers,
            json={"status": "pending"}
        )
    else:
        print("Failed to publish the post. Status Code:", publish_response.status_code)
        return None

    return definitive_url












def scrape_webpage_content(url):
    print(f"Attempting to scrape content from URL: {url}")
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser', from_encoding='utf-8')
            content = soup.find('article') or soup.find('div', {'id': 'mw-content-text'}) or soup.find('body').find('p')
            if content:
                print(f"Successfully fetched content from {url}")
                return content.text.strip()
            else:
                print(f"Content structure not found for {url}. Skipping...")
                return ""
        else:
            print(f"HTTP Error {response.status_code} for URL: {url}. Skipping...")
            return ""
    except requests.Timeout:
        print(f"Request timed out for URL: {url}. Skipping...")
        return ""
    except requests.ConnectionError:
        print(f"Connection error occurred while fetching URL: {url}. Skipping...")
        return ""
    except Exception as e:
        print(f"An error occurred while fetching URL: {url}: {str(e)}. Skipping...")
        return ""


def generate_combined_summary(contents):
    # Combine the initial summaries
    combined_summary_prompt = "Résumer ces articles en un récapitulatif exhaustif:\n\n" + "\n\n---\n\n".join(contents)
    
    print("Generating combined summary...")  # Inform the user that summary generation is starting.
    
    # Adapted to use generate_with_model_retry, assuming it can handle parameters for ChatCompletion
    combined_summary = generate_with_model_retry(
        prompt=combined_summary_prompt,
        model="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=1024,  # Adjust based on the complexity of your content, ensure this parameter is handled by generate_with_model_retry
        role="system"  # Make sure to adjust your generate_with_model_retry to handle this parameter if necessary
    ).strip()
    
    # Print the generated summary
    print("Generated Combined Summary:")
    print(combined_summary)
        
    return combined_summary




def fetch_latest_media_id(wp_media_endpoint, wp_auth_header):
    """
    Fetches the ID of the latest uploaded image from the WordPress media library.

    Args:
    wp_media_endpoint (str): WordPress media library endpoint.
    wp_auth_header (str): Authorization header for WordPress API requests.

    Returns:
    int: The ID of the latest uploaded media, or None if not found.
    """
    headers = {'Authorization': wp_auth_header}
    response = requests.get(wp_media_endpoint, headers=headers, params={'per_page': 1, 'orderby': 'date', 'order': 'desc'})
    
    if response.status_code == 200:
        media_items = response.json()
        if media_items:
            return media_items[0]['id']  # Return the ID of the latest uploaded media
    return None



def create_and_post_supporting_articles(paa_questions, main_article_url, main_article_keyword, tag_id, category_id):
    for question in paa_questions:
        # Generate content for the supporting article using the PAA question
        print(f"Generating supporting article for question: {question}")
        title, content = generate_content(
            keyword=question,
            is_supporting_article=True,
            main_article_url=main_article_url,
            main_article_keyword=main_article_keyword,
            main_article_question=question,  # Assuming you want to repeat the question in the supporting article
            main_article_answer=main_article_question_answer  # You may want to customize this part based on how you handle answers
        )
        
        
        # Generate a meta description for the supporting article
        meta_description = generate_meta_description(question)
        
        # Fetch the latest media ID if you plan to use images
        latest_media_id = fetch_latest_media_id(wp_media_endpoint, wp_auth_header)
        
        # Post the supporting article to WordPress
        post_id, supporting_article_url = post_to_wordpress(title, content, meta_description, latest_media_id, tag_id, category_id)
        
        if post_id:
            print(f"Supporting article posted successfully. URL: {supporting_article_url}")
        else:
            print("Failed to post supporting article.")






def process_keyword(keyword_data, category_name, wp_api_url, wp_auth_header, wp_media_endpoint):
    print("Processing keyword:", keyword_data['question'])
    
    # Fetch "People Also Ask" questions and other related information for the keyword
    people_also_ask_questions, top_search_results, related_searches = fetch_additional_information(keyword_data['question'])

    # Fetch or create a WordPress tag using the main article keyword
    main_article_keyword = keyword_data['question']  # This is your main keyword
    tag_id = fetch_or_create_wordpress_tag(main_article_keyword)

    # Generate and post the main article using the main_article_keyword directly
    title, content = generate_content(
        keyword=keyword_data['question'],
        is_supporting_article=False,
        main_article_url=None,
        main_article_keyword=main_article_keyword,
        main_article_question=None,
        main_article_answer=None
    )

    meta_description = generate_meta_description(keyword_data['question'])
    latest_media_id = fetch_latest_media_id(wp_media_endpoint, wp_auth_header)

    # Fetch or create the WordPress category using the provided category name
    if category_name:
        category_id = fetch_or_create_wordpress_category(category_name)
    else:
        category_id = None

    # Temporarily post the main article as published to get the definitive URL
    post_id, temp_url = post_to_wordpress(title, content, meta_description, latest_media_id, tag_id, category_id, status='publish')
    
    # Ensure the post was successfully created
    if post_id:
        print(f"Temporarily published article. Post ID: {post_id}, Temp URL: {temp_url}")
        
        # Immediately revert the article's status to pending and fetch the definitive URL
        definitive_url = publish_and_revert_post(post_id, wp_api_url, wp_auth_header)
        if definitive_url:
            print(f"Article reverted to pending. Definitive URL: {definitive_url}")
            main_article_url = definitive_url  # Use this URL for internal linking in supporting articles
            update_is_published(keyword_data['ID'])  # Mark the keyword as published in your database
            
            # With people_also_ask_questions populated, create and post supporting articles
            create_and_post_supporting_articles(people_also_ask_questions, main_article_url, main_article_keyword, tag_id, category_id)  # Passing tag_id to be used for supporting articles
        else:
            print("Failed to revert article to pending or fetch definitive URL.")
    else:
        print("Failed to post article.")


print("Starting main execution flow...")

# Assume args.limit is defined via argparse as shown previously
processed_keywords = 0
while processed_keywords < args.limit:
    keyword_data = fetch_keyword(subcategory=subcategory)
    if keyword_data:
        process_keyword(keyword_data, category_name, wp_api_url, wp_auth_header, wp_media_endpoint)
        processed_keywords += 1
    else:
        print("No more keywords to process.")
        break


    
    




# WordPress Response Post AI Automation Script

This Python script automates the process of researching, creating, and publishing question / answer based articles.

- It scrapes the web pages for top three results on google (uses serper.dev API to get the results) for the question
- creates a summary (openAI API) of the critical information contained in these 3 pages
- generates a short, snippet-optimized answer
- generates semantically relevant subheadings, and content
- scrapes relevant images from pexels (pexels API)
- manipulates the images by rotating, zoom, color filters... to create a unique file
- generates alt tags
- generates a summary table with relevant columns
- scrapes relevant additional questions (serper.dev API)
- answers each question
- uploads the media files to wordpress library, generates and publish as draft the complete wordpress article.

## Requirements

Before you begin, ensure you have Python 3 installed on your system. Additionally, you will need to install the following Python libraries:

- `mysql.connector`
- `openai`
- `requests`
- `beautifulsoup4`
- `Pillow`

You can install these libraries using pip:

pip install mysql-connector-python openai requests beautifulsoup4 Pillow

Setup

MySQL Database and OpenAI API Key:
Ensure that your MySQL database access credentials and your OpenAI API key are correctly configured within the script.
Python Libraries:
Install the required Python libraries as mentioned above.
Pexels API Key
Serper.dev api key


## Usage

The script is designed to accept command-line arguments for dynamic operation. Here are the options you can use:

- `--category`: The category name for the WordPress post.
- `--fetch`: Fetch a keyword with a specific subcategory.
- `--limit`: Limit the number of main keywords to process. Default is 1.
- `--site`: The site to post the article on. This argument is required.
- `--skipsupport`: if this is set, it will only process the main keywords and skip the supporting articles.
- `--language`: set to either fr/en/es, this will change the prompts language and serper api request to relevant country code.

### Example Command

```bash
python script.py --site yogowo --category "Health" --fetch "Nutrition" --limit 2 --language fr --skipsupport
```

## Features

Keyword Fetching: Fetches and processes keywords from a MySQL database.
SEO Optimization: Generates SEO-friendly tags and meta descriptions using the OpenAI API.
Image Adjustments and Upload: Adjusts images and uploads them to the WordPress media library.
Article Posting: Posts articles to WordPress with specified categories and tags.
Related Content Generation: Supports generating and posting related content based on "People Also Ask" questions.
Configuration

Modify the db_config and site_configs variables in the script to match your MySQL database and WordPress site configurations, respectively.

## Important Note

Ensure that you have valid credentials for the MySQL database, OpenAI API, and WordPress site configured in the script. Incorrect configurations may lead to execution errors.

For the Yoast meta description field to work, add the following to your theme's functions.php: 

```php
/*** Rest api field for yoast meta desc ***/
add_action('init', function() {
    register_post_meta('post', '_yoast_wpseo_metadesc', [
        'single' => true,
        'type' => 'string',
        'show_in_rest' => true,
    ]);
});
```

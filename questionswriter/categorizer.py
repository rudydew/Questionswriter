import openai
import mysql.connector

# Configuration
openai.api_key = '***REMOVED***'
db_config = {'user': '***REMOVED***', 'password': '***REMOVED***', 'host': '***REMOVED***', 'database': '***REMOVED***'}

def get_questions_batch(limit, offset):
    """Retrieve a batch of questions from the database."""
    print(f"Retrieving batch: limit={limit}, offset={offset}")
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("SELECT id, question FROM questions WHERE category IS NULL LIMIT %s OFFSET %s", (limit, offset))
    questions = cursor.fetchall()
    cursor.close()
    conn.close()
    print(f"Retrieved {len(questions)} questions")
    return questions

def update_database_with_categories(question_id, category, subcategory):
    """Update the database with the determined category and subcategory."""
    print(f"Updating database for question ID {question_id} with category '{category}' and subcategory '{subcategory}'")
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("UPDATE questions SET category = %s, subcategory = %s WHERE id = %s", (category, subcategory, question_id))
    conn.commit()
    cursor.close()
    conn.close()
    print("Database updated successfully.")

def categorize_with_openai(questions, model="gpt-3.5-turbo", max_tokens=1000):
    """
    Categorize questions using the OpenAI API, guiding the model to choose from a predefined list of categories and subcategories.
    """
    print(f"Categorizing {len(questions)} questions with OpenAI API")
    categorized_questions = []

    categories_hint = "Categories: Arts & Literature, Science & Mathematics, Social Sciences & Humanities, Technology & Engineering, Business & Finance, Health & Medicine, Sports & Recreation, Lifestyle, Cooking, & Leisure, Environment & Geography, Politics, Law & Government, Education & Career, Culture & Society. Subcategories include, but are not limited to, Visual Arts, Literature, Music, Film & Theater for Arts & Literature; Biology, Chemistry, Physics, Mathematics for Science & Mathematics, etc."

    for question_id, question_text in questions:
        try:
            prompt = f"{categories_hint}\n\nClassify the following question into one of the above categories and subcategories:\n\n{question_text}\n\nCategory and Subcategory:"
            response = openai.Completion.create(
                engine=model,
                prompt=prompt,
                max_tokens=max_tokens,
                n=1,
                stop=None,
                temperature=0.5
            )
            response_text = response.choices[0].text.strip()
            category_subcategory = response_text.split(',')
            category = category_subcategory[0].strip()
            subcategory = category_subcategory[1].strip() if len(category_subcategory) > 1 else 'General'

            print(f"Question ID {question_id} categorized as '{category}' and '{subcategory}'")
            categorized_questions.append((question_id, question_text, category, subcategory))
        except Exception as e:
            print(f"Error processing question {question_id}: {e}")

    return categorized_questions

def process_questions():
    batch_size = 50
    offset = 0
    while True:
        print(f"Processing batch starting from offset {offset}")
        questions = get_questions_batch(batch_size, offset)
        if not questions:
            print("No more questions to process. Exiting.")
            break
        categorized_questions = categorize_with_openai(questions)
        for question_id, category, subcategory in categorized_questions:
            update_database_with_categories(question_id, category, subcategory)
        offset += batch_size

# Run the processing function
process_questions()
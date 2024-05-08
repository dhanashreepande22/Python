from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
import pandas as pd
from collections import Counter
import re
import gdown  # Import gdown to use it for downloading files from Google Drive

app = FastAPI()

class Item(BaseModel):
    id: int
    prompt: str

@app.post("/identify-task/")
async def identify_task(item: Item):
    id = item.id
    
    # Download the CSV file using gdown
    url = 'https://drive.google.com/uc?id=1Ivswt6nFooeh1dcKAjel-ELepOjf6e1C'
    output = 'application_data.csv'
    gdown.download(url, output, quiet=False)

    # Download necessary NLTK resources
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('wordnet', quiet=True)

    # Define tasks and their specific steps
    tasks = {
        'UI/UX Design': ['Implement responsive design'],
        'API Integration': ['Implement HTTP client', 'Parse JSON data', 'Handle errors and exceptions', 'Cache data', 'Implement pull-to-refresh'],
        'Testing': ['Write unit tests', 'Implement widget tests', 'Fix issues'],
        'Debugging': ['Review logs', 'Use breakpoints', 'Apply fixes'],
        'Performance Optimization': ['Minimize layout rebuilds', 'Use lazy loading', 'Optimize database queries'],
        'Maintaining Codebase': ['Refactor code for readability', 'Update dependencies', 'Resolve merge conflicts'],
        'Adapting to Platform-Specific Features': ['Implement platform channels']
    }

    def load_data(filename):
        # Adjust the path to the location where the file is actually downloaded
        return pd.read_csv(filename)

        
    def extract_keywords(user_prompt):
        stop_words = set(stopwords.words('english'))
        return set(word_tokenize(user_prompt.lower())) - stop_words
    
    def extract_relevant_task(user_prompt):
        prompt_keywords = extract_keywords(user_prompt)
        max_overlap = 0
        best_match = ("No specific task matched", [])
        for task, steps in tasks.items():
            task_keywords = set(task.lower().split())
            overlap = len(prompt_keywords & task_keywords)
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = (task, steps)
        return best_match

    def compare_tags(input_tags, app_tags):
        input_count = Counter(input_tags)
        app_count = Counter(app_tags)
        return sum(min(input_count[tag], app_count[tag]) for tag in input_count if tag in app_count)

    def find_top_apps(df, task_name, steps, top_n=2):
        task_keywords = set(task_name.lower().split())
        steps_keywords = set(word.lower() for step in steps for word in word_tokenize(step))
        input_tags = task_keywords.union(steps_keywords)
        df['score'] = df['tags'].apply(lambda tags: compare_tags(input_tags, extract_keywords(tags)))
        return df.sort_values(by='score', ascending=False).head(top_n)['application_name']
    
    user_prompt = item.prompt
    filename = 'application_data.csv'  # This is the filename where gdown downloaded the file
    df = load_data(filename)
    task_name, steps = extract_relevant_task(user_prompt)
    top_apps = find_top_apps(df, task_name, steps)
    print("Identified Task:", task_name)
    print("Steps:", ', '.join(steps))
    print("\nTop Matching Applications:")
    print(top_apps.to_string(index=False))
    return {"status" : "success", "task": task_name, "steps": steps, "top_apps": top_apps.to_list()}
import os

def check_models_in_read():
    models_path = 'LLM-TPU/models'
    readme_path = 'LLM-TPU/README.md'
    
    with open(readme_path, 'r', encoding='utf-8') as file:
        readme_content = file.read()
    
    missing_models = []
    for model in os.listdir(models_path):
        if model not in readme_content:
            missing_models.append(model)
    
    if missing_models:
        print("Missing model references in README.md:")
        for model in missing_models:
            print(model)
        exit(1)  
    else:
        print("All models are properly referenced in README.md.")

if __name__ == '__main__':
    check_models_in_read()

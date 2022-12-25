import cohere
import random

def main():
    co = cohere.Client(get_key().strip())
    responses = []

    languages = get_languages()

    chosen_languages = []
    for i in range(1):
        lang = random.choice(languages)
        chosen_languages.append(lang)
        responses.append(respond(co, f'translate \"hello\" to {lang}:\n'))
        
    languages = co.detect_language(responses).results
    for i in range(len(responses)):
        print(f'Generated: {responses[i]}\nActual: {chosen_languages[i]}\nPredicted: {languages[i].language_name}\n')

def get_languages():
    with open('languages.txt', 'r') as languages:
        return [lang[0:lang.index('(') - 1] for lang in languages.readlines()]

def get_key():
    with open('key.env', 'r') as key:
        return key.read()

def respond(generator, prompt):
    response = generator.generate(model='xlarge', prompt=prompt, max_tokens=5, temperature=2, end_sequences=['.', '?', '!'])
    return response.generations[0].text

if __name__ == '__main__':
    main()

import cohere
import random

def main():
    co = cohere.Client(get_key().strip())
    responses = []

    languages = get_languages()

    for i in range(1):
        lang = random.choice(languages)
        responses.append(respond(co, f'speak in {lang}:\n'))
        
    languages = co.detect_language(responses).results
    print([f'{responses[i]} -- {lang} (ACTUAL) -- {languages[i].language_name} (PRED)' for i in range(len(responses))])

def get_languages():
    with open('languages.txt', 'r') as languages:
        return [lang[0:lang.index('(') - 1] for lang in languages.readlines()]

def get_key():
    with open('key.env', 'r') as key:
        return key.read()

def respond(generator, prompt):
    response = response = generator.generate(model='xlarge', prompt=prompt, max_tokens=50, temperature=2, end_sequences=['.', '?', '!'])
    return response.generations[0].text

if __name__ == '__main__':
    main()

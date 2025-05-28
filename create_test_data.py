import json

data = [
    {'toxic': 'This movie is absolute garbage, worst thing I\'ve ever seen!', 'clean': 'This movie was not enjoyable and didn\'t meet my expectations.'},
    {'toxic': 'You\'re a complete idiot who knows nothing!', 'clean': 'I disagree with your perspective on this matter.'},
    {'toxic': 'This is the most useless piece of junk software ever!', 'clean': 'This software has significant room for improvement.'},
    {'toxic': 'What a stupid and worthless idea that is!', 'clean': 'I don\'t think this idea would be effective.'}
]

with open('data/detox/test_data.jsonl', 'w') as f:
    for item in data:
        f.write(json.dumps(item) + '\n') 
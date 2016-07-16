import json

subreddits = ['askreddit', 'cscareerquestions', 'funny', 'videos', 'wtf', 'todayilearned', 'pics', 'science', 'worldnews']
files = {}

if __name__ == '__main__':
    f = open('RC_2015-01')
    i = 0
    while True:
        i += 1
        if i % 100000 == 0:
            print(i)
        try:
            line = next(f)
        except:
            break
        msg = json.loads(line)
        if msg['subreddit'].lower() in subreddits and msg['subreddit'] not in files:
            files[msg['subreddit']] = open(msg['subreddit'], 'w+')


        if msg['subreddit'] in files:
            files[msg['subreddit']].write(line)

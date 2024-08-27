import requests
import pandas as pd
import time

# Replace with your GitHub personal access token
TOKEN =('https:github.com/26chanda')
HEADERS = {'Authorization': f'token {TOKEN}'}
BASE_URL = 'https://api.github.com'

def get_comments(repo, page=1, per_page=100):
    """Fetch comments from a GitHub repository."""
    url = f'{BASE_URL}/repos/{repo}/issues/comments'
    params = {'page': page, 'per_page': per_page}
    response = requests.get(url, headers=HEADERS, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data: {response.status_code}")
        return []

def fetch_comments_from_repos(repos, num_comments=1000):
    """Fetch a specified number of comments from a list of repositories."""
    all_comments = []
    for repo in repos:
        page = 1
        while len(all_comments) < num_comments:
            comments = get_comments(repo, page=page)
            if not comments:
                break
            all_comments.extend(comments)
            page += 1
            time.sleep(1)  # Be polite and avoid hitting rate limits
    return all_comments[:num_comments]

def main():
    repos = ['github.com']  # Replace with desired repository names
    num_comments = 1000
    comments = fetch_comments_from_repos(repos, num_comments)

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(comments)
    df.to_csv('code_comments.csv', index=False)
    print(f"Saved {len(df)} comments to 'code_comments.csv'")

if __name__ == "__main__":
    main()

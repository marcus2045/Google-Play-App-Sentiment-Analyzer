# MAIN - START
# IMPORT STATEMENTS - START
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from google_play_scraper import app, Sort, reviews_all, exceptions
import matplotlib.pyplot as plt
nltk.download('vader_lexicon')

# IMPORT STATEMENTS - END

# CONFIGURATION - START
APP_ID = 'com.fitbit.FitbitMobile'  # Change this to any Google Play app ID
LANGUAGE = 'en'                     # Language for reviews
COUNTRY = 'us'                      # Country for reviews
SORT_ORDER = Sort.NEWEST            # Sort order for reviews
REVIEW_COUNT = 10000                # Number of reviews to fetch
# CONFIGURATION - END

# DATA IMPORT - START
try:
    # First, get app details to use for filename
    app_details = app(APP_ID)
    app_name = app_details['title'].replace(' ', '_').lower()
    
    # Fetch reviews
    reviews = reviews_all(
        APP_ID,
        sleep_milliseconds=0,
        lang=LANGUAGE,
        country=COUNTRY,
        sort=SORT_ORDER,
        count=REVIEW_COUNT
    )
    
    # Convert to DataFrame and save
    df = pd.json_normalize(reviews)
    filename = f'{app_name}_reviews.csv'
    df.to_csv(filename, index=False)
    
    print(f"Successfully fetched {len(reviews)} reviews for {app_details['title']}")
    print(f"Saved to {filename}")
    
    # Load the CSV for processing
    csv = pd.read_csv(filename)

except exceptions.NotFoundError:
    print(f"Error: App with ID '{APP_ID}' not found. Please check the app ID.")
    exit()
except Exception as e:
    print(f"An error occurred: {str(e)}")
    exit()

# DATA IMPORT - END

# FUNCTIONS - START
sia = SentimentIntensityAnalyzer()

def remove_chars_from_string(string, chars_to_remove):
    new_string = ""
    for char in string:
        if char not in chars_to_remove:
            new_string += char
    return new_string

def get_sentiment(review):
    """Calculates the sentiment score of a review."""
    review = remove_chars_from_string(str(review), '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    return sia.polarity_scores(review)['compound']

def plot_rating_distribution(df, app_name):
    """Creates a bar chart showing the distribution of ratings."""
    rating_counts = df['score'].value_counts().sort_index()
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(rating_counts.index, rating_counts.values, color=['red', 'orange', 'yellow', 'lightgreen', 'green'])
    plt.title(f'Rating Distribution for {app_name}', fontsize=16)
    plt.xlabel('Star Rating', fontsize=12)
    plt.ylabel('Number of Reviews', fontsize=12)
    plt.xticks([1, 2, 3, 4, 5])
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{app_name}_rating_distribution.png')
    plt.show()
    
    # Print the most popular rating
    most_common_rating = rating_counts.idxmax()
    print(f"\nThe most popular rating is: {most_common_rating} stars ({rating_counts[most_common_rating]} reviews)")
    
    return rating_counts
# FUNCTIONS - END

# SENTIMENT ANALYSIS - START
csv['sentiment_score'] = csv['content'].apply(get_sentiment)
sorted_csv = csv.sort_values(by='sentiment_score', ascending=False)

positive = sorted_csv.head(40)
negative = sorted_csv.tail(40)

print("\nTop 40 positive reviews:")
print(positive[['content', 'sentiment_score']])

print("\nTop 40 negative reviews:")
print(negative[['content', 'sentiment_score']])

# Save with sentiment scores
output_filename = f'{app_name}_with_sentiment.csv'
csv.to_csv(output_filename, index=False)
print(f"\nSentiment analysis complete. Results saved to {output_filename}")

# Print summary statistics
print(f"\nSentiment Analysis Summary for {app_details['title']}:")
print(f"Total reviews analyzed: {len(csv)}")
print(f"Average sentiment score: {csv['sentiment_score'].mean():.4f}")
print(f"Median sentiment score: {csv['sentiment_score'].median():.4f}")
print(f"Minimum sentiment score: {csv['sentiment_score'].min():.4f}")
print(f"Maximum sentiment score: {csv['sentiment_score'].max():.4f}")

# RATING DISTRIBUTION CHART - START
print(f"\nGenerating rating distribution chart...")
rating_counts = plot_rating_distribution(csv, app_details['title'])
print("Rating distribution:")
for rating in sorted(rating_counts.index):
    print(f"{rating} stars: {rating_counts[rating]} reviews ({rating_counts[rating]/len(csv)*100:.1f}%)")
# RATING DISTRIBUTION CHART - END

# MAIN - END

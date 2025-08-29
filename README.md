# Google-Play-App-Sentiment-Analyzer
A Python tool that analyzes user reviews from any Google Play app, performing sentiment analysis and visualizing rating distributions.


Features
Review Scraping: Fetches reviews from any Google Play app
Sentiment Analysis: Uses NLTK's VADER to analyze review sentiment
Rating Distribution: Visualizes star rating distribution with bar charts
top Reviews: Identifies and displays the most positive and negative reviews
Export Functionality: Saves results to CSV files for further analysis

Supported Apps
This tool works with any Android app on Google Play. Just replace the APP_ID with the target app's package name.

Some examples:
Fitbit: com.fitbit.FitbitMobile
Instagram: com.instagram.android
Spotify: com.spotify.music
WhatsApp: com.whatsapp

Dependencies
pandas: Data manipulation and analysis
nltk: Natural language processing and sentiment analysis
google-play-scraper: Fetching reviews from Google Play
matplotlib: Data visualization

License
This project is licensed under the MIT License - see the LICENSE file for details.

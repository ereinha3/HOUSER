import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from preprocess import get_df

# Load data
file_path = 'data/amazon/Gift_Cards.jsonl'
data = get_df(file_path)

average_rating_per_item = data.groupby('item')['rating'].mean().mean()
print('Average rating per item:', average_rating_per_item)

# Distribution of ratings
rating_counts = data['rating'].value_counts().sort_index()
plt.bar(rating_counts.index, rating_counts.values)
plt.xlabel('Rating (Stars)')
plt.ylabel('Number of Reviews')
plt.title('Distribution of Ratings')
plt.savefig('img/rating_distribution.png')
plt.clf() 
plt.close()

# User activity
user_review_counts = data['user'].value_counts().sort_index()  # Count users with the same number of reviews
print('Average reviews per user:', user_review_counts.mean())
plt.bar(user_review_counts.index, user_review_counts.values)
plt.xlabel('Number of Reviews per User')
plt.ylabel('Number of Users')
plt.title('Distribution of Reviews per User')
plt.savefig('img/user_review_distribution.png')
plt.clf() 
plt.close()


# Product popularity
print('computing avg reviews per item...')
product_review_counts = data['item'].value_counts()
print('computer avg reviews per item.')
print('Average reviews per item:', product_review_counts.mean())

# Bin the data to group items with similar numbers of reviews
# Use logarithmic bins to handle the wide range of values
bins = np.logspace(0, np.log10(product_review_counts.max()), num=50)
plt.hist(product_review_counts, bins=bins, edgecolor='black')
plt.xscale('log')  # Use logarithmic scale for x-axis
plt.yscale('log')  # Use logarithmic scale for y-axis
plt.xlabel('Number of Reviews per Item (Log Scale)')
plt.ylabel('Number of Items (Log Scale)')
plt.title('Distribution of Reviews per Product (Log Scale)')
plt.savefig('img/item_review_distribution.png')
plt.clf()  # Clear the current figure
plt.close()  # Close the figure to free up memory
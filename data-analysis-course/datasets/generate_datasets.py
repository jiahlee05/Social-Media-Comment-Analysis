# Generate Sample Datasets for Data Analysis Course

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

print("Generating sample datasets for the course...")

# ============================================
# Dataset 1: Social Media Engagement
# For Session 1
# ============================================

print("\n1. Generating social_media_engagement.csv...")

n_posts = 500

platforms = np.random.choice(['Instagram', 'Twitter', 'Facebook', 'TikTok'], n_posts,
                            p=[0.3, 0.25, 0.25, 0.2])
age_groups = np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+'], n_posts,
                             p=[0.25, 0.30, 0.25, 0.15, 0.05])

# Platform-based engagement (Instagram highest)
platform_multipliers = {'Instagram': 1.5, 'TikTok': 1.3, 'Facebook': 1.0, 'Twitter': 0.8}
likes = []
shares = []
comments = []

for platform in platforms:
    mult = platform_multipliers[platform]
    likes.append(int(np.random.normal(600, 200, 1)[0] * mult))
    shares.append(int(np.random.normal(120, 50, 1)[0] * mult))
    comments.append(int(np.random.normal(45, 20, 1)[0] * mult))

# Post hour
post_hours = np.random.choice(range(24), n_posts,
                             p=[0.01]*6 + [0.03]*3 + [0.06]*6 + [0.08]*6 + [0.04]*3)

df1 = pd.DataFrame({
    'post_id': range(1, n_posts + 1),
    'platform': platforms,
    'age_group': age_groups,
    'likes': np.maximum(0, likes),
    'shares': np.maximum(0, shares),
    'comments': np.maximum(0, comments),
    'post_hour': post_hours
})

df1.to_csv('social_media_engagement.csv', index=False)
print(f"   ✅ Created {len(df1)} rows")

# ============================================
# Dataset 2: News Consumption
# For Session 2
# ============================================

print("\n2. Generating news_consumption.csv...")

n_readers = 600

categories = np.random.choice(['Politics', 'Technology', 'Sports', 'Entertainment',
                              'Business', 'Health', 'Science'], n_readers,
                             p=[0.20, 0.18, 0.15, 0.15, 0.12, 0.12, 0.08])
devices = np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_readers,
                          p=[0.55, 0.35, 0.10])
age_groups = np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+'], n_readers,
                             p=[0.15, 0.30, 0.25, 0.20, 0.10])

# Device affects time spent
device_multipliers = {'Mobile': 0.8, 'Desktop': 1.2, 'Tablet': 1.0}
time_spent = []
articles_read = []
engagement_scores = []

for device in devices:
    mult = device_multipliers[device]
    time_spent.append(max(1, np.random.normal(15, 5, 1)[0] * mult))
    articles_read.append(max(1, int(np.random.normal(5, 2, 1)[0] * mult)))

ages = []
for ag in age_groups:
    if ag == '18-24':
        ages.append(np.random.randint(18, 25))
    elif ag == '25-34':
        ages.append(np.random.randint(25, 35))
    elif ag == '35-44':
        ages.append(np.random.randint(35, 45))
    elif ag == '45-54':
        ages.append(np.random.randint(45, 55))
    else:
        ages.append(np.random.randint(55, 75))

engagement_scores = np.random.uniform(1, 10, n_readers)

df2 = pd.DataFrame({
    'reader_id': range(1, n_readers + 1),
    'category': categories,
    'device': devices,
    'age_group': age_groups,
    'age': ages,
    'time_spent': np.round(time_spent, 1),
    'articles_read': articles_read,
    'engagement_score': np.round(engagement_scores, 2)
})

df2.to_csv('news_consumption.csv', index=False)
print(f"   ✅ Created {len(df2)} rows")

# ============================================
# Dataset 3: Advertising Experiment
# For Session 3
# ============================================

print("\n3. Generating advertising_experiment.csv...")

n_ads = 400

ad_types = np.random.choice(['Image', 'Video', 'Carousel', 'Story'], n_ads,
                           p=[0.25, 0.25, 0.25, 0.25])
genders = np.random.choice(['Male', 'Female'], n_ads, p=[0.48, 0.52])
age_groups = np.random.choice(['18-24', '25-34', '35-44', '45+'], n_ads,
                             p=[0.25, 0.35, 0.25, 0.15])

# Ad type affects conversion (Video best)
ad_type_effects = {
    'Video': {'mean': 0.08, 'std': 0.03},
    'Carousel': {'mean': 0.065, 'std': 0.025},
    'Image': {'mean': 0.055, 'std': 0.02},
    'Story': {'mean': 0.05, 'std': 0.02}
}

conversion_rates = []
engagements = []
costs = []

for ad_type in ad_types:
    effect = ad_type_effects[ad_type]
    conversion_rates.append(max(0, min(1, np.random.normal(effect['mean'], effect['std']))))
    engagements.append(max(0, int(np.random.normal(500, 150))))
    costs.append(max(0.5, np.random.normal(2.5, 0.8)))

df3 = pd.DataFrame({
    'ad_id': range(1, n_ads + 1),
    'ad_type': ad_types,
    'gender': genders,
    'age_group': age_groups,
    'engagement': engagements,
    'conversion_rate': np.round(conversion_rates, 4),
    'cost_per_click': np.round(costs, 2)
})

df3.to_csv('advertising_experiment.csv', index=False)
print(f"   ✅ Created {len(df3)} rows")

# ============================================
# Dataset 4: Social Media Comments
# For Session 4
# ============================================

print("\n4. Generating social_media_comments.csv...")

n_comments = 300

platforms = np.random.choice(['Instagram', 'Twitter', 'Facebook', 'YouTube'], n_comments,
                            p=[0.30, 0.25, 0.25, 0.20])

# Sample comments (positive, neutral, negative)
positive_comments = [
    "I absolutely love this product! Amazing quality!",
    "Best purchase I've ever made! Highly recommend!",
    "Excellent service and great customer support!",
    "This is fantastic! Exceeded my expectations!",
    "Outstanding! Will definitely buy again!",
    "So happy with this! Perfect for what I needed!",
    "Great value for money! Very satisfied!",
    "Wonderful experience! Thank you so much!",
    "Impressive quality! Love everything about it!",
    "Brilliant! Exactly what I was looking for!"
]

neutral_comments = [
    "It's okay, nothing special but does the job",
    "Average product, met basic expectations",
    "Not bad, not great, just okay",
    "It works fine for everyday use",
    "Decent quality for the price",
    "Neutral feelings about this purchase",
    "Pretty standard, no complaints",
    "It's alright, does what it says",
    "Acceptable quality, no issues so far",
    "Regular product, nothing extraordinary"
]

negative_comments = [
    "Very disappointed with the quality. Not worth it!",
    "Terrible experience! Would not recommend!",
    "Poor quality and bad customer service!",
    "Waste of money! Completely unsatisfied!",
    "Horrible! Nothing like advertised!",
    "Very unhappy with this purchase! Avoid!",
    "Awful product! Returned immediately!",
    "Worst purchase ever! Total disappointment!",
    "Unacceptable quality! Very frustrated!",
    "Pathetic! Don't waste your time!"
]

# Generate comments with realistic sentiment distribution
sentiments = np.random.choice(['Positive', 'Neutral', 'Negative'], n_comments,
                             p=[0.50, 0.30, 0.20])

comments = []
for sentiment in sentiments:
    if sentiment == 'Positive':
        comments.append(np.random.choice(positive_comments))
    elif sentiment == 'Neutral':
        comments.append(np.random.choice(neutral_comments))
    else:
        comments.append(np.random.choice(negative_comments))

df4 = pd.DataFrame({
    'comment_id': range(1, n_comments + 1),
    'platform': platforms,
    'comment': comments,
    'timestamp': [datetime.now() - timedelta(days=np.random.randint(0, 30))
                  for _ in range(n_comments)]
})

df4.to_csv('social_media_comments.csv', index=False)
print(f"   ✅ Created {len(df4)} rows")

# ============================================
# Dataset 5: Communication Experiment
# For Session 5
# ============================================

print("\n5. Generating communication_experiment.csv...")

n_participants = 350

conditions = np.random.choice(['Control', 'Emotional_Appeal', 'Logical_Appeal',
                              'Combined_Appeal'], n_participants, p=[0.25, 0.25, 0.25, 0.25])
genders = np.random.choice(['Male', 'Female', 'Other'], n_participants,
                          p=[0.48, 0.50, 0.02])

ages = np.random.randint(18, 65, n_participants)

# Condition affects persuasion (Combined best)
condition_effects = {
    'Control': {'mean': 4.5, 'std': 1.2},
    'Emotional_Appeal': {'mean': 5.8, 'std': 1.3},
    'Logical_Appeal': {'mean': 6.2, 'std': 1.1},
    'Combined_Appeal': {'mean': 7.1, 'std': 1.0}
}

persuasion_scores = []
credibility_scores = []
response_times = []

for condition in conditions:
    effect = condition_effects[condition]
    persuasion_scores.append(max(1, min(10, np.random.normal(effect['mean'], effect['std']))))
    credibility_scores.append(max(1, min(10, np.random.normal(6.0, 1.5))))
    response_times.append(max(1, np.random.normal(45, 15)))

# Add some missing values (realistic)
persuasion_scores = np.array(persuasion_scores)
credibility_scores = np.array(credibility_scores)
missing_indices = np.random.choice(n_participants, size=int(n_participants * 0.05), replace=False)
persuasion_scores[missing_indices[:len(missing_indices)//2]] = np.nan
credibility_scores[missing_indices[len(missing_indices)//2:]] = np.nan

df5 = pd.DataFrame({
    'participant_id': range(1, n_participants + 1),
    'condition': conditions,
    'age': ages,
    'gender': genders,
    'persuasion_score': np.round(persuasion_scores, 2),
    'credibility_score': np.round(credibility_scores, 2),
    'response_time': np.round(response_times, 1)
})

df5.to_csv('communication_experiment.csv', index=False)
print(f"   ✅ Created {len(df5)} rows (with {missing_indices.shape[0]} missing values)")

# ============================================
# Summary
# ============================================

print("\n" + "=" * 60)
print("✅ All datasets generated successfully!")
print("=" * 60)
print("\nDataset Summary:")
print(f"1. social_media_engagement.csv: {len(df1)} rows")
print(f"2. news_consumption.csv: {len(df2)} rows")
print(f"3. advertising_experiment.csv: {len(df3)} rows")
print(f"4. social_media_comments.csv: {len(df4)} rows")
print(f"5. communication_experiment.csv: {len(df5)} rows")
print(f"\nTotal: {len(df1) + len(df2) + len(df3) + len(df4) + len(df5)} rows")
print("\n데이터셋 생성 완료! 🎉")

# Generate Sample Datasets for Data Analysis Course
# Using only standard library

import csv
import random
from datetime import datetime, timedelta

random.seed(42)

print("Generating sample datasets for the course...")

# ============================================
# Dataset 1: Social Media Engagement
# ============================================

print("\n1. Generating social_media_engagement.csv...")

platforms = ['Instagram', 'Twitter', 'Facebook', 'TikTok']
age_groups = ['18-24', '25-34', '35-44', '45-54', '55+']
platform_multipliers = {'Instagram': 1.5, 'TikTok': 1.3, 'Facebook': 1.0, 'Twitter': 0.8}

with open('social_media_engagement.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['post_id', 'platform', 'age_group', 'likes', 'shares', 'comments', 'post_hour'])

    for i in range(1, 501):
        platform = random.choices(platforms, weights=[0.3, 0.25, 0.25, 0.2])[0]
        age_group = random.choices(age_groups, weights=[0.25, 0.30, 0.25, 0.15, 0.05])[0]
        mult = platform_multipliers[platform]

        likes = max(0, int(random.gauss(600, 200) * mult))
        shares = max(0, int(random.gauss(120, 50) * mult))
        comments = max(0, int(random.gauss(45, 20) * mult))
        post_hour = random.choices(range(24), weights=[0.01]*6 + [0.03]*3 + [0.06]*6 + [0.08]*6 + [0.04]*3)[0]

        writer.writerow([i, platform, age_group, likes, shares, comments, post_hour])

print("   ✅ Created 500 rows")

# ============================================
# Dataset 2: News Consumption
# ============================================

print("\n2. Generating news_consumption.csv...")

categories = ['Politics', 'Technology', 'Sports', 'Entertainment', 'Business', 'Health', 'Science']
devices = ['Mobile', 'Desktop', 'Tablet']
device_multipliers = {'Mobile': 0.8, 'Desktop': 1.2, 'Tablet': 1.0}

with open('news_consumption.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['reader_id', 'category', 'device', 'age_group', 'age',
                    'time_spent', 'articles_read', 'engagement_score'])

    for i in range(1, 601):
        category = random.choices(categories, weights=[0.20, 0.18, 0.15, 0.15, 0.12, 0.12, 0.08])[0]
        device = random.choices(devices, weights=[0.55, 0.35, 0.10])[0]
        age_group = random.choices(age_groups, weights=[0.15, 0.30, 0.25, 0.20, 0.10])[0]

        mult = device_multipliers[device]
        time_spent = round(max(1, random.gauss(15, 5) * mult), 1)
        articles_read = max(1, int(random.gauss(5, 2) * mult))
        engagement_score = round(random.uniform(1, 10), 2)

        if age_group == '18-24':
            age = random.randint(18, 24)
        elif age_group == '25-34':
            age = random.randint(25, 34)
        elif age_group == '35-44':
            age = random.randint(35, 44)
        elif age_group == '45-54':
            age = random.randint(45, 54)
        else:
            age = random.randint(55, 74)

        writer.writerow([i, category, device, age_group, age, time_spent,
                        articles_read, engagement_score])

print("   ✅ Created 600 rows")

# ============================================
# Dataset 3: Advertising Experiment
# ============================================

print("\n3. Generating advertising_experiment.csv...")

ad_types = ['Image', 'Video', 'Carousel', 'Story']
genders = ['Male', 'Female']
ad_age_groups = ['18-24', '25-34', '35-44', '45+']

ad_type_effects = {
    'Video': (0.08, 0.03),
    'Carousel': (0.065, 0.025),
    'Image': (0.055, 0.02),
    'Story': (0.05, 0.02)
}

with open('advertising_experiment.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['ad_id', 'ad_type', 'gender', 'age_group', 'engagement',
                    'conversion_rate', 'cost_per_click'])

    for i in range(1, 401):
        ad_type = random.choice(ad_types)
        gender = random.choices(genders, weights=[0.48, 0.52])[0]
        age_group = random.choices(ad_age_groups, weights=[0.25, 0.35, 0.25, 0.15])[0]

        mean, std = ad_type_effects[ad_type]
        conversion_rate = round(max(0, min(1, random.gauss(mean, std))), 4)
        engagement = max(0, int(random.gauss(500, 150)))
        cost = round(max(0.5, random.gauss(2.5, 0.8)), 2)

        writer.writerow([i, ad_type, gender, age_group, engagement,
                        conversion_rate, cost])

print("   ✅ Created 400 rows")

# ============================================
# Dataset 4: Social Media Comments
# ============================================

print("\n4. Generating social_media_comments.csv...")

comment_platforms = ['Instagram', 'Twitter', 'Facebook', 'YouTube']

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

with open('social_media_comments.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['comment_id', 'platform', 'comment', 'timestamp'])

    for i in range(1, 301):
        platform = random.choices(comment_platforms, weights=[0.30, 0.25, 0.25, 0.20])[0]
        sentiment_type = random.choices(['positive', 'neutral', 'negative'],
                                       weights=[0.50, 0.30, 0.20])[0]

        if sentiment_type == 'positive':
            comment = random.choice(positive_comments)
        elif sentiment_type == 'neutral':
            comment = random.choice(neutral_comments)
        else:
            comment = random.choice(negative_comments)

        days_ago = random.randint(0, 30)
        timestamp = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d %H:%M:%S')

        writer.writerow([i, platform, comment, timestamp])

print("   ✅ Created 300 rows")

# ============================================
# Dataset 5: Communication Experiment
# ============================================

print("\n5. Generating communication_experiment.csv...")

conditions = ['Control', 'Emotional_Appeal', 'Logical_Appeal', 'Combined_Appeal']
exp_genders = ['Male', 'Female', 'Other']

condition_effects = {
    'Control': (4.5, 1.2),
    'Emotional_Appeal': (5.8, 1.3),
    'Logical_Appeal': (6.2, 1.1),
    'Combined_Appeal': (7.1, 1.0)
}

with open('communication_experiment.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['participant_id', 'condition', 'age', 'gender',
                    'persuasion_score', 'credibility_score', 'response_time'])

    missing_persuasion = random.sample(range(1, 351), 10)
    missing_credibility = random.sample(range(1, 351), 10)

    for i in range(1, 351):
        condition = random.choice(conditions)
        gender = random.choices(exp_genders, weights=[0.48, 0.50, 0.02])[0]
        age = random.randint(18, 64)

        mean, std = condition_effects[condition]
        persuasion = round(max(1, min(10, random.gauss(mean, std))), 2)
        credibility = round(max(1, min(10, random.gauss(6.0, 1.5))), 2)
        response_time = round(max(1, random.gauss(45, 15)), 1)

        # Add missing values
        if i in missing_persuasion:
            persuasion = ''
        if i in missing_credibility:
            credibility = ''

        writer.writerow([i, condition, age, gender, persuasion,
                        credibility, response_time])

print("   ✅ Created 350 rows (with 20 missing values)")

# ============================================
# Summary
# ============================================

print("\n" + "=" * 60)
print("✅ All datasets generated successfully!")
print("=" * 60)
print("\nDataset Summary:")
print("1. social_media_engagement.csv: 500 rows")
print("2. news_consumption.csv: 600 rows")
print("3. advertising_experiment.csv: 400 rows")
print("4. social_media_comments.csv: 300 rows")
print("5. communication_experiment.csv: 350 rows")
print(f"\nTotal: 2150 rows")
print("\n데이터셋 생성 완료! 🎉")

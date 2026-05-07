#!/bin/bash

# Configuration
NUM_SAMPLES=100
# BASE_URL="https://open.bigmodel.cn/api/paas/v4/" # Optional: set if needed

# Define pairs: "topic:category"
# PAIRS=(
#     # blog_post: entrepreneurship, marketing, film, sports, gaming
#     "entrepreneurship:blog_post"
#     "marketing:blog_post"
#     "gaming:blog_post"
#     "sports:blog_post"

#     # restaurant (location) and history:place
#     "location:restaurant"
#     "history:place"

#     # event: history, corporate, visual_arts and sports:match
#     "history:event"
#     "corporate:event"
#     "visual_arts:event"
#     "sports:sport_match"

#     # person: computer_science, economics, corporate, government, history
#     "computer_science:person"
#     "corporate:person"
#     "government:person"
#     "history:person"
# )

PAIRS=(
    # scientific_paper: physics, biology, medicine, mathematics
    "physics:scientific_paper"
    "biology:scientific_paper"
    "medicine:scientific_paper"
    "mathematics:scientific_paper"

    # news_article: government, economics, corporate
    "government:news_article"
    "economics:news_article"
    "corporate:news_article"

    # works: film, music, literature
    "film:film_work"
    "music:music_work"
    "literature:works"

    # institution
    "philosophy:institution"
    "government:institution"
    "corporate:company"

    # law:legal_case and corporate:financial_report
    "law:legal_case"
    "corporate:financial_report"

    # blog_post: entrepreneurship, marketing, film, sports, gaming
    "entrepreneurship:blog_post"
    "marketing:blog_post"
    "gaming:blog_post"
    "sports:blog_post"

    # restaurant (location) and history:place
    "location:restaurant"
    "history:place"

    # event: history, corporate, visual_arts and sports:match
    "history:event"
    "corporate:event"
    "visual_arts:event"
    "sports:sport_match"

    # person: computer_science, economics, corporate, government, history
    "computer_science:person"
    "corporate:person"
    "government:person"
    "history:person"
)

# Output directory for results
OUTPUT_DIR="../results/full_ctx/"
mkdir -p "$OUTPUT_DIR"

echo "Starting Seed Generation Study for multiple topics/categories..."
echo "Samples per pair: $NUM_SAMPLES"
echo "Concurrency: $CONCURRENCY"
echo "Results will be saved in $OUTPUT_DIR"
echo "--------------------------------------------------"

for pair in "${PAIRS[@]}"; do
    # Split topic and category
    IFS=':' read -r topic category <<< "$pair"
    
    OUTPUT_FILE="${OUTPUT_DIR}/seed_stats_${topic}_${category}.json"
    
    echo "Running study for Topic: $topic, Category: $category"
    echo "Output: $OUTPUT_FILE"
    
    python seed_statistics.py \
        --num_samples "$NUM_SAMPLES" \
        --topic "$topic" \
        --category "$category" \
        --output_file "$OUTPUT_FILE" \
        --verbose
    
    if [ $? -eq 0 ]; then
        echo "Finished study for $topic:$category"
    else
        echo "Error: Study failed for $topic:$category"
    fi
    echo "--------------------------------------------------"
done

echo "All studies completed. Results are in $OUTPUT_DIR"

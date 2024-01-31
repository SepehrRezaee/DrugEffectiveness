This study presents a recommendation system for suggesting drugs for specific
medical conditions. The system leverages the BERT (Bidirectional Encoder Rep-
resentations from Transformers) model, a state-of-the-art Deep Neural Network,
to understand the context of various features of drugs, including ’EaseOfUse’,
’Effective’, ’Price’, ’Reviews’, and ’Satisfaction’. The system generates BERT
embeddings for these combined features and calculates the cosine similarity be-
tween different drugs to recommend the most suitable ones for a specific condition.
The performance of the recommendation system is evaluated using Mean Average
Precision (MAP) and Normalized Discounted Cumulative Gain (NDCG) metrics.
The results demonstrate the effectiveness of the proposed system in recommending
drugs for specific conditions, showcasing the power of BERT in handling such
tasks.

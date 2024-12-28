# Drug Recommendation System

## Overview
This project presents a recommendation system to suggest the most appropriate drugs for specific medical conditions. Using state-of-the-art machine learning techniques, particularly the BERT (Bidirectional Encoder Representations from Transformers) model, the system evaluates various drug features to generate accurate and personalized recommendations.

## Key Features
- **Machine Learning Model**: The recommendation system is powered by BERT embeddings, leveraging semantic understanding for enhanced drug recommendations.
- **Multi-Feature Analysis**: The system incorporates the following features:
  - Ease of Use
  - Effectiveness
  - Price
  - Reviews
  - Satisfaction
- **Evaluation Metrics**:
  - Mean Average Precision (MAP)
  - Normalized Discounted Cumulative Gain (NDCG)

## Dataset
The analysis and recommendations are based on a comprehensive dataset with the following variables:
- **Condition**: Medical condition for which the drug is used.
- **Drug**: Name of the drug.
- **EaseOfUse**: User ratings on the drug's ease of use.
- **Effective**: Ratings on the drug's effectiveness.
- **Price**: Average price of the drug.
- **Reviews**: Number of user reviews.
- **Satisfaction**: Overall satisfaction ratings.

## Methodology
1. **Data Preparation**:
   - Cleaned and preprocessed the dataset.
   - Created a combined feature column for embedding generation.
2. **Modeling**:
   - Generated embeddings using the BERT model for combined features.
   - Computed cosine similarity between embeddings to identify the most relevant drugs.
3. **Evaluation**:
   - Assessed the system's performance using MAP and NDCG metrics.

## Visualizations
Comprehensive visualizations were created to understand the data distribution and relationships, including:
- Histograms and box plots for numerical variables.
- Bar charts for categorical variables.
- Correlation matrices and heatmaps for feature relationships.
- Scatter and violin plots to explore interactions between features.

## Results
- **Performance Metrics**:
  - MAP: 1.000
  - NDCG: 1.000
- **Recommendations**: Example drugs recommended for the condition "fever" include:
  - Chlorpheniram-DM-Acetaminophen
  - Phenylephrine-DM-Acetaminophen
  - Cpm-Pseudoeph-DM-Acetaminophen

## Discussion
The system demonstrates the potential of BERT in processing drug-related data and making accurate recommendations. Key benefits include:
- Contextual understanding of features.
- High scalability for different medical conditions.
- Reliable and interpretable results.

## Future Directions
- Integration of real-time patient data.
- Use of advanced models like GPT or more specialized transformers.
- Enhancement of data privacy and security mechanisms.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook dataanalysis.ipynb
   ```

## Usage
1. Prepare the dataset and ensure it matches the required format.
2. Execute the notebook to preprocess data and generate recommendations.
3. Analyze the results and visualizations provided.

## Contributors
- **Sepehr Rezaee** (Shahid Beheshti University)
- **Mahdi Firouz** (Shahid Beheshti University)

## License
This project is licensed under the MIT License.

# AssignmentAI
Technical Report: Classification and Summarization Service
1. Data Handling
Preprocessing
Text Cleaning: Removed special characters, HTML tags, and excessive whitespace.
Tokenization: Used a tokenizer from the pre-trained BERT model for uniform token splitting.
Stopword Removal: Excluded non-informative words like "and," "the," etc., where necessary.
Lowercasing: Convert all text to lowercase to ensure consistency.
NER-Specific Preprocessing: Annotated named entities using a domain-specific knowledge base for better entity extraction performance.
Data Augmentation
Synonym Replacement: Replaced specific words with their synonyms to introduce variability.
Back Translation: Translated text to another language and back to English to diversify the dataset.
Paraphrasing: Used a paraphrasing model to generate additional text samples.
2. Modeling Choices
Classification Approach
Model: Fine-tuned BERT for multi-label classification due to its state-of-the-art performance on text data.
Why BERT?
Pre-trained on a vast corpus, making it suitable for domain-specific tasks with minimal fine-tuning.
Handles long-range dependencies and contextual relationships in text effectively.
Challenges:
Limited labeled data.
Imbalance in class distribution.
Solutions:
Implemented class weights in the loss function to address class imbalance.
Used cross-validation to ensure robustness and generalizability.
Summarization Approach
Model: Fine-tuned BART (Bidirectional and Auto-Regressive Transformers) for abstractive summarization.
Why BART?
Handles noisy input and generates fluent, concise summaries.
Pre-trained for summarization tasks, requiring fewer adjustments.
Challenges:
Generated summaries occasionally missed key points.
Solutions:
Used reinforcement learning to align model outputs with critical elements of the input text.
Entity Extraction
Model: Custom NER pipeline using SpaCy.
Why SpaCy?
Efficient for real-time applications.
Customizable for domain-specific entity types.
Challenges:
Overlap of entities in the same text.
Solutions:
Improved annotation quality and fine-tuned the pipeline with overlapping labels.
3. Performance Results
For Spacy model:
Precision: [1. 1. 1. 0.]
Recall: [1. 1. 1. 0.]
F1 Score: [1. 1. 1. 0.]
4. Error Analysis
Classification Errors
Confusion:
"FEATURE" vs. "PRICING_KEYWORD" due to overlapping contexts in sentences.
Example:
Input: "Our software integrates seamlessly with CompetitorX's platform."
Misclassification: Predicted "FEATURE" instead of "COMPETITOR."
Improvement Areas:
Enhance the dataset with more diverse examples.
Use domain-specific embeddings.
Entity Extraction Errors
Challenges:
Mislabeling overlapping entities.
Missed entities due to rare terms in the dataset.
Example:
Input: "CompetitorX's pricing is better."
Missed "CompetitorX" as an entity.
Proposed Fixes:
Better entity annotations and inclusion of rare terms in training data.
Summarization Errors
Challenges:
Summaries occasionally omitted key insights.
Example:
Input: "The software is user-friendly but lacks pricing flexibility."
Summary: "The software is user-friendly."
Proposed Fixes:
Fine-tune the summarization model with domain-specific summary annotations.
5. Future Work
Data Curation
Collect and annotate a larger, domain-specific dataset.
Address class imbalance with advanced augmentation techniques.
Advanced Fine-Tuning
Use prompt-tuning for better adaptability to downstream tasks.
Incorporate adapters for domain-specific fine-tuning.
Real-World Deployment
Optimize the model for edge devices.
Monitor deployed models for drift and retrain periodically to ensure continued performance.



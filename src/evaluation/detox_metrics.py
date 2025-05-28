"""
Evaluation metrics for text detoxification:
1. Style Transfer Accuracy (STA): Using Jigsaw toxicity classifier
2. Content Preservation (SIM): Using Wieting embeddings
3. Fluency (FL): Using RoBERTa-CoLA classifier
"""
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from typing import List, Dict, Union, Tuple
import numpy as np

class DetoxificationEvaluator:
    def __init__(self, device="cuda"):
        self.device = device
        
        # Load toxicity classifier
        self.toxicity_tokenizer = AutoTokenizer.from_pretrained("s-nlp/roberta_toxicity_classifier")
        self.toxicity_model = AutoModelForSequenceClassification.from_pretrained("s-nlp/roberta_toxicity_classifier")
        self.toxicity_model.to(device)
        
        # Load sentence embeddings model (Wieting's model)
        self.sim_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
        self.sim_model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
        self.sim_model.to(device)
        
        # Load fluency classifier (RoBERTa-CoLA)
        self.fluency_tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA")
        self.fluency_model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-CoLA")
        self.fluency_model.to(device)
        
        # Set all models to eval mode
        self.toxicity_model.eval()
        self.sim_model.eval()
        self.fluency_model.eval()

    @torch.no_grad()
    def compute_toxicity_scores(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Compute toxicity scores for a list of texts."""
        scores = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.toxicity_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.toxicity_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            toxic_probs = probs[:, 1].cpu().numpy()  # Assuming 1 is toxic class
            scores.extend(toxic_probs)
            
        return np.array(scores)

    @torch.no_grad()
    def compute_similarity_score(self, original_texts: List[str], generated_texts: List[str], batch_size: int = 32) -> float:
        """Compute content preservation scores using cosine similarity between embeddings."""
        def get_embeddings(texts):
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = self.sim_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.sim_model(**inputs)
                # Use mean pooling
                embeddings = torch.mean(outputs.last_hidden_state, dim=1)
                all_embeddings.append(embeddings)
                
            return torch.cat(all_embeddings, dim=0)

        orig_embeddings = get_embeddings(original_texts)
        gen_embeddings = get_embeddings(generated_texts)
        
        # Normalize embeddings
        orig_embeddings = F.normalize(orig_embeddings, p=2, dim=1)
        gen_embeddings = F.normalize(gen_embeddings, p=2, dim=1)
        
        # Compute cosine similarity
        similarities = torch.sum(orig_embeddings * gen_embeddings, dim=1)
        return similarities.cpu().numpy()

    @torch.no_grad()
    def compute_fluency_scores(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Compute fluency scores using RoBERTa-CoLA classifier."""
        scores = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.fluency_tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.fluency_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            acceptable_probs = probs[:, 1].cpu().numpy()  # Assuming 1 is acceptable class
            scores.extend(acceptable_probs)
            
        return np.array(scores)

    def evaluate(self, original_texts: List[str], generated_texts: List[str], batch_size: int = 32) -> Dict[str, float]:
        """
        Evaluate detoxification using all metrics.
        
        Returns:
            Dict containing:
            - 'style_transfer': percentage of non-toxic outputs
            - 'content_preservation': average similarity score
            - 'fluency': percentage of fluent sentences
        """
        # Compute toxicity scores (lower is better)
        toxicity_scores = self.compute_toxicity_scores(generated_texts, batch_size)
        style_transfer = (toxicity_scores < 0.5).mean()  # Using 0.5 as threshold
        
        # Compute content preservation
        similarities = self.compute_similarity_score(original_texts, generated_texts, batch_size)
        content_preservation = similarities.mean()
        
        # Compute fluency
        fluency_scores = self.compute_fluency_scores(generated_texts, batch_size)
        fluency = (fluency_scores > 0.5).mean()  # Using 0.5 as threshold
        
        return {
            'style_transfer': float(style_transfer),
            'content_preservation': float(content_preservation),
            'fluency': float(fluency),
            'toxicity_scores': toxicity_scores.tolist(),
            'similarity_scores': similarities.tolist(),
            'fluency_scores': fluency_scores.tolist()
        }

    def evaluate_file(self, original_file: str, generated_file: str, batch_size: int = 32) -> Dict[str, float]:
        """Evaluate detoxification using input files."""
        with open(original_file, 'r') as f:
            original_texts = [line.strip() for line in f]
            
        with open(generated_file, 'r') as f:
            generated_texts = [line.strip() for line in f]
            
        return self.evaluate(original_texts, generated_texts, batch_size) 
from sentence_transformers import CrossEncoder

class CrossEncoderRerank():
    def __init__(self, model_name="cross-encoder/stsb-roberta-large", device='cuda'):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query, candidates, top_n=3):
        pairs = [(query, candidate['page_content']) for candidate in candidates]
        scores = self.model.predict(pairs)
        for i, candidate in enumerate(candidates):
            candidate['score'] = scores[i]
        sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
        return sorted_candidates[:top_n]

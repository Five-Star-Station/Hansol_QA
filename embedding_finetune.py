import pandas as pd

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import MSEEvaluator

class LoadData:
    def __init__(self, dataset_path, BATCH_SIZE):
        self.train_dataset_path = dataset_path + 'train.csv'
        self.test_dataset_path = dataset_path + 'test.csv'
        self.batch_size = BATCH_SIZE
        
    def load_dataset(self):
        train_dataset = pd.read_csv(self.train_dataset_path)
        test_dataset = pd.read_csv(self.test_dataset_path)
        
        dataset = train_dataset
        
        examples = []
        for i in range(len(dataset)):
            query_1 = dataset['질문_1'][i]
            query_2 = dataset['질문_2'][i]
            
            answer_1 = dataset['답변_1'][i]
            answer_2 = dataset['답변_2'][i]
            answer_3 = dataset['답변_3'][i]
            answer_4 = dataset['답변_4'][i]
            
            examples.extend([
                InputExample(texts=[query_1, answer_1]),
                InputExample(texts=[query_1, answer_2]),
                InputExample(texts=[query_1, answer_3]),
                InputExample(texts=[query_1, answer_4]),
                InputExample(texts=[query_2, answer_1]),
                InputExample(texts=[query_2, answer_2]),
                InputExample(texts=[query_2, answer_3]),
                InputExample(texts=[query_2, answer_4]),
            ])
            
        loader = DataLoader(
            examples, batch_size=self.batch_size
        )
        
        return loader
    
    def eval_dataset(self, model):
        train_dataset = pd.read_csv(self.train_dataset_path)
        
        evaluator = MSEEvaluator(
            source_sentences=train_dataset['질문_1'],
            target_sentences=train_dataset['답변_5'],
            teacher_model=model,
        )
    
    
if __name__ == '__main__':
    
    model_id = "distiluse-base-multilingual-cased-v1"
    embedding_model = SentenceTransformer(model_id)
    
    loss = losses.MultipleNegativesRankingLoss(embedding_model)
    
    batch_size = 32
    epochs = 10
    
    data_loader = LoadData(dataset_path='./dataset/', BATCH_SIZE=batch_size)
    loader = data_loader.load_dataset()
    evaluator = data_loader.eval_dataset(embedding_model)
    
    warmup_steps = int(len(loader) * epochs * 0.1)
    
    embedding_model.fit(
        train_objectives=[(loader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path = './finetuned_embedding',
        show_progress_bar=True,
        evaluator=evaluator,
        evaluation_steps=2
    )
    
    
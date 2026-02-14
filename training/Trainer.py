import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

"""
Klasa zarządzająca procesem trenowania i ewaluacji modelu. 
PARAMETRY:
    model     - model do trenowania/ewaluacji
    optimizer - optymalizator wykorzystywany podczas trenowania
    scheduler - scheduler zmieniający współczynnik uczenia podczas trenowania
PROCEDURY:
    step_batch() - wykonuje krok trenowania na pojedynczym batchu danych, zwraca wartość straty (loss)
    evaluate()   - wykonuje ewaluację modelu na podanym dataloaderze, zwraca słownik z metrykami (accuracy, AUC, F1)
"""
class Trainer:
    def __init__(self, model, optimizer, scheduler):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


    def step_batch(self, batch, criterion):
        self.model.train()
        labels = torch.stack([b["label"] for b in batch]).to(self.device)
        self.optimizer.zero_grad(set_to_none=True)
        
        logits = self.model(batch)
        loss = criterion(logits, labels)
        loss.backward()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()                    

        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        all_logits, all_labels = [], []
        
        for batch in dataloader: 
            logits = self.model(batch)
            labels = torch.stack([b["label"] for b in batch]).to(self.device)
            all_logits.append(logits)
            all_labels.append(labels)
        
        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0).view(-1)
        
        probs = torch.softmax(logits, dim=-1)[:,1]
        preds = torch.argmax(logits, dim=-1)

        y_true = labels.cpu().numpy()
        y_pred = preds.cpu().numpy()
        y_prob = probs.cpu().numpy()

        acc = accuracy_score(y_true, y_pred)

        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = float('nan')
        
        try:
            f1 = f1_score(y_true, y_pred, average='binary')
        except Exception:
            f1 = float('nan')
        return {"acc": acc, "auc": auc, "f1": f1}
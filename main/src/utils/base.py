import torch
from tqdm import tqdm
from torch_geometric.utils import to_dense_adj, to_dense_batch
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import numpy as np
import os

def plot_confusion_matrix(confusion_matrix):
    plt.figure(figsize=(8, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix')
    plt.show()

class BaseEngine():
    def __init__(self, model, optimizer, loss_fn, penalty, device, train_loader, val_loader, test_loader, num_epochs, show_cm):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.penalty = penalty
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.show_cm = show_cm
        
        experiment_name = "new"
        os.makedirs(f"experiments/{experiment_name}", exist_ok=True)
        self.best_model_path = f"experiments/{experiment_name}/best_model.pt"
        
        # Set up logging
        log_file_path = f"experiments/{experiment_name}/log.txt"
        logging.basicConfig(filename=log_file_path, level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger()

        self.logger.info("Initialized BaseEngine")

    def train_fn(self, loader):
        self.model.train()
        total_loss = 0
        bar = tqdm(loader, bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', ncols=100)

        for i, data in enumerate(bar):
            data = data[0]
            target = data.y.long().to(self.device)
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x, _ = to_dense_batch(x, batch)
            x = x.to(self.device)
            adj = to_dense_adj(edge_index, batch).to(self.device)
            self.optimizer.zero_grad()
            out, _, adja = self.model(x, adj, indices=None)
            loss = self.loss_fn(out.float(), target) 
            if adja is not None:
                loss += self.penalty * torch.norm(adja, p=1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader.dataset)
        self.logger.info(f'Training loss: {avg_loss:.6f}')
        return avg_loss

    def evaluate(self, loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        bar = tqdm(loader)
        with torch.no_grad():
            for i, data in enumerate(bar):
                data = data[0]
                label = data.y
                x, edge_index, batch = data.x, data.edge_index, data.batch
                x, _ = to_dense_batch(x, batch)
                x = x.to(self.device)
                adj = to_dense_adj(edge_index, batch).to(self.device)
                logits, _, adja = self.model(x, adj, indices=None)
                pred = torch.max(logits, 1)[1]
                all_preds.extend(pred.tolist())
                all_labels.extend(label.tolist())
        accuracy = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        return accuracy, cm
    
    def train(self):
        train_loss = []
        best_val_accuracy = -np.inf
        for epoch in range(self.num_epochs):
            self.logger.info(f'Starting epoch {epoch + 1}/{self.num_epochs}')
            loss = self.train_fn(self.train_loader)
            val_accuracy, cm = self.evaluate(self.val_loader)
            train_accuracy, cm = self.evaluate(self.train_loader)
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save(self.model.state_dict(), self.best_model_path)
                self.logger.info(f"Best model saved at {self.best_model_path}")
            self.logger.info(f'Epoch: {epoch + 1}, Loss: {loss:.16f}, Train Accuracy: {train_accuracy*100:.6f}%, Val Accuracy: {val_accuracy*100:.6f}%')
            train_loss.append(loss)
        self.evaluateTest(self.best_model_path)

    def evaluateTest(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True)) 
        test_accuracy, cm = self.evaluate(self.test_loader)
        self.logger.info(f'Test Accuracy: {test_accuracy*100:.2f}%')
        if self.show_cm:
            plot_confusion_matrix(cm)

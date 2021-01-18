#trainers.py

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from MMD_utils import compute_gamma, compute_mmd_gram_matrix


class MMDTrainer:
    
    def __init__(self, model, optimizer, landmark_loader, alpha=1.0, beta=0.0, device=torch.device("cpu"), nystrom="LLSVM", regularizer="variance"):
        
        self.device = device
        self.nystrom = nystrom

        self.model = model
        self.optimizer = optimizer

        self.center = None
        self.reg_weight = 0
        self.alpha = alpha
        self.beta = beta
        self.regularizer = regularizer

        self.landmark_loader = landmark_loader
        

    def train(self, train_loader):
        self.model.train()
        
        

        if self.center == None:  # first iteration
            F_list = []

        loss_accum = 0
        svdd_loss_accum = 0
        reg_loss_accum = 0
        total_iters = 0

        for batch in train_loader:
            #print(batch)
            #print(".", end='')
            
            landmark_embeddings = []
            for landmark_batch in self.landmark_loader:
                landmark_batch_embeddings = self.model(landmark_batch)
                landmark_embeddings = landmark_embeddings + landmark_batch_embeddings

            gamma = compute_gamma(landmark_embeddings, device=self.device).detach() # no backpropagation for gamma
            #print(gamma)
            
            
            train_embeddings = self.model(batch)
            K_trainZ = compute_mmd_gram_matrix(train_embeddings, landmark_embeddings, gamma=gamma, device=self.device).to(self.device)

            if self.nystrom == "LLSVM":
                
                K_Z = compute_mmd_gram_matrix(landmark_embeddings, gamma=gamma, device=self.device).to(self.device)
                eigenvalues, U_Z = torch.symeig(K_Z, eigenvectors=True)

                #removed smallest 2/3 eigenvalues due to numerical instability
                no_of_eigens = len(eigenvalues)
                eigenvalues = eigenvalues[-no_of_eigens//3:]
                
                # if eigenvalues still negative, adjust - values small enough so that it does not affect
                m = min(eigenvalues).detach()
                if m < 0:
                    eigenvalues = eigenvalues - 2*m
                elif m == 0:
                    eigenvalues = eigenvalues + 1e-9
                
                U_Z = U_Z[:,-no_of_eigens//3:]
                T = torch.matmul(U_Z,torch.diag(eigenvalues**-0.5))
                F_train = torch.matmul(K_trainZ, T)

            elif self.nystrom == "RSVM":
                
                F_train = K_trainZ
                
            

            # if first iteration, compute center, and don't do any backprop
            if self.center == None:
                F_list.append(F_train)
            
            else:
                train_scores = torch.sum((F_train - self.center)**2, dim=1).cpu()
                svdd_loss = torch.mean(train_scores)
                
                if self.regularizer == "variance":
                    reg_loss = -(1/(F_train.shape[0]-1))*torch.sum(torch.var(F_train,dim=0))
                else:
                    raise ValueError("Unrecognized regularization type")

                loss = svdd_loss + self.reg_weight * reg_loss
                self.reg_weight = max(self.reg_weight, (self.alpha*self.reg_weight + (1-self.alpha)*self.beta*(svdd_loss/abs(reg_loss))).detach())
                #self.reg_weight = (self.alpha*self.reg_weight + (1-self.alpha)*self.beta*(svdd_loss/abs(reg_loss))).detach()
                    
                #backpropagate
                self.optimizer.zero_grad()
                loss.backward()    
                self.optimizer.step()
                
                loss_accum += loss.detach().cpu().numpy()
                svdd_loss_accum += svdd_loss.detach().cpu().numpy()
                reg_loss_accum += reg_loss.detach().cpu().numpy()
                total_iters += 1

        if self.center == None:
            full_F_list = torch.cat(F_list)
            self.center = torch.mean(full_F_list, dim=0).detach() # no backpropagation for center
            #print("center computed")

            average_loss = -1
            average_svdd_loss = -1
            average_reg_loss = -1
        
        else:
            average_loss = loss_accum/total_iters
            average_svdd_loss = svdd_loss_accum/total_iters
            average_reg_loss = reg_loss_accum/total_iters

        return average_loss, average_svdd_loss, average_reg_loss


    def test(self, test_loader):
        self.model.eval()
        
        with torch.no_grad():

            landmark_embeddings = []

            for landmark_batch in self.landmark_loader:
                landmark_batch_embeddings = self.model(landmark_batch)
                landmark_embeddings = landmark_embeddings + landmark_batch_embeddings

            gamma = compute_gamma(landmark_embeddings, device=self.device) # no backpropagation for gamma
            #print(gamma)

            if self.nystrom == "LLSVM":
                
                K_Z = compute_mmd_gram_matrix(landmark_embeddings, gamma=gamma, device=self.device).to(self.device)
                eigenvalues, U_Z = torch.symeig(K_Z, eigenvectors=True)

                #removed smallest 2/3 eigenvalues due to numerical instability
                no_of_eigens = len(eigenvalues)
                eigenvalues = eigenvalues[-no_of_eigens//3:]
                
                # if eigenvalues still negative, adjust - values small enough so that it does not affect
                m = min(eigenvalues)
                if m < 0:
                    eigenvalues = eigenvalues - 2*m
                elif m == 0:
                    eigenvalues = eigenvalues + 1e-9
                
                U_Z = U_Z[:,-no_of_eigens//3:]
                T = torch.matmul(U_Z,torch.diag(eigenvalues**-0.5))
            
            dists_list = []
            for batch in test_loader:

                R_embeddings = self.model(batch)
                K_RZ = compute_mmd_gram_matrix(R_embeddings, landmark_embeddings, gamma=gamma, device=self.device).to(self.device)
                
                if self.nystrom == "LLSVM":
                    F = torch.matmul(K_RZ, T)
                elif self.nystrom == "RSVM":
                    F = K_RZ
                
                batch_dists = torch.sum((F - self.center)**2, dim=1).cpu()
                dists_list.append(batch_dists)
            
            labels = torch.cat([batch.y for batch in test_loader])
            dists = torch.cat(dists_list)
            #print(dists)

            ap = average_precision_score(labels, dists)
            roc_auc = roc_auc_score(labels, dists)

            return ap, roc_auc, dists, labels

class MeanTrainer:
    
    def __init__(self, model, optimizer, alpha=1.0, beta=0.0, device=torch.device("cpu"), regularizer="variance"):
        
        self.device = device

        self.model = model
        self.optimizer = optimizer

        self.center = None
        self.reg_weight = 0
        self.alpha = alpha
        self.beta = beta
        self.regularizer = regularizer        

    def train(self, train_loader):
        self.model.train()
        
        if self.center == None:  # first iteration
            F_list = []

        loss_accum = 0
        svdd_loss_accum = 0
        reg_loss_accum = 0
        total_iters = 0

        for batch in train_loader:
            #print(".", end='')
            
            train_embeddings = self.model(batch)
            mean_train_embeddings = [torch.mean(emb, dim=0) for emb in train_embeddings] # Mean-ggregation: G_emb = mean(v_emb for v in G)
            F_train = torch.stack(mean_train_embeddings)
                
            # if first iteration, compute center, and don't do any backprop
            if self.center == None:
                F_list.append(F_train)
            
            else:
                train_scores = torch.sum((F_train - self.center)**2, dim=1).cpu()
                svdd_loss = torch.mean(train_scores)
                
                if self.regularizer == "variance":
                    reg_loss = -(1/(F_train.shape[0]-1))*torch.sum(torch.var(F_train,dim=0))
                else:
                    raise ValueError("Unrecognized regularization type")

                loss = svdd_loss + self.reg_weight * reg_loss
                self.reg_weight = max(self.reg_weight, (self.alpha*self.reg_weight + (1-self.alpha)*self.beta*(svdd_loss/abs(reg_loss))).detach())
                #self.reg_weight = (self.alpha*self.reg_weight + (1-self.alpha)*self.beta*(svdd_loss/abs(reg_loss))).detach()
                    
                #backpropagate
                self.optimizer.zero_grad()
                loss.backward()    
                self.optimizer.step()
                
                loss_accum += loss.detach().cpu().numpy()
                svdd_loss_accum += svdd_loss.detach().cpu().numpy()
                reg_loss_accum += reg_loss.detach().cpu().numpy()
                total_iters += 1

        if self.center == None: # first epoch only
            full_F_list = torch.cat(F_list)
            self.center = torch.mean(full_F_list, dim=0).detach() # no backpropagation for center
            #print("center computed")

            average_loss = -1
            average_svdd_loss = -1
            average_reg_loss = -1
        
        else:
            average_loss = loss_accum/total_iters
            average_svdd_loss = svdd_loss_accum/total_iters
            average_reg_loss = reg_loss_accum/total_iters

        return average_loss, average_svdd_loss, average_reg_loss


    def test(self, test_loader):
        self.model.eval()
        
        with torch.no_grad():

            dists_list = []
            for batch in test_loader:

                test_embeddings = self.model(batch)
                mean_test_embeddings = [torch.mean(emb, dim=0) for emb in test_embeddings] # Mean-aggregation: G_emb = mean(v_emb for v in G)
                F_test = torch.stack(mean_test_embeddings)
                
                batch_dists = torch.sum((F_test - self.center)**2, dim=1).cpu()
                dists_list.append(batch_dists)
            
            labels = torch.cat([batch.y for batch in test_loader])
            dists = torch.cat(dists_list)
            #print(dists)

            ap = average_precision_score(labels, dists)
            roc_auc = roc_auc_score(labels, dists)

            return ap, roc_auc, dists, labels
# GNN-anomaly

	python main.py

runs the GIN-loss based code. It extracts hidden representations of graphs, then measures the accuracy of MMD on these representations over all pairs of graphs. The ROC scores are output for different selections of MMD bandwidth.

	python main_svm.py

runs the SVM-loss based code.

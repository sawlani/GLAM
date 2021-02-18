Prerequisites: torch, torch-geometric

Example command to run a single model configuration,
	python main.py --data=PROTEINS --lr=0.01

To run a number of configurations, to use for model selection later, set the required configurations in config.txt and run:
    python main.py --use_config

Once the pickle file is generated for outputs of all configurations, use model selection to select models:
    python model_selection.py --data=PROTEINS --aggregation=MMD

Run both files with --help to see all parameters that can be input.
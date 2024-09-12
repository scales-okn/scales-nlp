from pathlib import Path
from configuration_maker import Config, ConfigKey


keys = [
	ConfigKey(
		name='PACER_DIR',
		group='main',
		key_type='path',
		default=None,
		description='Path to the directory for PACER data.',
	),

	ConfigKey(
		name='PACER_USERNAME',
		group='main',
		key_type='str',
		default=None,
		description='Username for PACER. Only needed if purchasing dockets from PACER.',
	),

	ConfigKey(
		name='PACER_PASSWORD',
		group='main',
		key_type='str',
		default=None,
		description='Password for PACER. Only needed if purchasing dockets from PACER.',
	),

	ConfigKey(
		name='HUGGING_FACE_TOKEN',
		group='main',
		default=None,
		description='Your Hugging Face auth token. Only needed if using a private model with pipelines.',
	),


	ConfigKey(
		name='MODEL_NAME',
		group='train-args',
		default='scales-okn/docket-language-model',
		description='Default model to use for training routines',
	),

	ConfigKey(
		name='MAX_LENGTH',
		group='train-args',
		key_type='int',
		default=256,
		description='Maximum token length for input examples',
	),

	ConfigKey(
		name='EVAL_SPLIT',
		group='train-args',
		key_type='float',
		default=0.2,
		description='Fraction of training data to use for validation',
	),

	ConfigKey(
		name='EPOCHS',
		group='train-args',
		key_type='int',
		default=5,
		description='Number of training epochs',
	),

	ConfigKey(
		name='SHUFFLE',
		group='train-args',
		key_type='bool',
		default=True,
		description='Whether to shuffle training data',
	),

	ConfigKey(
		name='TRAIN_BATCH_SIZE',
		group='train-args',
		key_type='int',
		default=4,
		description='Training batch size',
	),

	ConfigKey(
		name='EVAL_BATCH_SIZE',
		group='train-args',
		key_type='int',
		default=8,
		description='Validation batch size',
	),

	ConfigKey(
		name='GRADIENT_ACCUMULATION_STEPS',
		group='train-args',
		key_type='int',
		default=4,
		description='Number of steps to accumulate gradients before performing a backward pass',
	),

	ConfigKey(
		name='LEARNING_RATE',
		group='train-args',
		key_type='float',
		default=3e-5,
		description='Learning rate',
	),

	ConfigKey(
		name='WARMUP_RATIO',
		group='train-args',
		key_type='float',
		default=0.06,
		description='Proportion of training steps to perform linear learning rate warmup for',
	),

	ConfigKey(
		name='WEIGHT_DECAY',
		group='train-args',
		key_type='float',
		default=0.01,
		description='Optimizer weight decay',
	),

	ConfigKey(
		name='SAVE_STEPS',
		group='train-args',
		key_type='int',
		default=100,
		description='Save checkpoint and evaluate model every X steps',
	),

	ConfigKey(
		name='DEVELOPER_MODE',
		group='dev',
		key_type='bool',
		default=False,
		description='Enable developer mode',
	),

	ConfigKey(
		name='LABEL_DATA_DIR',
		group='dev',
		key_type='path',
		default=None,
		description='Path to use (if different) for label data',
	),

	ConfigKey(
		name='JUDGE_DATA_DIR',
		group='dev',
		key_type='path',
		default=None,
		description='Path to use (if different) for judge data',
	),
]


config = Config(
	path=Path.home() / '.cache' / 'scales-nlp' / 'config.json',
	config_keys=keys,
	cli_command='scales-nlp configure',
)

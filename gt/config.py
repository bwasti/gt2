"""
Configuration system for GT sharding strategies.

Loads YAML configs that specify how tensors should be sharded across workers.
Supports signal-based scoping for fine-grained control.
"""

import yaml
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class ShardConfig:
    """Configuration for how to shard a tensor."""
    axis: int = 0  # Which axis to shard along
    workers: Optional[List[int]] = None  # Which worker IDs to use (None = all)
    replicated: bool = False  # If True, tensor is replicated across workers


@dataclass
class SignalConfig:
    """Configuration for a signal scope."""
    shard: Optional[ShardConfig] = None
    backward_signal: Optional[str] = None  # Name of signal for backward pass


class Config:
    """
    Global configuration manager for GT.

    Loads YAML files that specify sharding strategies for named signals.

    Example config.yaml:
    ```
    forward_layer1:
      shard:
        axis: 0
        workers: [0, 1, 2, 3]

    backward_layer1:
      shard:
        axis: 0
        workers: [0, 1, 2, 3]

    pipeline_stage_1:
      shard:
        axis: 0
        workers: [0, 1]
      backward: pipeline_stage_1_bwd

    pipeline_stage_1_bwd:
      shard:
        axis: 0
        workers: [2, 3]
    ```
    """

    def __init__(self):
        self.signals: Dict[str, SignalConfig] = {}
        self._config_file: Optional[str] = None

    def load(self, config_file: str):
        """Load configuration from YAML file."""
        self._config_file = config_file

        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file, 'r') as f:
            raw_config = yaml.safe_load(f)

        if raw_config is None:
            return

        # Parse signal configurations
        for signal_name, signal_dict in raw_config.items():
            self.signals[signal_name] = self._parse_signal_config(signal_dict)

    def _parse_signal_config(self, signal_dict: Dict[str, Any]) -> SignalConfig:
        """Parse a signal configuration from dict."""
        shard_config = None
        if 'shard' in signal_dict:
            shard_dict = signal_dict['shard']
            shard_config = ShardConfig(
                axis=shard_dict.get('axis', 0),
                workers=shard_dict.get('workers', None),
                replicated=shard_dict.get('replicated', False)
            )

        backward_signal = signal_dict.get('backward', None)

        return SignalConfig(
            shard=shard_config,
            backward_signal=backward_signal
        )

    def get_signal(self, name: str) -> Optional[SignalConfig]:
        """Get configuration for a named signal."""
        return self.signals.get(name, None)

    def has_signal(self, name: str) -> bool:
        """Check if a signal is configured."""
        return name in self.signals

    def load_from_env(self, env_var: str = 'GT_CONFIG'):
        """
        Load configuration from environment variable.

        Args:
            env_var: Environment variable name (default: GT_CONFIG)
        """
        config_path = os.environ.get(env_var, None)
        if config_path:
            print(f"GT: Loading config from {config_path}")
            self.load(config_path)

    def clear(self):
        """Clear all configuration."""
        self.signals.clear()
        self._config_file = None


# Global config instance
_config = Config()


def load_config(config_file: str):
    """Load configuration from YAML file."""
    _config.load(config_file)


def get_config() -> Config:
    """Get the global config instance."""
    return _config


def get_signal_config(name: str) -> Optional[SignalConfig]:
    """Get configuration for a named signal."""
    return _config.get_signal(name)

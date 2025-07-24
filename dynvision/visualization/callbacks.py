from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import wandb
from tqdm import tqdm

from dynvision.visualization.plot_classifier_responses import (
    plot_classifier_responses,
)
from dynvision.visualization.plot_weight_distributions import (
    plot_weight_distributions,
)
from dynvision.utils import on_same_device


class MonitorWeightDistributions(pl.Callback):
    def on_validation_end(self, trainer, model):
        state_dict = model.state_dict()

        fig, ax = plot_weight_distributions(state_dict)

        trainer.logger.experiment.log(
            {"validation/weight_distributions": wandb.Image(fig)}
        )
        return None


class MonitorClassifierResponses(pl.Callback):

    def run_one_forward_pass(self, trainer, model):
        # Run one forward pass to generate responses
        model.eval()
        sample = next(iter(trainer.val_dataloaders))

        with on_same_device(
            x_0=sample[0], **{k: v for k, v in model.named_parameters()}
        ):
            model.validation_step(sample, batch_idx=0, store_responses=True)
            model.reset()

    def clear_responses(self, model):
        if hasattr(model, "responses"):
            del model.responses
        torch.cuda.empty_cache()

    def on_validation_end(self, trainer, model):
        if hasattr(model, "get_dataframe"):
            df = model.get_dataframe()

            # if not len(df):
            #     self.run_one_forward_pass(trainer, model)
            #     df = model.get_dataframe()
            #     self.clear_responses(model)

            if len(df):
                fig, ax = plot_classifier_responses(df)

                trainer.logger.experiment.log(
                    {"validation/classifier_response": wandb.Image(fig)}
                )
        return None

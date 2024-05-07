from typing import List

import numpy as np
import tensorflow as tf
from keras.models import Model, load_model
from keras.optimizers import Adam

from seg_tgce.data.oxford_pet.disturbance.model import (
    download_base_model,
    find_last_encoder_conv_layer,
    produce_disturbed_models,
)
from seg_tgce.data.oxford_pet.oxford_pet import get_data_multiple_annotators
from seg_tgce.loss.tgce import TcgeConfig, TcgeSs
from seg_tgce.metrics.dice_coefficient import DiceCoefficient
from seg_tgce.models.unet import unet_tgce
from seg_tgce.run.oxford_ma_runner.model_result import ModelResult
from seg_tgce.run.oxford_ma_runner.plotting import (
    epoch_progress_plotter,
    plot_losses_and_metrics,
)
from seg_tgce.run.runner import (
    Runner,
    RunningSessionParams,
    SessionPartialResults,
    SessionResults,
)


def fetch_models(noise_levels_snr: list[float]) -> List[Model]:
    model_path = download_base_model()
    model_ann = load_model(model_path, compile=False)

    last_conv_encoder_layer = find_last_encoder_conv_layer(model_ann)

    disturbance_models, measured_snr_values = produce_disturbed_models(
        noise_levels_snr, model_path, last_conv_encoder_layer
    )
    print(f"Measured snr values from produced models: {measured_snr_values}")
    return disturbance_models


class OxfordMARunner(Runner):
    def __init__(self, params: RunningSessionParams) -> None:
        self.params = params

    async def run(  # pylint: disable=too-many-locals, too-many-statements
        self,
    ) -> SessionResults:
        if self.params.extra is None:
            raise ValueError("Extra parameters are required for this runner.")
        entropy_gamma_values = self.params.extra["entropy_gamma_values"]
        noise_levels_snr = self.params.extra["noise_levels_snr"]

        model_results: list[ModelResult] = []

        disturbance_models = fetch_models(noise_levels_snr)

        train, val, _ = get_data_multiple_annotators(
            annotation_models=disturbance_models,
            target_shape=self.params.target_img_shape,
            batch_size=self.params.batch_size,
        )

        train_iterable = train.cache()
        val_iterable = val.cache()

        for entropy_gamma in entropy_gamma_values:
            print(f"========== Entropy Gamma: {entropy_gamma} ==========")
            optimizer = Adam()
            model_name = f"UNET-TGCE_SS_gamma_{entropy_gamma}"
            loss_fn = TcgeSs(
                q=0.1,
                config=TcgeConfig(
                    num_annotators=self.params.num_annotators, gamma=entropy_gamma
                ),
            )
            dice_fn = DiceCoefficient()
            losses = []
            dices = []
            val_losses = []
            val_dices = []

            model = unet_tgce(
                input_shape=self.params.target_img_shape + (3,),
                out_channels=2,
                n_scorers=self.params.num_annotators,
                name=model_name,
            )
            model.compile(loss=loss_fn, metrics=dice_fn, optimizer=optimizer)

            for epoch in range(self.params.n_epochs):
                print(f"Epoch {epoch+1}/{self.params.n_epochs}")
                epoch_loss = 0.0
                epoch_dice = 0.0
                epoch_val_loss = 0.0
                epoch_val_dice = 0.0
                epoch_losses = []
                epoch_dices = []
                for x_batch, y_batch in train_iterable:
                    with tf.GradientTape() as tape:
                        predictions = model(x_batch, training=True)
                        loss = loss_fn.call(y_batch, predictions)

                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(
                        zip(gradients, model.trainable_variables)  # type:ignore
                    )

                    epoch_loss += float(loss.numpy())
                    epoch_losses.append(loss.numpy())
                    y_pred_ = predictions[..., :2]
                    y_pred_ = y_pred_[..., tf.newaxis]
                    y_pred_ = tf.repeat(
                        y_pred_, repeats=[self.params.num_annotators], axis=-1
                    )
                    dice = tf.reduce_mean(dice_fn.call(y_batch, y_pred_))
                    epoch_dice += float(dice.numpy())
                    epoch_dices.append(dice.numpy())
                losses.append(np.array(epoch_losses).mean())
                dices.append(np.array(epoch_dices).mean())

                epoch_val_losses = []
                epoch_val_dices = []

                for x_batch_val, y_batch_val in val_iterable:
                    val_predictions = model(x_batch_val, training=False)
                    val_loss = loss_fn.call(y_batch_val, val_predictions)
                    val_total_loss = val_loss

                    epoch_val_loss += float(val_total_loss.numpy())
                    y_pred_val = val_predictions[..., :2]
                    y_pred_val = y_pred_val[..., tf.newaxis]
                    y_pred_val = tf.repeat(
                        y_pred_val, repeats=[self.params.num_annotators], axis=-1
                    )
                    val_dice = tf.reduce_mean(dice_fn.call(y_batch_val, y_pred_val))
                    epoch_val_dice += float(val_dice.numpy())

                    epoch_val_losses.append(val_total_loss.numpy())
                    epoch_val_dices.append(val_dice.numpy())

                val_losses.append(np.array(epoch_val_losses).mean())
                val_dices.append(np.array(epoch_val_dices).mean())

                print(
                    f"Training Loss: {np.array(epoch_losses).mean():.3f}, "
                    f"Training Dice: {np.array(epoch_dices).mean():.3f}"
                )
                y_batch = None
                x_batch = None
                print(
                    f"Validation Loss: {np.array(epoch_val_losses).mean():.3f}, "
                    f"Validation Dice: {np.array(epoch_val_dices).mean():.3f}"
                )
                if epoch % self.params.plotting_frequency == 0:
                    epoch_progress_plotter(
                        x_batch,
                        y_batch,
                        predictions,
                        np.random.randint(0, len(x_batch) - 1),
                        noise_levels_snr,
                        self.params.num_annotators,
                    )

            model_result = ModelResult(losses, dices, val_losses, val_dices)
            file_name = f"{model_name}.json"
            model_result.save_as_json(file_name)
            model_results.append(model_result)
            trained_model_path = f"{model_name}.keras"
            model.save(trained_model_path)
            plot_losses_and_metrics(losses, dices, val_losses, val_dices, model_name)

        return SessionResults(models=[], train_metadata={})  # type:ignore

    async def stop(self) -> SessionPartialResults:
        print("Stopping!")
        return SessionPartialResults(train_metadata={})

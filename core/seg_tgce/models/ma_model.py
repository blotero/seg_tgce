from keras.models import Model
from tensorflow import GradientTape


class ModelMultipleAnnotators(
    Model
):  # pylint: disable=abstract-method, too-few-public-methods
    def train_step(self, data):
        x, y = data

        with GradientTape() as tape:
            y_pred = self(x, training=True)
            loss, _, _ = self.compute_loss(y=y, y_pred=y_pred)

        gradients = tape.gradient(loss, self.trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_vars))
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

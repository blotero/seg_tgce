from keras.models import Model
from tensorflow import GradientTape


class ModelMultipleAnnotators(
    Model
):  # pylint: disable=abstract-method, too-few-public-methods

    def train_step(self, data):
        x, y = data

        with GradientTape() as tape:
            y_pred = self(x, training=True)
            self.loss.stage = "train"
            loss = self.compute_loss(y=y, y_pred=y_pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.loss.stage = "val"
        loss = self.compute_loss(y=y, y_pred=y_pred)

        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

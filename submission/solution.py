import os

import numpy as np

from features import StreamingFeatures

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "linear_model.npz")
PREDICTION_CLIP = 10.0      # last line of defence: predictions must be finite


class PredictionModel:
    def __init__(self):
        model = np.load(MODEL_PATH)
        self.mean = model["mean"].astype(np.float64)
        self.scale = model["scale"].astype(np.float64)
        self.coef = model["coef"].astype(np.float64)            # (2, n_features)
        self.intercept = model["intercept"].astype(np.float64)  # (2,)
        self.std_clip = float(model["std_clip"])
        self.n_features = self.coef.shape[1]

        self.features = StreamingFeatures()
        self.seq_ix = None
        self._z = np.empty(self.n_features)                     # reused scratch space

    def predict(self, data_point):
        # A new sequence means the memory features start again from nothing
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self.features.reset()

        # Update state on every row, before deciding whether to answer
        full = self.features.update(data_point.state)

        if not data_point.need_prediction:
            return None

        # Same standardisation as training: centre, rescale, clip
        z = self._z
        np.subtract(full[:self.n_features], self.mean, out=z)
        np.divide(z, self.scale, out=z)
        np.clip(z, -self.std_clip, self.std_clip, out=z)

        # The model itself: one dot product
        pred = self.coef @ z
        pred += self.intercept

        # Non-finite predictions are a submission error, so check and repair.
        # Checking the two outputs is far cheaper than scrubbing 175 inputs
        # on every row, and nothing else can turn a finite input into NaN.
        if not np.isfinite(pred).all():
            np.nan_to_num(z, copy=False, nan=0.0,
                          posinf=self.std_clip, neginf=-self.std_clip)
            pred = self.coef @ z + self.intercept
            np.nan_to_num(pred, copy=False, nan=0.0,
                          posinf=PREDICTION_CLIP, neginf=-PREDICTION_CLIP)

        np.clip(pred, -PREDICTION_CLIP, PREDICTION_CLIP, out=pred)
        return pred

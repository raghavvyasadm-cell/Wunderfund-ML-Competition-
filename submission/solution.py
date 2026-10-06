import os

import numpy as np

from features import FEATURE_VERSION, N_MEMORY, N_RAW, N_STATELESS, StreamingFeatures

HERE = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(HERE, "mlp_model.onnx")
NPZ_PATH = os.path.join(HERE, "linear_model.npz")
PREDICTION_CLIP = 10.0      # last line of defence: predictions must be finite
VALID_WIDTHS = (N_RAW, N_RAW + N_STATELESS, N_RAW + N_STATELESS + N_MEMORY)


class OnnxModel:
    """A network exported to ONNX. Standardisation is baked into the graph."""

    def __init__(self, path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1      # the container has one CPU
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(path, options,
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        meta = self.session.get_modelmeta().custom_metadata_map
        version = meta.get("feature_version", "unknown")
        if version != FEATURE_VERSION:
            raise ValueError(f"model was trained on features '{version}', "
                             f"but features.py is '{FEATURE_VERSION}' - retrain")
        self.n_features = int(meta.get("n_features", VALID_WIDTHS[-1]))
        self._batch = np.empty((1, self.n_features), dtype=np.float32)

    def __call__(self, full):
        self._batch[0] = full[:self.n_features]
        return self.session.run(None, {self.input_name: self._batch})[0][0]


class LinearModel:
    """The earlier model: standardise, then one dot product."""

    def __init__(self, path):
        model = np.load(path)
        self.mean = model["mean"].astype(np.float64)
        self.scale = model["scale"].astype(np.float64)
        self.coef = model["coef"].astype(np.float64)            # (2, n_features)
        self.intercept = model["intercept"].astype(np.float64)
        self.std_clip = float(model["std_clip"])
        self.n_features = self.coef.shape[1]

        version = str(model["feature_version"]) if "feature_version" in model else "unknown"
        if version != FEATURE_VERSION:
            raise ValueError(f"model was trained on features '{version}', "
                             f"but features.py is '{FEATURE_VERSION}' - retrain")
        self._z = np.empty(self.n_features)

    def __call__(self, full):
        z = self._z
        np.subtract(full[:self.n_features], self.mean, out=z)
        np.divide(z, self.scale, out=z)
        np.clip(z, -self.std_clip, self.std_clip, out=z)
        pred = self.coef @ z
        pred += self.intercept
        if not np.isfinite(pred).all():
            np.nan_to_num(z, copy=False, nan=0.0,
                          posinf=self.std_clip, neginf=-self.std_clip)
            pred = self.coef @ z + self.intercept
        return pred


class PredictionModel:
    def __init__(self):
        if os.path.exists(ONNX_PATH):
            self.model = OnnxModel(ONNX_PATH)
        elif os.path.exists(NPZ_PATH):
            self.model = LinearModel(NPZ_PATH)
        else:
            raise FileNotFoundError("no model file found next to solution.py")
        if self.model.n_features not in VALID_WIDTHS:
            raise ValueError(f"model expects {self.model.n_features} features, "
                             f"which matches no feature set in features.py")
        self.features = StreamingFeatures()
        self.seq_ix = None

    def predict(self, data_point):
        # A new sequence means the memory features start again from nothing
        if data_point.seq_ix != self.seq_ix:
            self.seq_ix = data_point.seq_ix
            self.features.reset()

        # Update state on every row, before deciding whether to answer
        full = self.features.update(data_point.state)

        if not data_point.need_prediction:
            return None

        pred = np.asarray(self.model(full), dtype=np.float64)

        # Non-finite predictions are a submission error, so check and repair
        if not np.isfinite(pred).all():
            pred = np.nan_to_num(pred, nan=0.0, posinf=PREDICTION_CLIP,
                                 neginf=-PREDICTION_CLIP)
        return np.clip(pred, -PREDICTION_CLIP, PREDICTION_CLIP).astype(np.float32)

class DummyModel:
    def predict(self, X):
        return [15 for _ in X]

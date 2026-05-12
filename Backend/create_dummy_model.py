import pickle

class DummyModel:
    def predict(self, X):
        # Always predict the index 15 which maps to 'Fungal infection' in the app mapping
        return [15 for _ in X]

if __name__ == '__main__':
    m = DummyModel()
    with open('model/svc.pkl', 'wb') as f:
        pickle.dump(m, f)
    print('Dummy model written to model/svc.pkl')
